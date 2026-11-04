"""Discrete-event engine: heapq of (time, priority, seq, event). Owns the clock, the queue and the main loop.
Calls the active rule set after every event to get start/rent decisions."""
import heapq
import math
from dataclasses import replace

from rules.base import DEADLINE_AWARE, EASY, IMMEDIATE, sort_key
from sim.cloud import CloudPool
from sim.cluster import allocate, best_fit, build_nodes, release
from sim.models import DONE, RUNNING
from sim.safety import SafetyChecker

# Event kinds; the number is the order of processing at equal times (finish frees resources first).
JOB_FINISH, CLOUD_READY, JOB_ARRIVAL = 0, 1, 2


def could_hold(node, job):
    """True if the node's total capacity is big enough for the job."""
    return node.cores >= job.cores and node.mem_gb >= job.mem_gb and node.gpus >= job.gpus


def reservation(now, job, nodes, jobs_by_id):
    """Earliest (node, time) the job would fit, freeing running jobs by estimated end. (None, inf) if never."""
    best_node, best_time = None, math.inf
    for n in nodes:
        if not could_hold(n, job):
            continue
        cores, mem, gpus, t = n.free_cores, n.free_mem, n.free_gpus, now
        running = sorted((jobs_by_id[i] for i in n.running),
                         key=lambda r: (max(now, r.start + r.est_runtime), r.id))
        for r in running:
            if cores >= job.cores and mem >= job.mem_gb and gpus >= job.gpus:
                break
            cores, mem, gpus = cores + r.cores, mem + r.mem_gb, gpus + r.gpus
            t = max(now, r.start + r.est_runtime)
        if t < best_time:
            best_node, best_time = n, t
    return best_node, best_time


def schedule_pass(now, queue, nodes, jobs_by_id, ruleset, margin, start_job, request_cloud):
    """One generic pass: order, start from the front, EASY backfill, then burst."""
    queue.sort(key=sort_key(ruleset.order))
    while queue:
        node = best_fit(nodes, queue[0])
        if node is None:
            break
        start_job(queue.pop(0), node)
    if not queue:
        return

    front = queue[0]
    reserved_node, reserved_time = reservation(now, front, nodes, jobs_by_id)

    if ruleset.backfill == EASY:
        for job in queue[1:]:
            if now + job.est_runtime <= reserved_time:
                node = best_fit(nodes, job)
            else:
                node = best_fit([n for n in nodes if n is not reserved_node], job)
            if node is not None:
                queue.remove(job)
                start_job(job, node)

    if ruleset.burst == IMMEDIATE:
        for job in list(queue):
            request_cloud(job)
    elif ruleset.burst == DEADLINE_AWARE:
        work_ahead = 0
        for job in list(queue):
            if job is front:
                projected = reserved_time
            else:
                n_fit = sum(1 for n in nodes if could_hold(n, job))
                projected = now + work_ahead / n_fit if n_fit else math.inf
            if job.deadline is not None and projected + job.est_runtime > job.deadline - margin:
                request_cloud(job)
            if job in queue:
                work_ahead += job.est_runtime


def simulate(jobs, cfg, ruleset):
    """Run all jobs to completion (drain) under one rule set. Returns (jobs, pool): copies of the
    jobs with start/end/where filled in, and the cloud pool with every instance that was rented."""
    jobs = [replace(j) for j in jobs]
    jobs_by_id = {j.id: j for j in jobs}
    nodes = build_nodes(cfg)
    nodes_by_id = {n.id: n for n in nodes}
    pool = CloudPool(cfg)
    margin = cfg["burst"]["deadline_margin_s"]
    safety = SafetyChecker()
    events, seq, queue, now = [], 0, [], 0

    def push(time, kind, payload):
        nonlocal seq
        heapq.heappush(events, (time, kind, seq, payload))
        seq += 1

    def start_job(job, node):
        safety.job_started(job)
        allocate(node, job)
        job.state, job.start, job.node_id = RUNNING, now, node.id
        push(now + job.true_runtime, JOB_FINISH, job.id)

    def request_cloud(job):
        """Rent an instance for a queued job; the job leaves the cluster queue for good."""
        inst = pool.request(job, now)
        if inst is not None:
            queue.remove(job)
            push(now + pool.boot_delay, CLOUD_READY, inst.id)

    for j in jobs:
        push(j.arrival, JOB_ARRIVAL, j.id)

    while events:
        now, kind, _, payload = heapq.heappop(events)
        if kind == JOB_ARRIVAL:
            queue.append(jobs_by_id[payload])
        elif kind == CLOUD_READY:
            inst = pool.instances[payload]
            pool.mark_ready(inst, now)
            job = jobs_by_id[inst.job_id]
            safety.job_started(job)
            job.state, job.start, job.on_cloud, job.instance_id = RUNNING, now, True, inst.id
            push(now + job.true_runtime, JOB_FINISH, job.id)
        elif kind == JOB_FINISH:
            job = jobs_by_id[payload]
            if job.on_cloud:
                pool.release(pool.instances[job.instance_id], now)
            else:
                release(nodes_by_id[job.node_id], job)
            job.state, job.end = DONE, now
        # Schedule once all events at this timestamp are in (e.g. several jobs arriving at t=0).
        if not events or events[0][0] > now:
            schedule_pass(now, queue, nodes, jobs_by_id, ruleset, margin, start_job, request_cloud)
        safety.check(now, nodes, jobs_by_id, pool)

    assert not queue, "simulation ended with jobs still queued"
    return jobs, pool
