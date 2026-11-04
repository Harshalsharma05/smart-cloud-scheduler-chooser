"""Discrete-event engine: heapq of (time, priority, seq, event). Owns the clock, the queue and the main loop.
Calls the active rule set after every event to get start/rent decisions."""
import heapq
from dataclasses import replace

from sim.cloud import CloudPool
from sim.cluster import allocate, best_fit, build_nodes, release
from sim.models import DONE, RUNNING
from sim.safety import SafetyChecker

# Event kinds; the number is the order of processing at equal times (finish frees resources first).
JOB_FINISH, CLOUD_READY, JOB_ARRIVAL = 0, 1, 2

NEVER, IMMEDIATE = "NEVER", "IMMEDIATE"


def schedule_pass(queue, nodes, start_job):
    """TEMPORARY (Phase 2): strict FCFS, no backfill. Replaced by RuleSet in Phase 4."""
    queue.sort(key=lambda j: (j.arrival, j.id))
    while queue:
        node = best_fit(nodes, queue[0])
        if node is None:
            break
        start_job(queue.pop(0), node)


def burst_pass(queue, burst, request_cloud):
    """TEMPORARY (Phase 3): NEVER or IMMEDIATE for every job still queued."""
    if burst == IMMEDIATE:
        for job in list(queue):
            request_cloud(job)


def simulate(jobs, cfg, burst=NEVER):
    """Run all jobs to completion (drain). Returns (jobs, pool): copies of the jobs with
    start/end/where filled in, and the cloud pool with every instance that was rented."""
    jobs = [replace(j) for j in jobs]
    jobs_by_id = {j.id: j for j in jobs}
    nodes = build_nodes(cfg)
    nodes_by_id = {n.id: n for n in nodes}
    pool = CloudPool(cfg)
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
        schedule_pass(queue, nodes, start_job)
        burst_pass(queue, burst, request_cloud)
        safety.check(now, nodes, jobs_by_id, pool)

    assert not queue, "simulation ended with jobs still queued"
    return jobs, pool
