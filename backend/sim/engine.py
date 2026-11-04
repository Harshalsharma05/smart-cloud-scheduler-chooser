"""Discrete-event engine: heapq of (time, priority, seq, event). Owns the clock, the queue and the main loop.
Calls the active rule set after every event to get start/rent decisions."""
import heapq
from dataclasses import replace

from sim.cluster import allocate, best_fit, build_nodes, release
from sim.models import DONE, RUNNING
from sim.safety import SafetyChecker

# Event kinds; the number is the order of processing at equal times (finish frees resources first).
JOB_FINISH, CLOUD_READY, JOB_ARRIVAL = 0, 1, 2


def schedule_pass(now, queue, nodes, start_job):
    """TEMPORARY (Phase 2): strict FCFS, no backfill, no cloud. Replaced by RuleSet in Phase 4."""
    queue.sort(key=lambda j: (j.arrival, j.id))
    while queue:
        node = best_fit(nodes, queue[0])
        if node is None:
            break
        start_job(queue.pop(0), node)


def simulate(jobs, cfg):
    """Run all jobs to completion (drain). Returns copies of the jobs with start/end/node_id filled in."""
    jobs = [replace(j) for j in jobs]
    jobs_by_id = {j.id: j for j in jobs}
    nodes = build_nodes(cfg)
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
        push(now + job.true_runtime, JOB_FINISH, (job.id, node))

    for j in jobs:
        push(j.arrival, JOB_ARRIVAL, j.id)

    while events:
        now, kind, _, payload = heapq.heappop(events)
        if kind == JOB_ARRIVAL:
            queue.append(jobs_by_id[payload])
        elif kind == JOB_FINISH:
            job_id, node = payload
            job = jobs_by_id[job_id]
            release(node, job)
            job.state, job.end = DONE, now
        # CLOUD_READY: no-op until Phase 3
        schedule_pass(now, queue, nodes, start_job)
        safety.check(now, nodes, jobs_by_id)

    assert not queue, "simulation ended with jobs still queued"
    return jobs
