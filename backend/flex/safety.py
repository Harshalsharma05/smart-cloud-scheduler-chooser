"""Safety checker for the paper engine. Raises AssertionError on the first violation."""
from .models import (CLOUD_BOOTING, CLOUD_RUNNING, DONE, PENDING, QUEUED, RUNNING, SUSPENDED)

CLOUD_STATES = (CLOUD_BOOTING, CLOUD_RUNNING)


def check(engine):
    """Full check of the engine state (cost: one pass over all jobs)."""
    last = getattr(engine, "_last_check_now", 0)
    assert engine.now >= last, f"clock went back: {engine.now} < {last}"
    engine._last_check_now = engine.now

    used = sum(engine.jobs[i].gpus for i in engine.running)
    assert used + engine.free == engine.G, f"GPU accounting: used {used} + free {engine.free} != G {engine.G}"
    assert 0 <= used <= engine.G and engine.free >= 0, "pool over capacity"

    queued_ids, running_ids = set(), set()
    for j in engine.jobs.values():
        assert j.n_starts <= 1, f"{j.job_id} started twice"
        assert j.n_resumes <= j.n_suspends, f"{j.job_id} resumed more than suspended"
        if j.state == PENDING:
            assert j.n_starts == 0, f"{j.job_id} pending but started"
        elif j.state == QUEUED:
            queued_ids.add(j.job_id)
            assert j.n_starts == 0 or j.n_resumes >= 1, f"{j.job_id} queued after a start"
        elif j.state == RUNNING:
            running_ids.add(j.job_id)
            assert not j.on_cloud, f"{j.job_id} back on-prem after a burst"
            assert j.first_start is not None and j.first_start >= j.submit, f"{j.job_id} bad start"
        elif j.state == SUSPENDED:
            assert not j.on_cloud and j.n_suspends > j.n_resumes, f"{j.job_id} bad suspend"
        elif j.state == DONE:
            assert j.end is not None and j.end >= j.submit, f"{j.job_id} bad times"
            assert j.work_done == j.wall_s + j.extra_work, f"{j.job_id} work not conserved"
            _check_segments(j)
        if j.state in CLOUD_STATES:
            assert j.on_cloud and j.job_id in engine.cloud.active, f"{j.job_id} cloud state without instance"
    assert queued_ids == {j.job_id for j in engine.queue}, "queue does not match QUEUED jobs"
    assert running_ids == engine.running, "running set does not match RUNNING jobs"
    if engine.cloud is not None:
        _check_cloud(engine)


def _check_segments(j):
    """Segments are ordered, never go back on-prem after the cloud, and add up to the job's work."""
    prev_end, seen_cloud, work = -1, False, 0.0
    for start, end, where, speed in j.segments:
        assert start >= prev_end and end > start, f"{j.job_id} segments overlap or are empty"
        prev_end = end
        if where in ("boot", "cloud"):
            seen_cloud = True
        assert not (seen_cloud and where == "onprem"), f"{j.job_id} returned on-prem after a burst"
        if where in ("onprem", "cloud"):
            work += speed * (end - start)
    target = j.wall_s + j.extra_work
    # the finish time is rounded up to a whole second, so the last segment may overshoot by < 1 s of work
    assert target - 1e-6 <= work <= target + 1.0 + 1e-6, f"{j.job_id} segment work {work} != {target}"


def _check_cloud(engine):
    cloud = engine.cloud
    ids = [i.job_id for i in cloud.instances]
    assert len(ids) == len(set(ids)), "more than one cloud instance for a job"
    assert sum(i.gpus for i in cloud.active.values()) <= cloud.params.max_gpus, "cloud GPU cap exceeded"
    for inst in cloud.instances:
        j = engine.jobs[inst.job_id]
        assert j.on_cloud and inst.gpus == j.gpus, f"instance of {inst.job_id} does not match its job"
        if inst.released is not None:
            assert inst.ready is not None and inst.released >= inst.ready >= inst.requested, \
                f"instance of {inst.job_id} released before READY"
            assert j.state == DONE, f"instance of {inst.job_id} released but job not done"
        else:
            assert j.state in CLOUD_STATES and inst.job_id in cloud.active, f"{inst.job_id} instance not tracked"
            assert (inst.ready is not None) == (j.state == CLOUD_RUNNING), f"{inst.job_id} READY mismatch"
