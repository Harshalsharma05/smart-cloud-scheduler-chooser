"""Safety checker for the paper engine. Raises AssertionError on the first violation."""
from .models import DONE, PENDING, QUEUED, RUNNING


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
        if j.state == PENDING:
            assert j.n_starts == 0, f"{j.job_id} pending but started"
        elif j.state == QUEUED:
            queued_ids.add(j.job_id)
            assert j.n_starts == 0, f"{j.job_id} queued after a start"
        elif j.state == RUNNING:
            running_ids.add(j.job_id)
            assert j.first_start is not None and j.first_start >= j.submit, f"{j.job_id} bad start"
        elif j.state == DONE:
            assert j.end is not None and j.end >= j.first_start >= j.submit, f"{j.job_id} bad times"
            assert j.work_done == j.wall_s + j.extra_work, f"{j.job_id} work not conserved"
    assert queued_ids == {j.job_id for j in engine.queue}, "queue does not match QUEUED jobs"
    assert running_ids == engine.running, "running set does not match RUNNING jobs"
