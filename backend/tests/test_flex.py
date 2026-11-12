"""Tests for the paper engine (backend/flex)."""
import random

from flex.engine import Engine
from flex.models import FJob, load_config
from traces.synthetic import generate

H = 3600


def job(jid, gpus, hours, submit=0):
    return FJob(job_id=jid, user="u", submit=submit, gpus=gpus, wall_s=int(hours * H), status="Pass")


def starts(eng):
    return {j.job_id: j.first_start for j in eng.jobs.values()}


def test_fcfs_and_skip_ahead():
    # FCFS: A fills the pool, B and C wait until A ends.
    eng = Engine([job("A", 2, 10), job("B", 1, 1), job("C", 1, 1)], G=2, check_every=1)
    eng.run_to_drain()
    assert starts(eng) == {"A": 0, "B": 10 * H, "C": 10 * H}
    # Skip-ahead: B (2 GPUs) does not fit, so C (1 GPU) jumps ahead.
    eng = Engine([job("A", 1, 10), job("B", 2, 1), job("C", 1, 1)], G=2, check_every=1)
    eng.run_to_drain()
    assert starts(eng) == {"A": 0, "B": 10 * H, "C": 0}


def test_synthetic_day_is_deterministic_and_safe():
    cfg = load_config()
    runs = []
    for _ in range(2):
        jobs = generate(random.Random(cfg["seed"]), days=1)
        eng = Engine(jobs, cfg["synthetic"]["pool_gpus"], check_every=1)
        eng.run_to_drain()
        runs.append([(j.job_id, j.first_start, j.end) for j in eng.jobs.values()])
    assert len(runs[0]) > 100 and runs[0] == runs[1]
    assert all(end is not None for _, _, end in runs[0])
