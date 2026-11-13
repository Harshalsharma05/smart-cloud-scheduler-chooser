"""Tests for the paper engine (backend/flex)."""
import random

from flex.cloud import Cloud, CloudParams, total_cost
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


def test_cap_and_suspend_timing():
    half = lambda j, f: 0.5          # speed function for the test (the real curve comes in S3/S11)
    # Cap 0.5 from 1 h to 3 h: 1 h + 2 h at half speed (1 h of work) + 8 h left -> ends at 11 h.
    eng = Engine([job("A", 1, 10)], G=1, check_every=1)
    eng.run_until(1 * H)
    eng.set_cap(0.5, half)
    eng.run_until(3 * H)
    eng.set_cap(1.0)
    eng.run_to_drain()
    assert eng.jobs["A"].end == 11 * H
    # Suspended from 1 h to 3 h: ends at 12 h, plus restore_s if given.
    for restore in (0, 90):
        eng = Engine([job("A", 1, 10)], G=1, check_every=1)
        eng.run_until(1 * H)
        eng.suspend(eng.jobs["A"])
        eng.run_until(3 * H)
        eng.resume(eng.jobs["A"], restore_s=restore)
        eng.run_to_drain()
        assert eng.jobs["A"].end == 12 * H + restore


def test_burst_running_job_is_billed_from_ready():
    params = CloudParams(price_per_gpu_h=3.0, boot_delay_s=300, migrate_s=120, min_billed_s=60, max_gpus=1000)
    eng = Engine([job("A", 2, 2)], G=2, check_every=1, cloud=Cloud(params))
    eng.run_until(1 * H)             # 1 h of work remains
    assert eng.burst(eng.jobs["A"])
    assert eng.free == 2             # on-prem GPUs freed at once
    eng.run_to_drain()
    assert eng.jobs["A"].end == 1 * H + 300 + 3600 + 120
    assert abs(total_cost(eng.cloud, eng.now) - 2 * 3.0 * 3720 / 3600) < 1e-9
