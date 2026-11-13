"""Philly-like SYNTHETIC jobs for testing the engine. Every number below is synthetic; none goes into the paper."""
import bisect
import math
import random
import statistics
from pathlib import Path

import pandas as pd

from flex.models import FJob

# All constants are SYNTHETIC (chosen to look roughly like a shared GPU cluster, not fitted to data).
GPU_SIZES = [1, 2, 4, 8, 16]                  # synthetic
GPU_WEIGHTS = [0.60, 0.10, 0.15, 0.10, 0.05]  # synthetic
MEDIAN_WALL_S = 3600                          # synthetic: median 1 h
SIGMA = 1.4                                   # synthetic: lognormal shape
MAX_WALL_S = 3 * 86400                        # synthetic: cap 3 days
MIN_WALL_S = 60                               # synthetic
JOBS_PER_HOUR = 7.0                           # synthetic: mean arrival rate (about 70% load on 64 GPUs)
DIURNAL_AMP = 0.5                             # synthetic: day/night swing of the arrival rate
PEAK_HOUR = 15                                # synthetic
N_USERS = 50                                  # synthetic
FAIL_RATE_BETA = (1.5, 8.5)                   # synthetic: per-user failure rate ~ Beta, mean 0.15
KILLED_PROB = 0.15                            # synthetic
FAILED_LEN_FACTOR = 0.25                      # synthetic: Failed jobs are shorter
UTIL_BETA = (2.0, 3.0)                        # synthetic: mean GPU utilisation ~ Beta
GRID = [0, 60, 600, 3600, 14400, 43200]       # elapsed grid of the prediction contract (same as Track 1)
HISTORY = 32                                  # max history jobs per user (same as Track 1)
FLOOR_S = 60                                  # floor of every exported remaining runtime (same as Track 1)


def generate(rng, days=1.0, n_jobs=None):
    """Seeded synthetic jobs. Stops after `days` of arrivals, or after n_jobs jobs if given."""
    fail_rate = [rng.betavariate(*FAIL_RATE_BETA) for _ in range(N_USERS)]
    jobs, t, i = [], 0.0, 0
    while n_jobs is None or i < n_jobs:
        hour = (t / 3600.0) % 24
        rate = JOBS_PER_HOUR * (1 + DIURNAL_AMP * math.cos(2 * math.pi * (hour - PEAK_HOUR) / 24)) / 3600.0
        t += rng.expovariate(rate)
        if n_jobs is None and t >= days * 86400:
            break
        u = rng.randrange(N_USERS)
        wall = rng.lognormvariate(math.log(MEDIAN_WALL_S), SIGMA)
        x = rng.random()
        if x < fail_rate[u]:
            status, wall = "Failed", wall * FAILED_LEN_FACTOR
        elif x < fail_rate[u] + KILLED_PROB:
            status, wall = "Killed", wall * rng.uniform(0.2, 1.0)
        else:
            status = "Pass"
        wall = int(min(max(wall, MIN_WALL_S), MAX_WALL_S))
        jobs.append(FJob(job_id=f"s{i:06d}", user=f"u{u:02d}", submit=int(t),
                         gpus=rng.choices(GPU_SIZES, GPU_WEIGHTS)[0], wall_s=wall, status=status,
                         util=rng.betavariate(*UTIL_BETA)))
        i += 1
    return jobs


def assign_deadlines(jobs, rng, cfg):
    """D12: a seeded random share of jobs gets deadline = submit + slack x wall_s, slack ~ U(slack_min, slack_max),
    independent of status. cfg = flex.yaml `deadlines:`. Both draws are made for every job so the stream is stable."""
    for j in jobs:
        hit, slack = rng.random() < cfg["fraction"], rng.uniform(cfg["slack_min"], cfg["slack_max"])
        j.deadline = int(j.submit + slack * j.wall_s) if hit else None
    return jobs


def synthetic_jobs(cfg):
    """The synthetic trace the experiment runner uses: `days` of arrivals plus deadlines. Separate seeded streams."""
    jobs = generate(random.Random(cfg["seed"]), days=cfg["synthetic"]["days"])
    return assign_deadlines(jobs, random.Random(cfg["seed"] + 1), cfg["deadlines"])


def write_predictions(jobs, out_dir):
    """Write synthetic oracle.csv and last2.csv with the Track 1 rules. `jobs` must have been simulated
    (first_start and end set): history = the user's jobs finished by the observation time, newest last, at most 32.
    t_obs = submit at elapsed 0, else first_start + elapsed. Rows exist only while e < wall_s.
    Users with no history get the global median wall_s and global fail/cancel rates (synthetic has no train split)."""
    assert all(j.first_start is not None and j.end is not None for j in jobs), "simulate the jobs first"
    median_wall = statistics.median(j.wall_s for j in jobs)
    glob_fail = sum(j.status == "Failed" for j in jobs) / len(jobs)
    glob_cancel = sum(j.status == "Killed" for j in jobs) / len(jobs)
    by_user = {}
    for j in sorted(jobs, key=lambda j: (j.end, j.job_id)):
        by_user.setdefault(j.user, []).append(j)
    ends = {u: [j.end for j in js] for u, js in by_user.items()}
    oracle_rows, last2_rows = [], []
    for j in sorted(jobs, key=lambda j: j.job_id):
        for e in GRID:
            if e >= j.wall_s:
                break
            rt = float(max(FLOOR_S, j.wall_s - e))
            oracle_rows.append((j.job_id, e, rt, rt, rt, float(j.status == "Failed"), float(j.status == "Killed")))
            t_obs = j.submit if e == 0 else j.first_start + e
            n_done = bisect.bisect_right(ends[j.user], t_obs)
            hist = by_user[j.user][max(0, n_done - HISTORY):n_done]
            if hist:
                est = statistics.fmean(h.wall_s for h in hist[-2:])
                p_fail = sum(h.status == "Failed" for h in hist) / len(hist)
                p_cancel = sum(h.status == "Killed" for h in hist) / len(hist)
            else:
                est, p_fail, p_cancel = median_wall, glob_fail, glob_cancel
            r = max(float(FLOOR_S), est - e)
            last2_rows.append((j.job_id, e, r, r, r, p_fail, p_cancel))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["job_id", "elapsed_s", "rt_p10_s", "rt_p50_s", "rt_p90_s", "p_fail", "p_cancel"]
    for name, rows in (("oracle", oracle_rows), ("last2", last2_rows)):
        pd.DataFrame(rows, columns=cols).to_csv(out_dir / f"{name}.csv", index=False)
    return out_dir


if __name__ == "__main__":
    import random
    import sys
    import time

    from flex.engine import Engine
    from flex.models import load_config

    cfg = load_config()
    jobs = generate(random.Random(cfg["seed"]), n_jobs=20000)
    eng = Engine(jobs, cfg["synthetic"]["pool_gpus"], check_every=cfg["check_every"])
    t0 = time.perf_counter()
    eng.run_to_drain()
    print(f"{len(jobs)} synthetic jobs, G={eng.G}, {eng.n_events} events, simulated {eng.now / 86400:.0f} days, "
          f"run time {time.perf_counter() - t0:.2f} s")
    if "--predictions" in sys.argv:      # python -m traces.synthetic --predictions
        exp_jobs = synthetic_jobs(cfg)   # the trace the experiment runner uses, simulated without events
        base = Engine(exp_jobs, cfg["synthetic"]["pool_gpus"], check_every=cfg["check_every"])
        base.run_to_drain()
        out = write_predictions(list(base.jobs.values()),
                                Path(__file__).resolve().parents[2] / cfg["paths"]["data"] / "synthetic" / "predictions")
        print(f"wrote synthetic oracle.csv and last2.csv for {len(exp_jobs)} jobs to", out)
