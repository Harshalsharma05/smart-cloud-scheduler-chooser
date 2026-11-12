"""Philly-like SYNTHETIC jobs for testing the engine. Every number below is synthetic; none goes into the paper."""
import math

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


if __name__ == "__main__":
    import random
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
