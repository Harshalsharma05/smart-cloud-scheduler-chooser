"""Seeded synthetic job generator (random.Random(seed) only, no global random)."""
import argparse
import math
import random

from sim.models import Job
from workloads.scenarios import HORIZON_S, SCENARIOS

CPU_CORES = [1, 2, 4, 8, 16, 32, 64]
CPU_WEIGHTS = [25, 20, 20, 15, 10, 7, 3]
MEDIAN_RUNTIME_S = 1800
SIGMA = 1.0                     # log-normal shape
MAX_RUNTIME_S = 12 * 3600


def _rate_per_hour(scn, t):
    hour = t / 3600
    for start, end, rate in scn["rates"]:
        if start <= hour < end:
            return rate
    return 0


def generate(scenario: str, seed: int) -> list[Job]:
    """Return the jobs of one scenario, sorted by arrival, ids 1..n."""
    scn = SCENARIOS[scenario]
    rng = random.Random(seed)
    jobs, t = [], 0.0
    while True:
        t += rng.expovariate(_rate_per_hour(scn, t) / 3600)
        if t >= HORIZON_S:
            break
        if rng.random() < scn["gpu_frac"]:
            gpus = rng.randint(1, 4)
            cores, mem = 8 * gpus, 32 * gpus
        else:
            gpus = 0
            cores = rng.choices(CPU_CORES, CPU_WEIGHTS)[0]
            mem = 2 * cores
        true_rt = int(min(MAX_RUNTIME_S, max(60, rng.lognormvariate(math.log(MEDIAN_RUNTIME_S), SIGMA))))
        est_rt = int(true_rt * rng.uniform(1.0, 3.0))
        deadline = None
        if rng.random() < scn["deadline_frac"]:
            deadline = int(t) + int(rng.uniform(*scn["slack"]) * true_rt)
        n = len(jobs) + 1
        jobs.append(Job(id=n, arrival=int(t), cores=cores, mem_gb=mem, gpus=gpus,
                        true_runtime=true_rt, est_runtime=est_rt, deadline=deadline,
                        owner=f"user{rng.randint(1, 10)}", name=f"job{n}"))
    return jobs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="rush", choices=list(SCENARIOS))
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    jobs = generate(args.scenario, args.seed)
    print(f"scenario={args.scenario} seed={args.seed} jobs={len(jobs)}")
    for j in jobs[:10]:
        print(f"id={j.id} t={j.arrival} cores={j.cores} mem={j.mem_gb} gpus={j.gpus} "
              f"true={j.true_runtime} est={j.est_runtime} deadline={j.deadline}")
