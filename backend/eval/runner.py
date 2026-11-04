"""Runs every rule set on the same scenario+seed and returns a comparison. CLI:
python -m eval.runner --scenario rush --seed 1"""
import argparse

from eval.metrics import run_metrics, score
from rules.library import RULESETS
from sim.engine import simulate
from sim.models import load_config
from workloads.generator import generate
from workloads.scenarios import SCENARIOS

MISS_REF, SLOWDOWN_REF, COST_REF = "B2", "B2", "B5"


def timeline(jobs, pool) -> dict:
    end = max(j.end for j in jobs)
    return {
        "jobs": [{"id": j.id, "start": j.start, "end": j.end,
                  "where": j.instance_id if j.on_cloud else j.node_id, "on_cloud": j.on_cloud,
                  "late": j.deadline is not None and j.end > j.deadline,
                  "arrival": j.arrival, "deadline": j.deadline, "est_runtime": j.est_runtime,
                  "cores": j.cores, "mem_gb": j.mem_gb, "gpus": j.gpus} for j in jobs],
        "cloud": [{"id": i.id, "type": i.type.name, "requested": i.requested_at,
                   "ready": i.ready_at, "released": i.released_at, "job_id": i.job_id,
                   "price_per_h": i.type.price_per_h, "cost": round(pool.cost(i, end), 2)}
                  for i in pool.instances.values()],
    }


def run_comparison(scenario, seed, rulesets=None, timeline_for=None, profile=None, cfg=None) -> dict:
    """Run the rule sets on one scenario+seed. Returns the POST /run response shape."""
    cfg = cfg or load_config()
    profile = profile or cfg["score"]["default_profile"]
    weights = cfg["score"]["profiles"][profile]
    rulesets = rulesets or list(RULESETS)
    timeline_for = timeline_for or ("B4" if "B4" in rulesets else rulesets[0])
    jobs = generate(scenario, seed)

    runs, metrics = {}, {}
    for rid in dict.fromkeys(rulesets + [MISS_REF, SLOWDOWN_REF, COST_REF, timeline_for]):  # ordered, no repeats
        runs[rid] = simulate(jobs, cfg, RULESETS[rid])
        metrics[rid] = run_metrics(*runs[rid], cfg)
    miss_ref = max(metrics[MISS_REF]["miss_rate"], cfg["score"]["miss_ref_floor"])
    slowdown_ref = metrics[SLOWDOWN_REF]["avg_bounded_slowdown"]
    cost_ref = metrics[COST_REF]["cloud_cost"]

    results = []
    for rid in rulesets:
        m = metrics[rid]
        results.append({
            "ruleset": rid,
            "late_pct": round(100 * m["miss_rate"], 1),
            "avg_wait_min": round(m["avg_wait_min"], 1),
            "avg_bounded_slowdown": round(m["avg_bounded_slowdown"], 2),
            "cloud_cost": round(m["cloud_cost"], 1),
            "cloud_instance_hours": round(m["cloud_instance_hours"], 2),
            "score": round(score(m, miss_ref, slowdown_ref, cost_ref, weights), 3),
        })
    return {"scenario": scenario, "seed": seed, "profile": profile, "n_jobs": len(jobs), "results": results,
            "timeline": {"ruleset": timeline_for, **timeline(*runs[timeline_for])}}


def print_table(out):
    print(f"scenario={out['scenario']} seed={out['seed']} profile={out['profile']} jobs={out['n_jobs']}")
    cols = ["late_pct", "avg_wait_min", "avg_bounded_slowdown", "cloud_cost", "cloud_instance_hours", "score"]
    heads = ["late %", "wait min", "slowdown", "cost", "inst h", "score J"]
    print(f"{'rule':5}" + "".join(f"{h:>10}" for h in heads))
    for r in out["results"]:
        print(f"{r['ruleset']:5}" + "".join(f"{r[c]:>10}" for c in cols))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="rush", choices=list(SCENARIOS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rulesets", nargs="+", default=list(RULESETS), choices=list(RULESETS))
    ap.add_argument("--profile", default=None, choices=list(load_config()["score"]["profiles"]))
    args = ap.parse_args()
    print_table(run_comparison(args.scenario, args.seed, args.rulesets, profile=args.profile))
