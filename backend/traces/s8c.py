"""S8c: measurement-only diagnostics on the Philly baseline (python -m traces.s8c, from backend/). Changes no model logic."""
import copy
import dataclasses

import numpy as np
import pandas as pd

from flex.cloud import Cloud
from flex.engine import Engine
from flex.models import load_config
from flex.paired import baseline_snapshots, run_base
from flex.policies import run_policy, world_from_config
from flex.predictions import load_predictions
from traces.philly import ROOT, _busy_per_minute, load_philly


def event_setup(L, cfg, world):
    ev = [e for e in L["events"] if e.duration_h == 2 and abs(e.cut - 0.2) < 1e-9][0]
    eng = Engine(copy.deepcopy(L["jobs"]), L["G"], check_every=cfg["check_every"], cloud=Cloud(world.cloud))
    (_, snap), = baseline_snapshots(eng, [ev.t_start])
    return ev, snap


def burst_diagnostic(L, cfg, oracle):
    world = world_from_config(cfg, oracle)
    ev, snap = event_setup(L, cfg, world)
    row, base, pol = run_policy(snap, ev, "burst", world, "oracle", "predictor", cfg["seed"], return_runs=True)
    insts = pol.engine.cloud.instances
    rows = []
    for inst in insts:
        j = pol.engine.jobs[inst.job_id]
        rows.append(dict(job_id=j.job_id, gpus=j.gpus, remaining_h=(j.burst_rem or 0) / 3600.0, status=j.status,
                         carry=j.carry, warmup=j.warmup,
                         billed_gpu_h=inst.gpus * ((inst.released if inst.released is not None else pol.engine.now) - inst.ready) / 3600.0
                         if inst.ready is not None else 0.0))
    df = pd.DataFrame(rows)
    print(f"\n1. burst-only, default event {ev.event_id}: {len(insts)} instances, {df.job_id.nunique()} jobs, "
          f"cloud_usd {row['cloud_usd']:.0f}, cloud_gpu_h {row['cloud_gpu_h']:.1f}, price {world.cloud.price_per_gpu_h}/GPU-h")
    print(f"   GPUs per burst job: total {df.gpus.sum()}, median {df.gpus.median():.0f}, max {df.gpus.max()}")
    print(f"   carry-over: {int(df.carry.sum())} of {len(df)}; warm-up (incl. carry): {int(df.warmup.sum())}")
    print(f"   remaining wall > 24 h: {int((df.remaining_h > 24).sum())} jobs, {df[df.remaining_h > 24].billed_gpu_h.sum():.0f} GPU-h of "
          f"{df.billed_gpu_h.sum():.0f}; > 7 d: {int((df.remaining_h > 168).sum())}")
    print(f"   remaining wall (h) quantiles 10/50/90/max: {np.percentile(df.remaining_h, [10, 50, 90]).round(1)} {df.remaining_h.max():.0f}")
    print("   by status:", df.groupby("status").agg(n=("gpus", "size"), gpu_h=("billed_gpu_h", "sum")).round(0).to_dict("index"))
    return df


def capacity_sensitivity(L, cfg):
    ts, te = L["test_start"], L["test_end"]
    out = []
    for f in (1.0, 0.95, 0.9, 0.85):
        eng = Engine(copy.deepcopy(L["jobs"]), int(round(f * L["G"])), check_every=cfg["check_every"])
        eng.run_to_drain()
        segs = [(s, e, j.gpus) for j in eng.jobs.values() for s, e, w, _ in j.segments if w == "onprem"]
        busy = _busy_per_minute([x[0] for x in segs], [x[1] for x in segs], [x[2] for x in segs], ts, te)
        w = np.array([j.first_start - j.submit for j in eng.jobs.values() if not j.warmup], dtype=float)
        out.append(dict(pool_factor=f, pool_gpus=eng.G, busy_gpus_mean=busy.mean(), wait_mean_min=w.mean() / 60,
                        wait_p50_min=np.percentile(w, 50) / 60, wait_p90_min=np.percentile(w, 90) / 60, frac_wait_gt_60s=(w > 60).mean()))
    df = pd.DataFrame(out).round(3)
    path = ROOT / "results" / "philly_baseline"
    path.mkdir(parents=True, exist_ok=True)
    df.to_csv(path / "capacity_sensitivity.csv", index=False)
    print("\n2. capacity sensitivity (baseline only; the engine uses its G for scheduling AND idle/host power, power unused here):")
    print(df.to_string(index=False))


def long_carry(L, cfg):
    raw = pd.read_parquet(ROOT / cfg["paths"]["data"] / "processed" / "philly_jobs.parquet")
    e0 = L["epoch0"]
    d = raw[~raw["never_started"] & (raw["num_gpus"] > 0)]
    d = d[(d["submit_time"] < e0) & (d["first_start_time"] < e0)]
    end = d["first_start_time"] + d["wall_s"].where(~d["censored"], L["last_end"] - d["first_start_time"])
    long = d[(end - e0 > 14 * 86400) & (end > e0)]
    print(f"\n3. carry-over with > 14 days left: {len(long)} jobs, {int(long.num_gpus.sum())} GPUs "
          f"(of {sum(j.gpus for j in L['jobs'] if j.carry)} carried), censored (end = trace end): {int(long.censored.sum())}")


def policy_table(L, cfg, oracle, last2):
    print("\n4. default event, delay and mix:")
    cols = ["policy", "predictor", "shed_mwh", "compliance", "usd_per_mwh", "cost_usd", "cloud_usd", "n_suspended", "n_burst",
            "dgpuh_good", "dgpuh_all", "missed_deadlines"]
    rows = []
    for label, pred, name, wf in (("last2", last2, "last2", 0), ("oracle w_fail=1", oracle, "oracle", 1)):
        world = world_from_config(cfg, pred)
        world = dataclasses.replace(world, cost=dataclasses.replace(world.cost, w_fail=wf))
        ev, snap = event_setup(L, cfg, world)
        base = run_base(snap, ev, world.params, world.h_after_s)
        for p in ("delay", "mix"):
            r = run_policy(snap, ev, p, world, name, "predictor", cfg["seed"], base)
            rows.append(dict(setting=label, **{c: r[c] for c in cols if c != "predictor"}))
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def main():
    cfg = load_config()
    L = load_philly(cfg)
    oracle = load_predictions(ROOT / cfg["paths"]["predictions"] / "oracle.csv")
    last2 = load_predictions(ROOT / cfg["paths"]["predictions"] / "last2.csv")
    burst_diagnostic(L, cfg, oracle)
    capacity_sensitivity(L, cfg)
    long_carry(L, cfg)
    policy_table(L, cfg, oracle, last2)


if __name__ == "__main__":
    main()
