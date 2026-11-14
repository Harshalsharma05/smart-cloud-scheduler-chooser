"""S14b: is the learned-predictor result an artifact of the lookup rule (latest grid row <= elapsed, subtract the gap,
floor at 60 s) for jobs past the last grid point? Measurement only: nothing in flex/ changes.

Default cell (10 events), mix with last2 / gbdt / lightgbm / transformer (p50) and transformer (p90 cost). For the jobs
the mix bursts, and for ALL running jobs at the event start: elapsed, share beyond 43,200 s, share at the 60 s floor, share
with the gap larger than the grid row's p50 ("exhausted"), actual remaining time, share of the cloud $.
Run from backend/:  python -m traces.s14b      (about 10 min: one baseline pass over the trace)
Output: results/s14b/jobs.csv (one row per event x predictor x running job) and the printed tables.
"""
import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from flex import cloud as cloudlib
from flex.experiments import DEFAULT_EVENT, ROOT, cell, cloud_variant, load_predictors, load_setup, run_events
from flex.models import load_config
from flex.policies import progress, run_policy, world_from_config

LAST_GRID = 43200
PREDS = ["last2", "gbdt", "lightgbm", "transformer"]
VARIANTS = [("last2", "p50"), ("gbdt", "p50"), ("lightgbm", "p50"), ("transformer", "p50"), ("transformer", "p90")]


def lookup_detail(pred, job, elapsed):
    """Looked-up p50/p90 plus the raw grid row used (last row with elapsed_s <= elapsed) and the gap."""
    entry = pred.rows.get(job.job_id)
    if entry is None:
        p10, p50, p90, _, _ = pred.lookup(job, elapsed)
        return dict(fallback=True, p50=p50, p90=p90, row_p50=np.nan, row_p90=np.nan, gap=np.nan, last_grid=np.nan)
    grid, vals = entry
    _, p50, p90, _, _ = pred.lookup(job, elapsed)
    i = max(0, np.searchsorted(grid, elapsed, side="right") - 1)
    return dict(fallback=False, p50=p50, p90=p90, row_p50=vals[i][1], row_p90=vals[i][2], gap=elapsed - grid[i], last_grid=grid[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="contract", choices=["contract", "hold_beyond_grid"])
    rule = ap.parse_args().rule
    cfg = load_config()
    cfg, jobs, pool, events = load_setup(cfg, "philly", False)
    preds = load_predictors(cfg, "philly")
    for pr in preds.values():
        pr.lookup_rule = rule
    base_world = world_from_config(cfg, preds["last2"])
    world = cloud_variant(base_world, cfg["cloud"]["price_per_gpu_h"], cfg["cloud"]["boot_delay_s"], 1000)
    rows = []

    def per_event(snap, ev, base):
        running = [snap.jobs[i] for i in snap.running]
        el = {j.job_id: progress(snap, j) for j in running}
        burst = {}
        for pname, q in VARIANTS:
            w = dataclasses.replace(world, predictor=preds[pname], cost_quantile=q)
            _, _, pol = run_policy(snap, ev, "mix", w, pname, "predictor", cfg["seed"], base, return_runs=True)
            burst[(pname, q)] = {i.job_id: cloudlib.instance_cost(pol.engine.cloud.params, i, pol.engine.now) for i in pol.engine.cloud.instances}
        for pname in PREDS:
            for j in running:
                d = lookup_detail(preds[pname], j, el[j.job_id])
                d.update(event_id=ev.event_id, predictor=pname, job_id=j.job_id, gpus=j.gpus, elapsed=el[j.job_id], status=j.status,
                         actual_rem=max(0.0, j.wall_s + j.elapsed0 - el[j.job_id]), carry=j.carry)
                for q in ("p50", "p90"):
                    b = burst.get((pname, q))
                    d[f"burst_{q}"] = None if b is None else j.job_id in b
                    d[f"cost_{q}"] = np.nan if b is None else b.get(j.job_id, 0.0)
                rows.append(d)
        return []
    run_events(cfg, jobs, pool, world, cell(events, *DEFAULT_EVENT), per_event, "s14b")
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "s14b" / ("" if rule == "contract" else "hold")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "jobs.csv", index=False)
    tables(df, out)


def shares(d, w=None):
    """Job-count shares for one set of rows (needs the columns of lookup_detail)."""
    beyond = d["elapsed"] > LAST_GRID
    floor = (d["p50"] == 60) & ~d["fallback"]
    exh = (d["gap"] > d["row_p50"]) & ~d["fallback"]
    return beyond, floor, exh


def tables(df, out):
    pd.set_option("display.width", 220, "display.max_columns", 40)
    res = []
    for pname, q in VARIANTS:
        d = df[df["predictor"] == pname]
        for label, sub in (("burst", d[d[f"burst_{q}"] == True]), ("all running", d)):  # noqa: E712
            beyond, floor, exh = shares(sub)
            r = dict(variant=f"{pname}-{q}", set=label, n=len(sub), elapsed_med_h=sub["elapsed"].median() / 3600, beyond_43200=beyond.mean(),
                     floor60=floor.mean(), exhausted=exh.mean(), fallback=sub["fallback"].mean(),
                     floor_gpu_share=sub.loc[floor, "gpus"].sum() / max(1, sub["gpus"].sum()),
                     actual_rem_med_h=sub["actual_rem"].median() / 3600, actual_rem_p90_h=sub["actual_rem"].quantile(.9) / 3600)
            if label == "burst":
                c = sub[f"cost_{q}"]
                tot = c.sum()
                r.update(cloud_usd=tot, usd_beyond=c[beyond].sum() / tot if tot else np.nan, usd_floor=c[floor].sum() / tot if tot else np.nan,
                         usd_exhausted=c[exh].sum() / tot if tot else np.nan, usd_fallback=c[sub["fallback"]].sum() / tot if tot else np.nan)
            res.append(r)
    t = pd.DataFrame(res)
    print(t.round(3).to_string(index=False))
    t.to_csv(out / "tables.csv", index=False)
    print("\nJobs beyond the last grid point: p50 / p90 at their last grid row (median, h) and the looked-up p50 (median, h)")
    rows = []
    for pname in PREDS:
        d = df[(df["predictor"] == pname) & (df["elapsed"] > LAST_GRID) & ~df["fallback"]]
        for label, sub in (("all running", d), ("burst (mix p50)", d[d["burst_p50"] == True])):  # noqa: E712
            rows.append(dict(predictor=pname, set=label, n=len(sub), row_p50_h=sub["row_p50"].median() / 3600, row_p90_h=sub["row_p90"].median() / 3600,
                             looked_up_p50_h=sub["p50"].median() / 3600, actual_rem_h=sub["actual_rem"].median() / 3600))
    t5 = pd.DataFrame(rows)
    print(t5.round(2).to_string(index=False))
    t5.to_csv(out / "beyond_grid.csv", index=False)


if __name__ == "__main__":
    main()
