"""S12b: measurement-only diagnostics of the S12 results (python -m traces.s12b, from backend/). No model change, nothing tuned.

1. burst-only (last2, as in E1): peak cloud GPUs (booting + running) per event, how often the cloud GPU cap refused a
   burst (cloud.can_request is wrapped here only to count refusals); three 30% events re-run with max_cloud_gpus = 3000.
2. 30% events 8 and 9: lowest facility power if every running job were suspended, against the target P*.
3. mix (last2) on three events: predicted remaining runtime (p50) of the jobs it burst vs the actual remaining.
"""
import copy
import dataclasses

import numpy as np
import pandas as pd

from flex import cloud as cloudlib
from flex.cloud import Cloud, instance_cost
from flex.engine import Engine
from flex.metrics import target_kw
from flex.models import load_config
from flex.paired import baseline_snapshots, run_base
from flex.policies import progress, run_policy, world_from_config
from flex.power import PowerTracker
from flex.predictions import load_predictions
from traces.philly import ROOT, load_philly

refusals = {"n": 0}
_orig_can_request = cloudlib.can_request


def counting_can_request(cloud, gpus):
    ok = _orig_can_request(cloud, gpus)
    if not ok:
        refusals["n"] += 1
    return ok


def peak_cloud_gpus(instances):
    """Maximum of GPUs requested-and-not-released (booting + running) over time."""
    ev = []
    for i in instances:
        ev += [(i.requested, i.gpus)] + ([(i.released, -i.gpus)] if i.released is not None else [])
    cur = peak = 0
    for _, d in sorted(ev, key=lambda x: (x[0], x[1])):     # releases before requests at the same second
        cur += d
        peak = max(peak, cur)
    return peak


def main():
    cfg = load_config()
    L = load_philly(cfg)
    last2 = load_predictions(ROOT / cfg["paths"]["predictions"] / "last2.csv")
    world = world_from_config(cfg, last2)
    cap_now = world.cloud.max_gpus
    events = L["events"]
    eng = Engine(L["jobs"], L["G"], check_every=cfg["check_every"], cloud=Cloud(world.cloud))
    by_start = {}
    for e in events:
        by_start.setdefault(e.t_start, []).append(e)
    diag3000 = {"d1h_c30_0", "d2h_c30_0", "d4h_c30_0"}
    floor_ev = {f"d{d}h_c30_{k}" for d in (1, 2, 4) for k in (8, 9)}
    mix_ev = {"d2h_c20_0", "d1h_c10_0", "d4h_c30_0"}
    p = world.params
    floor_kw = p.pue * L["G"] * (p.p_idle_w + p.h_idle_w) / 1000.0
    cloudlib.can_request = counting_can_request
    rows1, rows1b, rows2, rows3, jobs3 = [], [], [], [], []
    for t_s, snap in baseline_snapshots(eng, by_start):
        for ev in by_start[t_s]:
            base = run_base(snap, ev, p, world.h_after_s)
            p_star = target_kw(base.tracker, ev.t_start, ev.t_end, ev.cut)
            refusals["n"] = 0
            row, _, pol = run_policy(snap, ev, "burst", world, "last2", "predictor", cfg["seed"], base, return_runs=True)
            tgt = ev.cut * row["base_mean_kw"] * ev.duration_h / 1000.0
            rows1.append(dict(event_id=ev.event_id, duration_h=ev.duration_h, cut=ev.cut, peak_cloud_gpus=peak_cloud_gpus(pol.engine.cloud.instances),
                              refusals=refusals["n"], n_burst=row["n_burst"], shed_pct=100 * row["shed_mwh"] / tgt, compliance=row["compliance"],
                              usd_per_mwh=row["usd_per_mwh"], cloud_usd=row["cloud_usd"]))
            if ev.event_id in diag3000:
                w3 = dataclasses.replace(world, cloud=dataclasses.replace(world.cloud, max_gpus=3000))
                refusals["n"] = 0
                r3, _, pol3 = run_policy(snap, ev, "burst", w3, "last2", "predictor", cfg["seed"], base, return_runs=True)
                rows1b.append(dict(event_id=ev.event_id, cap=1000, shed_pct=rows1[-1]["shed_pct"], usd_per_mwh=row["usd_per_mwh"], peak=rows1[-1]["peak_cloud_gpus"],
                                   compliance=row["compliance"], cloud_usd=row["cloud_usd"], n_burst=row["n_burst"]))
                rows1b.append(dict(event_id=ev.event_id, cap=3000, shed_pct=100 * r3["shed_mwh"] / tgt, usd_per_mwh=r3["usd_per_mwh"],
                                   peak=peak_cloud_gpus(pol3.engine.cloud.instances), compliance=r3["compliance"], cloud_usd=r3["cloud_usd"], n_burst=r3["n_burst"]))
            if ev.event_id in floor_ev:
                dl = run_policy(snap, ev, "delay", world, "last2", "predictor", cfg["seed"], base)
                rows2.append(dict(event_id=ev.event_id, p_star_kw=p_star, floor_kw=floor_kw, base_mean_kw=row["base_mean_kw"],
                                  infeasible=p_star < floor_kw, delay_peak_kw=p_star + dl["shortfall_kw"], delay_compliance=dl["compliance"],
                                  delay_shed_pct=100 * dl["shed_mwh"] / tgt, running_jobs=len(snap.running),
                                  running_gpus=sum(snap.jobs[i].gpus for i in snap.running)))
            if ev.event_id in mix_ev:
                rm, _, polm = run_policy(snap, ev, "mix", world, "last2", "predictor", cfg["seed"], base, return_runs=True)
                for inst in polm.engine.cloud.instances:
                    j = polm.engine.jobs[inst.job_id]
                    sj = snap.jobs.get(inst.job_id)
                    was_running = sj is not None and inst.job_id in snap.running
                    elapsed = progress(snap, sj) if was_running else (sj.elapsed0 + sj.work_done if sj is not None else 0)
                    _, p50, _, _, _ = last2.lookup(j, elapsed)
                    jobs3.append(dict(event_id=ev.event_id, job_id=inst.job_id, gpus=inst.gpus, running=was_running, has_rows=inst.job_id in last2.rows,
                                      pred_p50_s=p50, actual_rem_s=j.burst_rem, billed_usd=instance_cost(polm.engine.cloud.params, inst, polm.engine.now),
                                      pred_usd=world.cloud.price_per_gpu_h * inst.gpus * (p50 + world.cloud.migrate_s) / 3600.0))
                rows3.append(dict(event_id=ev.event_id, n_burst=rm["n_burst"], cloud_usd=rm["cloud_usd"], usd_per_mwh=rm["usd_per_mwh"]))
        print(f"  t_s {t_s}: {len(rows1)}/{len(events)} events", flush=True)
    out = ROOT / "results" / "s12b"
    out.mkdir(parents=True, exist_ok=True)
    d1, d2, d3 = pd.DataFrame(rows1), pd.DataFrame(rows2), pd.DataFrame(jobs3)
    d1.to_csv(out / "burst_cap.csv", index=False)
    pd.DataFrame(rows1b).to_csv(out / "burst_cap3000.csv", index=False)
    d2.to_csv(out / "floor.csv", index=False)
    d3.to_csv(out / "mix_burst_jobs.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"\n1a. burst-only (last2), cloud GPU cap {cap_now}: peak cloud GPUs and refusals by cut")
    print(d1.groupby("cut").agg(events=("event_id", "size"), peak_median=("peak_cloud_gpus", "median"), peak_max=("peak_cloud_gpus", "max"),
                                events_with_refusal=("refusals", lambda s: int((s > 0).sum())), events_peak_ge_95pct=("peak_cloud_gpus", lambda s: int((s >= 0.95 * cap_now).sum())),
                                shed_pct_median=("shed_pct", "median"), compliant=("compliance", lambda s: int((s >= 0.99).sum()))).round(1).to_string())
    print("\n1b. DIAGNOSTIC: burst-only on three 30% events, max_cloud_gpus 1000 vs 3000")
    print(pd.DataFrame(rows1b).round(1).to_string(index=False))
    print(f"\n2. floor = PUE x G x (P_idle + H_idle) = {floor_kw:.0f} kW (all {L['G']} GPUs idle, no busy host power)")
    print(d2.round(1).to_string(index=False))
    print("\n3. mix (last2) burst jobs")
    print(pd.DataFrame(rows3).round(1).to_string(index=False))
    for ev_id, g in d3.groupby("event_id"):
        r = g[g.has_rows]
        floor60 = (r.pred_p50_s <= 60).mean() if len(r) else float("nan")
        under = (r.pred_p50_s < 0.5 * r.actual_rem_s)
        print(f"\n  {ev_id}: {len(g)} burst jobs ({int(g.running.sum())} were running), {int((~g.has_rows).sum())} without prediction rows (oracle fallback)")
        print(f"    jobs with rows: p50 at 60 s floor {floor60:.0%}; predicted < half of actual {under.mean():.0%}; predicted >= actual {(r.pred_p50_s >= r.actual_rem_s).mean():.0%}")
        print(f"    actual remaining h (median/p90): {r.actual_rem_s.median() / 3600:.2f}/{r.actual_rem_s.quantile(.9) / 3600:.2f}; predicted p50 h (median/p90): {r.pred_p50_s.median() / 3600:.2f}/{r.pred_p50_s.quantile(.9) / 3600:.2f}")
        print(f"    billed $ total {g.billed_usd.sum():,.0f} vs predicted $ {g.pred_usd.sum():,.0f} (rows only: {r.billed_usd.sum():,.0f} vs {r.pred_usd.sum():,.0f}); "
              f"$ on floor jobs {r[r.pred_p50_s <= 60].billed_usd.sum():,.0f}, on under-predicted (< half) {r[under].billed_usd.sum():,.0f}, no rows {g[~g.has_rows].billed_usd.sum():,.0f}")
    cloudlib.can_request = _orig_can_request


if __name__ == "__main__":
    main()
