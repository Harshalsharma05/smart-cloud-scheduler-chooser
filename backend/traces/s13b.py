"""S13b: measurement-only diagnostics of the S13 results (python -m traces.s13b, from backend/). No model change.

1. migrate_s and restore_s varied SEPARATELY (0.5x, 1x, 2x) for burst-only and delay (last2) on the 10 default-cell events.
2. delay+oracle and mix+oracle per event: jobs suspended / burst by status, remaining wall of the burst jobs, cost split.
3. Absolute table (median $ and MWh per policy) from the E1 runs.
"""
import dataclasses

import numpy as np
import pandas as pd

from flex.cloud import Cloud, instance_cost
from flex.engine import Engine
from flex.metrics import delay_gpuh, missed_deadlines
from flex.models import load_config
from flex.paired import baseline_snapshots, run_base
from flex.policies import progress, run_policy, world_from_config
from flex.predictions import load_predictions
from traces.philly import ROOT, load_philly

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def burst_jobs(pol, snap, pred, world):
    """One row per cloud instance: status, gpus, actual remaining at burst, predicted p50, billed seconds."""
    rows = []
    for inst in pol.engine.cloud.instances:
        j = pol.engine.jobs[inst.job_id]
        sj = snap.jobs.get(inst.job_id)
        running = sj is not None and inst.job_id in snap.running
        elapsed = progress(snap, sj) if running else (sj.elapsed0 + sj.work_done if sj is not None else 0)
        p50 = pred.lookup(j, elapsed)[1]
        end = inst.released if inst.released is not None else pol.engine.now
        rows.append(dict(job_id=inst.job_id, status=j.status, gpus=inst.gpus, running=running, rem_s=j.burst_rem, p50_s=p50,
                         billed_s=(end - inst.ready) if inst.ready is not None else 0,
                         usd=instance_cost(pol.engine.cloud.params, inst, pol.engine.now)))
    return pd.DataFrame(rows)


def cost_split(row, base_eng, pol_eng, cost):
    """cloud $, delay $ (c_delay x dGPUh_good with w_fail as configured), deadline $ of one policy run."""
    good, total = delay_gpuh(base_eng, pol_eng)
    _, miss_gpuh = missed_deadlines(base_eng, pol_eng)
    d_good = good + cost.w_fail * (total - good)
    return dict(cloud_usd=row["cloud_usd"], delay_usd=cost.c_delay * d_good, deadline_usd=cost.m * cost.c_delay * miss_gpuh)


def main():
    cfg = load_config()
    L = load_philly(cfg)
    pr = {n: load_predictions(ROOT / cfg["paths"]["predictions"] / f"{n}.csv") for n in ("last2", "oracle")}
    world = world_from_config(cfg, pr["last2"])
    cell = [e for e in L["events"] if e.duration_h == 2 and abs(e.cut - 0.2) < 1e-9]
    eng = Engine(L["jobs"], L["G"], check_every=cfg["check_every"], cloud=Cloud(world.cloud))
    mig0, res0 = world.cloud.migrate_s, world.restore_s
    variants = {"mig x0.5": (mig0 * 0.5, res0), "mig x1": (mig0, res0), "mig x2": (mig0 * 2, res0),
                "res x0.5": (mig0, res0 * 0.5), "res x2": (mig0, res0 * 2)}
    p1, bj1, p2, p3 = [], [], [], []
    by_start = {}
    for e in cell:
        by_start.setdefault(e.t_start, []).append(e)
    for t_s, snap in baseline_snapshots(eng, by_start):
        for ev in by_start[t_s]:
            base = run_base(snap, ev, world.params, world.h_after_s)
            tgt = ev.cut * base.tracker.mean_kw(ev.t_start, ev.t_end) / 0.8 * ev.duration_h / 1000.0 * 0.8   # = cut x base mean x h
            for name, (mg, rs) in variants.items():
                w = dataclasses.replace(world, restore_s=int(rs), cloud=dataclasses.replace(world.cloud, migrate_s=int(mg)))
                for pol_name in ("burst", "delay"):
                    row, _, pol = run_policy(snap, ev, pol_name, w, "last2", "predictor", cfg["seed"], base, return_runs=True)
                    p1.append(dict(variant=name, policy=pol_name, event_id=ev.event_id, usd_per_mwh=row["usd_per_mwh"], cost_usd=row["cost_usd"],
                                   n_burst=row["n_burst"], n_suspended=row["n_suspended"], cloud_usd=row["cloud_usd"], shed_mwh=row["shed_mwh"],
                                   missed=row["missed_deadlines"]))
                    if pol_name == "burst":
                        bj = burst_jobs(pol, snap, pr["last2"], w)
                        bj["variant"], bj["event_id"] = name, ev.event_id
                        bj1.append(bj)
            for pname in ("oracle",):
                wo = dataclasses.replace(world, predictor=pr[pname])
                for pol_name in ("delay", "mix"):
                    row, _, pol = run_policy(snap, ev, pol_name, wo, pname, "predictor", cfg["seed"], base, return_runs=True)
                    sus = {s: sum(1 for j in pol.engine.jobs.values() if j.n_suspends > 0 and j.status == s) for s in ("Pass", "Failed", "Killed")}
                    bj = burst_jobs(pol, snap, pr[pname], wo)
                    bs = bj.status.value_counts().to_dict() if len(bj) else {}
                    sp = cost_split(row, base.engine, pol.engine, wo.cost)
                    p2.append(dict(event_id=ev.event_id, policy=pol_name + "+oracle", susp_pass=sus["Pass"], susp_failed=sus["Failed"], susp_killed=sus["Killed"],
                                   burst_pass=bs.get("Pass", 0), burst_failed=bs.get("Failed", 0), burst_killed=bs.get("Killed", 0),
                                   burst_rem_h_median=bj.rem_s.median() / 3600 if len(bj) else np.nan, event_h=ev.duration_h,
                                   cost_usd=row["cost_usd"], shed_mwh=row["shed_mwh"], usd_per_mwh=row["usd_per_mwh"], **sp))
    d1, d2 = pd.DataFrame(p1), pd.DataFrame(p2)
    bj = pd.concat(bj1, ignore_index=True)
    out = ROOT / "results" / "s13b"
    out.mkdir(parents=True, exist_ok=True)
    d1.to_csv(out / "migrate_restore.csv", index=False); bj.to_csv(out / "burst_jobs_migrate.csv", index=False); d2.to_csv(out / "oracle_mechanism.csv", index=False)

    print("\n1. migrate_s / restore_s separately (10 default-cell events, last2; medians over events, sums where stated)")
    g = d1.groupby(["policy", "variant"], sort=False)
    t = g.agg(usd_per_mwh=("usd_per_mwh", "median"), cost_usd=("cost_usd", "median"), n_burst=("n_burst", "median"), n_susp=("n_suspended", "median"),
              cloud_usd=("cloud_usd", "median"), shed_mwh=("shed_mwh", "median"), missed=("missed", "median")).round(2)
    print(t.to_string())
    bj["excess_s"] = bj["billed_s"] - bj["rem_s"]
    bj["floor"] = bj["p50_s"] <= 60
    bt = bj.groupby("variant", sort=False).agg(jobs=("job_id", "size"), billed_h_median=("billed_s", lambda s: s.median() / 3600), billed_h_mean=("billed_s", lambda s: s.mean() / 3600),
                                              rem_h_median=("rem_s", lambda s: s.median() / 3600), excess_s_median=("excess_s", "median"),
                                              gpus_mean=("gpus", "mean"), floor_share=("floor", "mean"), usd_per_job=("usd", "mean")).round(2)
    print("\n   burst jobs (burst-only): billed time = remaining + migrate_s?  (excess_s = billed - remaining at burst)")
    print(bt.to_string())
    # same jobs across variants: jobs burst in both mig x1 and mig x2 for the same event
    a = bj[bj.variant == "mig x1"].set_index(["event_id", "job_id"])
    b = bj[bj.variant == "mig x2"].set_index(["event_id", "job_id"])
    both = a.index.intersection(b.index)
    print(f"   jobs burst in both mig x1 and mig x2: {len(both)} (of {len(a)} / {len(b)}); billed-time difference on the shared jobs: median {np.median(b.loc[both].billed_s - a.loc[both].billed_s):.0f} s (migrate_s +120 s)")
    only1, only2 = a.index.difference(b.index), b.index.difference(a.index)
    print(f"   only in x1: {len(only1)} jobs, median remaining {a.loc[only1].rem_s.median() / 3600:.1f} h, mean GPUs {a.loc[only1].gpus.mean():.1f}; "
          f"only in x2: {len(only2)} jobs, median remaining {b.loc[only2].rem_s.median() / 3600:.1f} h, mean GPUs {b.loc[only2].gpus.mean():.1f}")
    print(f"   shared-job remaining h median {a.loc[both].rem_s.median() / 3600:.1f}; usd on only-x1 {a.loc[only1].usd.sum():,.0f}, only-x2 {b.loc[only2].usd.sum():,.0f}, shared x1 {a.loc[both].usd.sum():,.0f}, shared x2 {b.loc[both].usd.sum():,.0f}")

    print("\n2. oracle mechanism, per event (w_fail = 0)")
    cols = ["event_id", "policy", "susp_pass", "susp_failed", "susp_killed", "burst_pass", "burst_failed", "burst_killed", "burst_rem_h_median", "cloud_usd", "delay_usd", "deadline_usd", "cost_usd", "shed_mwh"]
    print(d2[cols].round(2).to_string(index=False))
    pv = d2.pivot(index="event_id", columns="policy", values="cost_usd")
    pm = d2.pivot(index="event_id", columns="policy", values="shed_mwh")
    diff = pd.DataFrame(dict(delay_usd=pv["delay+oracle"], mix_usd=pv["mix+oracle"], mix_minus_delay_usd=pv["mix+oracle"] - pv["delay+oracle"],
                             shed_delay=pm["delay+oracle"], shed_mix=pm["mix+oracle"]))
    print("\n   absolute difference per event (mix+oracle - delay+oracle), $:")
    print(diff.round(2).to_string())
    print(f"   median difference {diff.mix_minus_delay_usd.median():.1f} $; sum {diff.mix_minus_delay_usd.sum():.1f} $; events mix cheaper {(diff.mix_minus_delay_usd < 0).sum()}/10")
    print("   medians by policy:\n", d2.groupby("policy")[["cloud_usd", "delay_usd", "deadline_usd", "cost_usd", "shed_mwh", "burst_rem_h_median"]].median().round(2).to_string())

    e1 = pd.read_csv(ROOT / "results" / "e1" / "runs.csv", dtype={"predictor": str, "setting": str}).fillna({"predictor": "", "setting": ""})
    e1 = e1[(e1.duration_h == 2) & (e1.cut.sub(0.2).abs() < 1e-9)]
    e1["who"] = np.where(e1.predictor == "oracle", e1.policy + "+oracle", e1.policy + np.where(e1.policy == "cap", "", "/last2"))
    t3 = e1.groupby("who").agg(events=("event_id", "size"), cost_usd_w0=("cost_usd", "median"), cost_usd_w1=("cost_usd_wfail1", "median"), shed_mwh=("shed_mwh", "median"),
                               cloud_usd=("cloud_usd", "median"), usd_per_mwh_w0=("usd_per_mwh", "median"), compliance=("compliance", "median")).round(3)
    print("\n3. absolute table, default cell (10 events), medians:")
    print(t3.to_string())
    t3.to_csv(out / "absolute_table.csv")


if __name__ == "__main__":
    main()
