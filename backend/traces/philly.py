"""Philly jobs for the engine (SIM_PLAN S8). Reads data/processed/philly_jobs.parquet, philly_machines.parquet and
(optional) philly_job_util.parquet (S9); TEST_START / TEST_END come from ml/config.py (TEST_END is exclusive).

Decisions (S8):
- Time: job-log times are UTC epoch seconds. Simulated t = epoch - epoch0, epoch0 = TEST_START - warmup_days (flex.yaml).
  Local time for events (America/Los_Angeles) is done by flex.grid.
- Warm-up: every job with submit in [epoch0, TEST_END) is loaded; those with submit < TEST_START get warmup=True
  (no deadline, excluded from metrics). Jobs submitted before epoch0 are ignored (counted and reported).
  Warm-up jobs and censored jobs have no prediction rows, so predictors fall back to oracle values (counted).
- Carry-over (S8b/S8d): a job with first_start < epoch0 < end keeps its real job_id and becomes a job submitted at t = 0
  with wall_s = end - epoch0 (end = first_start + wall_s; trace end for censored jobs), warmup=True, carry=True,
  elapsed0 = epoch0 - first_start (added to work_done for the prediction lookup, so rows of the real job apply),
  submit_real = its trace submit time (tie-breaks only). If the carry-overs exceed G, the latest-started ones are dropped
  (counted). Jobs submitted before epoch0 that start after epoch0 are still ignored (counted, not simulated).
- Censored jobs (end not observed) have no known outcome. Their trace status is not an outcome, so they are given
  status "Pass" (useful work) in the cost metrics and in the oracle p_fail / p_cancel. Their trace statuses are reported.
- Dropped: never_started jobs, jobs with 0 GPUs, jobs with wall_s <= 0 (all counted in the report).
  Censored jobs (no end) are kept with wall_s = (last end_time in the log) - first_start_time.
- Deadlines: D12, only for test-window jobs, drawn in (submit, job_id) order from Random(seed + 1).
- util: philly_job_util.parquet (S9) if it exists (unmeasured jobs: the n_minutes-weighted cluster average), else util_default
  from flex.yaml (an ASSUMPTION, logged).
"""
import copy
import importlib.util
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from flex.grid import epoch_from_iso, generate_events
from flex.models import FJob, load_config
from traces.synthetic import assign_deadlines

ROOT = Path(__file__).resolve().parents[2]
JOB_COLUMNS = ["job_id", "user", "vc", "submit_time", "first_start_time", "end_time", "wall_s", "run_s", "num_gpus",
               "num_attempts", "status", "censored", "never_started"]


def ml_config():
    """ml/config.py read as a module (constants only; nothing is copied)."""
    spec = importlib.util.spec_from_file_location("ml_config", ROOT / "ml" / "config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def utc_epoch(date_text):
    return epoch_from_iso(f"{date_text}T00:00:00+00:00")


def check_columns(df, columns, name):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SystemExit(f"{name}: missing contract columns {missing}; stop and report to Track 1")


def load_philly(cfg=None):
    """Returns a dict: jobs (FJob list), G, epoch0, test_start, test_end (simulated s), events, util_source, report, real."""
    cfg = cfg or load_config()
    mc = ml_config()
    data = ROOT / cfg["paths"]["data"] / "processed"
    raw = pd.read_parquet(data / "philly_jobs.parquet")
    machines = pd.read_parquet(data / "philly_machines.parquet")
    check_columns(raw, JOB_COLUMNS, "philly_jobs.parquet")
    check_columns(machines, ["machine_id", "num_gpus"], "philly_machines.parquet")
    G = int(machines["num_gpus"].sum())

    test_start_utc, test_end_utc = utc_epoch(mc.TEST_START), utc_epoch(mc.TEST_END)
    epoch0 = test_start_utc - int(cfg["philly"]["warmup_days"]) * 86400
    assert epoch0 == utc_epoch(mc.SIM_START), "ml/config.py SIM_START must equal TEST_START - philly.warmup_days"
    rep = {"rows_in_file": len(raw)}
    umap, fallback, util_src = {}, float(cfg["philly"]["util_default"]), None
    util_path = data / "philly_job_util.parquet"
    if util_path.exists():                                   # S9: measured; unmeasured jobs get the cluster average
        ut = pd.read_parquet(util_path)
        umap = ut.set_index("job_id")["gpu_util_mean"].to_dict()
        fallback = float(np.average(ut["gpu_util_mean"], weights=ut["n_minutes"]))
        util_src = f"philly_job_util.parquet; unmeasured jobs use the cluster average {fallback:.3f}"

    last_end = int(raw["end_time"].max())                  # for censored jobs
    d = raw[~raw["never_started"]]
    rep["never_started_dropped"] = len(raw) - len(d)
    rep["zero_gpu_dropped"] = int((d["num_gpus"] <= 0).sum())
    d = d[d["num_gpus"] > 0].copy()
    d["wall_s"] = d["wall_s"].where(~d["censored"], last_end - d["first_start_time"])   # censored: observed so far
    rep["censored_kept"] = int(d["censored"].sum())
    rep["censored_trace_status"] = d[d["censored"]]["status"].value_counts().to_dict()
    d["status"] = d["status"].where(~d["censored"], "Pass")          # no observed outcome: treat as useful
    rep["wall_nonpositive_dropped"] = int((d["wall_s"] <= 0).sum())
    d = d[d["wall_s"] > 0]
    rep["gpus_above_G"] = int((d["num_gpus"] > G).sum())

    # jobs submitted before epoch0 are ignored; say how many were still running then
    before = d[d["submit_time"] < epoch0]
    end_abs = before["first_start_time"] + before["wall_s"]
    running = before[(before["first_start_time"] < epoch0) & (end_abs > epoch0)].copy()
    rep["before_epoch0_total"] = len(before)
    rep["before_epoch0_queued_at_epoch0_ignored"] = int((before["first_start_time"] >= epoch0).sum())
    rep["carry_candidates"] = len(running)
    rep["carry_candidates_gpus"] = int(running["num_gpus"].sum())
    rep["carry_remaining_gt_14d"] = int((end_abs[running.index] - epoch0 > 14 * 86400).sum())
    running["remaining"] = (end_abs[running.index] - epoch0).astype(int)
    carry, used = [], 0
    for r in running.sort_values(["first_start_time", "job_id"]).itertuples(index=False):
        if used + r.num_gpus > G:
            continue
        used += int(r.num_gpus)
        carry.append(FJob(job_id=r.job_id, user=r.user, submit=0, gpus=int(r.num_gpus), wall_s=int(r.remaining),
                          status=r.status, util=float(umap.get(r.job_id, fallback)), warmup=True, carry=True,
                          elapsed0=int(epoch0 - r.first_start_time), submit_real=int(r.submit_time - epoch0)))
    rep["carry_jobs"] = len(carry)
    rep["carry_gpus"] = used
    rep["carry_truncated"] = len(running) - len(carry)

    d = d[(d["submit_time"] >= epoch0) & (d["submit_time"] < test_end_utc)].sort_values(["submit_time", "job_id"])
    util = [float(umap.get(j, fallback)) for j in d["job_id"]]
    if util_src:
        util_source = f"{util_src} ({int(d['job_id'].isin(umap).sum())} of {len(d)} jobs measured)"
    else:
        util_source = f"util_default = {fallback} (ASSUMPTION; philly_job_util.parquet not found)"

    jobs = [FJob(job_id=r.job_id, user=r.user, submit=int(r.submit_time - epoch0), gpus=int(r.num_gpus),
                 wall_s=int(r.wall_s), status=r.status, util=float(u_), warmup=bool(r.submit_time < test_start_utc))
            for r, u_ in zip(d.itertuples(index=False), util)]
    jobs = carry + jobs
    test_jobs = [j for j in jobs if not j.warmup]
    assign_deadlines(test_jobs, random.Random(cfg["seed"] + 1), cfg["deadlines"])    # already in (submit, job_id) order

    test_start, test_end = test_start_utc - epoch0, test_end_utc - epoch0
    events = generate_events(random.Random(cfg["seed"]), epoch0, test_start, test_end, cfg["grid"])
    rep.update(jobs=len(jobs), warmup_jobs=len(jobs) - len(test_jobs) - len(carry), carry_pseudo_jobs=len(carry),
               test_jobs=len(test_jobs), G=G)
    real = d.set_index("job_id")[["first_start_time", "run_s", "censored"]]       # real timings, for sanity checks only
    all_started = raw[~raw["never_started"] & (raw["num_gpus"] > 0)]          # for the real busy-GPU curve
    return dict(jobs=jobs, G=G, epoch0=epoch0, test_start_utc=test_start_utc, test_end_utc=test_end_utc, all_started=all_started,
                last_end=last_end, test_start=test_start, test_end=test_end, events=events,
                util_source=util_source, report=rep, real=real)


def _busy_per_minute(starts, ends, gpus, w0, w1):
    """GPUs busy at every minute boundary of [w0, w1) from (start, end, gpus) triples (same time origin as w0, w1)."""
    n = (w1 - w0 + 59) // 60
    diff = np.zeros(n + 1)
    lo = np.clip(np.ceil((np.asarray(starts, dtype=float) - w0) / 60.0), 0, n).astype(int)
    hi = np.clip(np.ceil((np.asarray(ends, dtype=float) - w0) / 60.0), 0, n).astype(int)
    np.add.at(diff, lo, gpus)
    np.add.at(diff, hi, -np.asarray(gpus, dtype=float))
    return np.cumsum(diff)[:n]


def validation(L, eng, out_dir=None):
    """Baseline (simulated) vs real, test window: busy GPUs (mean, p95) and wait (mean, p50, p90, share > 60 s).
    Real busy GPUs use first_start_time and end_time of ALL started jobs (censored: trace end); waits use the kept test jobs."""
    ts, te, e0 = L["test_start"], L["test_end"], L["epoch0"]
    a = L["all_started"]
    end = a["end_time"].fillna(L["last_end"]).astype(float)
    real_busy = _busy_per_minute(a["first_start_time"].astype(float), end, a["num_gpus"].to_numpy(),
                                 L["test_start_utc"], L["test_end_utc"])
    segs = [(s_, e_, j.gpus) for j in eng.jobs.values() for s_, e_, where, _ in j.segments if where == "onprem"]
    sim_busy = _busy_per_minute([x[0] for x in segs], [x[1] for x in segs], [x[2] for x in segs], ts, te)
    test = [j for j in eng.jobs.values() if not j.warmup]
    rw = np.array([L["real"].loc[j.job_id, "first_start_time"] - (j.submit + e0) for j in test], dtype=float)
    sw = np.array([j.first_start - j.submit for j in test], dtype=float)
    rows = [("busy_gpus_mean", real_busy.mean(), sim_busy.mean()), ("busy_gpus_p95", np.percentile(real_busy, 95), np.percentile(sim_busy, 95)),
            ("utilisation_mean", real_busy.mean() / L["G"], sim_busy.mean() / L["G"]),
            ("wait_mean_min", rw.mean() / 60, sw.mean() / 60), ("wait_p50_min", np.percentile(rw, 50) / 60, np.percentile(sw, 50) / 60),
            ("wait_p90_min", np.percentile(rw, 90) / 60, np.percentile(sw, 90) / 60),
            ("frac_wait_gt_60s", (rw > 60).mean(), (sw > 60).mean())]
    df = pd.DataFrame(rows, columns=["metric", "real", "simulated"]).round(3)
    print(f"baseline validation, test window ({len(test)} jobs; G = {L['G']}):")
    print(df.to_string(index=False))
    out = Path(out_dir) if out_dir else ROOT / "results" / "philly_baseline"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "validation.csv", index=False)
    return df


def main():
    """Done-check of S8: python -m traces.philly (from backend/). Prints the S8 report and one timed paired run per policy."""
    from flex.cloud import Cloud
    from flex.engine import Engine
    from flex.paired import baseline_snapshots, run_base
    from flex.policies import run_policy, world_from_config
    from flex.power import PowerTracker
    from flex.predictions import load_predictions

    cfg = load_config()
    t0 = time.perf_counter()
    L = load_philly(cfg)
    print(f"load: {time.perf_counter() - t0:.1f} s")
    for k, v in L["report"].items():
        print(f"  {k}: {v}")
    print("  util:", L["util_source"])
    jobs, G, ts, te = L["jobs"], L["G"], L["test_start"], L["test_end"]
    print(f"  events: {len(L['events'])}, first start day {L['events'][0].t_start / 86400:.2f}, "
          f"window days {ts / 86400:.0f}-{te / 86400:.0f}")

    world = world_from_config(cfg, None)
    eng = Engine(copy.deepcopy(jobs), G, check_every=cfg["check_every"], cloud=Cloud(world.cloud))   # the engine mutates its jobs
    tracker = PowerTracker(world.params)
    tracker.attach(eng)
    t0 = time.perf_counter()
    eng.run_to_drain()
    run_s = time.perf_counter() - t0
    print(f"baseline run: {len(jobs)} jobs, {eng.n_events} events, run time {run_s:.1f} s, drained at day {eng.now / 86400:.1f}")

    validation(L, eng)
    test = [j for j in eng.jobs.values() if not j.warmup]
    real = L["real"]
    nc = [j for j in test if not real.loc[j.job_id, "censored"]]
    share = sum(j.wall_s > 1.1 * real.loc[j.job_id, "run_s"] for j in nc) / len(nc)
    print(f"share of non-censored test jobs with wall_s > 1.1 x run_s: {share:.3f} ({len(nc)} jobs)")
    pcs = tracker.pieces(ts, te)
    print(f"facility power in the window: mean {tracker.mean_kw(ts, te):.0f} kW, peak {max(p for _, p in pcs):.0f} kW")
    for name in ("oracle", "last2"):
        p = ROOT / cfg["paths"]["predictions"] / f"{name}.csv"
        if p.exists():
            pr = load_predictions(p)
            print(f"{name}.csv: {sum(j.job_id not in pr.rows for j in jobs)} of {len(jobs)} jobs have no rows (oracle fallback)")

    # one paired run of each policy on a default event (2 h, 20%) with the oracle
    oracle = load_predictions(ROOT / cfg["paths"]["predictions"] / "oracle.csv")
    world = world_from_config(cfg, oracle)
    ev = [e for e in L["events"] if e.duration_h == 2 and abs(e.cut - 0.2) < 1e-9][0]
    eng2 = Engine(jobs, G, check_every=cfg["check_every"], cloud=Cloud(world.cloud))
    t0 = time.perf_counter()
    (t_s, snap), = baseline_snapshots(eng2, [ev.t_start])
    print(f"event {ev.event_id}: run to t_s + snapshot {time.perf_counter() - t0:.1f} s")
    t0 = time.perf_counter()
    base = run_base(snap, ev, world.params, world.h_after_s)
    print(f"  baseline continuation {time.perf_counter() - t0:.1f} s")
    for name in ("delay", "cap", "burst", "mix"):
        t0 = time.perf_counter()
        row = run_policy(snap, ev, name, world, "oracle", "predictor", cfg["seed"], base)
        print(f"  {name}: {time.perf_counter() - t0:.1f} s  (shed {row['shed_mwh']:.3f} MWh, "
              f"compliance {row['compliance']:.2f}, $/MWh {row['usd_per_mwh']}, cost ${row['cost_usd']:.0f}, "
              f"cloud ${row['cloud_usd']:.0f}, dGPUh_good {row['dgpuh_good']:.1f}, missed {row['missed_deadlines']}, "
              f"suspended {row['n_suspended']}, burst {row['n_burst']})")


if __name__ == "__main__":
    main()
