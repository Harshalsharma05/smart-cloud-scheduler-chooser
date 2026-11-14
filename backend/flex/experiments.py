"""Experiment runner: python -m flex.experiments <e1|e2|e3|e4|e5> --data synthetic|philly --predictor last2 [--quick]

Writes results/<exp>/runs.csv (one row per event x policy x setting) and results/<exp>/summary.csv (median and
IQR over events). Events are processed one at a time from the baseline snapshots (memory). The wall clock is read
here, in the runner, never inside the simulator: runtime_s is the time of one policy run (the baseline
continuation of an event is shared by its policies).
"""
import argparse
import copy
import dataclasses
import random
import time
from pathlib import Path

import pandas as pd

from .cloud import Cloud
from .engine import Engine
from .grid import epoch_from_iso, generate_events
from .metrics import COLUMNS
from .models import load_config
from .paired import baseline_snapshots, run_base
from .policies import run_policy, world_from_config
from .predictions import load_predictions

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVENT = (2, 0.2)            # D13: 2 h, 20%
HARD_EVENT = (4, 0.3)               # E3: 4 h, 30%
ALL_PREDICTORS = ["last2", "gbdt", "lightgbm", "transformer", "oracle"]
BLIND_SELECTORS = ["none-largest", "random"]
SUMMARY_METRICS = ["usd_per_mwh", "cost_usd", "usd_per_mwh_wfail1", "cost_usd_wfail1", "shed_mwh", "compliance", "shortfall_kw", "rebound_mwh", "cloud_usd",
                   "cloud_waste_fail_usd", "cloud_waste_short_usd", "dgpuh_good", "missed_deadlines", "d_wait_min",
                   "d_slowdown", "n_burst", "n_suspended"]
KEYS = ["duration_h", "cut", "policy", "selector", "predictor", "setting"]


# ---- setup ----

def predictions_dir(cfg, data):
    return ROOT / cfg["paths"]["data"] / "synthetic" / "predictions" if data == "synthetic" else ROOT / cfg["paths"]["predictions"]


def load_setup(cfg, data, quick):
    """Jobs, pool size, epoch, the baseline-engine factory and all events for one data source."""
    if data == "philly":                                           # S8/S9c: real trace, pool from philly.pool_gpus
        from traces.philly import load_philly
        cfg = copy.deepcopy(cfg)
        if quick:
            cfg["grid"]["events_per_cell"] = 3
        L = load_philly(cfg)
        print(f"Philly: {len(L['jobs'])} jobs, pool G = {L['G']} (installed {L['G_installed']}), {len(L['events'])} events")
        return cfg, L["jobs"], L["G"], L["events"]
    from traces.synthetic import synthetic_jobs, write_predictions
    cfg = copy.deepcopy(cfg)
    if quick:
        cfg["grid"]["events_per_cell"] = 3
    jobs = synthetic_jobs(cfg)
    pool = cfg["synthetic"]["pool_gpus"]
    pdir = predictions_dir(cfg, data)
    if not (pdir / "oracle.csv").exists() or not (pdir / "last2.csv").exists():
        base = Engine(synthetic_jobs(cfg), pool, check_every=0)       # simulate without events to get start/end times
        base.run_to_drain()
        write_predictions(list(base.jobs.values()), pdir)
        print("wrote synthetic predictions to", pdir)
    epoch0 = epoch_from_iso(cfg["synthetic"]["epoch0_utc"])
    t_hi = int(cfg["synthetic"]["days"] * 86400)
    events = generate_events(random.Random(cfg["seed"]), epoch0, 0, t_hi, cfg["grid"])
    return cfg, jobs, pool, events


def load_predictors(cfg, data):
    """Every prediction file that exists; the missing names are listed, never faked."""
    pdir = predictions_dir(cfg, data)
    found = {n: load_predictions(pdir / f"{n}.csv") for n in ALL_PREDICTORS if (pdir / f"{n}.csv").exists()}
    missing = [n for n in ALL_PREDICTORS if n not in found]
    if missing:
        print("prediction files not found (not run):", ", ".join(missing))
    return found


def cell(events, duration_h, cut):
    return [e for e in events if e.duration_h == duration_h and abs(e.cut - cut) < 1e-9]


# ---- driver ----

def run_events(cfg, jobs, pool, world, events, per_event, label):
    """For each baseline snapshot (one event start at a time) and each event starting then:
    one shared baseline continuation, then per_event(snapshot, event, base) -> rows."""
    eng = Engine(jobs, pool, check_every=cfg["check_every"], cloud=Cloud(world.cloud))
    by_start = {}
    for e in events:
        by_start.setdefault(e.t_start, []).append(e)
    rows, done, t0 = [], 0, time.perf_counter()
    for t_s, snap in baseline_snapshots(eng, by_start):
        for ev in by_start[t_s]:
            base = run_base(snap, ev, world.params, world.h_after_s)
            rows += per_event(snap, ev, base)
            done += 1
        print(f"  {label}: {done}/{len(events)} events, {time.perf_counter() - t0:.0f} s", flush=True)
    return rows


def timed_row(cfg, exp, snap, ev, name, world, base, predictor_name="", selector="predictor", setting=""):
    t = time.perf_counter()
    row = run_policy(snap, ev, name, world, predictor_name, selector, cfg["seed"], base)
    row.update(exp=exp, setting=setting, runtime_s=round(time.perf_counter() - t, 3), seed=cfg["seed"])
    return row


# ---- experiments ----

def e1(cfg, jobs, pool, events, predictors, predictor, quick):
    """Policies delay, cap, burst, mix on every event; mix with the oracle as the upper bound."""
    world = world_from_config(cfg, predictors[predictor])

    def per_event(snap, ev, base):
        rows = [timed_row(cfg, "e1", snap, ev, n, world, base, predictor) for n in ("delay", "cap", "burst", "mix")]
        if "oracle" in predictors and predictor != "oracle":
            w = dataclasses.replace(world, predictor=predictors["oracle"])
            rows += [timed_row(cfg, "e1", snap, ev, n, w, base, "oracle", setting="upper-bound") for n in ("mix", "delay")]
        return rows
    return run_events(cfg, jobs, pool, world, events, per_event, "e1")


def e2(cfg, jobs, pool, events, predictors, predictor, quick):
    """Default event: cloud price x boot delay. delay and cap do not depend on the cloud, so they run once per event.
    c_delay stays at its default (the base cloud price) while the cloud price is swept."""
    world = world_from_config(cfg, predictors[predictor])
    price_factors, boots = ([0.5, 1, 2], [60, 300]) if quick else ([0.25, 0.5, 1, 2, 4], [60, 300, 900])
    settings = [(pf, b) for pf in price_factors for b in boots]
    label = lambda pf, b: f"price_x{pf}_boot{b}"

    def per_event(snap, ev, base):
        rows = []
        for name in ("delay", "cap"):
            r = timed_row(cfg, "e2", snap, ev, name, world, base, predictor)
            rows += [dict(r, setting=label(pf, b)) for pf, b in settings]
        for pf, b in settings:
            cp = dataclasses.replace(world.cloud, price_per_gpu_h=world.cloud.price_per_gpu_h * pf, boot_delay_s=b)
            w = dataclasses.replace(world, cloud=cp)
            rows += [timed_row(cfg, "e2", snap, ev, n, w, base, predictor, setting=label(pf, b)) for n in ("burst", "mix")]
        return rows
    return run_events(cfg, jobs, pool, world, cell(events, *DEFAULT_EVENT), per_event, "e2")


def e3(cfg, jobs, pool, events, predictors, predictor, quick):
    """Burst-only and mix by selector: none-largest, random, and every predictor file that exists."""
    world = world_from_config(cfg, predictors[predictor])
    chosen = cell(events, *DEFAULT_EVENT) + cell(events, *HARD_EVENT)

    def per_event(snap, ev, base):
        rows = []
        for name in ("burst", "mix"):
            for sel in BLIND_SELECTORS:
                rows.append(timed_row(cfg, "e3", snap, ev, name, world, base, "", sel))
            for pname, pred in predictors.items():
                rows.append(timed_row(cfg, "e3", snap, ev, name, dataclasses.replace(world, predictor=pred), base, pname))
        return rows
    return run_events(cfg, jobs, pool, world, chosen, per_event, "e3")


# ---- output ----

def summarise(runs):
    """Median, 25th and 75th percentile over events for each (cell, policy, selector, predictor, setting)."""
    df = runs.copy()
    for k in ("selector", "predictor", "setting"):
        df[k] = df[k].fillna("")
    g = df.groupby(KEYS, dropna=False)
    parts = [g.size().rename("n_events"), g["usd_per_mwh"].apply(lambda s: s.isna().sum()).rename("n_na")]
    for m in SUMMARY_METRICS:
        parts += [g[m].median().rename(f"{m}_median"), g[m].quantile(0.25).rename(f"{m}_q25"),
                  g[m].quantile(0.75).rename(f"{m}_q75")]
    return pd.concat(parts, axis=1).reset_index()


def write_results(exp, rows, out_dir):
    out = Path(out_dir) / exp
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(rows, columns=COLUMNS)
    runs.to_csv(out / "runs.csv", index=False)
    summary = summarise(runs)
    summary.to_csv(out / "summary.csv", index=False)
    return runs, summary


def e1_table(runs):
    """The quick-run table: per policy over all events, and how often the mix matches the best single action."""
    df = runs.copy()
    df["who"] = df["policy"] + df["predictor"].fillna("").map(lambda p: f"/{p}" if p else "")
    t = df.groupby("who").agg(events=("event_id", "size"), usd_per_mwh_median=("usd_per_mwh", "median"),
                              n_na=("usd_per_mwh", lambda s: int(s.isna().sum())), shed_mwh_median=("shed_mwh", "median"),
                              compliance_median=("compliance", "median"), cloud_usd_median=("cloud_usd", "median"))
    print(t.round(4).to_string())
    main = df[(df["setting"].fillna("") == "")]
    key = ["duration_h", "cut", "event_id"]
    mix = main[main["policy"] == "mix"].set_index(key)["usd_per_mwh"]
    singles = main[main["policy"].isin(["delay", "cap", "burst"])].groupby(key)["usd_per_mwh"].min()
    both = pd.concat([mix.rename("mix"), singles.rename("best_single")], axis=1).dropna()
    wins = int((both["mix"] <= both["best_single"] * (1 + 1e-9)).sum())
    print(f"mix $/MWh <= best single action on {wins} of {len(both)} events with both defined "
          f"({len(mix) - len(both)} events where the mix or all singles are n/a)")
    return t


def e4_from_e1(out_dir, summary_only=False):
    """E4 reads the E1 runs: user impact and rebound."""
    runs = pd.read_csv(Path(out_dir) / "e1" / "runs.csv")
    cols = ["event_id", "duration_h", "cut", "policy", "selector", "predictor", "setting", "d_wait_min", "d_slowdown",
            "rebound_mwh", "missed_deadlines", "dgpuh_good", "dgpuh_all", "n_suspended", "n_burst", "cap_level"]
    out = Path(out_dir) / "e4"
    out.mkdir(parents=True, exist_ok=True)
    runs[cols].to_csv(out / "runs.csv", index=False)
    summarise(runs).to_csv(out / "summary.csv", index=False)
    return runs


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m flex.experiments")
    ap.add_argument("exp", choices=["e1", "e2", "e3", "e4", "e5"])
    ap.add_argument("--data", choices=["synthetic", "philly"], required=True)
    ap.add_argument("--predictor", default="last2")
    ap.add_argument("--quick", action="store_true", help="3 events per cell (and smaller sweeps)")
    ap.add_argument("--out", default=None, help="results directory (default: <repo>/results)")
    args = ap.parse_args(argv)
    cfg = load_config()
    out_dir = Path(args.out) if args.out else ROOT / cfg["paths"]["results"]
    if args.exp == "e5":
        raise SystemExit("e5 (carbon) is phase S15")
    if args.exp == "e4":
        if not (out_dir / "e1" / "runs.csv").exists():
            main(["e1", "--data", args.data, "--predictor", args.predictor] + (["--quick"] if args.quick else [])
                 + (["--out", str(out_dir)] if args.out else []))
        e4_from_e1(out_dir)
        print("wrote", out_dir / "e4")
        return
    cfg, jobs, pool, events = load_setup(cfg, args.data, args.quick)
    predictors = load_predictors(cfg, args.data)
    if args.predictor not in predictors:
        raise SystemExit(f"no predictions/{args.predictor}.csv for --data {args.data}")
    t0 = time.perf_counter()
    rows = {"e1": e1, "e2": e2, "e3": e3}[args.exp](cfg, jobs, pool, events, predictors, args.predictor, args.quick)
    runs, summary = write_results(args.exp, rows, out_dir)
    print(f"{args.exp}: {len(runs)} rows in {time.perf_counter() - t0:.0f} s -> {out_dir / args.exp}")
    if args.exp == "e1":
        e1_table(runs)


if __name__ == "__main__":
    main()
