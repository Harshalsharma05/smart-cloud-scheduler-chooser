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
        print(f"Philly: {len(L['jobs'])} jobs, pool G = {L['G']} (installed {L['G_installed']}), {len(L['events'])} events; "
              f"carry-over {L['report']['carry_jobs']} jobs, truncated {L['report']['carry_truncated']}")
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
    for pr in found.values():
        pr.lookup_rule = cfg.get("lookup_rule", "contract")
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


def floor_kw(params, G):
    """Lowest facility power: every GPU idle, host idle power, times PUE (nothing running on-prem)."""
    return params.pue * G * (params.p_idle_w + params.h_idle_w) / 1000.0


def timed_row(cfg, exp, snap, ev, name, world, base, predictor_name="", selector="predictor", setting="", pool=None):
    t = time.perf_counter()
    row = run_policy(snap, ev, name, world, predictor_name, selector, cfg["seed"], base)
    row.update(exp=exp, setting=setting, runtime_s=round(time.perf_counter() - t, 3), seed=cfg["seed"])
    if pool is not None:                                       # S13: P* below the all-idle floor cannot be met by any policy
        row["feasible"] = bool(row["target_kw"] >= floor_kw(world.params, pool) - 1e-9)
    return row


# ---- experiments ----

def e1(cfg, jobs, pool, events, predictors, predictor, quick):
    """S16a. Per event: cap (no predictor); delay, burst, mix with last2, transformer and the oracle (upper bound);
    mix with the p90 cloud bill (transformer); in the default cell also delay with gbdt and lightgbm (attribution control)."""
    world = world_from_config(cfg, predictors[predictor])
    names = [n for n in ("last2", "transformer", "oracle") if n in predictors]

    def per_event(snap, ev, base):
        rows = [timed_row(cfg, "e1", snap, ev, "cap", world, base, pool=pool)]
        for pname in names:
            w = dataclasses.replace(world, predictor=predictors[pname])
            rows += [timed_row(cfg, "e1", snap, ev, n, w, base, pname, pool=pool) for n in ("delay", "burst", "mix")]
            if pname == "transformer":
                rows.append(timed_row(cfg, "e1", snap, ev, "mix", dataclasses.replace(w, cost_quantile="p90"), base, pname, setting="cq=p90", pool=pool))
        if (ev.duration_h, ev.cut) == DEFAULT_EVENT:
            for pname in ("gbdt", "lightgbm"):
                if pname in predictors:
                    rows.append(timed_row(cfg, "e1", snap, ev, "delay", dataclasses.replace(world, predictor=predictors[pname]), base, pname,
                                          setting="attribution", pool=pool))
        return rows
    return run_events(cfg, jobs, pool, world, events, per_event, "e1")


PRICES = [0.43, 1.5, 3.06, 6.12]      # USD per GPU-h: spot (S11 source), 0.5x, base, 2x
BOOTS = [60, 300, 900]
CAPS = [1000, 3000]


def setting_text(pool, price, boot, cap):
    return f"pool={pool};price={price};boot={boot};cap={cap}"


def cloud_variant(world, price, boot, cap):
    return dataclasses.replace(world, cloud=dataclasses.replace(world.cloud, price_per_gpu_h=price, boot_delay_s=boot, max_gpus=cap))


def policy_rows(cfg, exp, snap, ev, base, world, predictors, pool, price, boot, cap, last2_burst=False):
    """S13c, one cloud setting: mix with last2, transformer, oracle; mix-p90 (transformer); burst with transformer and oracle
    (and last2 only when last2_burst, i.e. on the default setting)."""
    w = cloud_variant(world, price, boot, cap)
    st = setting_text(pool, price, boot, cap)
    rows = []
    for pname in ("last2", "transformer", "oracle"):
        rows.append(timed_row(cfg, exp, snap, ev, "mix", dataclasses.replace(w, predictor=predictors[pname]), base, pname, setting=st, pool=pool))
    rows.append(timed_row(cfg, exp, snap, ev, "mix", dataclasses.replace(w, predictor=predictors["transformer"], cost_quantile="p90"), base,
                          "transformer", setting=st + ";cq=p90", pool=pool))
    for pname in ("transformer", "oracle") + (("last2",) if last2_burst else ()):
        rows.append(timed_row(cfg, exp, snap, ev, "burst", dataclasses.replace(w, predictor=predictors[pname]), base, pname, setting=st, pool=pool))
    return rows


def delay_rows(cfg, exp, snap, ev, base, world, predictors, pool, settings):
    """delay with last2, transformer and oracle does not depend on the cloud: run once, copy to every setting."""
    out = []
    for pname in ("last2", "transformer", "oracle"):
        w = dataclasses.replace(world, predictor=predictors[pname])
        r = timed_row(cfg, exp, snap, ev, "delay", w, base, pname, pool=pool)
        out += [dict(r, setting=st) for st in settings]
    return out


def e2(cfg, jobs, pool, events, predictors, predictor, quick, prices=None):
    """S13c sweep A, default cell (all 10 events): price x boot delay. c_delay stays at its default while the cloud price is swept.
    Pool = auto, cloud cap 1000. `prices` restricts the sweep to some prices (split runs)."""
    world = world_from_config(cfg, predictors[predictor])
    prices, boots = ([1.5, 3.06], [60, 300]) if quick else (prices or PRICES, BOOTS)
    settings = [(p, b) for p in prices for b in boots]
    base_price, base_boot = cfg["cloud"]["price_per_gpu_h"], cfg["cloud"]["boot_delay_s"]

    def per_event(snap, ev, base):
        rows = delay_rows(cfg, "e2", snap, ev, base, world, predictors, pool, [setting_text(pool, p, b, 1000) for p, b in settings])
        for p, b in settings:
            rows += policy_rows(cfg, "e2", snap, ev, base, world, predictors, pool, p, b, 1000, last2_burst=(p == base_price and b == base_boot))
        return rows
    return run_events(cfg, jobs, pool, world, cell(events, *DEFAULT_EVENT), per_event, "e2")


def e2b(cfg, jobs, pool, events, predictors, predictor, quick):
    """S13c sweep B for ONE pool (the pool comes from philly.pool_gpus, set by --pool): price x cloud cap at boot 300 s.
    Cap 1000 at four prices (pool x price map) and cap 3000 at the base price (saturation). w_fail 0 / 1 are both in the rows."""
    world = world_from_config(cfg, predictors[predictor])
    base_price = cfg["cloud"]["price_per_gpu_h"]
    boot = cfg["cloud"]["boot_delay_s"]
    combos = [(p, boot, 1000) for p in PRICES] + [(base_price, boot, 3000)]

    def per_event(snap, ev, base):
        rows = delay_rows(cfg, "e2b", snap, ev, base, world, predictors, pool, [setting_text(pool, p, b, c) for p, b, c in combos])
        for p, b, c in combos:
            rows += policy_rows(cfg, "e2b", snap, ev, base, world, predictors, pool, p, b, c, last2_burst=(p == base_price and c == 1000))
        return rows
    return run_events(cfg, jobs, pool, world, cell(events, *DEFAULT_EVENT), per_event, "e2b")


def perturbations(cfg):
    """T4: name -> (changed cfg copy, cap curve variant). One change at a time; defaults are in the sweep-A cell price 3.06 / boot 300."""
    def mod(fn, variant=None):
        c = copy.deepcopy(cfg)
        fn(c)
        return c, variant
    P = {}
    for m in (0, 3):
        P[f"m={m}"] = mod(lambda c, m=m: c["cost"].__setitem__("m", m))
    for f in (0.5, 2):
        P[f"c_delay x{f}"] = mod(lambda c, f=f: c["cost"].__setitem__("c_delay_factor", f))
    P["pessimistic cap curve"] = mod(lambda c: None, "pessimistic")
    P["p_idle 40 W"] = mod(lambda c: c["power"].__setitem__("p_idle_w", 40))       # 70 W is the default
    for f in (0.5, 1.5):
        P[f"host power x{f}"] = mod(lambda c, f=f: (c["power"].__setitem__("h_idle_w", c["power"]["h_idle_w"] * f),
                                                    c["power"].__setitem__("h_busy_w", c["power"]["h_busy_w"] * f)))
    P["PUE 1.3"] = mod(lambda c: c["power"].__setitem__("pue", 1.3))               # 1.58 is the default
    for f in (0.5, 2):
        P[f"migrate/restore x{f}"] = mod(lambda c, f=f: (c["cloud"].__setitem__("migrate_s", int(c["cloud"]["migrate_s"] * f)),
                                                        c["cloud"].__setitem__("restore_s", int(c["cloud"]["restore_s"] * f))))
    return P


def t4(cfg, jobs, pool, events, predictors, predictor, quick, only=None):
    """S13c T4: delay (last2, transformer, oracle), burst (transformer, oracle), mix (last2, transformer, oracle) and mix-p90
    (transformer) on the default cell under each perturbation."""
    perts = {k: v for k, v in perturbations(cfg).items() if only is None or k in only}

    def per_event(snap, ev, base0):
        rows = []
        for name, (c, variant) in perts.items():
            world = world_from_config(c, predictors[predictor], variant)
            base = run_base(snap, ev, world.params, world.h_after_s)
            st = f"pert={name}"
            for pname in ("last2", "transformer", "oracle"):
                w = dataclasses.replace(world, predictor=predictors[pname])
                rows.append(timed_row(c, "t4", snap, ev, "delay", w, base, pname, setting=st, pool=pool))
                rows.append(timed_row(c, "t4", snap, ev, "mix", w, base, pname, setting=st, pool=pool))
                if pname != "last2":
                    rows.append(timed_row(c, "t4", snap, ev, "burst", w, base, pname, setting=st, pool=pool))
                if pname == "transformer":
                    rows.append(timed_row(c, "t4", snap, ev, "mix", dataclasses.replace(w, cost_quantile="p90"), base, pname, setting=st + ";cq=p90", pool=pool))
        return rows
    world = world_from_config(cfg, predictors[predictor])
    return run_events(cfg, jobs, pool, world, cell(events, *DEFAULT_EVENT), per_event, "t4")


def nocloud_world(world, predictor, quantile="p50"):
    """S13d control: the same mix controller and predictor, cloud switched off (max_cloud_gpus = 0). Every planned burst is refused,
    so the mix can only delay (a refused burst becomes a suspension) and cap. The controller logic is not changed."""
    w = cloud_variant(world, world.cloud.price_per_gpu_h, world.cloud.boot_delay_s, 0)
    return dataclasses.replace(w, predictor=predictor, cost_quantile=quantile)


def e1nc(cfg, jobs, pool, events, predictors, predictor, quick):
    """S13d (1): mix-nocloud on the E1 events with last2, transformer (p50 and p90) and the oracle."""
    world = world_from_config(cfg, predictors[predictor])

    def per_event(snap, ev, base):
        rows = [timed_row(cfg, "e1nc", snap, ev, "mix", nocloud_world(world, predictors[p]), base, p, setting="nocloud", pool=pool)
                for p in ("last2", "transformer", "oracle")]
        rows.append(timed_row(cfg, "e1nc", snap, ev, "mix", nocloud_world(world, predictors["transformer"], "p90"), base, "transformer",
                              setting="nocloud;cq=p90", pool=pool))
        return rows
    return run_events(cfg, jobs, pool, world, events, per_event, "e1nc")


def e2nc(cfg, jobs, pool, events, predictors, predictor, quick):
    """S13d (2): mix-nocloud (transformer p90, oracle) on the default cell for ONE pool (--pool) and prices 0.43 and 3.06."""
    world = world_from_config(cfg, predictors[predictor])
    boot = cfg["cloud"]["boot_delay_s"]

    def per_event(snap, ev, base):
        rows = []
        for price in (0.43, 3.06):
            for p, q in (("transformer", "p90"), ("oracle", "p50")):
                w = nocloud_world(cloud_variant(world, price, boot, 0), predictors[p], q)
                rows.append(timed_row(cfg, "e2nc", snap, ev, "mix", w, base, p, setting=setting_text(pool, price, boot, 0) + (";cq=p90" if q == "p90" else ""), pool=pool))
        return rows
    return run_events(cfg, jobs, pool, world, cell(events, *DEFAULT_EVENT), per_event, "e2nc")


E3_PARTS = {"default": (DEFAULT_EVENT, 1000, False), "hard": (HARD_EVENT, 3000, False), "oos": (DEFAULT_EVENT, 1000, True)}
LEARNED = ["gbdt", "lightgbm", "transformer"]         # last2 has p90 = p50, so its p90 variant would be identical


def e3(cfg, jobs, pool, events, predictors, predictor, quick, part="default", min_submit=None):
    """S14. Burst-only and mix by selector (none-largest, random, every predictor file), mix also with the p90 cloud bill
    for the learned predictors, plus delay (last2, oracle) as reference rows. Parts: default = 2 h / 20% with cloud cap 1000;
    hard = 4 h / 30% with cap 3000; oos = the default cell with burst candidates limited to jobs submitted at or after
    the test start (min_submit, simulated seconds), out-of-sample for the learned predictors."""
    (dur, cut), cap, oos = E3_PARTS[part]
    base_world = world_from_config(cfg, predictors[predictor])
    price, boot = cfg["cloud"]["price_per_gpu_h"], cfg["cloud"]["boot_delay_s"]
    world = cloud_variant(base_world, price, boot, cap)
    if oos:
        world = dataclasses.replace(world, burst_min_submit=min_submit)
    st = f"cap={cap}" + (";burst>=test_start" if oos else "")

    def per_event(snap, ev, base):
        rows = []
        for name in ("burst", "mix"):
            for sel in BLIND_SELECTORS:
                rows.append(timed_row(cfg, "e3", snap, ev, name, world, base, "", sel, st, pool))
            for pname, pred in predictors.items():
                rows.append(timed_row(cfg, "e3", snap, ev, name, dataclasses.replace(world, predictor=pred), base, pname, setting=st, pool=pool))
        for pname in LEARNED:
            if pname in predictors:
                w = dataclasses.replace(world, predictor=predictors[pname], cost_quantile="p90")
                rows.append(timed_row(cfg, "e3", snap, ev, "mix", w, base, pname, setting=st + ";cq=p90", pool=pool))
        if not oos:
            for pname in ("last2", "oracle"):
                if pname in predictors:
                    rows.append(timed_row(cfg, "e3", snap, ev, "delay", dataclasses.replace(world, predictor=predictors[pname]), base, pname, setting=st, pool=pool))
        return rows
    return run_events(cfg, jobs, pool, world, cell(events, dur, cut), per_event, f"e3-{part}")


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
            "rebound_mwh", "missed_deadlines", "dgpuh_good", "dgpuh_all", "n_suspended", "n_burst", "cap_level", "feasible"]
    out = Path(out_dir) / "e4"
    out.mkdir(parents=True, exist_ok=True)
    runs[cols].to_csv(out / "runs.csv", index=False)
    summarise(runs).to_csv(out / "summary.csv", index=False)
    return runs


def baseline_waits(cfg, jobs, pool, out_dir, label):
    """Baseline (no event) wait of the test-window jobs for this pool: mean, p50, p90 (min) and share waiting > 60 s."""
    import numpy as np
    eng = Engine(copy.deepcopy(jobs), pool, check_every=cfg["check_every"])
    eng.run_to_drain()
    w = np.array([j.first_start - j.submit for j in eng.jobs.values() if not j.warmup], dtype=float)
    row = dict(pool=label, pool_gpus=pool, jobs=len(w), wait_mean_min=w.mean() / 60, wait_p50_min=np.percentile(w, 50) / 60,
               wait_p90_min=np.percentile(w, 90) / 60, frac_wait_gt_60s=float((w > 60).mean()),
               carry_jobs=sum(j.carry for j in jobs), carry_gpus=sum(j.gpus for j in jobs if j.carry))
    out = Path(out_dir) / "e2b"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).round(3).to_csv(out / f"baseline_{label}.csv", index=False)
    print("baseline waits:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m flex.experiments")
    ap.add_argument("exp", choices=["e1", "e1merge", "e1nc", "e2", "e2b", "e2nc", "e3", "e4", "e5", "t4"])
    ap.add_argument("--data", choices=["synthetic", "philly"], required=True)
    ap.add_argument("--predictor", default="last2")
    ap.add_argument("--quick", action="store_true", help="3 events per cell (and smaller sweeps)")
    ap.add_argument("--out", default=None, help="results directory (default: <repo>/results)")
    ap.add_argument("--pool", default=None, help="e2b: installed | auto | 0.95 | 0.9 (x auto); sets philly.pool_gpus")
    ap.add_argument("--only", default=None, help="t4: perturbation names separated by |")
    ap.add_argument("--prices", default=None, help="e2: prices (USD/GPU-h) separated by |, for split runs")
    ap.add_argument("--duration", type=float, default=None, help="e1: only events of this length (h), for split runs")
    ap.add_argument("--cut", type=float, default=None, help="e1: only events with this cut, for split runs")
    ap.add_argument("--lookup-rule", default=None, choices=["contract", "hold_beyond_grid"], help="e3: S14c prediction lookup rule beyond the last grid point")
    ap.add_argument("--part", default="default", choices=list(E3_PARTS), help="e3: default | hard | oos")
    ap.add_argument("--tag", default="", help="suffix of the runs file for split runs (runs_<tag>.csv)")
    args = ap.parse_args(argv)
    cfg = load_config()
    out_dir = Path(args.out) if args.out else ROOT / cfg["paths"]["results"]
    if args.exp == "e5":
        raise SystemExit("e5 (carbon) is phase S15")
    if args.exp == "e1merge":                                # S16a: merge the split E1 runs (runs_<tag>.csv) into runs.csv and summary.csv
        files = sorted((out_dir / "e1").glob("runs_*.csv"))
        runs = pd.concat([pd.read_csv(f, dtype={"selector": str, "predictor": str, "setting": str}) for f in files], ignore_index=True)
        runs.to_csv(out_dir / "e1" / "runs.csv", index=False)
        summarise(runs).to_csv(out_dir / "e1" / "summary.csv", index=False)
        print(f"merged {len(files)} files, {len(runs)} rows, {runs['event_id'].nunique()} events")
        return
    if args.exp == "e4":
        if not (out_dir / "e1" / "runs.csv").exists():
            main(["e1", "--data", args.data, "--predictor", args.predictor] + (["--quick"] if args.quick else [])
                 + (["--out", str(out_dir)] if args.out else []))
        e4_from_e1(out_dir)
        print("wrote", out_dir / "e4")
        return
    if args.exp in ("e2b", "e2nc"):
        from traces.philly_pool import load_pool
        d = load_pool()
        auto = d["pool_gpus"]
        cfg["philly"]["pool_gpus"] = {"installed": d["installed_gpus"], "auto": "auto"}.get(args.pool, None) or int(round(float(args.pool) * auto))
    cfg, jobs, pool, events = load_setup(cfg, args.data, args.quick)
    if args.exp in ("e1", "e1nc") and (args.duration is not None or args.cut is not None):
        events = [e for e in events if (args.duration is None or e.duration_h == args.duration) and (args.cut is None or abs(e.cut - args.cut) < 1e-9)]
    predictors = load_predictors(cfg, args.data)
    if args.predictor not in predictors:
        raise SystemExit(f"no predictions/{args.predictor}.csv for --data {args.data}")
    t0 = time.perf_counter()
    if args.exp == "e2b":
        baseline_waits(cfg, jobs, pool, out_dir, args.pool)
    if args.exp == "e3":
        from traces.philly import ml_config, utc_epoch
        mc = ml_config()
        min_sub = utc_epoch(mc.TEST_START) - utc_epoch(mc.SIM_START) if args.data == "philly" else None
        rule = args.lookup_rule or cfg.get("lookup_rule", "contract")
        for pr in predictors.values():
            pr.lookup_rule = rule
        args.tag = args.tag or (args.part + ("_hold" if rule != "contract" else ""))
        rows = e3(cfg, jobs, pool, events, predictors, args.predictor, args.quick, args.part, min_sub)
    elif args.exp == "t4":
        rows = t4(cfg, jobs, pool, events, predictors, args.predictor, args.quick, args.only.split("|") if args.only else None)
    else:
        extra = dict(prices=[float(x) for x in args.prices.split("|")]) if (args.exp == "e2" and args.prices) else {}
        rows = {"e1": e1, "e1nc": e1nc, "e2": e2, "e2b": e2b, "e2nc": e2nc}[args.exp](cfg, jobs, pool, events, predictors, args.predictor, args.quick, **extra)
    if args.tag:                                          # split runs: one file per piece, merged by the analysis script
        out = Path(out_dir) / args.exp
        out.mkdir(parents=True, exist_ok=True)
        runs = pd.DataFrame(rows, columns=COLUMNS)
        runs.to_csv(out / f"runs_{args.tag}.csv", index=False)
        print(f"{args.exp}[{args.tag}]: {len(runs)} rows in {time.perf_counter() - t0:.0f} s -> {out}")
        return
    runs, summary = write_results(args.exp, rows, out_dir)
    print(f"{args.exp}: {len(runs)} rows in {time.perf_counter() - t0:.0f} s -> {out_dir / args.exp}")
    if args.exp == "e1":
        e1_table(runs)


if __name__ == "__main__":
    main()
