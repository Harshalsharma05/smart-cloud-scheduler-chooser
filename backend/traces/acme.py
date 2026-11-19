"""AcmeTrace Seren jobs for the engine (step 35, S-A1). Reads data/processed/acme_jobs.parquet (Track 1, A1);
ACME_SIM_START / ACME_TEST_START / ACME_TEST_END come from ml/config.py (read, never copied).

Same rules as the Philly loader (traces/philly.py) unless stated:
- Seren GPU jobs only. Time: UTC epoch seconds; simulated t = epoch - epoch0, epoch0 = ACME_SIM_START. Jobs submitted in
  [epoch0, ACME_TEST_START) are warm-up (no deadline, not measured); events only in [ACME_TEST_START, ACME_TEST_END),
  start times in China time (flex.yaml acme.grid.tz).
- Carry-over: a job running at epoch0 keeps its id, is submitted at t = 0 with wall_s = end - epoch0 and elapsed0 =
  epoch0 - first_start (as Philly). If the carry-overs exceed G, the latest-started ones are dropped (counted).
- Status (decision M1): waste = raw_state FAILED / NODE_FAIL / TIMEOUT -> "Failed"; COMPLETED, CANCELLED and PREEMPTED
  -> "Pass" (useful work in the cost metrics). The controller uses P(useful) = 1 - p_fail (acme.p_useful = fail_only).
- Power (M2): one cluster-wide utilisation acme.util for every job (per-job utilisation is not in the job trace),
  calibrated by `python -m traces.acme` so the simulated mean GPU power over the test window is acme.target_gpu_power_w.
- Dropped: never-started jobs, 0 GPUs, wall_s <= 0 (counted). Seren has no censored jobs (checked).
"""
import copy
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from flex.grid import epoch_from_iso, generate_events
from flex.models import FJob, load_config
from traces.philly import _busy_per_minute, check_columns, ml_config
from traces.synthetic import assign_deadlines

ROOT = Path(__file__).resolve().parents[2]
JOB_COLUMNS = ["job_id", "user", "cluster", "type", "submit_time", "first_start_time", "end_time", "wall_s", "num_gpus",
               "status", "raw_state", "censored", "never_started"]
WASTE_STATES = ("FAILED", "NODE_FAIL", "TIMEOUT")
POWER_PKL = ROOT / "data" / "raw" / "acme" / "AcmeTrace" / "data" / "utilization" / "util_pkl"


def acme_epoch(text):
    """UTC epoch of an ml/config.py ACME_* string such as '2023-06-21 16:00' (UTC)."""
    return epoch_from_iso(text.replace(" ", "T") + ":00+00:00")


def acme_config(cfg):
    """A copy of cfg with the acme: section applied (power, cloud price and cap, event time zone, P(useful) rule).
    Philly keys that acme: does not name (PUE, cap curve, overheads, grid shape, costs, deadlines) stay as they are."""
    cfg = copy.deepcopy(cfg)
    a = cfg["acme"]
    cfg["power"].update(a["power"])
    cfg["cloud"].update(a["cloud"])
    cfg["grid"].update(a["grid"])
    cfg["p_useful"] = a["p_useful"]
    return cfg


def load_acme(cfg):
    """cfg must already be acme_config(cfg). Returns jobs, G, epoch0, test window (simulated s), events, side info, report."""
    mc = ml_config()
    a = cfg["acme"]
    raw = pd.read_parquet(ROOT / cfg["paths"]["data"] / "processed" / "acme_jobs.parquet")
    check_columns(raw, JOB_COLUMNS, "acme_jobs.parquet")
    raw = raw[raw["cluster"] == a["cluster"]]
    G = int(a["pool_gpus"])
    u = float(a["util"])
    epoch0, ts_utc, te_utc = acme_epoch(mc.ACME_SIM_START), acme_epoch(mc.ACME_TEST_START), acme_epoch(mc.ACME_TEST_END)
    rep = {"rows_seren": len(raw), "censored": int(raw["censored"].sum())}
    if rep["censored"]:
        raise SystemExit("acme_jobs.parquet: Seren has censored jobs; the loader assumes none (stop and report)")
    d = raw[~raw["never_started"] & (raw["num_gpus"] > 0)].copy()
    rep["never_started_or_zero_gpu_dropped"] = len(raw) - len(d)
    for c in ("submit_time", "first_start_time", "end_time", "wall_s"):
        d[c] = d[c].astype("int64")
    rep["wall_nonpositive_dropped"] = int((d["wall_s"] <= 0).sum())
    d = d[d["wall_s"] > 0]
    d["sim_status"] = np.where(d["raw_state"].isin(WASTE_STATES), "Failed", "Pass")
    rep["gpus_above_G"] = int((d["num_gpus"] > G).sum())
    info = d.set_index("job_id")[["type", "raw_state", "num_gpus"]]

    before = d[d["submit_time"] < epoch0]
    end_abs = before["first_start_time"] + before["wall_s"]
    running = before[(before["first_start_time"] < epoch0) & (end_abs > epoch0)].copy()
    rep["before_epoch0_total"] = len(before)
    rep["before_epoch0_queued_at_epoch0_ignored"] = int((before["first_start_time"] >= epoch0).sum())
    rep["carry_candidates"] = len(running)
    rep["carry_candidates_gpus"] = int(running["num_gpus"].sum())
    running["remaining"] = (end_abs[running.index] - epoch0).astype(int)
    carry, used = [], 0
    for r in running.sort_values(["first_start_time", "job_id"]).itertuples(index=False):
        if used + r.num_gpus > G:
            continue
        used += int(r.num_gpus)
        carry.append(FJob(job_id=r.job_id, user=r.user, submit=0, gpus=int(r.num_gpus), wall_s=int(r.remaining),
                          status=r.sim_status, util=u, warmup=True, carry=True,
                          elapsed0=int(epoch0 - r.first_start_time), submit_real=int(r.submit_time - epoch0)))
    rep.update(carry_jobs=len(carry), carry_gpus=used, carry_truncated=len(running) - len(carry))

    d = d[(d["submit_time"] >= epoch0) & (d["submit_time"] < te_utc)].sort_values(["submit_time", "job_id"])
    jobs = carry + [FJob(job_id=r.job_id, user=r.user, submit=int(r.submit_time - epoch0), gpus=int(r.num_gpus),
                         wall_s=int(r.wall_s), status=r.sim_status, util=u, warmup=bool(r.submit_time < ts_utc))
                    for r in d.itertuples(index=False)]
    test_jobs = [j for j in jobs if not j.warmup]
    assign_deadlines(test_jobs, random.Random(cfg["seed"] + 1), cfg["deadlines"])
    test_start, test_end = ts_utc - epoch0, te_utc - epoch0
    events = generate_events(random.Random(cfg["seed"]), epoch0, test_start, test_end, cfg["grid"])
    rep.update(jobs=len(jobs), warmup_jobs=len(jobs) - len(test_jobs) - len(carry), test_jobs=len(test_jobs), G=G, util=u,
               test_status=pd.Series([j.status for j in test_jobs]).value_counts().to_dict())
    real = d.set_index("job_id")[["first_start_time"]]
    all_started = raw[~raw["never_started"] & (raw["num_gpus"] > 0)]
    return dict(jobs=jobs, G=G, epoch0=epoch0, test_start_utc=ts_utc, test_end_utc=te_utc, test_start=test_start,
                test_end=test_end, events=events, info=info, report=rep, real=real, all_started=all_started)


def _safe_load(path):
    """The power pickles hold only numpy arrays; a restricted unpickler refuses anything else."""
    allowed = {("numpy.core.multiarray", "_reconstruct"), ("numpy", "ndarray"), ("numpy", "dtype"),
               ("numpy._core.multiarray", "_reconstruct")}

    class U(pickle.Unpickler):
        def find_class(self, m, n):
            if (m, n) in allowed:
                return super().find_class(m, n)
            raise pickle.UnpicklingError(f"blocked {m}.{n}")

    with open(path, "rb") as f:
        return np.asarray(U(f).load())


def power_sources():
    """The CDF readings behind acme.power and acme.target_gpu_power_w (W): GPU mean / p10, GPU-node server p1 / p10 / p90 / p99."""
    g = _safe_load(POWER_PKL / "gpu_power_seren.pkl")
    s = _safe_load(POWER_PKL / "server_power.pkl")
    q = lambda a, p: float(np.interp(p, a[1], a[0]))
    return {"gpu_mean": float(g[0].mean()), "gpu_p10": q(g, 10), "srv_p1": q(s, 1), "srv_p10": q(s, 10),
            "srv_p90": q(s, 90), "srv_p99": q(s, 99)}


def host_power(src, p_idle, p_max, lo="srv_p1", hi="srv_p99"):
    """(h_idle, h_busy) per GPU from an 8-GPU node: low end = idle host + 8 idle GPUs; high end = idle host + 8 x (p_max + h_busy)."""
    host_idle = src[lo] - 8 * p_idle
    return host_idle / 8, (src[hi] - host_idle - 8 * p_max) / 8


def busy_gpu_seconds(eng, t0, t1):
    """On-prem busy GPU-seconds in [t0, t1) of a finished run, from the job segments."""
    tot = 0
    for j in eng.jobs.values():
        for s_, e_, where, _ in j.segments:
            if where == "onprem":
                tot += j.gpus * max(0, min(e_, t1) - max(s_, t0))
    return tot


def validation(L, eng, tracker, cfg, out_dir):
    """Table I for Acme: real vs simulated busy GPUs, utilisation, waits (test window); calibration of u."""
    ts, te, e0, G = L["test_start"], L["test_end"], L["epoch0"], L["G"]
    a = L["all_started"]
    real_busy = _busy_per_minute(a["first_start_time"].astype(float), a["end_time"].astype(float), a["num_gpus"].to_numpy(),
                                 L["test_start_utc"], L["test_end_utc"])
    segs = [(s_, e_, j.gpus) for j in eng.jobs.values() for s_, e_, where, _ in j.segments if where == "onprem"]
    sim_busy = _busy_per_minute([x[0] for x in segs], [x[1] for x in segs], [x[2] for x in segs], ts, te)
    test = [j for j in eng.jobs.values() if not j.warmup]
    rw = np.array([L["real"].loc[j.job_id, "first_start_time"] - (j.submit + e0) for j in test], dtype=float)
    sw = np.array([j.first_start - j.submit for j in test], dtype=float)
    rows = [("busy_gpus_mean", real_busy.mean(), sim_busy.mean()), ("busy_gpus_p95", np.percentile(real_busy, 95), np.percentile(sim_busy, 95)),
            ("utilisation_mean", real_busy.mean() / G, sim_busy.mean() / G),
            ("wait_mean_min", rw.mean() / 60, sw.mean() / 60), ("wait_p50_min", np.percentile(rw, 50) / 60, np.percentile(sw, 50) / 60),
            ("wait_p90_min", np.percentile(rw, 90) / 60, np.percentile(sw, 90) / 60),
            ("frac_wait_gt_60s", (rw > 60).mean(), (sw > 60).mean())]
    val = pd.DataFrame(rows, columns=["metric", "real", "simulated"])
    p = cfg["power"]
    busy_share = busy_gpu_seconds(eng, ts, te) / (G * (te - ts))
    target = float(cfg["acme"]["target_gpu_power_w"])
    u_cal = (target - p["p_idle_w"]) / ((p["p_max_w"] - p["p_idle_w"]) * busy_share)
    u = float(cfg["acme"]["util"])
    gpu_mean_w = p["p_idle_w"] + busy_share * u * (p["p_max_w"] - p["p_idle_w"])
    src = power_sources()
    hi_, hb_ = host_power(src, p["p_idle_w"], p["p_max_w"])
    hi10, hb10 = host_power(src, p["p_idle_w"], p["p_max_w"], "srv_p10", "srv_p90")
    cal = pd.DataFrame([dict(busy_share_sim=busy_share, target_gpu_power_w=target, u_calibrated=u_cal, u_used=u,
                             sim_mean_gpu_power_w=gpu_mean_w, facility_mean_kw=tracker.mean_kw(ts, te),
                             floor_kw=p["pue"] * G * (p["p_idle_w"] + p["h_idle_w"]) / 1000.0,
                             cdf_gpu_mean_w=src["gpu_mean"], cdf_gpu_p10_w=src["gpu_p10"], srv_p1_w=src["srv_p1"], srv_p99_w=src["srv_p99"],
                             h_idle_p1_w=hi_, h_busy_p99_w=hb_, srv_p10_w=src["srv_p10"], srv_p90_w=src["srv_p90"],
                             h_idle_p10_w=hi10, h_busy_p90_w=hb10)])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    val.round(3).to_csv(out / "validation.csv", index=False)
    cal.round(4).to_csv(out / "calibration.csv", index=False)
    print(f"baseline validation, test window ({len(test)} jobs; G = {G}):")
    print(val.round(3).to_string(index=False))
    print(cal.round(3).T.to_string(header=False))
    return val, cal


def main():
    """python -m traces.acme (from backend/): load, one baseline run, Table I check and the u calibration."""
    from flex.engine import Engine
    from flex.policies import world_from_config
    from flex.power import PowerTracker

    cfg = acme_config(load_config())
    t0 = time.perf_counter()
    L = load_acme(cfg)
    print(f"load: {time.perf_counter() - t0:.1f} s")
    for k, v in L["report"].items():
        print(f"  {k}: {v}")
    print(f"  events: {len(L['events'])}, starts (China time):",
          sorted({pd.Timestamp(L['epoch0'] + e.t_start, unit='s', tz='UTC').tz_convert('Asia/Shanghai').strftime('%m-%d %H:%M') for e in L['events']}))
    world = world_from_config(cfg, None)
    eng = Engine(copy.deepcopy(L["jobs"]), L["G"], check_every=cfg["check_every"])
    tracker = PowerTracker(world.params)
    tracker.attach(eng)
    t0 = time.perf_counter()
    eng.run_to_drain()
    print(f"baseline run: {len(L['jobs'])} jobs, {eng.n_events} events, {time.perf_counter() - t0:.1f} s")
    validation(L, eng, tracker, cfg, ROOT / cfg["paths"]["results"] / "acme_baseline")


if __name__ == "__main__":
    main()
