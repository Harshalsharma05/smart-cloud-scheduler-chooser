"""A1: load the AcmeTrace job traces into data/processed/acme_jobs.parquet and write ml/reports/A1_acme_report.md.

Run from the repo root:  ml\\.venv\\Scripts\\python ml\\acme_load.py
Data: data/raw/acme/AcmeTrace (github.com/InternLM/AcmeTrace, CC-BY-4.0; Hu et al., NSDI 2024). Philly data is only read.

Time zone: Seren strings carry +08:00 (China time), Kalos strings carry +00:00 (UTC). Both are parsed with their own
offset into UTC epoch seconds. Evidence that Kalos' true local time is also China time is in the report (hour-of-day table).
"""
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

RAW = config.DATA_RAW / "acme" / "AcmeTrace" / "data"
OUT = config.DATA_PROCESSED / "acme_jobs.parquet"
REPORT = config.REPORTS / "A1_acme_report.md"
LOCAL_TZ = {"seren": "Asia/Shanghai", "kalos": "Asia/Shanghai", "philly": "America/Los_Angeles"}
STATE_MAP = {"COMPLETED": "Pass", "CANCELLED": "Killed", "PREEMPTED": "Killed",
             "FAILED": "Failed", "TIMEOUT": "Failed", "NODE_FAIL": "Failed"}
DAY_SEED, N_DAYS = 0, 10
LONG_S = 12 * 3600
COLS = ["job_id", "user", "vc", "cluster", "type", "submit_time", "first_start_time", "end_time", "wall_s", "num_gpus",
        "status", "raw_state", "censored", "never_started", "node_num"]


# ------------------------------------------------------------------- loading
def epoch(col):
    """Offset-aware time strings -> UTC epoch seconds (Int64, NA for missing)."""
    t = pd.to_datetime(col, utc=True)
    return ((t - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().round()).astype("Int64")


def load_cluster(name):
    """One cluster's raw job trace -> table with the philly_jobs columns plus cluster, type, raw_state, node_num.
    All jobs (also CPU-only ones); the caller filters num_gpus >= 1."""
    d = pd.read_csv(RAW / "job_trace" / f"trace_{name}.csv", dtype={"job_id": str, "user": str})
    out = pd.DataFrame({
        "job_id": d["job_id"], "user": d["user"],
        "vc": name,  # `queue` in the raw file is the queue wait in seconds, not a queue name: use the cluster
        "cluster": name, "type": d["type"],
        "submit_time": epoch(d["submit_time"]), "first_start_time": epoch(d["start_time"]), "end_time": epoch(d["end_time"]),
        "num_gpus": d["gpu_num"].astype("int64"), "raw_state": d["state"], "node_num": d["node_num"].astype("int64")})
    out["status"] = out["raw_state"].map(STATE_MAP)  # RUNNING (unfinished) stays null
    out["never_started"] = out["first_start_time"].isna()
    out["censored"] = out["end_time"].isna() & ~out["never_started"]
    out["wall_s"] = (out["end_time"] - out["first_start_time"]).astype("Int64")
    unknown = set(d["state"]) - set(STATE_MAP) - {"RUNNING"}
    assert not unknown, f"unmapped states {unknown}"
    return out[COLS]


def load_all():
    return pd.concat([load_cluster("seren"), load_cluster("kalos")], ignore_index=True)


# ----------------------------------------------------------------- utilities
def gpu_hours(df):
    return (df["wall_s"].astype("float64").fillna(0).clip(lower=0) * df["num_gpus"]) / 3600


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def md(df, floats=2):
    cols = list(df.columns)
    L = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("-" if np.isnan(v) else f"{v:,.{floats}f}")
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{v:,}")
            else:
                cells.append(str(v))
        L.append("| " + " | ".join(cells) + " |")
    return "\n".join(L)


def share_table(df, key, label):
    """Shares of jobs and of GPU-hours by `key` (status/type/raw state), non-censored jobs with wall_s > 0 for GPU-hours."""
    g = df.assign(gh=gpu_hours(df), _k=df[key].fillna("(unfinished)"))
    t = g.groupby("_k").agg(jobs=("job_id", "size"), gpu_hours=("gh", "sum"))
    t["% jobs"] = 100 * t["jobs"] / t["jobs"].sum()
    t["% GPU-hours"] = 100 * t["gpu_hours"] / t["gpu_hours"].sum()
    t = t.sort_values("gpu_hours", ascending=False).reset_index().rename(columns={"_k": label})
    return t


def wall_percentiles(df):
    """Wall-time percentiles (minutes) by mapped status among finished jobs with wall_s > 0."""
    f = df[(~df["censored"]) & (~df["never_started"]) & (df["wall_s"].astype("float64") > 0)]
    rows = []
    for st, g in [("all", f)] + list(f.groupby("status")):
        w = g["wall_s"].astype("float64") / 60
        rows.append({"status": st, "jobs": len(g), "p10": w.quantile(.1), "p50": w.quantile(.5), "p90": w.quantile(.9),
                     "p99": w.quantile(.99), "mean": w.mean()})
    return pd.DataFrame(rows)


def running_at_slots(df, cluster, days, tz):
    """Mean running jobs / GPUs and the share of running GPUs held by jobs already running > 12 h, at 30-min slots
    17:00, 17:30, ..., 19:30 local time on each day. A job is running at t if start <= t < end (censored: no end)."""
    s = df[~df["never_started"]]
    start = s["first_start_time"].astype("float64").to_numpy()
    end = s["end_time"].astype("float64").fillna(np.inf).to_numpy()
    gpus = s["num_gpus"].to_numpy().astype("float64")
    rows = []
    for d in days:
        for k in range(6):
            t = pd.Timestamp(d, tz=tz) + pd.Timedelta(hours=17, minutes=30 * k)
            ts = t.timestamp()
            run = (start <= ts) & (end > ts)
            long = run & (ts - start > LONG_S)
            rows.append({"day": str(d), "running_jobs": run.sum(), "running_gpus": gpus[run].sum(),
                         "long_gpus": gpus[long].sum(), "long_jobs": long.sum()})
    r = pd.DataFrame(rows)
    return {"cluster": cluster, "slots": len(r), "mean_running_jobs": r["running_jobs"].mean(),
            "mean_running_gpus": r["running_gpus"].mean(),
            "share_gpus_long_12h": r["long_gpus"].sum() / max(r["running_gpus"].sum(), 1),
            "share_jobs_long_12h": r["long_jobs"].sum() / max(r["running_jobs"].sum(), 1),
            "per_day_share": r.groupby("day").apply(lambda x: x["long_gpus"].sum() / max(x["running_gpus"].sum(), 1), include_groups=False)}


def random_days(first, last, seed=DAY_SEED, n=N_DAYS):
    days = pd.date_range(first, last, freq="D")
    return sorted(random.Random(seed).sample(list(days.date), n))


def largest_jobs(df):
    g = df[(df["num_gpus"] >= 1)]
    gh = gpu_hours(g)
    rows = [{"threshold": f">= {k} GPUs", "jobs": int((g["num_gpus"] >= k).sum()),
             "% GPU-hours": 100 * gh[g["num_gpus"] >= k].sum() / gh.sum()} for k in (8, 64, 128, 256, 512, 1024)]
    return g["num_gpus"].max(), pd.DataFrame(rows)


def cdf_stats(path, rows=(0, 1)):
    """Percentiles of a saved CDF curve (row 0 = power in W at 1000 CDF points, row 1 = CDF %). The pickles hold only numpy
    arrays; a restricted unpickler refuses anything else."""
    allowed = {("numpy.core.multiarray", "_reconstruct"), ("numpy", "ndarray"), ("numpy", "dtype"),
               ("numpy._core.multiarray", "_reconstruct")}

    class U(pickle.Unpickler):
        def find_class(self, m, n):
            if (m, n) in allowed:
                return super().find_class(m, n)
            raise pickle.UnpicklingError(f"blocked {m}.{n}")

    with open(path, "rb") as f:
        a = np.asarray(U(f).load())
    x, y = a[rows[0]], a[rows[1]]
    q = lambda p: float(np.interp(p, y, x))
    return {"min": float(x.min()), "p1": q(1), "p10": q(10), "p25": q(25), "p50": q(50), "p75": q(75), "p90": q(90),
            "p99": q(99), "max": float(x.max()), "mean (approx, from CDF)": float(np.mean(x))}


# ------------------------------------------------------------- split proposal
SPLIT = {"train_end": "2023-06-01", "val_end": "2023-06-15", "test_end": "2023-07-16"}


def split_text(gpu):
    """Proposed Acme split (not implemented): counts of GPU jobs per split for the proposed bounds."""
    bounds = {k: pd.Timestamp(v, tz="Asia/Shanghai").timestamp() for k, v in SPLIT.items()}
    st = gpu["submit_time"].astype("float64")
    part = np.select([st < bounds["train_end"], st < bounds["val_end"], st < bounds["test_end"]], ["train", "val", "test"], "after")
    g = gpu.assign(split=part, long=gpu["wall_s"].astype("float64") > 600)
    t = g.groupby(["cluster", "split"]).agg(jobs=("job_id", "size"), jobs_wall_gt_10min=("long", "sum"),
                                            users=("user", "nunique"), censored=("censored", "sum")).reset_index()
    t["split"] = pd.Categorical(t["split"], ["train", "val", "test", "after"])
    t = t.sort_values(["cluster", "split"])
    return ["Proposal (by submit time, China time UTC+8, half-open, the same shape as Philly's split):", "",
            f"- **train**: submit < {SPLIT['train_end']}  (Seren Mar-May)",
            f"- **val**: {SPLIT['train_end']} <= submit < {SPLIT['val_end']}  (2 weeks)",
            f"- **test**: {SPLIT['val_end']} <= submit < {SPLIT['test_end']}  (31 days, as long as Philly's test window)",
            f"- Not used: submit >= {SPLIT['test_end']} (late July and August; the repo's power curves are August data, but they are curves, not time series).",
            f"- The simulation would start 7 days before the test window, on 2023-06-08, as in Philly (`SIM_START`), so carry-over and val-period jobs get predictions.", "",
            "Jobs per split under this proposal:", "", md(t, 0), "",
            "Points for the decision (these are not made here):",
            "1. **Seren only for the ML and simulation, Kalos as a second case.** Kalos starts on 2023-05-17, has only about 20,000 GPU jobs, "
            "and 96% of its GPU-hours are pretraining. A pooled table would need `cluster` as a feature (there is no `vc`).",
            "2. **The time split is uneven in volume.** June 26-July 2 alone has about 100,000 Seren jobs (89% Eval), so the test window is much "
            "larger than train; moving `train_end` later (to 2023-06-15) "
            "would balance train and test at the cost of a 2-week val set in July.",
            "3. **Most Acme jobs are very short** (Seren median 2.3 min; 82% of Seren GPU jobs run 600 s or less), so the grid "
            "0, 60, 600, 3600, 14400, 43200 s gives most jobs only 1-2 rows, and the 60 s floor on remaining runtime matters much more than in Philly.",
            "4. `PREEMPTED` (9.7% of Seren GPU jobs) is mapped to Killed here; a model for failure risk would treat it separately or drop it.", ""]


# -------------------------------------------------------------------- report
def main():
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    allj = load_all()
    n_cpu = allj[allj["num_gpus"] == 0].groupby("cluster").size()
    gpu = allj[allj["num_gpus"] >= 1].reset_index(drop=True)
    gpu.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(gpu):,} GPU jobs ({gpu['cluster'].value_counts().to_dict()}); CPU-only dropped: {n_cpu.to_dict()}")

    ph = pd.read_parquet(config.DATA_PROCESSED / "philly_jobs.parquet")
    ph = ph[~ph["never_started"]].copy()
    ph["cluster"] = "philly"
    ph["first_start_time"] = ph["first_start_time"].astype("Int64")
    sets = {"seren": gpu[gpu["cluster"] == "seren"], "kalos": gpu[gpu["cluster"] == "kalos"], "philly": ph[ph["num_gpus"] >= 1]}

    # random days (fixed seed), inside each trace's fully observed range
    ranges = {"seren": ("2023-03-02", "2023-08-15"), "kalos": ("2023-05-18", "2023-08-15"), "philly": ("2017-10-03", "2017-12-14")}
    run = {k: running_at_slots(sets[k], k, random_days(*ranges[k]), LOCAL_TZ[k]) for k in sets}

    L = ["# A1: AcmeTrace job characterisation (with Philly alongside)", "",
         "Source: AcmeTrace job traces (github.com/InternLM/AcmeTrace, CC-BY-4.0; Hu et al., \"Characterization of Large Language Model "
         "Development in the Datacenter\", NSDI 2024). Only the ~109 MB job traces were downloaded; the 80 GB utilisation data on Hugging Face was not.",
         "Philly numbers are computed here from `data/processed/philly_jobs.parquet` (read only). All Acme tables use GPU jobs only (`num_gpus >= 1`).", ""]

    # ---- loading decisions
    L += ["## 1. Loading decisions", "",
          "- Files: `trace_seren.csv` (818,327 rows) and `trace_kalos.csv` (62,413 rows). "
          f"CPU-only jobs (`gpu_num = 0`) dropped from the parquet: Seren {n_cpu.get('seren', 0):,}, Kalos {n_cpu.get('kalos', 0):,}. "
          f"`data/processed/acme_jobs.parquet`: {len(gpu):,} GPU jobs.",
          "- Columns as in `philly_jobs.parquet` (job_id, user, vc, submit_time, first_start_time, end_time, wall_s, num_gpus, status, censored, "
          "never_started) plus `cluster`, `type`, `raw_state`, `node_num`. `num_attempts` and `run_s` do not exist in Acme and are omitted.",
          "- `vc` = cluster name: the raw `queue` column is the queue wait in seconds, not a queue name.",
          "- `wall_s` = `end_time - first_start_time` (= the raw `duration`). Jobs with `wall_s <= 0` are kept, as in Philly.",
          "- State map: COMPLETED -> Pass; CANCELLED -> Killed; FAILED, TIMEOUT, NODE_FAIL -> Failed (as requested). "
          "**PREEMPTED -> Killed**: the request did not cover it, but 43,838 Seren jobs have it (about 5% of rows). "
          "It is an interruption by the scheduler, not a failure of the job, so it goes with Killed; the raw state is kept in `raw_state`. "
          "RUNNING (3 Kalos jobs, no end time) gets `status` = null and `censored` = true.",
          "- Duplicate job ids: none. No job starts before it is submitted and none ends before it starts.", ""]

    # ---- timezone
    hr = {}
    for c in ("seren", "kalos"):
        s = sets[c]
        t = pd.to_datetime(s["submit_time"].astype("float64"), unit="s", utc=True)
        hr[f"{c} (UTC hour)"] = t.dt.hour.value_counts(normalize=True).sort_index()
        hr[f"{c} (China hour)"] = t.dt.tz_convert("Asia/Shanghai").dt.hour.value_counts(normalize=True).sort_index()
    h = pd.DataFrame(hr).fillna(0).mul(100).round(1)
    h.insert(0, "hour", h.index)
    L += ["## 2. Time zone", "",
          "- Seren strings end in `+08:00` (China Standard Time, no daylight saving). Kalos strings end in `+00:00` (UTC). "
          "Every string is parsed with its own offset into UTC epoch seconds, so both clusters are on one clock.",
          "- Check that Kalos' UTC label is real and not a mislabelled local time: the share of submissions by hour of day (below) peaks at "
          "01-10 UTC for Kalos, which is 09-18 China time, the same working-day shape Seren shows in China time (10-19, low at 07). "
          "So both clusters are China-time workloads; \"local time\" for the 17:00-20:00 analysis means UTC+8.", "",
          "Share of GPU-job submissions by hour of day (%):", "", md(h, 1), ""]

    # ---- jobs per cluster and month
    mon = gpu.assign(month=pd.to_datetime(gpu["submit_time"].astype("float64"), unit="s", utc=True).dt.strftime("%Y-%m"))
    mt = mon.pivot_table(index="month", columns="cluster", values="job_id", aggfunc="size", fill_value=0).reset_index()
    cpu_all = allj[allj["num_gpus"] == 0]
    L += ["## 3. GPU jobs per cluster and month (by submit time, UTC)", "", md(mt, 0), "",
          f"Trace ranges (submit time, UTC): Seren {pd.to_datetime(sets['seren']['submit_time'].min(), unit='s')} to "
          f"{pd.to_datetime(sets['seren']['submit_time'].max(), unit='s')}; Kalos {pd.to_datetime(sets['kalos']['submit_time'].min(), unit='s')} to "
          f"{pd.to_datetime(sets['kalos']['submit_time'].max(), unit='s')}. Philly: Aug-Dec 2017 (117,325 jobs, T1 report).", ""]

    # ---- shares
    L += ["## 4. State and type shares, by jobs and by GPU-hours", "",
          "GPU-hours = `wall_s * num_gpus / 3600` (unfinished jobs count 0).", ""]
    for c in ("seren", "kalos"):
        L += [f"### {c.capitalize()} - mapped status", md(share_table(sets[c], "status", "status")), "",
              f"### {c.capitalize()} - raw state", md(share_table(sets[c], "raw_state", "raw state")), "",
              f"### {c.capitalize()} - job type", md(share_table(sets[c], "type", "type")), ""]
    L += ["### Philly - status", md(share_table(sets["philly"], "status", "status")), ""]

    # ---- wall time percentiles
    L += ["## 5. Wall-time percentiles by status (minutes; finished jobs with wall_s > 0)", ""]
    for c in ("seren", "kalos", "philly"):
        L += [f"### {c.capitalize()}", md(wall_percentiles(sets[c])), ""]

    # ---- failed+killed share
    fk = []
    for c in ("seren", "kalos", "philly"):
        g = sets[c]
        gh = gpu_hours(g)
        tot = gh.sum()
        fk.append({"dataset": c, "GPU-hours": tot, "% Failed": 100 * gh[g["status"] == "Failed"].sum() / tot,
                   "% Killed": 100 * gh[g["status"] == "Killed"].sum() / tot,
                   "% Failed+Killed": 100 * gh[g["status"].isin(["Failed", "Killed"])].sum() / tot})
    s_ = sets["seren"]
    gh = gpu_hours(s_)
    no_pre = 100 * gh[s_["status"].isin(["Failed", "Killed"]) & (s_["raw_state"] != "PREEMPTED")].sum() / gh.sum()
    L += ["## 6. Share of GPU-hours used by Failed + Killed jobs", "", md(pd.DataFrame(fk), 1), "",
          f"Sensitivity (Seren): without PREEMPTED the Failed+Killed share is {no_pre:.1f}%.", ""]

    # ---- slots
    L += ["## 7. Running work at 17:00-19:30 local time (30-min slots, 10 random days per trace)", "",
          f"Days drawn with `random.Random({DAY_SEED})` from each trace's fully observed range (Seren 2023-03-02..08-15, Kalos 2023-05-18..08-15, "
          "Philly 2017-10-03..12-14), weekends included. Slots: 17:00, 17:30, ..., 19:30 (6 per day, 60 per trace). Local time: China (UTC+8) for Acme; "
          "America/Los_Angeles for Philly, the time zone used for the grid events in the paper. A job is running at t if start <= t < end "
          "(Philly: first start to end, so the gaps between attempts count as running). \"Long\" = already running for more than 12 h at t.", "",
          md(pd.DataFrame([{k: v for k, v in r.items() if k != "per_day_share"} for r in run.values()]).assign(
              share_gpus_long_12h=lambda d: 100 * d["share_gpus_long_12h"], share_jobs_long_12h=lambda d: 100 * d["share_jobs_long_12h"]
          ).rename(columns={"share_gpus_long_12h": "% running GPUs held by >12 h jobs", "share_jobs_long_12h": "% running jobs >12 h"}), 1), "",
          "Per-day share of running GPUs held by >12 h jobs (%), mean over the 6 slots of each day:", ""]
    for k, r in run.items():
        L.append(f"- {k}: " + ", ".join(f"{d} {100 * v:.0f}" for d, v in r["per_day_share"].items()))
    L += ["", "Acme's share swings between 0% and 100% from day to day because a handful of pretraining jobs decide it "
          "(Kalos has only about 2 jobs running at these hours, so its per-day value is one or two jobs). Philly stays high every day.", ""]

    # ---- largest jobs
    L += ["## 8. Largest jobs (GPUs per job)", ""]
    for c in ("seren", "kalos", "philly"):
        mx, t = largest_jobs(sets[c])
        L += [f"### {c.capitalize()}: maximum {mx:,} GPUs", md(t, 1), ""]
        if c != "philly":
            top = sets[c].nlargest(5, "num_gpus")[["job_id", "type", "raw_state", "num_gpus", "wall_s"]]
            L += ["Five largest jobs:", "", md(top, 0), ""]

    # ---- power files
    L += ["## 9. GPU power files in the repo", "",
          "The repo has no raw power samples (those are in the 80 GB Hugging Face data, not downloaded). "
          "`data/utilization/util_pkl/gpu_power_{seren,kalos}.pkl` and `server_power.pkl` hold only **CDF curves** (1000 points, power in W "
          "against CDF %), used for the paper's Figure on power. The notebook says they use **August 2023** data. "
          "They give the distribution of GPU power over all GPU samples, not power conditioned on idle or busy, so an exact idle and a busy mean "
          "cannot be computed from the repo. The low end of the curve shows the idle floor and the A100 TDP of 400 W (marked in the notebook) is the busy ceiling.", ""]
    P = RAW / "utilization" / "util_pkl"
    rows = [{"series": lab, **cdf_stats(P / f, r)} for lab, f, r in [
        ("Seren GPU power (W)", "gpu_power_seren.pkl", (0, 1)), ("Kalos GPU power (W)", "gpu_power_kalos.pkl", (0, 1)),
        ("GPU node server power (W)", "server_power.pkl", (0, 1)), ("CPU node server power (W)", "server_power.pkl", (2, 3))]]
    L += [md(pd.DataFrame(rows), 1), "",
          "Reading (approximate, from the CDFs): the lowest quarter of GPU samples sit at 60-66 W, which is the **idle floor** (about 63 W); "
          "the median is 97-101 W; the 90th percentile is 395-404 W, close to the A100 TDP of 400 W, which is the **busy level**. "
          "The mean over all samples is about 169 W (Seren) and 192 W (Kalos). These are the means over all samples, not idle-only or busy-only means. "
          "Philly has no power data at all.", ""]

    L += ["## 10. Headline comparison", ""]
    head = []
    for c in ("philly", "seren", "kalos"):
        g = sets[c]
        gh = gpu_hours(g)
        f = g[(~g["censored"]) & (g["wall_s"].astype("float64") > 0)]
        head.append({"": c, "GPU jobs": len(g), "GPU-hours": gh.sum(), "% GPU-h Failed+Killed": 100 * gh[g["status"].isin(["Failed", "Killed"])].sum() / gh.sum(),
                     "median wall (min)": f["wall_s"].astype("float64").median() / 60, "p99 wall (h)": f["wall_s"].astype("float64").quantile(.99) / 3600,
                     "max GPUs": int(g["num_gpus"].max()), "mean running jobs 17-20h": run[c]["mean_running_jobs"],
                     "mean running GPUs 17-20h": run[c]["mean_running_gpus"], "% GPUs held by >12h jobs": 100 * run[c]["share_gpus_long_12h"]})
    L += [md(pd.DataFrame(head), 1), ""]

    # ---- split proposal
    t = pd.to_datetime(gpu["submit_time"].astype("float64"), unit="s", utc=True)
    wk = gpu.assign(week=t.dt.tz_localize(None).dt.to_period("W-SUN").dt.start_time.dt.date).pivot_table(
        index="week", columns="cluster", values="job_id", aggfunc="size", fill_value=0).reset_index()
    L += ["## 11. Proposed time split (not implemented)", "", "GPU jobs per week (submit time, UTC, week starting Monday):", "", md(wk, 0), ""]
    L += split_text(gpu)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
