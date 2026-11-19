"""T2: leak-free training examples, one row per (job, elapsed grid point).

Run from the repo root:  ml\\.venv\\Scripts\\python ml\\examples.py
Definitions: ML_PLAN.md section 2. hist_idx holds row indices into philly_jobs.parquet (original row order).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def utc_epoch(date_str):
    return int(pd.Timestamp(date_str, tz="UTC").timestamp())


def default_bounds():
    """Half-open split bounds on submit_time (epoch seconds), from config.py."""
    return {
        "train_end": utc_epoch(config.TRAIN_END),
        "val_start": utc_epoch(config.VAL_START),
        "val_end": utc_epoch(config.VAL_END),
        "test_start": utc_epoch(config.TEST_START),
        "test_end": utc_epoch(config.TEST_END),
    }


def split_of(submit, b):
    if submit < b["train_end"]:
        return "train"
    if b["val_start"] <= submit < b["val_end"]:
        return "val"
    if b["test_start"] <= submit < b["test_end"]:
        return "test"
    return None


def build_examples(jobs, attempts, bounds=None, grid=None, H=None):
    """jobs: philly_jobs table, attempts: philly_attempts table -> examples DataFrame."""
    bounds = bounds or default_bounds()
    grid = grid or config.GRID
    H = H or config.H
    jobs = jobs.reset_index(drop=True)
    wall = jobs["wall_s"].astype("float64").to_numpy()
    submit = jobs["submit_time"].astype("float64").to_numpy()
    start = jobs["first_start_time"].astype("float64").to_numpy()
    end = jobs["end_time"].astype("float64").to_numpy()
    ngpu = jobs["num_gpus"].to_numpy()
    status = jobs["status"].to_numpy()

    finished = (~jobs["never_started"].to_numpy()) & (~jobs["censored"].to_numpy()) & (wall > 0)
    # example jobs: started, finished, positive wall and GPUs, submitted before the end of the test window
    use = finished & (ngpu > 0) & (submit < bounds["test_end"])

    # per-user history pool, sorted by end_time (jobs with a known end and wall > 0)
    pool = {}
    for u, idx in jobs[finished].groupby("user").indices.items():
        rows = jobs.index[finished][idx].to_numpy()
        order = np.argsort(end[rows], kind="stable")
        rows = rows[order]
        pool[u] = (rows, end[rows])

    # attempt start times per job (sorted) for attempts_so_far
    att = attempts.dropna(subset=["start_time"]).sort_values(["job_id", "start_time"])
    att_starts = {k: v.to_numpy(dtype="float64") for k, v in att.groupby("job_id")["start_time"]}

    out = {k: [] for k in ["job_row", "elapsed_s", "t_obs", "queue_wait_s", "attempts_so_far", "hist_idx",
                           "hist_count", "hist_mean_log_wall", "hist_median_log_wall", "hist_last1_wall",
                           "hist_last2_wall", "hist_fail_rate", "hist_cancel_rate"]}
    users = jobs["user"].to_numpy()
    job_ids = jobs["job_id"].to_numpy()
    empty = np.array([], dtype=np.int32)
    for j in np.flatnonzero(use):
        starts = att_starts.get(job_ids[j], np.array([]))
        rows_u, ends_u = pool.get(users[j], (empty, np.array([])))
        for e in grid:
            if e >= wall[j]:
                break
            t_obs = submit[j] if e == 0 else start[j] + e
            pos = np.searchsorted(ends_u, t_obs, side="right")  # history jobs: end_time <= t_obs
            h = rows_u[max(0, pos - H):pos]
            h = h[h != j].astype(np.int32)  # never the job itself
            if len(h):
                lw = np.log1p(wall[h])
                hs = status[h]
                agg = (len(h), lw.mean(), np.median(lw), wall[h[-1]], wall[h[-2]] if len(h) > 1 else np.nan,
                       (hs == "Failed").mean(), (hs == "Killed").mean())
            else:
                agg = (0,) + (np.nan,) * 6
            out["job_row"].append(j)
            out["elapsed_s"].append(e)
            out["t_obs"].append(t_obs)
            out["queue_wait_s"].append(0.0 if e == 0 else start[j] - submit[j])
            out["attempts_so_far"].append(int(np.searchsorted(starts, t_obs, side="right")))
            out["hist_idx"].append(h)
            for k, v in zip(["hist_count", "hist_mean_log_wall", "hist_median_log_wall", "hist_last1_wall",
                             "hist_last2_wall", "hist_fail_rate", "hist_cancel_rate"], agg):
                out[k].append(v)

    ex = pd.DataFrame(out)
    r = ex["job_row"].to_numpy()
    ts = pd.to_datetime(ex["t_obs"], unit="s")  # hour/day of the submit time (UTC)
    sub_ts = pd.to_datetime(submit[r], unit="s")
    ex.insert(0, "job_id", job_ids[r])
    ex["user"] = users[r]
    ex["vc"] = pd.Categorical(jobs["vc"].to_numpy()[r])
    if "type" in jobs.columns:  # Acme workload type (not in Philly)
        ex["type"] = pd.Categorical(jobs["type"].to_numpy()[r])
    ex["num_gpus"] = ngpu[r]
    ex["submit_hour"] = sub_ts.hour.to_numpy()
    ex["submit_dow"] = sub_ts.day_of_week.to_numpy()
    ex["submit_time"] = submit[r].astype("int64")
    ex["wall_s"] = wall[r].astype("int64")
    ex["remaining_s"] = (wall[r] - ex["elapsed_s"].to_numpy()).astype("int64")
    ex["status"] = status[r]
    ex["split"] = [split_of(s, bounds) for s in submit[r]]
    ex["t_obs"] = ex["t_obs"].astype("int64")
    ex["queue_wait_s"] = ex["queue_wait_s"].astype("int64")
    del ts
    return ex


def write_report(ex, jobs, n_candidates, path):
    L = ["# T2 examples report", "",
         f"- Example table: **{len(ex):,} rows** from {ex['job_id'].nunique():,} jobs "
         f"(started, non-censored, `wall_s > 0`, `num_gpus > 0`, submitted before {config.TEST_END}; "
         f"{n_candidates:,} started non-censored jobs before that date, so {n_candidates - ex['job_id'].nunique():,} excluded).",
         "- `split` is by `submit_time`; jobs submitted in a gap or after the test window get no split (none expected: the bounds are contiguous).",
         "", "## Rows per split and grid point", "",
         "| split | " + " | ".join(f"e={e}" for e in config.GRID) + " | total |",
         "|---|" + "---|" * (len(config.GRID) + 1)]
    for s in ["train", "val", "test"]:
        sub = ex[ex["split"] == s]
        cnt = sub["elapsed_s"].value_counts()
        L.append(f"| {s} | " + " | ".join(f"{int(cnt.get(e, 0)):,}" for e in config.GRID) + f" | {len(sub):,} |")
    tot = ex["elapsed_s"].value_counts()
    L.append("| all | " + " | ".join(f"{int(tot.get(e, 0)):,}" for e in config.GRID) + f" | {len(ex):,} |")
    hl = ex["hist_idx"].map(len)
    L += ["", "## History", "",
          f"- Rows with empty history: {100 * (hl == 0).mean():.2f}% overall; "
          + ", ".join(f"{s} {100 * (hl[ex['split'] == s] == 0).mean():.2f}%" for s in ["train", "val", "test"]),
          f"- Median history length: {hl.median():.0f} (mean {hl.mean():.1f}, max {hl.max()}, cap H = {config.H})",
          f"- Rows with a full history (H = {config.H}): {100 * (hl == config.H).mean():.1f}%", "",
          "## Notes", "",
          "- `t_obs = submit_time` at e = 0, `first_start_time + e` otherwise; history = same user's jobs with `end_time <= t_obs`, newest last, at most 32, never the job itself.",
          "- `hist_idx` are row indices into `philly_jobs.parquet`. History jobs are started, non-censored jobs with `wall_s > 0`.",
          "- `submit_hour` / `submit_dow` are from the job's `submit_time` in UTC (the job log is UTC).",
          f"- Targets: `remaining_s = wall_s - elapsed_s` (models predict `log1p`), `status`. Floor of 60 s is applied only at export (T2b)."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    import argparse

    import dataset as D
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="philly", choices=["philly", "acme"])
    ds = D.get(ap.parse_args().dataset)
    jobs, attempts = D.load_jobs(ds), D.load_attempts(ds)
    ex = build_examples(jobs, attempts, bounds=ds.bounds)
    out = D.examples_path(ds)
    ex.to_parquet(out, index=False)
    if ds.name == "philly":
        b = default_bounds()
        n_cand = int(((~jobs["never_started"]) & (~jobs["censored"]) & (jobs["submit_time"] < b["test_end"])).sum())
        write_report(ex, jobs, n_cand, config.REPORTS / "T2_examples_report.md")
    print(f"{len(ex):,} rows -> {out}")
    print(ex["split"].value_counts().to_dict())


if __name__ == "__main__":
    main()
