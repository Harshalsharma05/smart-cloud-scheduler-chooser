"""T2b: export the two simple predictors, `oracle` and `last2`, in the contract CSV format.

Run from the repo root:  ml\\.venv\\Scripts\\python ml\\export_simple.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from examples import utc_epoch  # noqa: E402

COLS = ["job_id", "elapsed_s", "rt_p10_s", "rt_p50_s", "rt_p90_s", "p_fail", "p_cancel"]


def warmup_jobs(jobs, test_start):
    """Valid jobs (T2 filter) submitted before TEST_START that end after it: running or still queued at TEST_START."""
    valid = (~jobs["never_started"]) & (~jobs["censored"]) & (jobs["wall_s"].astype("float64") > 0) & (jobs["num_gpus"] > 0)
    m = valid & (jobs["submit_time"].astype("float64") < test_start) & (jobs["end_time"].astype("float64") > test_start)
    return jobs[m]


def oracle_frame(ex):
    """Ground truth: remaining = wall_s - e (floored), status as 0/1 probabilities."""
    rt = np.maximum(config.MIN_RUNTIME_S, ex["remaining_s"].to_numpy()).astype("int64")
    return pd.DataFrame({"job_id": ex["job_id"].to_numpy(), "elapsed_s": ex["elapsed_s"].to_numpy(),
                         "rt_p10_s": rt, "rt_p50_s": rt, "rt_p90_s": rt,
                         "p_fail": (ex["status"] == "Failed").astype(float).to_numpy(),
                         "p_cancel": (ex["status"] == "Killed").astype(float).to_numpy()})


def last2_frame(ex, fallback_wall, fallback_fail, fallback_cancel):
    """Tsafrir last-two: mean wall_s of the user's last two finished jobs (one job: that job; none: train median)."""
    n = ex["hist_count"].to_numpy()
    l1, l2 = ex["hist_last1_wall"].to_numpy(dtype=float), ex["hist_last2_wall"].to_numpy(dtype=float)
    est = np.where(n >= 2, (l1 + l2) / 2, np.where(n == 1, l1, fallback_wall))
    rt = np.maximum(config.MIN_RUNTIME_S, est - ex["elapsed_s"].to_numpy())
    pf = np.where(n > 0, ex["hist_fail_rate"].to_numpy(dtype=float), fallback_fail)
    pc = np.where(n > 0, ex["hist_cancel_rate"].to_numpy(dtype=float), fallback_cancel)
    return pd.DataFrame({"job_id": ex["job_id"].to_numpy(), "elapsed_s": ex["elapsed_s"].to_numpy(),
                         "rt_p10_s": rt, "rt_p50_s": rt, "rt_p90_s": rt, "p_fail": pf, "p_cancel": pc})


def train_fallbacks(ex):
    """Median wall_s and failure / cancel rates over train-split jobs."""
    tj = ex[(ex["split"] == "train") & (ex["elapsed_s"] == 0)]
    return float(tj["wall_s"].median()), float((tj["status"] == "Failed").mean()), float((tj["status"] == "Killed").mean())


def main():
    jobs = pd.read_parquet(config.DATA_PROCESSED / "philly_jobs.parquet")
    ex = pd.read_parquet(config.DATA_PROCESSED / "ml_examples.parquet")
    config.PREDICTIONS.mkdir(parents=True, exist_ok=True)

    test = ex[ex["split"] == "test"]
    wu = warmup_jobs(jobs, utc_epoch(config.TEST_START))
    wu_rows = ex[ex["job_id"].isin(wu["job_id"])]
    oracle = pd.concat([test, wu_rows]).drop_duplicates(["job_id", "elapsed_s"])

    fw, ff, fc = train_fallbacks(ex)
    outs = {"oracle": oracle_frame(oracle), "last2": last2_frame(test, fw, ff, fc)}
    for name, df in outs.items():
        df = df.sort_values(["job_id", "elapsed_s"]).reset_index(drop=True)[COLS]
        df.to_csv(config.PREDICTIONS / f"{name}.csv", index=False)
        print(f"{name}: {len(df):,} rows, {df['job_id'].nunique():,} distinct jobs")

    running = int((wu["first_start_time"].astype("float64") <= utc_epoch(config.TEST_START)).sum())
    print(f"test: {len(test):,} rows, {test['job_id'].nunique():,} distinct jobs")
    print(f"warm-up jobs: {len(wu):,} (running at TEST_START {running:,}, queued at TEST_START {len(wu) - running:,}); "
          f"oracle rows added: {len(oracle) - len(test):,}")
    print(f"last2 train fallbacks: median wall {fw:.0f} s, fail {ff:.3f}, cancel {fc:.3f}; "
          f"test rows with empty history: {int((test['hist_count'] == 0).sum())}")


if __name__ == "__main__":
    main()
