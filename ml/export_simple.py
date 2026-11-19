"""T2b: export the two simple predictors, `oracle` and `last2`, in the contract CSV format.

Run from the repo root:  ml\\.venv\\Scripts\\python ml\\export_simple.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from examples import utc_epoch  # noqa: E402

COLS = ["job_id", "elapsed_s", "rt_p10_s", "rt_p50_s", "rt_p90_s", "p_fail", "p_cancel"]


def valid_mask(jobs):
    """T2 filter: started, not censored, wall_s > 0, num_gpus > 0."""
    return (~jobs["never_started"]) & (~jobs["censored"]) & (jobs["wall_s"].astype("float64") > 0) & (jobs["num_gpus"] > 0)


def export_jobs(jobs, ds=None):
    """Valid jobs overlapping [SIM_START, TEST_END): end_time > SIM_START and submit_time < TEST_END (dataset ds, default Philly)."""
    sim_start, test_end = (ds.sim_start, ds.test_end) if ds else (config.SIM_START, config.TEST_END)
    m = (valid_mask(jobs) & (jobs["end_time"].astype("float64") > utc_epoch(sim_start))
         & (jobs["submit_time"].astype("float64") < utc_epoch(test_end)))
    return jobs[m]


def export_keys(jobs, ex, ds=None):
    """The ml_examples rows (any split) of the export jobs: the key set of every prediction file."""
    return ex[ex["job_id"].isin(export_jobs(jobs, ds)["job_id"])]


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
    import dataset as D
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="philly", choices=["philly", "acme"])
    ds = D.get(ap.parse_args().dataset)
    jobs = D.load_jobs(ds)
    ex = pd.read_parquet(D.examples_path(ds))
    config.PREDICTIONS.mkdir(parents=True, exist_ok=True)
    keys = export_keys(jobs, ex, ds)

    fw, ff, fc = train_fallbacks(ex)
    outs = {"oracle": oracle_frame(keys), "last2": last2_frame(keys, fw, ff, fc)}
    for name, df in outs.items():
        df = df.sort_values(["job_id", "elapsed_s"]).reset_index(drop=True)[COLS]
        df.to_csv(D.pred_path(ds, name), index=False)
        print(f"{ds.prefix}{name}: {len(df):,} rows, {df['job_id'].nunique():,} distinct jobs")
    print("export rows per split:", keys["split"].value_counts().to_dict())
    print("export jobs per split:", keys.drop_duplicates("job_id")["split"].value_counts().to_dict())
    print(f"last2 train fallbacks: median wall {fw:.0f} s, fail {ff:.3f}, cancel {fc:.3f}; "
          f"export rows with empty history: {int((keys['hist_count'] == 0).sum())}")


if __name__ == "__main__":
    main()
