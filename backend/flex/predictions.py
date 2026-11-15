"""Reader for predictions/<name>.csv and the contract lookup rule (SIM_PLAN S5).

Contract: columns job_id, elapsed_s, rt_p10_s, rt_p50_s, rt_p90_s, p_fail, p_cancel; rt_* is the remaining runtime
(seconds) at elapsed_s, one row per elapsed grid point while the job runs.
Lookup: take the latest row with elapsed_s <= elapsed, subtract the gap (elapsed - elapsed_s) from rt_*, floor at 60 s.
Option lookup_rule="hold_beyond_grid" (S14c, SIM_PLAN): beyond 43,200 s return the last row unchanged (not for the oracle).
Jobs with no rows fall back to oracle values from the job itself (and are counted). Option missing_pred="uninformed" (W7
robustness): such jobs get 30-day remaining-runtime quantiles and p_fail = p_cancel = 0 instead (never for the oracle).
"""
import bisect
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("flex.predictions")

COLUMNS = ["job_id", "elapsed_s", "rt_p10_s", "rt_p50_s", "rt_p90_s", "p_fail", "p_cancel"]
FLOOR_S = 60
GRID_MAX_S = 43200                  # last elapsed grid point of the contract
LOOKUP_RULES = ("contract", "hold_beyond_grid")
MISSING_PRED = ("oracle", "uninformed")
UNINFORMED_RT_S = 30 * 86400        # W7: "long and useful", so a job without rows is never burst as about to finish


def oracle_values(job, elapsed):
    """Oracle prediction for a job from its own ground truth: (p10, p50, p90, p_fail, p_cancel)."""
    rt = max(FLOOR_S, job.wall_s + job.elapsed0 - elapsed)       # elapsed includes elapsed0
    return rt, rt, rt, float(job.status == "Failed"), float(job.status == "Killed")


class Predictions:
    def __init__(self, name, rows):
        """rows: dict job_id -> (list of elapsed_s, list of (p10, p50, p90, p_fail, p_cancel))."""
        self.name, self.rows = name, rows
        self.fallback_jobs = set()
        self.lookup_rule = "contract"       # S14c: "hold_beyond_grid" = last row unchanged past 43,200 s (never for the oracle)
        self.missing_pred = "oracle"        # W7: "uninformed" = fixed long, useful values for jobs without rows (never for the oracle)

    def lookup(self, job, elapsed):
        """(p10, p50, p90, p_fail, p_cancel) for `job` at `elapsed` seconds of progress."""
        entry = self.rows.get(job.job_id)
        if entry is None:
            self.fallback_jobs.add(job.job_id)
            if self.missing_pred == "uninformed" and self.name != "oracle":
                return UNINFORMED_RT_S, UNINFORMED_RT_S, UNINFORMED_RT_S, 0.0, 0.0
            return oracle_values(job, elapsed)
        grid, vals = entry
        if self.lookup_rule == "hold_beyond_grid" and elapsed > GRID_MAX_S and self.name != "oracle":
            return vals[-1]                 # no gap subtraction beyond the grid; between grid points the contract rule stays
        i = bisect.bisect_right(grid, elapsed) - 1
        if i < 0:
            raise ValueError(f"{self.name}: job {job.job_id} has no row at or before elapsed={elapsed}")
        gap = elapsed - grid[i]
        p10, p50, p90, p_fail, p_cancel = vals[i]
        return (max(FLOOR_S, p10 - gap), max(FLOOR_S, p50 - gap), max(FLOOR_S, p90 - gap), p_fail, p_cancel)

    def log_fallback_count(self):
        log.info("%s: %d jobs without prediction rows used oracle values", self.name, len(self.fallback_jobs))
        return len(self.fallback_jobs)


def load_predictions(path, name=None):
    """Read one prediction CSV (job_id is read as a string). Extra rows (e.g. oracle's train/val jobs) are fine."""
    path = Path(path)
    df = pd.read_csv(path, dtype={"job_id": str})
    if list(df.columns) != COLUMNS:
        raise ValueError(f"{path}: columns {list(df.columns)} != {COLUMNS}")
    if df.isna().any().any():
        raise ValueError(f"{path}: contains NaN")
    if df.duplicated(["job_id", "elapsed_s"]).any():
        raise ValueError(f"{path}: duplicate (job_id, elapsed_s) keys")
    df = df.sort_values(["job_id", "elapsed_s"], kind="stable")
    rows = {}
    for jid, g in df.groupby("job_id", sort=False):
        rows[jid] = (g["elapsed_s"].tolist(), list(map(tuple, g.iloc[:, 2:].to_numpy().tolist())))
    log.info("%s: %d rows, %d jobs", path.name, len(df), len(rows))
    return Predictions(name or path.stem, rows)


if __name__ == "__main__":
    # Real-file check: python -m flex.predictions [oracle last2 ...]  (run from backend/)
    import sys

    from .models import FJob

    root = Path(__file__).resolve().parents[2]
    for name in sys.argv[1:] or ["oracle", "last2"]:
        path = root / "predictions" / f"{name}.csv"
        if not path.exists():
            print(f"{name}: {path} not found")
            continue
        p = load_predictions(path)
        n_rows = sum(len(g) for g, _ in p.rows.values())
        print(f"{name}: {n_rows} rows, {len(p.rows)} distinct jobs")
        jid = next(j for j, (g, _) in sorted(p.rows.items()) if g[-1] >= 3600)    # a job that runs past 1 h
        dummy = FJob(job_id=jid, user="?", submit=0, gpus=1, wall_s=0, status="Pass")
        print(f"  job {jid}: rows at elapsed {p.rows[jid][0]}")
        for e in (0, 59, 600, 5000):
            print(f"  lookup at {e:>5} s: {p.lookup(dummy, e)}")
