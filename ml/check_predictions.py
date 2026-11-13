"""Shared done-check for every prediction export.

Usage:  python ml/check_predictions.py predictions/last2.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from export_simple import COLS, export_keys  # noqa: E402


def check(path):
    """Return a list of problems (empty = OK)."""
    path = Path(path)
    probs = []
    df = pd.read_csv(path, dtype={"job_id": str})
    if list(df.columns) != COLS:
        return [f"columns are {list(df.columns)}, expected {COLS}"]
    if df.isna().any().any():
        n = df.isna().sum()
        probs.append(f"NaN values: {n[n > 0].to_dict()}")
        df = df.dropna()
    p10, p50, p90 = df["rt_p10_s"], df["rt_p50_s"], df["rt_p90_s"]
    ok = (p10 >= config.MIN_RUNTIME_S) & (p10 <= p50) & (p50 <= p90)
    if not ok.all():
        probs.append(f"{int((~ok).sum())} rows violate 60 <= p10 <= p50 <= p90")
    bad = (df["p_fail"] < 0) | (df["p_cancel"] < 0) | (df["p_fail"] + df["p_cancel"] > 1 + 1e-9)
    if bad.any():
        probs.append(f"{int(bad.sum())} rows violate 0 <= p_fail, 0 <= p_cancel, p_fail + p_cancel <= 1")
    if not df["elapsed_s"].isin(config.GRID).all():
        probs.append("elapsed_s values outside GRID")
    dup = df.duplicated(["job_id", "elapsed_s"])
    if dup.any():
        probs.append(f"{int(dup.sum())} duplicate (job_id, elapsed_s) keys")

    ex = pd.read_parquet(config.DATA_PROCESSED / "ml_examples.parquet", columns=["job_id", "elapsed_s", "split"])
    jobs = pd.read_parquet(config.DATA_PROCESSED / "philly_jobs.parquet")
    expected = set(map(tuple, export_keys(jobs, ex)[["job_id", "elapsed_s"]].to_numpy()))
    got = set(map(tuple, df[["job_id", "elapsed_s"]].to_numpy()))
    missing, extra = expected - got, got - expected
    if missing:
        probs.append(f"{len(missing)} expected export keys missing, e.g. {sorted(missing)[:2]}")
    if extra:
        probs.append(f"{len(extra)} unexpected keys, e.g. {sorted(extra)[:2]}")
    return probs


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        sys.exit(__doc__)
    problems = check(args[0])
    if problems:
        print(f"{args[0]}: {len(problems)} problem(s)")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print(f"OK {args[0]}")
