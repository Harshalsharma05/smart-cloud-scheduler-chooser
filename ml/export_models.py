"""T5: write predictions/gbdt.csv, lightgbm.csv, transformer.csv over the export keys (ML_PLAN T2c), check all five
files, re-compute the test metrics from the CSVs and compare them with ml/results/predictor_metrics.csv.

    python ml/export_models.py
"""
import datetime
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_predictions as CP  # noqa: E402
import config  # noqa: E402
import evaluate as E  # noqa: E402
import export_simple as S  # noqa: E402

TOL = 1e-3
SOURCES = {
    "oracle": "ground truth (ml/export_simple.py)",
    "last2": "Tsafrir last-two (ml/export_simple.py)",
    "gbdt": "runs/gbdt/*.joblib (ml/baselines.py)",
    "lightgbm": "runs/lightgbm/*.txt (ml/baselines.py)",
    "transformer": "runs/tf_v1/best.pt (ml/predict.py)",
}


def git(*a):
    return subprocess.check_output(["git", *a], cwd=config.ROOT, text=True).strip()


def main():
    ex = pd.read_parquet(config.DATA_PROCESSED / "ml_examples.parquet")
    jobs = pd.read_parquet(config.DATA_PROCESSED / "philly_jobs.parquet")
    exp = S.export_keys(jobs, ex).reset_index(drop=True)
    per_split = exp["split"].value_counts().to_dict()
    print(f"export rows {len(exp):,}: {per_split}")
    preds = E.build_predictions(exp, jobs, ex)  # fitted models are loaded from runs/, not re-fitted
    for name in ("gbdt", "lightgbm", "transformer"):
        df = preds[name].sort_values(["job_id", "elapsed_s"]).reset_index(drop=True)[S.COLS]
        df.to_csv(config.PREDICTIONS / f"{name}.csv", index=False)
        print(f"{name}: {len(df):,} rows")

    # checker on all five files
    bad = 0
    for name in SOURCES:
        probs = CP.check(config.PREDICTIONS / f"{name}.csv")
        print(("OK " if not probs else "FAIL ") + f"predictions/{name}.csv", *probs)
        bad += bool(probs)
    if bad:
        sys.exit("checker failed; stopping")

    # consistency: test metrics recomputed from the CSVs vs ml/results/predictor_metrics.csv
    truth = exp[exp["split"] == "test"][["job_id", "elapsed_s", "remaining_s", "status"]].copy()
    truth["status"] = truth["status"].map({s: i for i, s in enumerate(E.STATUSES)})
    ref = pd.read_csv(config.RESULTS / "predictor_metrics.csv")
    ref = ref[ref["elapsed_s"].astype(str) == "all"].set_index("predictor")
    rows, worst = [], 0.0
    for name in ("gbdt", "lightgbm", "transformer"):
        df = pd.read_csv(config.PREDICTIONS / f"{name}.csv", dtype={"job_id": str})
        m = truth.merge(df, on=["job_id", "elapsed_s"], how="left")
        assert m["rt_p50_s"].notna().all()
        got = E.all_metrics(m["remaining_s"].to_numpy(), m["status"].to_numpy(), m)
        for k in ("pinball", "coverage_p10_p90", "logloss"):
            d = abs(got[k] - float(ref.loc[name, k]))
            worst = max(worst, d)
            rows.append((name, k, got[k], float(ref.loc[name, k]), d))
    cons = pd.DataFrame(rows, columns=["predictor", "metric", "from_csv", "evaluate", "abs_diff"])
    print(cons.to_string(index=False))
    if worst > TOL:
        sys.exit(f"CONSISTENCY FAILED: max abs diff {worst:.2e} > {TOL}; CSVs left as written, not adjusted")
    print(f"consistency OK: max abs diff {worst:.2e} (tolerance {TOL})")

    # report + README
    commit, date = git("rev-parse", "--short", "HEAD"), datetime.date.today().isoformat()
    st = lambda n: {"rows": len(pd.read_csv(config.PREDICTIONS / f"{n}.csv", usecols=["job_id"]))}
    tf = json.loads((config.RUNS / "tf_v1" / "summary.json").read_text())
    L = ["# T5 prediction exports", "",
         f"All five files cover the export keys (ML_PLAN T2c): valid jobs with `end_time > SIM_START` ({config.SIM_START}) and "
         f"`submit_time < TEST_END` ({config.TEST_END}), every grid point with `e < wall_s`. Rows per split: "
         + ", ".join(f"{k} {v:,}" for k, v in sorted(per_split.items())) + f"; total {len(exp):,}.", "",
         "**Train rows are in-sample for gbdt, lightgbm and transformer, and val rows were used for early stopping. Only test rows "
         "are clean.** (The 1,252 train rows are jobs still running at SIM_START, so they are long jobs and not a typical in-sample set.)", "",
         "| file | source | exported | rows | code commit |", "|---|---|---|---|---|"]
    for n, src in SOURCES.items():
        L.append(f"| predictions/{n}.csv | {src} | {date} | {st(n)['rows']:,} | {commit} |")
    L += ["", f"Transformer checkpoint: `runs/tf_v1/best.pt` (trained at commit {tf['git_commit']}, {tf['epochs_run']} epochs, best val loss {tf['best_val_loss']:.4f}). "
          "gbdt and lightgbm models were saved by T4 and loaded as they are; nothing was re-fitted.", "",
          "Checker: `python ml/check_predictions.py predictions/<name>.csv` printed OK for all five files.", "",
          f"## Consistency check (test metrics recomputed from the exported CSVs vs `ml/results/predictor_metrics.csv`, tolerance {TOL})", "",
          E.md_table(cons, floats=6), "", f"Max abs difference: {worst:.2e}.", ""]
    text = "\n".join(L)
    (config.REPORTS / "T5_exports.md").write_text(text, encoding="utf-8")
    (config.PREDICTIONS / "README.md").write_text(text, encoding="utf-8")
    print("wrote ml/reports/T5_exports.md and predictions/README.md")


if __name__ == "__main__":
    main()
