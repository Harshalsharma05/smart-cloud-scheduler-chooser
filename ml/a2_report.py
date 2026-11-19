"""A2: write ml/reports/A2_acme_predictions.md from the saved Acme tables, run summaries and prediction files.

    python ml/a2_report.py
Philly numbers come from its exported CSVs (read only); its metric files are not touched.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import dataset as D  # noqa: E402
import evaluate as E  # noqa: E402
import export_simple as S  # noqa: E402

COLS = ["pinball", "coverage_p10_p90", "mae_log", "logloss", "auroc_failed", "auroc_cancel", "brier_pfail", "mae_p50_min"]


def test_metrics(ds, name):
    """Test-split metrics (extra columns included) recomputed from predictions/<prefix><name>.csv."""
    ex = pd.read_parquet(D.examples_path(ds), columns=["job_id", "elapsed_s", "remaining_s", "status", "split"])
    t = ex[ex["split"] == "test"].copy()
    t["status"] = t["status"].map({s: i for i, s in enumerate(E.STATUSES)})
    df = pd.read_csv(D.pred_path(ds, name), dtype={"job_id": str})
    m = t.merge(df, on=["job_id", "elapsed_s"], how="left")
    assert m["rt_p50_s"].notna().all()
    return E.all_metrics(m["remaining_s"].to_numpy(), m["status"].to_numpy(), m, extra=True)


def main():
    ds = D.ACME
    ex = pd.read_parquet(D.examples_path(ds))
    jobs = D.load_jobs(ds)
    keys = S.export_keys(jobs, ex, ds)
    hl = ex["hist_idx"].map(len)
    L = ["# A2: Acme (Seren) predictions", "",
         "Same pipeline as Philly through a dataset switch (`ml/dataset.py`; `--dataset acme`). Philly code paths, data, predictions and results "
         "are unchanged (checked: `tf_v1` val predictions identical to the saved file, Philly `predictor_metrics.csv/.md` and the five Philly "
         "prediction CSVs byte-identical after the changes).", "",
         "## Set-up", "",
         f"- Seren GPU jobs only (`acme_jobs.parquet`, {len(jobs):,} jobs). Split by submit time, China time (UTC+8) midnights stored as UTC in `ml/config.py`: "
         f"train < 2023-06-15, val 2023-06-15 to 06-29, test 2023-06-29 to 07-30 (half-open). Simulation start `ACME_SIM_START` = 2023-06-22.",
         "- Elapsed grid, leakage rules (t_obs, history = same user's jobs with `end <= t_obs`, at most 32, never the job itself) and the 60 s floor are the Philly ones.",
         "- Labels: `p_fail` = FAILED / NODE_FAIL / TIMEOUT; `p_cancel` = CANCELLED / PREEMPTED (the A1 mapping, PREEMPTED -> Killed).",
         "- Features: Philly's, minus what Acme lacks: `run_s`, `num_attempts` (history tokens) and `attempts_so_far`; `vc` is the constant `seren` "
         "so the flat models drop it (the transformer keeps the embedding, which is a constant). `submit_hour` and `submit_dow` are UTC, as in Philly. "
         "`type` (workload type) is added as a categorical for all learned models, see below.", ""]

    # type
    L += ["## Is `type` known at submission?", "",
          "The README only says \"workload type in LLM development\". The NSDI 2024 paper (Hu et al., Sec. 3) says the jobs are categorised \"according to their "
          "production division and metadata (e.g., job names)\". Production division and job name are set by the user when the job is submitted, so `type` is "
          "treated as known at submission and is used. **Caveat:** the labelling rule is not published, so this cannot be verified from the data. "
          "As a check, a LightGBM without `type` (`lightgbm_notype`, same everything else) is evaluated below; it is almost as good, so the result does not "
          "depend on `type`.", ""]

    # counts
    L += ["## Example counts", "",
          "| split | example rows | jobs | rows with empty history | key-set rows | key-set jobs |", "|---|---|---|---|---|---|"]
    for sp in ("train", "val", "test"):
        e, k = ex[ex["split"] == sp], keys[keys["split"] == sp]
        L.append(f"| {sp} | {len(e):,} | {e['job_id'].nunique():,} | {100 * (hl[e.index] == 0).mean():.2f}% | {len(k):,} | {k['job_id'].nunique():,} |")
    L.append(f"| all | {len(ex):,} | {ex['job_id'].nunique():,} | {100 * (hl == 0).mean():.2f}% | {len(keys):,} | {keys['job_id'].nunique():,} |")
    cnt = ex.groupby(["split", "elapsed_s"]).size().unstack(fill_value=0).reindex(columns=config.GRID, fill_value=0)
    cnt.columns = [f"e={c}" for c in cnt.columns]
    L += ["", "Example rows per grid point (every row needs `e < wall_s`):", "", E.md_table(cnt.reset_index(), 0), "",
          "**Key set** (same rule as Philly's T2c): Seren jobs with `end_time > ACME_SIM_START` and `submit_time < ACME_TEST_END`, started, not censored, "
          f"`wall_s > 0`, `num_gpus > 0`, at all grid points with `e < wall_s`: {len(keys):,} rows, {keys['job_id'].nunique():,} jobs. "
          "Only 6 train rows are in it (one job submitted before 2023-06-15 that was still running on 06-22), so **in-sample rows are almost absent**; "
          f"the {int((keys['split'] == 'val').sum()):,} val rows were used for early stopping (the flat models and the transformer), and the "
          f"{int((keys['split'] == 'test').sum()):,} test rows ({100 * (keys['split'] == 'test').mean():.0f}% of the file) are clean.", ""]

    # training
    tf = json.loads((config.RUNS / "acme_tf_v1" / "summary.json").read_text())
    best_ep = min(tf["history"], key=lambda h: h["val_loss"])["epoch"]
    L += ["## Training", "",
          f"- **transformer** (`acme_tf_v1`; hyper-parameters as `tf_v1`: batch 512, lr 1e-3, patience 3, seed 0, fp32 on the GTX 1650): "
          f"{tf['params']:,} parameters (Philly {69702:,}); {tf['epochs_run']} epochs run, **best epoch {best_ep}** (val loss {tf['best_val_loss']:.4f}, "
          f"pinball {min(tf['history'], key=lambda h: h['val_loss'])['val_pinball']:.4f}), {tf['seconds_per_epoch']:.1f} s per epoch, "
          f"**{sum(h['seconds'] for h in tf['history']) / 60:.1f} min** in total. Val loss is noisy across epochs "
          f"({', '.join('%.3f' % h['val_loss'] for h in tf['history'])}).",
          ]
    for n in ("acme_gbdt", "acme_lightgbm", "acme_lightgbm_notype"):
        sm = json.loads((config.RUNS / n / "summary.json").read_text())
        L.append(f"- **{n}**: {sm['seconds']:.0f} s; best iterations q10/q50/q90/status = "
                 + "/".join(str(sm["val"][k]["best_iter"]) for k in ("q0.1", "q0.5", "q0.9", "status"))
                 + f"; features {len(sm['features'])}.")
    L.append("")

    # metrics
    names = ["last2", "gbdt", "lightgbm", "transformer", "oracle"]
    rows = []
    for n in names:
        rows.append({"dataset": "Acme Seren", "predictor": n, "n": 0, **test_metrics(D.ACME, n)})
    ab = pd.read_csv(config.RESULTS / "acme_predictor_metrics.csv")
    ab = ab[(ab["elapsed_s"].astype(str) == "all")].set_index("predictor")
    rows.append({"dataset": "Acme Seren", "predictor": "lightgbm_notype (ablation)", **ab.loc["lightgbm_notype", ["n"] + COLS].to_dict()})
    ph = [{"dataset": "Philly", "predictor": n, **test_metrics(D.PHILLY, n)} for n in names]
    pri = ab.loc["prior"]
    cols = ["dataset", "predictor", "n"] + COLS
    A = pd.DataFrame(rows)[cols]
    P = pd.DataFrame(ph)[cols]
    L += ["## Test-split metrics (the headline; nothing tuned on test)", "",
          "Pinball = mean over q10/q50/q90 in log1p space; coverage target 0.80; `mae_log` = mean |log1p(p50) - log1p(truth)|; log-loss is 3-class; "
          "AUROC for Failed vs not (`p_fail`) and Killed vs not (`p_cancel`); remaining runtime floored at 60 s. "
          "Acme and Philly side by side (Philly recomputed from its exported CSVs; it matches the T4 table).", "",
          "### Acme (Seren), test", "", E.md_table(A.drop(columns="dataset"), 3), "",
          f"Status prior (train class frequencies per grid point): log-loss {pri['logloss']:.3f}, AUROC Failed {pri['auroc_failed']:.3f}, AUROC Killed {pri['auroc_cancel']:.3f}.", "",
          "### Philly, test (T4)", "", E.md_table(P.drop(columns="dataset"), 3), ""]
    phi = pd.read_csv(config.RESULTS / "predictor_metrics.csv")
    phi = phi[phi["elapsed_s"].astype(str) == "all"].set_index("predictor")
    L += [f"Philly status prior: log-loss {phi.loc['prior', 'logloss']:.3f}, AUROC Failed {phi.loc['prior', 'auroc_failed']:.3f}.", ""]

    # per grid point for transformer & lightgbm (test)
    pg = pd.read_csv(config.RESULTS / "acme_predictor_metrics.csv")
    pg = pg[(pg["elapsed_s"].astype(str) != "all") & pg["predictor"].isin(["last2", "lightgbm", "transformer"])]
    L += ["## Per grid point (test)", "", E.md_table(pg[["predictor", "elapsed_s", "n", "pinball", "coverage_p10_p90", "mae_log", "logloss", "auroc_failed", "auroc_cancel"]], 3), ""]
    (config.REPORTS / "A2_acme_predictions.md").write_text("\n".join(L) + "\n@@SURPRISES@@\n", encoding="utf-8")
    print(E.md_table(A.drop(columns="dataset"), 3))
    print(E.md_table(P.drop(columns="dataset"), 3))
    print(E.md_table(pg[["predictor", "elapsed_s", "n", "pinball", "coverage_p10_p90", "mae_log", "logloss", "auroc_failed", "auroc_cancel"]], 3))


if __name__ == "__main__":
    main()
