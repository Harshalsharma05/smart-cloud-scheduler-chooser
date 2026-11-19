# T5 prediction exports

All five files cover the export keys (ML_PLAN T2c): valid jobs with `end_time > SIM_START` (2023-06-21 16:00) and `submit_time < TEST_END` (2023-07-29 16:00), every grid point with `e < wall_s`. Rows per split: test 332,176, train 6, val 99,577; total 431,759.

**Train rows are in-sample for gbdt, lightgbm and transformer, and val rows were used for early stopping. Only test rows are clean.** (The 1,252 train rows are jobs still running at SIM_START, so they are long jobs and not a typical in-sample set.)

| file | source | exported | rows | code commit |
|---|---|---|---|---|
| predictions/acme_oracle.csv | ground truth (ml/export_simple.py) | 2026-11-19 | 431,759 | 08d163c |
| predictions/acme_last2.csv | Tsafrir last-two (ml/export_simple.py) | 2026-11-19 | 431,759 | 08d163c |
| predictions/acme_gbdt.csv | runs/acme_gbdt/*.joblib (ml/baselines.py) | 2026-11-19 | 431,759 | 08d163c |
| predictions/acme_lightgbm.csv | runs/acme_lightgbm/*.txt (ml/baselines.py) | 2026-11-19 | 431,759 | 08d163c |
| predictions/acme_transformer.csv | runs/acme_tf_v1/best.pt (ml/predict.py) | 2026-11-19 | 431,759 | 08d163c |

Transformer checkpoint: `runs/acme_tf_v1/best.pt` (trained at commit 08d163c, 7 epochs, best val loss 0.9907). gbdt and lightgbm models were saved by their training run and loaded as they are; nothing was re-fitted.

Checker: `python ml/check_predictions.py predictions/acme_<name>.csv` printed OK for all five files.

## Consistency check (test metrics recomputed from the exported CSVs vs `ml/results/predictor_metrics.csv`, tolerance 0.001)

| predictor | metric | from_csv | evaluate | abs_diff |
|---|---|---|---|---|
| gbdt | pinball | 0.276251 | 0.276251 | 0.000000 |
| gbdt | coverage_p10_p90 | 0.841774 | 0.841774 | 0.000000 |
| gbdt | logloss | 0.605189 | 0.605189 | 0.000000 |
| lightgbm | pinball | 0.276003 | 0.276003 | 0.000000 |
| lightgbm | coverage_p10_p90 | 0.842722 | 0.842722 | 0.000000 |
| lightgbm | logloss | 0.596223 | 0.596223 | 0.000000 |
| transformer | pinball | 0.274901 | 0.274901 | 0.000000 |
| transformer | coverage_p10_p90 | 0.832959 | 0.832959 | 0.000000 |
| transformer | logloss | 0.562236 | 0.562236 | 0.000000 |

Max abs difference: 3.17e-08.
