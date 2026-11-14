# T5 prediction exports

All five files cover the export keys (ML_PLAN T2c): valid jobs with `end_time > SIM_START` (2017-11-08) and `submit_time < TEST_END` (2017-12-16), every grid point with `e < wall_s`. Rows per split: test 71,806, train 1,252, val 38,393; total 111,451.

**Train rows are in-sample for gbdt, lightgbm and transformer, and val rows were used for early stopping. Only test rows are clean.** (The 1,252 train rows are jobs still running at SIM_START, so they are long jobs and not a typical in-sample set.)

| file | source | exported | rows | code commit |
|---|---|---|---|---|
| predictions/oracle.csv | ground truth (ml/export_simple.py) | 2026-11-14 | 111,451 | e983718 |
| predictions/last2.csv | Tsafrir last-two (ml/export_simple.py) | 2026-11-14 | 111,451 | e983718 |
| predictions/gbdt.csv | runs/gbdt/*.joblib (ml/baselines.py) | 2026-11-14 | 111,451 | e983718 |
| predictions/lightgbm.csv | runs/lightgbm/*.txt (ml/baselines.py) | 2026-11-14 | 111,451 | e983718 |
| predictions/transformer.csv | runs/tf_v1/best.pt (ml/predict.py) | 2026-11-14 | 111,451 | e983718 |

Transformer checkpoint: `runs/tf_v1/best.pt` (trained at commit f283962, 9 epochs, best val loss 1.3243). gbdt and lightgbm models were saved by T4 and loaded as they are; nothing was re-fitted.

Checker: `python ml/check_predictions.py predictions/<name>.csv` printed OK for all five files.

## Consistency check (test metrics recomputed from the exported CSVs vs `ml/results/predictor_metrics.csv`, tolerance 0.001)

| predictor | metric | from_csv | evaluate | abs_diff |
|---|---|---|---|---|
| gbdt | pinball | 0.418514 | 0.418514 | 0.000000 |
| gbdt | coverage_p10_p90 | 0.779252 | 0.779252 | 0.000000 |
| gbdt | logloss | 0.682314 | 0.682314 | 0.000000 |
| lightgbm | pinball | 0.426386 | 0.426386 | 0.000000 |
| lightgbm | coverage_p10_p90 | 0.801660 | 0.801660 | 0.000000 |
| lightgbm | logloss | 0.678261 | 0.678261 | 0.000000 |
| transformer | pinball | 0.441537 | 0.441537 | 0.000000 |
| transformer | coverage_p10_p90 | 0.777762 | 0.777762 | 0.000000 |
| transformer | logloss | 0.789330 | 0.789330 | 0.000000 |

Max abs difference: 2.54e-08.
