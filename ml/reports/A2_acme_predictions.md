# A2: Acme (Seren) predictions

Same pipeline as Philly through a dataset switch (`ml/dataset.py`; `--dataset acme`). Philly code paths, data, predictions and results are unchanged (checked: `tf_v1` val predictions identical to the saved file, Philly `predictor_metrics.csv/.md` and the five Philly prediction CSVs byte-identical after the changes).

## Set-up

- Seren GPU jobs only (`acme_jobs.parquet`, 450,590 jobs). Split by submit time, China time (UTC+8) midnights stored as UTC in `ml/config.py`: train < 2023-06-15, val 2023-06-15 to 06-29, test 2023-06-29 to 07-30 (half-open). Simulation start `ACME_SIM_START` = 2023-06-22.
- Elapsed grid, leakage rules (t_obs, history = same user's jobs with `end <= t_obs`, at most 32, never the job itself) and the 60 s floor are the Philly ones.
- Labels: `p_fail` = FAILED / NODE_FAIL / TIMEOUT; `p_cancel` = CANCELLED / PREEMPTED (the A1 mapping, PREEMPTED -> Killed).
- Features: Philly's, minus what Acme lacks: `run_s`, `num_attempts` (history tokens) and `attempts_so_far`; `vc` is the constant `seren` so the flat models drop it (the transformer keeps the embedding, which is a constant). `submit_hour` and `submit_dow` are UTC, as in Philly. `type` (workload type) is added as a categorical for all learned models, see below.

## Is `type` known at submission?

The README only says "workload type in LLM development". The NSDI 2024 paper (Hu et al., Sec. 3) says the jobs are categorised "according to their production division and metadata (e.g., job names)". Production division and job name are set by the user when the job is submitted, so `type` is treated as known at submission and is used. **Caveat:** the labelling rule is not published, so this cannot be verified from the data. As a check, a LightGBM without `type` (`lightgbm_notype`, same everything else) is evaluated below; it is almost as good, so the result does not depend on `type`.

## Example counts

| split | example rows | jobs | rows with empty history | key-set rows | key-set jobs |
|---|---|---|---|---|---|
| train | 259,199 | 136,460 | 0.11% | 6 | 1 |
| val | 154,812 | 88,965 | 0.01% | 99,577 | 56,642 |
| test | 332,176 | 175,233 | 0.01% | 332,176 | 175,233 |
| all | 746,187 | 400,658 | 0.04% | 431,759 | 231,876 |

Example rows per grid point (every row needs `e < wall_s`):

| split | e=0 | e=60 | e=600 | e=3600 | e=14400 | e=43200 |
|---|---|---|---|---|---|---|
| test | 175233 | 115525 | 30366 | 7765 | 2544 | 743 |
| train | 136460 | 90385 | 24459 | 4825 | 2021 | 1049 |
| val | 88965 | 45160 | 14167 | 3871 | 1897 | 752 |

**Key set** (same rule as Philly's T2c): Seren jobs with `end_time > ACME_SIM_START` and `submit_time < ACME_TEST_END`, started, not censored, `wall_s > 0`, `num_gpus > 0`, at all grid points with `e < wall_s`: 431,759 rows, 231,876 jobs. Only 6 train rows are in it (one job submitted before 2023-06-15 that was still running on 06-22), so **in-sample rows are almost absent**; the 99,577 val rows were used for early stopping (the flat models and the transformer), and the 332,176 test rows (77% of the file) are clean.

## Training

- **transformer** (`acme_tf_v1`; hyper-parameters as `tf_v1`: batch 512, lr 1e-3, patience 3, seed 0, fp32 on the GTX 1650): 68,998 parameters (Philly 69,702); 7 epochs run, **best epoch 4** (val loss 0.9907, pinball 0.4039), 17.0 s per epoch, **2.0 min** in total. Val loss is noisy across epochs (1.154, 1.061, 1.032, 0.991, 1.028, 0.998, 1.068).
- **acme_gbdt**: 100 s; best iterations q10/q50/q90/status = 138/281/111/42; features 13.
- **acme_lightgbm**: 10 s; best iterations q10/q50/q90/status = 153/76/73/44; features 13.
- **acme_lightgbm_notype**: 12 s; best iterations q10/q50/q90/status = 122/174/116/89; features 12.

## Test-split metrics (the headline; nothing tuned on test)

Pinball = mean over q10/q50/q90 in log1p space; coverage target 0.80; `mae_log` = mean |log1p(p50) - log1p(truth)|; log-loss is 3-class; AUROC for Failed vs not (`p_fail`) and Killed vs not (`p_cancel`); remaining runtime floored at 60 s. Acme and Philly side by side (Philly recomputed from its exported CSVs; it matches the T4 table).

### Acme (Seren), test

| predictor | n | pinball | coverage_p10_p90 | mae_log | logloss | auroc_failed | auroc_cancel | brier_pfail | mae_p50_min |
|---|---|---|---|---|---|---|---|---|---|
| last2 | 332,176 | 0.545 | 0.189 | 1.090 | 3.783 | 0.890 | 0.791 | 0.104 | 1.867 |
| gbdt | 332,176 | 0.276 | 0.842 | 0.834 | 0.605 | 0.912 | 0.829 | 0.088 | 1.614 |
| lightgbm | 332,176 | 0.276 | 0.843 | 0.835 | 0.596 | 0.913 | 0.834 | 0.087 | 1.595 |
| transformer | 332,176 | 0.275 | 0.833 | 0.832 | 0.562 | 0.925 | 0.847 | 0.082 | 1.870 |
| oracle | 332,176 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| lightgbm_notype (ablation) | 332,176 | 0.276 | 0.849 | 0.841 | 0.608 | 0.913 | 0.820 | 0.086 | 1.634 |

Status prior (train class frequencies per grid point): log-loss 0.928, AUROC Failed 0.635, AUROC Killed 0.583.

### Philly, test (T4)

| predictor | n | pinball | coverage_p10_p90 | mae_log | logloss | auroc_failed | auroc_cancel | brier_pfail | mae_p50_min |
|---|---|---|---|---|---|---|---|---|---|
| last2 | 71806 | 1.034 | 0.080 | 2.069 | 2.343 | 0.772 | 0.777 | 0.155 | 52.608 |
| gbdt | 71806 | 0.419 | 0.779 | 1.311 | 0.682 | 0.797 | 0.755 | 0.134 | 43.495 |
| lightgbm | 71806 | 0.426 | 0.802 | 1.343 | 0.678 | 0.800 | 0.758 | 0.133 | 44.978 |
| transformer | 71806 | 0.442 | 0.778 | 1.388 | 0.789 | 0.805 | 0.759 | 0.138 | 65.168 |
| oracle | 71806 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |

Philly status prior: log-loss 0.888, AUROC Failed 0.542.

## Per grid point (test)

| predictor | elapsed_s | n | pinball | coverage_p10_p90 | mae_log | logloss | auroc_failed | auroc_cancel |
|---|---|---|---|---|---|---|---|---|
| last2 | 0 | 175233 | 0.405 | 0.267 | 0.811 | 3.600 | 0.894 | 0.844 |
| last2 | 60 | 115525 | 0.514 | 0.124 | 1.028 | 3.296 | 0.901 | 0.753 |
| last2 | 600 | 30366 | 0.982 | 0.056 | 1.964 | 4.595 | 0.829 | 0.655 |
| last2 | 3600 | 7765 | 1.708 | 0.012 | 3.416 | 9.433 | 0.825 | 0.535 |
| last2 | 14400 | 2544 | 2.167 | 0.002 | 4.333 | 10.503 | 0.842 | 0.471 |
| last2 | 43200 | 743 | 2.806 | 0.001 | 5.612 | 7.413 | 0.712 | 0.552 |
| lightgbm | 0 | 175233 | 0.235 | 0.853 | 0.693 | 0.618 | 0.904 | 0.867 |
| lightgbm | 60 | 115525 | 0.284 | 0.843 | 0.864 | 0.524 | 0.916 | 0.792 |
| lightgbm | 600 | 30366 | 0.428 | 0.775 | 1.370 | 0.630 | 0.856 | 0.715 |
| lightgbm | 3600 | 7765 | 0.436 | 0.877 | 1.353 | 0.903 | 0.816 | 0.591 |
| lightgbm | 14400 | 2544 | 0.405 | 0.879 | 1.209 | 0.924 | 0.806 | 0.516 |
| lightgbm | 43200 | 743 | 0.464 | 0.760 | 1.432 | 0.957 | 0.668 | 0.606 |
| transformer | 0 | 175233 | 0.234 | 0.837 | 0.693 | 0.569 | 0.920 | 0.883 |
| transformer | 60 | 115525 | 0.285 | 0.849 | 0.867 | 0.516 | 0.916 | 0.810 |
| transformer | 600 | 30366 | 0.419 | 0.765 | 1.330 | 0.578 | 0.909 | 0.709 |
| transformer | 3600 | 7765 | 0.439 | 0.772 | 1.352 | 0.902 | 0.891 | 0.568 |
| transformer | 14400 | 2544 | 0.389 | 0.863 | 1.256 | 0.834 | 0.837 | 0.620 |
| transformer | 43200 | 743 | 0.389 | 0.851 | 1.210 | 0.893 | 0.744 | 0.653 |

## What surprised me

1. **The three learned models tie on runtime.** Test pinball is 0.275 (transformer), 0.276 (gbdt) and 0.276 (lightgbm) and coverage is 0.83-0.84 for all. With one seed this gap is noise. On Philly the GBDTs clearly beat the transformer on runtime; here they do not.
2. **The transformer is best on status.** Log-loss 0.562 against 0.596 (lightgbm) and 0.605 (gbdt); AUROC Failed 0.925 against 0.913 / 0.912; AUROC Killed 0.847 against 0.834 / 0.829. On Philly the GBDTs had the better log-loss (0.678 against 0.789) and the transformer was only slightly ahead on AUROC. Its median error of p50 in minutes is still worse (1.87 against 1.60 min) although its log-space error is the same, so the difference is in the long tail.
3. **Status is far easier to predict on Acme than on Philly.** AUROC Failed is 0.91-0.93 (Philly 0.80) and the status prior log-loss is 0.928 (Philly 0.888), so the models gain much more over the prior here (0.56-0.60). `type` does not explain it (item 4). I did not investigate further; very repetitive per-user jobs (evaluation sweeps, debug runs) are a plausible reason but this is not verified.
4. **`type` adds little.** LightGBM without `type` has the same pinball (0.276), a log-loss of 0.608 against 0.596 and an AUROC Killed of 0.820 against 0.834. It helps the cancel prediction a little and nothing else.
5. **`last2` is bad, and worse than on Philly.** Coverage 0.19, log-loss 3.78 (users with no recent failures get `p_fail` near 0, which the log-loss punishes; Philly 2.34). At e = 3600 s its log-loss is 9.4. Track 2 should not read `last2` status values as probabilities.
6. **Jobs are short, so most rows are at e = 0 and e = 60.** 175,233 of the 332,176 test rows are at e = 0 and only 743 at e = 43,200. The runtime numbers are therefore dominated by short jobs. Absolute pinball and MAE are not comparable with Philly (Seren median wall time 2.3 min against 21.8 min); compare ranks, not values.
7. **Almost no in-sample rows.** The key set starts on 2023-06-22, a week after the train split ends, so it has only 6 train rows. This is cleaner than Philly (1,252 train rows). Val is 23% of the key set and test 77%.
8. **The key set is much bigger than Philly's:** 431,759 rows over 231,876 jobs (Philly 111,451 rows, 37,262 jobs). The CSVs are 14-41 MB each. The window is 38 days (2023-06-22 to 2023-07-30).
9. **The transformer's val loss is noisy** (1.154, 1.061, 1.032, 0.991, 1.028, 0.998, 1.068), so the best epoch (4) was partly luck. With patience 3 the run stopped after 7 epochs, 2 minutes in total.
10. **One seed only**, as requested; no confidence intervals. The differences between the three learned models in items 1-2 should not be quoted as significant.

## Files

- `predictions/acme_{oracle,last2,gbdt,lightgbm,transformer}.csv`: the contract format, key set above; each passes `python ml/check_predictions.py` (the checker picks the Acme key set from the `acme_` file name). Export report: `ml/reports/A2_exports.md`. The test metrics recomputed from the CSVs match `ml/results/acme_predictor_metrics.csv` to 3e-8.
- `ml/results/acme_predictor_metrics.{csv,md}`, `acme_predictor_metrics_by_split.csv`; `data/processed/acme_ml_examples.parquet`; `runs/acme_tf_v1`, `runs/acme_gbdt`, `runs/acme_lightgbm`, `runs/acme_lightgbm_notype`.

