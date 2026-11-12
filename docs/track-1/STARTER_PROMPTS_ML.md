# Track 1 starter prompts for Claude Code

Open Claude Code in the repo root. In every new session, paste the **session-start prompt** first, then the phase prompt. The order of phases across both tracks is in `docs/WORK_ORDER.md`.

Paste one prompt per Claude Code session (or per phase), in order. Each prompt asks Claude Code to stop after the phase and print a report. Paste that report into the Track 1 review chat before starting the next phase.

**Common report format.** Every prompt asks for this; paste it back as-is:

```
PHASE: <id>
FILES CHANGED: <list>
WHAT WAS BUILT: <3–6 lines>
DONE-CHECK: <the command(s) run and their output, shortened>
TESTS: <pytest -q output line>
NUMBERS: <any counts/metrics the phase asks for>
SURPRISES / DECISIONS MADE: <anything that differs from the plan, or that you had to decide>
COMMIT: <hash + message>
```

---

## Session-start prompt (every new Track 1 session)

```
This is a Track 1 (ML) session. Read CLAUDE.md, then docs/track-1/ML_PLAN.md (sections 0-2, 8 and the phase I give you). For anything in ml/, ML_PLAN.md is the source of truth: if CLAUDE.md still says "current goal: progress-check demo / IMPLEMENTATION.md is the source of truth", that part does not apply to ml/ work.

Rules for this session:
- Do only the phase I name. Do not start the next one.
- Work only in ml/, data/, predictions/, runs/ and ml/reports/. Do not change backend/, frontend/ or paper/. You may edit .gitignore (T0) and tick steps in docs/WORK_ORDER.md.
- Contract files and column names are fixed (ML_PLAN section 2 and T1/T2b). If the data forces a different choice, stop and ask me.
- Leakage rules are never relaxed. No test-split metrics before T4.
- Keep it simple: plain functions and dataclasses, plain PyTorch, only the tests the phase lists.
- I am on Windows. Give me any commands I must run myself.

At the end of the phase:
1. Run the phase's done-check and show the output.
2. Commit as the phase prompt says.
3. Tick that step in docs/WORK_ORDER.md and append one progress-log line (date, step, commit hash, one-line result); commit as "work order: step <N> done".
4. Print the report in the format at the top of docs/track-1/STARTER_PROMPTS_ML.md, and stop.
```

---

## T0: Housekeeping

```
Read CLAUDE.md and docs/track-1/ML_PLAN.md (sections 0, 2 and T0). Do only phase T0.

0. First commit the untracked planning docs unchanged (docs/WORK_ORDER.md, docs/track-1/, docs/track-2/) as "docs: track plans and work order".
1. Append to .gitignore only what is missing (results/ is already there): data/, predictions/, *.pt, *.ckpt, runs/, ml/.venv/, ml/results/*.parquet
2. Create ml/ with requirements.txt (pandas, pyarrow, numpy, scikit-learn, lightgbm, torch, pytest, tqdm), config.py (paths, GRID, H=32, split dates, seed, as in ML_PLAN section 2) and tests/__init__.py.
3. Create ml/.venv with Python 3.11 (py -3.11 -m venv ml\.venv). Run nvidia-smi first and install the torch CUDA wheel that matches it (from download.pytorch.org). If CUDA torch fails, install the CPU wheel and tell me.
4. Create empty folders data/raw, data/processed, predictions (git-ignored).
5. Do not touch backend/ or frontend/.

Done-check: python -c "import torch, pandas, lightgbm, sklearn; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))" and git status (no data files tracked).
Commit as "T0: ml/ skeleton, venv, gitignore". Then stop and print the report in the format from docs/track-1/STARTER_PROMPTS_ML.md.
```

---

## T1: Download and clean the Philly job log

Before running this, download `trace-data.tar.gz` from https://github.com/msr-fiddle/philly-traces (use the file's "Download" button, since it is a Git LFS file) into `data/raw/`.

```
Read CLAUDE.md and docs/track-1/ML_PLAN.md (sections 1, 2 and T1). Do only phase T1.

data/raw/trace-data.tar.gz is downloaded. Unpack it into data/raw/philly/ with tar -xzf and list the file sizes (expect ~6.6 GB total).

Write ml/philly_clean.py exactly as in T1:
- inspect timestamp formats first and print a few raw examples; convert all times to UTC epoch seconds (America/Los_Angeles if no tz is given), count the ambiguous/NaT values around the 5 Nov 2017 DST change
- attempts: drop invalid ones as specified; machine and gpu_ids strings
- jobs: first_start_time, end_time, wall_s, run_s, num_gpus (max over attempts, report how often attempts differ), num_attempts, status, censored, never_started; job_id as string
- write data/processed/philly_jobs.parquet, philly_attempts.parquet and philly_machines.parquet (required; columns machine_id, num_gpus, gpu_mem_gb)
- machine-ID check: the % of attempt machines found in machine_id, with 3 example values from each. If they don't match, stop and report; do not invent a mapping
- light timezone sanity check against the first ~2M rows of cluster_gpu_util, as described; report the result, and don't loop on it if it is unclear
- write ml/reports/T1_data_report.md with every item listed in T1 (including jobs per week, so we can confirm the split dates, machines by GPU memory size, total GPUs, and the machine-ID match)

Add the one test from T1 (a tiny hand-written JSON, 4 cases) in ml/tests/test_t1_clean.py.
Do not change the contract column names. If the data forces a different choice than the plan, stop and ask me.

Done-check: the script runs end to end, the row count is ~117k, the report exists, pytest ml/tests -q passes.
Commit as "T1: Philly cleaning to shared parquet tables + data report". Stop and print the report (put the key numbers from T1_data_report.md in NUMBERS).
```

---

## T2: Training examples and time split

```
Read CLAUDE.md and docs/track-1/ML_PLAN.md (sections 2 and T2, and contract section 7 in the plan's section 10). Do only phase T2. First commit the updated docs/track-1 files as "docs: T2 validity filter".

Write ml/examples.py:
- valid jobs only (started, not censored, wall_s > 0, num_gpus > 0; see the 'Valid job' bullet in ML_PLAN T2), submit_time < TEST_END (half-open); the same filter for history jobs; report valid and excluded counts per split
- one row per (job, grid point) where e < wall_s
- t_obs = submit_time if e == 0 else first_start_time + e
- current-job features, history (per user sorted by end_time, numpy.searchsorted, end_time <= t_obs, last 32, never the job itself) stored as hist_idx (row indices into philly_jobs), and aggregated history features, all as listed in T2
- split column from submit_time using the dates in ml/config.py
- write data/processed/ml_examples.parquet and ml/reports/T2_examples_report.md (rows per split and per grid point, % empty history, median history length)

Add the leakage test from T2 in ml/tests/test_t2_leakage.py, on a small synthetic jobs table.

Done-check: runs in a few minutes, pytest ml/tests -q passes, the row counts are reported.
Commit as "T2: leak-free training examples + time split". Stop and print the report.
```

---

## T2b: Export oracle and last2 + checker

```
Read CLAUDE.md and docs/track-1/ML_PLAN.md (T2b and section 10). Do only phase T2b.

1. ml/check_predictions.py <csv> [--allow-extra-keys]: all checks in T2b; prints OK or a list of problems; exit code 1 on problems.
2. ml/export_simple.py: writes predictions/oracle.csv (test jobs + train/val jobs still running at TEST_START) and predictions/last2.csv (Tsafrir last-two with the fallbacks and user failure/cancel rates from T2b). Floor at 60 s.
3. One test in ml/tests/test_t2b_exports.py (oracle = wall_s - e floored; last2 on a 3-job toy user).

Done-check: python ml/check_predictions.py predictions/oracle.csv --allow-extra-keys and python ml/check_predictions.py predictions/last2.csv both print OK. Report the row counts and the number of distinct test jobs.
Commit as "T2b: oracle and last2 predictions + checker". Stop and print the report.
```

---

## T3a: Transformer code + local smoke run

```
Read CLAUDE.md and docs/track-1/ML_PLAN.md (T3a). Do only phase T3a.

Write ml/model.py, ml/train.py and ml/predict.py exactly as specified in T3a (history tokens + [CUR] token, 2-layer encoder d_model=64, 4 heads, non-crossing quantile head with pinball loss in log1p space, 3-class status head with CE, fp32 by default with an --amp flag, early stopping on val loss, a checkpoint every epoch in runs/<name>/, summary.json, --max-rows for debugging, fixed seed). No user-ID embedding. Keep it to plain PyTorch, with no Lightning or other frameworks.

Add the one test from T3a (ml/tests/test_t3_model.py: shapes, non-crossing quantiles, loss falls on 256 examples within 200 steps; run it on CPU so it is fast).

Done-check:
1. python ml/train.py --name smoke --max-rows 20000 --epochs 2 on the GTX 1650; show the train/val loss per epoch.
2. Time one full epoch on all train rows (python ml/train.py --name timing --epochs 1). Report seconds per epoch, GPU memory used and the parameter count.
3. pytest ml/tests -q passes.
Do not compute any test-split metrics.
Commit as "T3a: transformer model, training and prediction scripts". Stop and print the report, including your recommendation: local or Kaggle for the full run (rule in T3a).
```

---

## T3b: Full training run

**If training locally:**

```
Read docs/track-1/ML_PLAN.md (T3b). Run python ml/train.py --name tf_v1 (fp32, defaults from config). When it finishes, run predict.py on the val split and print val pinball loss and val status log-loss for tf_v1 and for last2 (computed from ml_examples on val). Do not compute test metrics. If tf_v1 is not better than last2 on val, stop and report, and do not tune.
Commit runs/tf_v1/summary.json only (checkpoints are git-ignored) as "T3b: tf_v1 trained". Stop and print the report.
```

**If training on Kaggle:**

```
Read docs/track-1/ML_PLAN.md (T3b). Write ml/KAGGLE.md: the exact steps to (1) zip ml/ code and create one PRIVATE Kaggle dataset with ml_examples.parquet, philly_jobs.parquet and the zip; (2) notebook cells for GPU T4 that unzip, pip install only what Kaggle lacks, and run python train.py --data <input paths> --out /kaggle/working --amp --name tf_v1; (3) downloading runs/tf_v1/ back. Make sure train.py accepts --data and --out paths. Then stop, and I will run Kaggle myself.
```

After you've brought `runs/tf_v1/` back:

```
runs/tf_v1/ is back from Kaggle. Run predict.py on val and print val pinball loss and val status log-loss for tf_v1 and last2. No test metrics. Commit summary.json as "T3b: tf_v1 trained (Kaggle T4)". Stop and print the report.
```

---

## T4: Baselines and evaluation

```
Read CLAUDE.md and docs/track-1/ML_PLAN.md (T4). Do only phase T4.

1. ml/baselines.py: gbdt (sklearn HistGradientBoosting, quantile 0.1/0.5/0.9 + classifier) and lightgbm (quantile + multiclass), both on the same flat feature table (current-job + aggregated history features from ml_examples), early stopping on val, quantiles sorted per row, probabilities clipped so p_fail + p_cancel <= 1. Save the models under runs/gbdt/ and runs/lightgbm/.
2. ml/evaluate.py: on the TEST split, for last2, gbdt, lightgbm, transformer (tf_v1) and oracle: mean pinball (log space), median abs error of p50 in minutes, [p10,p90] coverage, 3-class log-loss, AUROC Failed-vs-not, Brier for p_fail; overall and per grid point. Write ml/results/predictor_metrics.csv and predictor_metrics.md.
3. One test, ml/tests/test_t4_metrics.py (metric functions on a 5-row toy example).

Report the numbers exactly as they come out. Do not tune on test.
Done-check: the table has all 5 predictors, the oracle row is perfect, pytest passes.
Commit as "T4: GBDT/LightGBM baselines + test evaluation". Stop and print the report, and paste predictor_metrics.md into NUMBERS.
```

---

## T5: Export remaining predictions

```
Read docs/track-1/ML_PLAN.md (T5). Do only phase T5.

Write predictions/gbdt.csv, predictions/lightgbm.csv and predictions/transformer.csv for the test keys (reuse predict.py / baselines.py; floor at 60 s). Run ml/check_predictions.py on all five files (oracle with --allow-extra-keys). Write ml/reports/T5_exports.md (and a copy at predictions/README.md): per file, its source model/checkpoint, date, row count and git commit.

Done-check: five OK lines.
Commit as "T5: all prediction files exported". Stop and print the report.
```
