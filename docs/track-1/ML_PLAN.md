# Track 1 plan: ML / transformer (Burst to Flex, CCGrid 2027)

Version 1.1, 12 Nov 2026 (machines parquet required; session-start prompt; progress log). Scope: phases T0–T5. Code lives in `ml/`. This file is the source of truth for work in `ml/`.
Contract with Track 2: `claude/PROJECT_CONTEXT.md` section 7 (v2). If this plan and the contract disagree, the contract wins. Raise the difference in the Track 1 review chat; don't decide it in code.

---

## 0. How to use this plan

- Do **one phase at a time**, in this order: T0 → T1 → T2 → T2b → T3a → T3b → T4 → T5.
- Every phase ends with a **done-check**, a **commit**, and a **report**. After any commit it is safe to pause or switch to Track 2.
- Effort figures are rough hours for one Claude Code session plus your review. They are not dates.
- "Unattended" marks steps that run while you do something else (download, Kaggle training).
- Anything uncertain or nice-to-have goes to section 9 ("Later"), not into the current phase.
- Every Claude Code session starts with the **Track 1 session-start prompt** (`STARTER_PROMPTS_ML.md`), then the phase prompt. The order of phases across both tracks is in `docs/WORK_ORDER.md`. At the end of each phase, Claude Code ticks that step and adds one line to its progress log.
- Until Track 2's S0 updates `CLAUDE.md`, its old header ("progress-check demo, IMPLEMENTATION.md is the source of truth") does not apply to `ml/`. The session-start prompt says so.

### What Track 2 is waiting for

| After phase | Track 2 can start |
|---|---|
| T1 | Philly loader, motivation charts, utilisation join, power model |
| T2b | Controller and experiments using `oracle` and `last2` |
| T5 | E3 with all predictors |

---

## 1. Facts about the Philly trace (checked 11 Nov 2026)

- **Source:** https://github.com/msr-fiddle/philly-traces (Microsoft Research, Project Fiddle).
  - A sanitised subset of the workload in Jeon et al., USENIX ATC 2019.
- **License:** CC-BY-4.0. Using it is fine, and we must cite the ATC'19 paper. The data is never committed to our repo.
- **File:** `trace-data.tar.gz`, about 0.98 GB compressed and 6.6 GB unpacked.
  - It is stored in the repo with Git LFS, so download it with the file's "Download" button on GitHub, or with `git lfs`.
  - It contains 5 files:
    - `cluster_job_log` (JSON)
    - `cluster_gpu_util` (CSV, per minute, `gpu0_util`…`gpu7_util`, `"NA"` when offline)
    - `cluster_cpu_util` (CSV)
    - `cluster_mem_util` (CSV)
    - `cluster_machine_list` (CSV: `machineId`, number of GPUs, single-GPU memory)
- **Coverage:** jobs submitted 2017-08-07 to 2017-12-22, 117,325 jobs.
- **Job log fields:**
  - `jobid`, `user` (hashed), `vc` (hashed), `submitted_time`.
  - `status`: `Pass` / `Killed` / `Failed`.
  - `attempts`: a list of attempts, each with `start_time`, `end_time`, and `detail` (a list of `{ip, gpus}`).
- **Known quirks (from the trace README):**
  - Some jobs have no attempts.
  - Some attempts lack a start or end time.
  - A `None` last `end_time` means the job was still running when the trace ended.
- **GPU model: not confirmed.**
  - Neither the trace nor the ATC'19 paper names a GPU model. The paper says only "thousands of GPUs of the same model".
  - The paper's controlled experiment uses P100s, but that is not the production cluster.
  - The machine list shows **12 GB and 24 GB** GPU memory. Some other simulators *assume* P100 (12 GB) for the 2-GPU servers and P40 (24 GB) for the 8-GPU servers, but that is their assumption, not a fact.
  - → Track 2 keeps "one representative GPU power ± 20%". T1 reports how many machines have each GPU memory size.
- **Job statuses (ATC'19 §2.3):**
  - Passed means the job completed.
  - Killed means the user terminated it.
  - Unsuccessful (`Failed` in the trace) means it still failed after a fixed number of retries.
  - So a job with several attempts usually means retries after failures. That is a useful, leak-free feature: the number of attempts started so far.

---

## 2. Definitions used everywhere (put them in `ml/config.py`)

- **Elapsed grid:** `GRID = [0, 60, 600, 3600, 14400, 43200]` seconds.
- **Elapsed** = time since the job's first start (`first_start_time`).
- **Observation time** of an example (job *j*, grid point *e*):
  - `t_obs = submit_time` if *e* = 0. This is the safe choice: it works whether the job is still queued or just starting.
  - `t_obs = first_start_time + e` if *e* > 0.
  - *Clarification sent to Track 2; see section 10.*
- **Row exists** only if the job is still running at *e*, that is `e < wall_s`. Grid point 0 always exists for a job that started.
- **Targets:**
  - `remaining_s = wall_s − e`. Models predict `log1p(remaining_s)`.
  - `status` ∈ {Pass, Failed, Killed}.
- **History** of an example: the same user's jobs with `end_time ≤ t_obs`, newest last, at most `H = 32`. Never the job itself.
- **Time split (defaults; T1 confirms them from weekly job counts):**
  - train: submitted before 2017-11-01
  - val: 2017-11-01 to 2017-11-14 (used only for early stopping and tuning)
  - test: 2017-11-15 to 2017-12-15
  - Jobs submitted after 2017-12-15 are not used, to limit censoring at the end of the trace.
  - Track 2 uses the same test window: the constants `TEST_START` and `TEST_END` in `ml/config.py`.
- **Floor:** every exported remaining runtime is at least 60 s.

---

## T0. Housekeeping (≈0.5–1 h)

**Goal:** a clean place to work.

**Tasks:**
- First, commit the untracked planning docs as they are (`docs/WORK_ORDER.md`, `docs/track-1/`, `docs/track-2/`) in their own commit, so later `git status` checks start clean.
- Add to `.gitignore` (only what is missing; `results/` is already there): `data/`, `predictions/`, `*.pt`, `*.ckpt`, `runs/`, `ml/.venv/`, `ml/results/*.parquet`.
- Create `ml/` with:
  - `requirements.txt`: pandas, pyarrow, numpy, scikit-learn, lightgbm, torch, pytest, tqdm.
  - `config.py`: paths, `GRID`, `H`, split dates, seeds.
  - `tests/`.
- Create `ml/.venv` with Python 3.11 (`py -3.11 -m venv ml\.venv`).
  - Install torch with the CUDA wheel that matches `nvidia-smi`.
  - Use the CPU wheel only if CUDA fails, and say so.
- Create empty `data/raw/`, `data/processed/`, `predictions/` (git-ignored).

**Done-check:** `python -c "import torch, pandas, lightgbm, sklearn; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` prints `True` and the GTX 1650. `git status` is clean after the commit. Step 1 is ticked in `docs/WORK_ORDER.md`.

**Pause point:** after the commit.

---

## T1. Download and clean the Philly job log (≈3–4 h, the download is unattended)

**Goal:** the shared tables in the contract, plus a short report on the data.

**Step 1 (you, manual):**
- Download `trace-data.tar.gz` into `data/raw/`.
- Check that about 8 GB of disk is free.
- Claude Code then unpacks it into `data/raw/philly/` (`tar -xzf`; Windows has `tar`) and lists the file sizes. The total should be about 6.6 GB.

**Step 2, `ml/philly_clean.py`:**
- Read `cluster_job_log`.
- **Timestamps:**
  - Inspect the string formats first.
  - Convert every time to UTC epoch seconds.
  - If job-log times have no timezone, treat them as `America/Los_Angeles` (the GPU-util CSV marks PST/PDT).
  - Daylight saving time ends on 5 Nov 2017, which is inside our data. Count the ambiguous and NaT values and report them.
- **Attempts:**
  - Drop attempts with a missing start or end time, except a missing *last* end, which marks a censored job.
  - `machine` = the `ip`(s) in `detail`, joined with `;`.
  - `gpu_ids` = `machine:gpu` pairs, joined with `;`.
- **Per job:**
  - `first_start_time` = the earliest attempt start.
  - `end_time` = the last attempt end.
  - `wall_s = end_time − first_start_time`.
  - `run_s` = the sum of attempt durations.
  - `num_attempts` = the number of valid attempts.
  - `num_gpus` = the maximum over attempts of the GPUs used. Report how often attempts differ.
- **Keep every job** in `philly_jobs.parquet`, and add two columns (an additive change, flagged to Track 2):
  - `censored` (bool): the job was still running at the end of the trace.
  - `never_started` (bool): the job has no valid attempt.
- **Also write:**
  - `philly_attempts.parquet`.
  - `philly_machines.parquet` (**required**: Track 2 uses it for the GPU pool size). Columns `machine_id` (string), `num_gpus` (int), `gpu_mem_gb` (int, parsed from strings like `" 24GB"`).
- **Machine-ID check:** report the % of attempt machines (`detail.ip`) that appear in `machine_id`, and give 3 example values from each. Track 2's utilisation join needs the IDs to match. If they don't match, stop and report; don't invent a mapping.
- **Timezone sanity check (light):**
  - Read only the first ~2M rows of `cluster_gpu_util`.
  - For about 50 attempts in that period, compare mean utilisation on their GPUs in the 10 min before vs after the attempt start.
  - If utilisation clearly rises after the start, the timezone choice is right. Report the result; don't loop on it if it is unclear.

**Outputs:**
- `data/processed/philly_jobs.parquet`: `job_id` (string), `user, vc, submit_time, first_start_time, end_time, wall_s, run_s, num_gpus, num_attempts, status, censored, never_started`.
- `data/processed/philly_attempts.parquet`: `job_id, attempt, start_time, end_time, machine, gpu_ids`.
- `data/processed/philly_machines.parquet`: `machine_id, num_gpus, gpu_mem_gb`.
- `ml/reports/T1_data_report.md`:
  - counts by status
  - jobs per week (to confirm the split)
  - % never started / censored / dropped attempts
  - `num_gpus` and `wall_s` distributions (median, p90)
  - number of users
  - machines by GPU memory (12 GB / 24 GB) and GPU count; total GPUs (the pool size G for Track 2)
  - the machine-ID match result
  - timezone decision and the sanity-check result
  - how many test-window jobs are censored

**Test (1):** a tiny hand-written JSON with 4 jobs. Each case must give the right row values:
- a job with 2 attempts
- a job with no attempts
- a job with a missing last end time
- an attempt with a missing start

**Done-check:**
- The script runs end to end.
- About 117k rows; the count matches the README.
- The report exists.
- `pytest ml/tests -q` passes.

**Pause point:** after the commit. **→ Track 2 is unblocked for the loader and power model.**

---

## T2. Training examples and time split (≈3–4 h)

**Goal:** a leak-free example table that every predictor uses.

**`ml/examples.py`:**
- **Valid job** (from the T1 report): started (`never_started` false), not `censored`, `wall_s > 0` and `num_gpus > 0`. This drops 194 censored, 22 with `wall_s ≤ 0`, 72 with `num_gpus = 0`, and the never-started jobs. Report valid and excluded counts per split.
- **Which jobs:** valid jobs with `submit_time < TEST_END` (half-open bounds from `ml/config.py`). The same validity filter applies to history jobs, so jobs killed in the queue do not count toward a user's history or failure rates (see Later).
- **One row per (job, grid point)**, kept where `e < wall_s` (section 2).
- **Current-job features** (all known at `t_obs`):
  - `num_gpus`, `vc` (as a category)
  - submit hour of day and day of week
  - `queue_wait_s = first_start_time − submit_time` (0 if *e* = 0)
  - `elapsed_s`
  - `attempts_so_far` (attempts with `start_time ≤ t_obs`)
- **History:**
  - Per user, sort finished jobs by `end_time`.
  - Use `numpy.searchsorted` to find the jobs with `end_time ≤ t_obs`, and keep the last 32.
  - Store this as `hist_idx`, a list of row indices into the jobs table, in a list column.
- **History token features** (computed at load time, from the history job):
  - `log1p(wall_s)`, `log1p(run_s)`, `num_gpus`, `num_attempts`
  - status (one-hot)
  - `log1p(queue_wait_s)`
  - `log1p(t_obs − its end_time)` (how long ago it ended)
  - same-VC flag
- **Aggregated history features** (for the tree baselines and `last2`):
  - count
  - mean and median `log1p(wall_s)`
  - last-1 and last-2 `wall_s`
  - fail rate and cancel rate
  - Empty history → NaN, which the trees handle natively.
- **`split`** ∈ {train, val, test}, by `submit_time` (section 2).

**Outputs:**
- `data/processed/ml_examples.parquet`
- a short block appended to `ml/reports/T2_examples_report.md`:
  - rows per split and per grid point
  - % of rows with empty history
  - median history length

**Test (1, the important one):** a leakage test on a small synthetic jobs table. For every example:
- every history job has `end_time ≤ t_obs`
- the job itself is never in its own history
- no train/val row has a test-period `submit_time`
- grid rows exist only for `e < wall_s`

**Done-check:**
- The script runs in a few minutes.
- The tests pass.
- The row counts look sensible, roughly 300k–600k rows in total; report the real number.

**Pause point:** after the commit.

---

## T2b. Export `oracle` and `last2` + the prediction checker (≈1.5–2 h)

**Goal:** Track 2 is never blocked.

**`ml/check_predictions.py <file>`** (the shared done-check for every export):
- columns exactly `job_id, elapsed_s, rt_p10_s, rt_p50_s, rt_p90_s, p_fail, p_cancel`
- no NaN
- `60 ≤ p10 ≤ p50 ≤ p90`
- `0 ≤ p_fail`, `0 ≤ p_cancel`, and `p_fail + p_cancel ≤ 1`
- `elapsed_s` ∈ GRID
- the key set `(job_id, elapsed_s)` equals the expected test key set from `ml_examples.parquet`
- It prints OK or a list of problems.

**`ml/export_simple.py`:**
- **`oracle`:**
  - `rt_p10 = rt_p50 = rt_p90 = max(60, wall_s − e)`
  - `p_fail = 1` if Failed, `p_cancel = 1` if Killed, else 0
  - *(Superseded by T2c: the export key set is now all valid jobs overlapping `[SIM_START, TEST_END)`.)* It covers the test jobs **plus the train/val jobs still running at `TEST_START`**.
  - The checker accepts these extra keys for `oracle` only.
- **`last2` (Tsafrir et al., TPDS 2007):**
  - The estimate is the mean `wall_s` of the user's last two finished jobs.
  - With one job, use that job.
  - With no history, use the median `wall_s` of the train split.
  - `rt_* = max(60, estimate − e)`.
  - `p_fail` and `p_cancel` = the user's failure and cancel rates over finished history jobs. With no history, use the train-split rates.

**Outputs:** `predictions/oracle.csv`, `predictions/last2.csv`.

**Test (1):**
- The oracle's remaining value at *e* equals `wall_s − e` (floored).
- `last2` on a 3-job toy user gives the hand-computed value.

**Done-check:** `check_predictions.py` prints OK for both files.

**Pause point:** after the commit. **→ Tell the Track 2 chat that `oracle` and `last2` are ready.**

---

## T2c. Re-export `oracle` and `last2` for all jobs overlapping the simulation window (≈1.5–2 h)

**Why (Track 2 request, 13 Nov):** Track 2 starts its simulation on `SIM_START` = 2017-11-08, seven days before `TEST_START`, so jobs already running or waiting then need prediction rows too (carry-over jobs, val-period jobs). With test-only keys, `last2` was half oracle at early events. This replaces the T2b key rule ("test keys, plus oracle warm-up extras") for all five files.

**New export key set (one rule for every prediction file):**
- *Export jobs* = **valid jobs** (T2 filter: started, not censored, `wall_s > 0`, `num_gpus > 0`) with `end_time > SIM_START` and `submit_time < TEST_END`. Never-started jobs are excluded.
- *Export keys* = every `ml_examples` row (all grid points with `e < wall_s`, any split) of those jobs.
- The causal rule is unchanged: history = the same user's jobs with `end_time ≤ t_obs`.
- Add `SIM_START = "2017-11-08"` to `ml/config.py`. Track 2 reads it and never copies it.

**Tasks:**
- Add one shared function (e.g. `export_keys(jobs, ex)`) used by the checker and by every exporter.
- `check_predictions.py`: the expected key set is now the export keys for **all five files**. Remove the `--allow-extra-keys` special case (no extras exist any more).
- `export_simple.py`: write `oracle.csv` and `last2.csv` over the export keys. The values are defined exactly as in T2b.
- Update the T2b test if it relied on the old key rule.

**Report must contain:**
- Export jobs and rows per split (train, val, test). For `gbdt`, `lightgbm` and `transformer`, **train rows are in-sample**, and val rows were used for early stopping. Only test rows are clean.
- For jobs overlapping `[SIM_START, TEST_END)` that are **not** in the export set, the count by reason: never started, censored, `wall_s ≤ 0`, `num_gpus = 0`. (Censored jobs hold GPUs but have no runtime; Track 2 handles them.)
- `oracle` and `last2` row counts, with the checker output.

**Done-check:** `check_predictions.py` prints OK for `oracle.csv` and `last2.csv` with the new key set. `pytest ml/tests -q` passes.

**Pause point:** after the commit. **→ Tell the Track 2 chat the new row counts and the by-reason counts.**

---

## T3a. Transformer: code + local smoke training on the GTX 1650 (≈4–6 h)

**Goal:** a correct, small model that trains on your laptop.

**`ml/model.py`:**
- **Inputs:** up to 32 history tokens + 1 `[CUR]` token.
  - `[CUR]` carries the current-job features, including `elapsed_s` and `attempts_so_far`.
  - An empty history is just padding plus `[CUR]`.
- **Embedding:**
  - Numeric features go through a small linear layer.
  - Categorical features (status, VC) get embeddings.
  - Add a type embedding (history vs current).
  - No positional encoding is needed beyond the "how long ago" feature, which already orders the tokens.
  - **No user-ID embedding**, because test users may be new. The history is the user signal.
- **Encoder:** `nn.TransformerEncoder`, 2 layers, `d_model` = 64, 4 heads, feed-forward 128, dropout 0.1, with a padding mask.
- **Output:** read the `[CUR]` output vector into two heads.
  - **Runtime head:** 3 outputs. `q50` is the output directly; `q10 = q50 − softplus(a)` and `q90 = q50 + softplus(b)`, so the quantiles can never cross. Pinball loss in `log1p` space.
  - **Status head:** 3 logits, cross-entropy.
  - Total loss = pinball + 1.0 × CE. Keep the weight in `config.py`.

**`ml/train.py`:**
- AdamW, learning rate 1e-3, batch 512.
- Early stopping on validation loss (patience 3), up to 30 epochs.
- A checkpoint every epoch in `runs/<name>/`.
- **Precision:** fp32 by default; the `--amp` flag (Kaggle only) turns on mixed precision.
- Fixed seed.
- Saves `runs/<name>/summary.json`: config, parameter count, epochs, best validation loss, time per epoch, GPU name, git commit.
- `--max-rows N` for quick debugging.

**`ml/predict.py`:** loads a checkpoint and writes predictions for any split, in the contract's CSV format.

**Test (1):** on 256 examples:
- the forward pass has the right shapes
- the quantiles never cross
- the loss falls clearly within 200 steps (it can overfit a tiny batch)

**Done-check:**
- `python ml/train.py --max-rows 20000 --epochs 2` runs on the 1650, and the validation loss goes down.
- Then time **one full epoch** on all train rows and write the time in the report.
  - If 30 epochs × that time is under about 1.5 h, train locally in T3b.
  - Otherwise use Kaggle.

**Pause point:** after the commit.

---

## T3b. Full training run (≈1 h of your time; training is unattended)

**Goal:** one trained transformer checkpoint with a recorded summary.

- **Local:** `python ml/train.py --name tf_v1` (fp32), and leave it running.
- **Kaggle T4** (if needed), following `ml/KAGGLE.md`, which Claude Code writes in this phase:
  - Upload `ml_examples.parquet`, `philly_jobs.parquet` and a zip of `ml/` as one **private** Kaggle dataset.
  - In a notebook with GPU T4, run `python train.py --data /kaggle/input/... --out /kaggle/working --amp --name tf_v1`.
  - Download `runs/tf_v1/` back into the repo.
- Either way, run `predict.py` on the **validation** split and print validation pinball loss and validation status log-loss. Don't look at test metrics yet.

**Done-check:**
- `runs/tf_v1/summary.json` exists.
- The validation numbers are printed and are better than `last2` on validation. If they are not, stop and report; don't tune endlessly.

**Pause point:** while training runs, you can run a Track 2 Claude Code session.

---

## T4. Baselines and evaluation (≈3–4 h)

**Goal:** a fair comparison, with the same features, splits and heads for every model.

**`ml/baselines.py`:** one table of flat features for both models (the current-job features plus the aggregated history features from T2).
- **`gbdt`** ("QSSF-style", after Hu et al. SC'21): scikit-learn `HistGradientBoostingRegressor` with `loss="quantile"`, one model per quantile (0.1 / 0.5 / 0.9), plus `HistGradientBoostingClassifier` for status.
- **`lightgbm`:** LightGBM with `objective="quantile"` (alpha 0.1 / 0.5 / 0.9), plus `objective="multiclass"`.
- **Both:**
  - Early stopping on val.
  - Sort the three quantiles per row so they never cross.
  - Clip the probabilities, then renormalise so `p_fail + p_cancel ≤ 1`.
- Honest note for the paper: both are gradient-boosted trees on the same features. The comparison shows library and implementation differences, not a different idea. See "Later" if they come out identical.

**`ml/evaluate.py`:** for every predictor (`last2`, `gbdt`, `lightgbm`, `transformer`; `oracle` as a sanity row), on the **test split**, overall and per grid point:
- **Runtime:**
  - mean pinball loss (log space)
  - median absolute error of p50 in minutes
  - coverage of [p10, p90] (target ≈ 80%)
- **Status:**
  - log-loss (3 classes)
  - AUROC for "Failed vs not"
  - Brier score for `p_fail`
- **Per-split table (new, T2c):** the same metrics over the export rows split by train (in-sample), val (early stopping) and test (clean), for the learned models. The headline table stays test-only.
- **Outputs:**
  - `ml/results/predictor_metrics.csv`, which the Track 2 paper figures can read
  - `ml/results/predictor_metrics.md`, a table ready to paste into the paper draft

**Test (1):** the metric functions give known values on a 5-row toy example. For instance, pinball loss is 0 for a perfect prediction, and coverage is 100% when the interval contains every target.

**Done-check:**
- The metrics table has all predictors.
- The oracle row is perfect, or near perfect for coverage.
- The numbers are reported as they are, even if the transformer is not the best.

**Pause point:** after the commit.

---

## T5. Export the remaining prediction files (≈1.5–2 h)

**Goal:** finish the contract.

- Write `predictions/gbdt.csv`, `predictions/lightgbm.csv` and `predictions/transformer.csv` for the **export keys** (T2c). Train-period rows are **in-sample** for all three models and val rows were used for early stopping; the report says so and gives the row counts per split.
- Run `check_predictions.py` on **all five files**.
- Write `predictions/README.md` (git-ignored with the folder, so also copy it to `ml/reports/T5_exports.md`). Per file, it lists:
  - the source checkpoint or model
  - the date created
  - the row count
  - the git commit

**Done-check:** five OK lines from the checker. **→ Tell the Track 2 chat that E3 can run.**

**Pause point:** the end of Track 1 for the first draft.

---

## 7. Effort summary

| Phase | Your hours (rough) | Unattended part |
|---|---|---|
| T0 | 0.5–1 | — |
| T1 | 3–4 | download, unpack |
| T2 | 3–4 | building examples (minutes) |
| T2b | 1.5–2 | — |
| T3a | 4–6 | — |
| T3b | ~1 | the training run |
| T4 | 3–4 | tree training (minutes) |
| T5 | 1.5–2 | — |
| **Total** | **≈18–24 h** | |

## 8. Rules for every phase

- Leakage rules (contract section 7) are never relaxed for convenience.
- No test-split metrics before T4. Tuning uses val only.
- Same seed + same data → the same output files.
- Keep it simple:
  - plain functions and dataclasses
  - one model file
  - no config frameworks, experiment trackers or class hierarchies
- Tests: only the ones listed here, about 6 in total.
- Report numbers honestly. If something is worse than expected, say so and stop. Don't hide it with tuning.

## 9. Later (revisit after the first full paper draft)

- 3 seeds for the transformer and the trees, reporting mean ± std. This is cheap, so do it before the final paper if there is time.
- Ablations: the transformer with no history (only `[CUR]`), and with history length 8 vs 32. These show whether the history matters.
- Calibration of the quantile intervals (split conformal on val) and of the status probabilities (temperature scaling).
- Proper treatment of censored jobs (survival-style loss) instead of dropping them.
- Small hyperparameter search (depth 2/4, width 64/128, loss weight).
- Use never-started jobs (killed or failed in the queue, 4.5% of all jobs, 9% of the test window) as a history signal for the user's failure and cancel rates.
- Utilisation-based features (needs Track 2's join and careful leakage rules).
- Predictions at any elapsed time instead of a fixed grid.
- If `gbdt` and `lightgbm` give nearly identical numbers, keep only one in the paper and say why.
- Inference latency per job, as a "cheap enough to deploy" line in the paper.
- Helios trace (after 1 Dec).
- Topic A (users gaming history-based predictors) could reuse this pipeline.

## 10. Contract clarifications to confirm with Track 2 (before T2)

*(Item 9 was added on 13 Nov: the export key set changed in T2c.)*

1. **Observation time:** `t_obs = submit_time` at *e* = 0, and `first_start_time + e` otherwise. The contract text says "submit time + elapsed", which is too early for jobs that waited in the queue. It is still leak-free, but it wastes information.
2. **Test window:** `TEST_START` = 2017-11-15 and `TEST_END` = 2017-12-15 (submission dates, confirmed by the T1 report). The simulator uses the same window.
3. **`oracle.csv` also covers train and val jobs still running at `TEST_START`.** The other files cover test jobs only.
4. **Rows exist only while the job runs** (`e < wall_s`). All files have the same key set, and the files already floor at 60 s.
5. `job_id` is a string (Philly IDs look like `application_..._...`).
6. **Additive columns** in `philly_jobs.parquet`: `censored` and `never_started`. `philly_machines.parquet` is **required** (contract v3), with columns `machine_id, num_gpus, gpu_mem_gb`. *(Merged into contract v3.)*
7. **GPU model is not confirmed** by the trace or the paper. The machine list has 12 GB and 24 GB GPUs, so keep the ± 20% power sensitivity.
8. **New, non-contract output** for the paper: `ml/results/predictor_metrics.csv`.
9. **Export key set (T2c, requested by Track 2):** every prediction file covers all valid jobs with `end_time > SIM_START` (2017-11-08) and `submit_time < TEST_END`, at all grid points `e < wall_s`, whatever their split. Predictions for train-period jobs are in-sample for the learned models; E3 results should be checked on clean jobs (submitted on or after `TEST_START`) as a robustness check. `PROJECT_CONTEXT.md` section 7 must be updated: the line "Predictions exist only for test-period jobs" and the oracle warm-up rule are replaced by this rule.
