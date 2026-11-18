# CLAUDE.md

## What this project is
Started as the minor project "Smart Scheduler Chooser" (a cluster + cloud simulator with a React demo). It is now the
research code for the IEEE CCGrid 2027 paper "Cloud Bursting Rarely Pays for Demand Response in On-Premise GPU
Clusters": when the grid asks a GPU cluster to cut power, compare delay, power-cap and burst-to-cloud in $ per MWh shed
on the Microsoft Philly trace, with a controller that picks jobs to burst using predicted remaining runtime and failure risk.

Two tracks, one Claude Code session at a time:
- Track 1 (ML, folder `ml/`): plan `docs/track-1/ML_PLAN.md`, prompts `docs/track-1/STARTER_PROMPTS_ML.md`.
- Track 2 (simulator and paper, folders `backend/flex/`, `backend/traces/`, `paper/`): plan `docs/track-2/SIM_PLAN.md`,
  `docs/track-2/PAPER_PLAN.md`, prompts `docs/track-2/STARTER_PROMPTS_T2.md`.
The only link between the tracks is the interface contract (data/processed/*.parquet, predictions/<name>.csv).

The progress-check demo (IMPLEMENTATION.md, phases 0-7, git tag `progress-check-demo`) is finished and **frozen**.

## End of every step: hand-off (both tracks)
Two Claude Code sessions (Track 1, Track 2) stay open; only one is active at a time. The order of steps across tracks
is `docs/WORK_ORDER.md`. When a step's done-check passes and it is committed: tick the step and log it in
WORK_ORDER.md, then print the phase report followed by the **HAND-OFF block defined in `docs/WORK_ORDER.md`
section 2b** (who reviews, cross-track note, manual work, and the exact NEXT prompt and which session to paste it into).
Then stop. Never start the next step on your own.

## How to work
- Do **one phase from IMPLEMENTATION.md at a time**. At the end: run that phase's check, summarise what was built in a few lines, and stop for review.
- Don't start the next phase or add features that aren't in the current phase.
- If something in the plan is unclear or seems wrong, ask instead of redesigning.

## Keep it simple (academic prototype, not industrial)
- No database, no auth, no Docker, no message queues, no websockets, no async job runners.
- No abstract base-class hierarchies or plugin systems. Plain functions and dataclasses. One `RuleSet` dataclass configured by fields.
- Tests: only the few listed in each phase's check (about 5 in total). No coverage targets, no mocking frameworks.
- Frontend: one page, plain CSS, Recharts for charts. No router, no Redux, no UI framework unless asked.
- Prefer readable code over clever code. Short docstrings; no long comment blocks.

## Rules the demo simulator must follow (frozen code in `sim/`, `rules/`, `workloads/`, `eval/` only)
- Simulated time is integer seconds; never read the wall clock inside the simulator.
- All randomness comes from a `random.Random(seed)` that is passed in. Same config + seed → identical results.
- Schedulers see `est_runtime`, never `true_runtime`.
- The safety checker runs after every event: no node over capacity, no job started twice, cloud cost only for instances between READY and RELEASED.
- Running jobs are never preempted or moved.

## Layout
```
backend/
  sim/        models, engine, cluster, cloud, safety
  rules/      RuleSet + library (B2, B3, B4, B5)
  workloads/  generator + scenarios
  eval/       metrics + runner (CLI)
  api/        FastAPI app
  configs/    default.yaml (world settings)
  tests/
  flex/       paper engine: models, engine, cloud, safety, power, grid, predictions, policies, metrics, paired, experiments
  traces/     synthetic Philly-like workload, Philly loader, utilisation join
  configs/    flex.yaml (paper world)
frontend/     React + Vite (created in Phase 0)
ml/           Track 1 (transformer and baselines)
paper/        LaTeX, figures/ (scripts + out/), tables/, numbers.tex
docs/         track-1/, track-2/ plans
data/, predictions/, results/   shared, git-ignored
```

## Commands
```
cd backend
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
python -m workloads.generator --scenario rush --seed 1
python -m eval.runner --scenario rush --seed 1
pytest -q
uvicorn api.main:app --reload                         # http://localhost:8000/docs

cd frontend
npm install
npm run dev                                           # http://localhost:5173
```

## ML track (folder `ml/`): Track 1 of the CCGrid 2027 paper

**Source of truth for anything in `ml/`: `docs/track-1/ML_PLAN.md`.** It overrides IMPLEMENTATION.md for ML work. Prompts are in `docs/track-1/STARTER_PROMPTS_ML.md`.

### How to work
- One phase from ML_PLAN.md at a time (T0, T1, T2, T2b, T3a, T3b, T4, T5). At the end: run the phase's done-check, commit, print the report in the format from STARTER_PROMPTS_ML.md, and stop.
- If the data or the plan forces a choice the plan doesn't cover, stop and ask. Don't redesign.
- Don't touch `backend/`, `frontend/` or `paper/` from an ML phase.

### Interface contract (do not change silently)
- Track 1 writes `data/processed/philly_jobs.parquet`, `data/processed/philly_attempts.parquet` and `predictions/<name>.csv` (`job_id, elapsed_s, rt_p10_s, rt_p50_s, rt_p90_s, p_fail, p_cancel`; names: oracle, last2, gbdt, lightgbm, transformer).
- `rt_*` = remaining runtime in seconds at `elapsed_s`, floored at 60. Elapsed grid: 0, 60, 600, 3600, 14400, 43200 s. Rows only while the job runs.
- Every prediction file must pass `python ml/check_predictions.py <file>`.
- Only `oracle` may use ground truth.

### Leakage rules (never relax them)
- Observation time: `t_obs = submit_time` at elapsed 0, else `first_start_time + elapsed`.
- History = the same user's jobs with `end_time <= t_obs`, at most 32, never the job itself.
- Time split from `ml/config.py` (train < val < test by submit time). No random split. No test metrics before T4; tune on val only.
- No user-ID embedding (test users can be new).

### Code rules
- Python 3.11 in `ml/.venv` (separate from `backend/.venv`). Run scripts from the repo root: `python ml/<script>.py`.
- Plain functions and dataclasses; plain PyTorch (no Lightning, Hydra, W&B, MLflow). Constants live in `ml/config.py`.
- GTX 1650: fp32 only. `--amp` (mixed precision) only on Kaggle T4.
- Same seed + same data → same outputs. Every training run writes `runs/<name>/summary.json`.
- Tests: only those listed in ML_PLAN.md (about 6). Never commit data, checkpoints or prediction CSVs (git-ignored).

### Commands
    py -3.11 -m venv ml\.venv && ml\.venv\Scripts\activate
    pip install -r ml/requirements.txt          # torch: CUDA wheel matching nvidia-smi
    python ml/philly_clean.py                   # T1
    python ml/examples.py                       # T2
    python ml/export_simple.py                  # T2b
    python ml/check_predictions.py predictions/last2.csv
    python ml/train.py --name tf_v1             # T3 (add --amp only on Kaggle)
    python ml/evaluate.py                       # T4
    pytest ml/tests -q

## Paper work: simulator and paper (Track 2)

**Source of truth for `backend/flex/`, `backend/traces/` and `paper/`: `docs/track-2/SIM_PLAN.md` (design decisions
D1-D16, phases S0-S16) and `docs/track-2/PAPER_PLAN.md` (W0-W8).** They override IMPLEMENTATION.md for this work.

### How to work
- One phase at a time (S0-S16, W0). At the end: run the phase's done-check, report (at most 15 lines), commit, stop.
- If the plan doesn't cover a choice, or a decision looks wrong, stop and ask. Don't redesign.
- Never change the frozen demo code: `sim/`, `rules/`, `workloads/`, `eval/`, `api/`, `frontend/` (bug fixes only, when asked).
  The 4 old tests must keep passing.
- Don't touch `ml/` from a Track 2 phase. `ml/config.py` may be read (TEST_START/TEST_END).

### Rules for the paper engine (`backend/flex/`)
- Integer seconds of simulated time; no wall clock in the simulator. One `random.Random(seed)` passed in; same config + seed
  -> identical results.
- GPU pool (count GPUs only), FCFS with skip-ahead, no runtime estimates in the base scheduler.
- Running jobs **may** be suspended, resumed, power-capped or burst to the cloud (SIM_PLAN D8-D11). Progress is tracked as
  `work_done` in nominal seconds; elapsed time for prediction lookup = `work_done`.
- Ground truth (`wall_s`, `status`) is used by the engine and by the `oracle` predictor only; policies read predictions.
- The safety checker covers: GPUs used <= pool, each job in exactly one state, work conservation at finish, cloud billed
  only READY -> RELEASED, clock monotone. Full checks every event in tests, every `check_every` events on Philly.
- Every parameter in `configs/flex.yaml` has a `# source: ..., accessed ...` or `# ASSUMPTION` comment. Never invent values.
- Synthetic data (`traces/synthetic.py`, `data/synthetic/`) is for testing only. No synthetic number goes into the paper.
- If a contract file is missing or has the wrong columns, stop and report; never work around it silently.

### Keep it simple
Plain functions and dataclasses, no class hierarchies, no databases, no new dependencies beyond requirements.txt.
Tests: only those in each phase's done-check (about 10 in `backend/tests/test_flex.py`).

### Paper rules
IEEEtran conference, 10 pages including references, double-blind (no names, university, repo links or PDF metadata).
Every number in the text comes from `paper/numbers.tex` or a figure script; never type a result by hand.

### Commands (from `backend/`)
    pip install -r requirements.txt
    pytest -q
    python -m flex.experiments e1 --data synthetic --quick
    python -m flex.experiments e1 --data philly --predictor last2
    python -m traces.philly_util                      # S9
    python ..\paper\figures\make_numbers.py           # S16
