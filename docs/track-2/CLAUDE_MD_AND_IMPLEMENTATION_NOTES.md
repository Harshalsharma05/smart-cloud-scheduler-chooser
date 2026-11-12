# Text for CLAUDE.md and IMPLEMENTATION.md (Track 2)

Applied in phase S0, together with Track 1's ML section (`docs/track-1/CLAUDE_MD_ML_SECTION.md`), so CLAUDE.md is edited only once (WORK_ORDER step 3). There are four edits; they don't overlap with Track 1's text.

---

## Edit 1 — CLAUDE.md: replace the sections "What this project is" and "Current goal" with

```markdown
## What this project is
Started as the minor project "Smart Scheduler Chooser" (a cluster + cloud simulator with a React demo). It is now the
research code for the IEEE CCGrid 2027 paper "Burst to Flex: Cloud Offloading as Grid Flexibility for On-Premise GPU
Clusters": when the grid asks a GPU cluster to cut power, compare delay, power-cap and burst-to-cloud in $ per MWh shed
on the Microsoft Philly trace, with a controller that picks jobs to burst using predicted remaining runtime and failure risk.

Two tracks, one Claude Code session at a time:
- Track 1 (ML, folder `ml/`): plan `docs/track-1/ML_PLAN.md`, prompts `docs/track-1/STARTER_PROMPTS_ML.md`.
- Track 2 (simulator and paper, folders `backend/flex/`, `backend/traces/`, `paper/`): plan `docs/track-2/SIM_PLAN.md`,
  `docs/track-2/PAPER_PLAN.md`, prompts `docs/track-2/STARTER_PROMPTS_T2.md`.
The only link between the tracks is the interface contract (data/processed/*.parquet, predictions/<name>.csv).

The progress-check demo (IMPLEMENTATION.md, phases 0-7, git tag `progress-check-demo`) is finished and **frozen**.
```

## Edit 2 — CLAUDE.md: rename the heading "Rules the simulator must follow" to

```markdown
## Rules the demo simulator must follow (frozen code in `sim/`, `rules/`, `workloads/`, `eval/` only)
```
The bullets stay as they are. The paper engine has its own rules (Edit 3).

## Edit 3 — CLAUDE.md: add this section after Track 1's ML section

```markdown
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
```

## Edit 4 — CLAUDE.md "Layout": add these lines to the tree

```
backend/
  flex/       paper engine: models, engine, cloud, safety, power, grid, predictions, policies, metrics, paired, experiments
  traces/     synthetic Philly-like workload, Philly loader, utilisation join
  configs/    flex.yaml (paper world)
ml/           Track 1 (transformer and baselines)
paper/        LaTeX, figures/ (scripts + out/), tables/, numbers.tex
docs/         track-1/, track-2/ plans
data/, predictions/, results/   shared, git-ignored
```

---

## IMPLEMENTATION.md — add at the very top

```markdown
> **Status (Nov 2026): finished and frozen.** This file describes the progress-check demo (phases 0-7 done,
> git tag `progress-check-demo`). It is **superseded for the paper work**: the CCGrid 2027 simulator and paper follow
> `docs/track-2/SIM_PLAN.md` and `docs/track-2/PAPER_PLAN.md`; the ML work follows `docs/track-1/ML_PLAN.md`.
> Phases 8-9 below ("stretch") will not be built. Do not change the demo code (`sim/`, `rules/`, `workloads/`,
> `eval/`, `api/`, `frontend/`) except for bug fixes.
```

## README.md (optional, one line)
Replace "Status: skeleton only. See IMPLEMENTATION.md, Phase 0." with:
"Status: the demo (IMPLEMENTATION.md) is done and frozen; the paper work is planned in `docs/track-1/` and `docs/track-2/`."
