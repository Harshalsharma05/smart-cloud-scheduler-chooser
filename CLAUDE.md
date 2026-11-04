# CLAUDE.md

## What this project is
Smart Scheduler Chooser: an academic minor project. A Python discrete-event simulator of a computing cluster that can also rent cloud machines, with a menu of scheduling rule sets, a FastAPI backend and a React (Vite) frontend. Later a "chooser" (rules / ML / LLM) will pick the rule set every 30 simulated minutes.

**Current goal:** the progress-check demo. The plan, phases, data model, rule sets and API contract are in `IMPLEMENTATION.md`. That file is the source of truth. `../v0.3/` is background reading for the long-term research plan; do not implement its full scope.

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

## Rules the simulator must follow
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
frontend/     React + Vite (created in Phase 0)
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
