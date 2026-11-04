# Implementation plan: up to the progress check

**Project:** Smart Scheduler Chooser (minor project, PDEU).
**Goal for the progress check:** a visible demo where the same jobs are run under different scheduling rule sets on a simulated cluster with rentable cloud, and the results (late jobs, waiting, cloud bill) are shown side by side in a React page. The point to show: *no single rule set wins in every situation*, which is why a chooser is needed later.

This is an **academic prototype**. Correct and simple beats complete and clever. Polish comes later.

---

## Scope

**In (this plan):** simulator core, cloud pool, 4 rule sets, 3 synthetic scenarios, metrics + score, runner, FastAPI, one React page.
**Stretch (only if Phases 1–7 are done):** a simple rule chooser that switches every 30 min; save/replay + oracle for the headroom number.
**Out (after the progress check):** ML chooser, LLM chooser, look-ahead chooser, attack study, real trace (Parallel Workloads Archive), rule sets B6/B8, statistics, advisor screen.

Background docs (`../v0.3/`) describe the full research plan. **Do not build the v0.3 scope now**; this file wins wherever they differ.

---

## Fixed decisions (do not re-debate during implementation)

- **Language:** Python 3.11+. Backend run from `backend/`.
- **Time:** integer seconds of simulated time. No wall-clock time anywhere in the simulator.
- **Randomness:** one `random.Random(seed)` passed in. Same config + seed must give the same result.
- **World:** `backend/configs/default.yaml` (cluster 16 CPU + 4 GPU + 2 high-mem nodes; cloud catalogue of 4 types; boot delay 90 s; cap 22 instances; per-second billing, 60 s minimum, billed from READY to RELEASED).
- **Cloud:** one job per rented instance; pick the cheapest instance type that fits the job; release immediately when the job ends.
- **Runtime estimates:** schedulers see `est_runtime`; the engine uses `true_runtime`. Jobs are not killed if they overrun the estimate.
- **No preemption, no migration.** Running jobs are never moved.
- **Placement:** always best-fit on the cluster (node left with the least spare cores after placing).
- **Runs drain:** after the last arrival, keep simulating until every job has finished (so no unfinished-job penalty is needed).

---

## Data model (`sim/models.py`)

```
Job:          id, arrival, cores, mem_gb, gpus, true_runtime, est_runtime,
              deadline (int | None), owner (str), name (str)
              + runtime fields: state, start, end, node_id | instance_id, on_cloud
Node:         id, cls, cores, mem_gb, gpus, free_cores, free_mem, free_gpus, running (set of job ids)
InstanceType: name, cores, mem_gb, gpus, price_per_h
CloudInstance:id, type, requested_at, ready_at, released_at, job_id, state
```
`owner` and `name` are unused now; they exist so the later attack study does not need a model change.

---

## Rule sets (`rules/`)

One `RuleSet` dataclass with three fields: `order`, `backfill`, `burst`. No subclass per rule set.

| ID | Order | Backfill | Burst (when to rent cloud) |
|---|---|---|---|
| B2 | FCFS (arrival, then id) | EASY | NEVER |
| B3 | SJF (est_runtime, then arrival) | EASY | NEVER |
| B4 | EDF (deadline; no-deadline jobs last, FCFS among them) | EASY | DEADLINE_AWARE |
| B5 | FCFS | EASY | IMMEDIATE |

**Scheduling pass** (run after every event):
1. Sort the queue by `order`.
2. Start jobs from the front while they fit (best-fit).
3. If the front job does not fit and `backfill == EASY`:
   - reservation = earliest time enough resources free up for the front job, computed from running jobs' `start + est_runtime`;
   - start any later job that fits now **and** (finishes by estimate before the reservation **or** does not use resources the front job needs). A simple, conservative version is fine: allow a backfill job only if it finishes before the reservation.
4. Burst policy for each queued job that cannot start now:
   - NEVER: nothing.
   - IMMEDIATE: request a cloud instance now.
   - DEADLINE_AWARE: if the job has a deadline and `projected_start + est_runtime > deadline - margin`, request an instance. `projected_start` = the EASY reservation time for the front job, or "now + sum of est_runtime of jobs ahead / number of fitting nodes" as a rough estimate for others. Rough is fine.
   - Never request more than `max_instances` live instances. A job with a pending instance is not started on the cluster.

---

## Scenarios (`workloads/scenarios.py`)

24 simulated hours each. Start values below; **tune them until the four rule sets clearly differ** (that tuning is part of the work).

| Scenario | Arrivals | Mix | Deadlines |
|---|---|---|---|
| quiet | ~15 jobs/h, steady | 15% GPU jobs | 20% of jobs, slack 3–6× runtime |
| rush | ~15 jobs/h, with a 6-hour burst of ~80 jobs/h | 20% GPU | 30%, slack 2–4× |
| deadline_heavy | ~40 jobs/h | 20% GPU | 70%, slack 1.3–2.5× |

- Cores per CPU job: choose from {1, 2, 4, 8, 16, 32, 64}; GPU jobs need 1–4 GPUs and 8 cores per GPU.
- True runtime: log-normal, median ~30 min, capped at 12 h.
- Estimated runtime = true × factor drawn from 1.0–3.0 (users overestimate).
- Deadline = arrival + slack × true runtime (use true runtime here; it is the "nominal" runtime).

---

## Metrics and score (`eval/metrics.py`)

Per run:
- `jobs` total, `late_pct` (deadline jobs finishing after deadline / deadline jobs)
- `avg_wait_min` (start − arrival)
- `avg_bounded_slowdown` = mean of max(1, (wait + runtime) / max(runtime, 10 s))
- `cloud_cost` (units), `cloud_instance_hours`
- `score J` = w_miss·(miss_rate / miss_ref) + w_slowdown·(slowdown / slowdown_ref) + w_cost·(cost / cost_ref)
  - References, all from the same scenario+seed: `miss_ref` = max(B2's miss rate, 0.05), `slowdown_ref` = B2's slowdown, `cost_ref` = B5's cost (use 1 if zero). Lower J is better.
  - Weight profiles (`configs/default.yaml`, `score.profiles`):

    | Profile | w_miss | w_slowdown | w_cost |
    |---|---|---|---|
    | balanced (default) | 0.4 | 0.3 | 0.3 |
    | deadline_first | 0.6 | 0.3 | 0.1 |
    | cost_first | 0.2 | 0.3 | 0.5 |
  - `python -m eval.runner --scenario rush --seed 1 --profile deadline_first`

---

## API contract (`api/main.py`)

- `GET /health` → `{"ok": true}`
- `GET /scenarios` → `[{"id": "rush", "label": "Rush hour", "description": "..."}]`
- `GET /rulesets` → `[{"id": "B2", "order": "FCFS", "backfill": "EASY", "burst": "NEVER", "label": "..."}]`
- `GET /profiles` → `[{"id": "balanced", "w_miss": 0.4, "w_slowdown": 0.3, "w_cost": 0.3}, ...]`
- `POST /run` body `{"scenario": "rush", "seed": 1, "rulesets": ["B2","B3","B4","B5"], "timeline_for": "B4", "profile": "balanced"}` →
  (every field optional; defaults as shown. `profile` picks the score weights. Unknown ids → 400.)
```json
{
  "scenario": "rush", "seed": 1, "profile": "balanced", "n_jobs": 612,
  "results": [
    {"ruleset": "B2", "late_pct": 31.2, "avg_wait_min": 48.5, "avg_bounded_slowdown": 6.1,
     "cloud_cost": 0.0, "cloud_instance_hours": 0.0, "score": 0.52}
  ],
  "timeline": {
    "ruleset": "B4",
    "jobs":  [{"id": 1, "start": 120, "end": 1900, "where": "cpu-3", "on_cloud": false, "late": false}],
    "cloud": [{"id": "c1", "type": "gpu_1", "requested": 3600, "ready": 3690, "released": 7200}]
  }
}
```
Runs are synchronous (a run takes seconds). CORS allows `http://localhost:5173`.

---

## Phases

Work **one phase at a time**. At the end of each phase: run the check, commit, stop.

### Phase 0: Setup (≈30 min)
- `cd backend && python -m venv .venv`, install `requirements.txt`.
- `git init` at the repo root, first commit.
- Frontend: from the repo root run `npm create vite@latest frontend -- --template react`, then `npm install recharts`.
- **Check:** `python -c "import fastapi, yaml"` works; `npm run dev` shows the Vite page.

### Phase 1: Models + workload generator
- `sim/models.py`, `workloads/generator.py`, `workloads/scenarios.py`, a loader for `configs/default.yaml`.
- **Check:** `python -m workloads.generator --scenario rush --seed 1` prints job count and the first 10 jobs; running it twice gives identical output.

### Phase 2: Engine + cluster + safety checker
- Event loop on `heapq` with events JOB_ARRIVAL, JOB_FINISH, CLOUD_READY (CLOUD_READY can be a no-op until Phase 3).
- Cluster allocation (best-fit), safety checker called after every event.
- Temporary rule: FCFS, no backfill, no cloud.
- Record every job's start/end/where for the timeline.
- **Check (test `tests/test_basic.py`):** the slide example. One free node; A (10 h, no deadline), B (1 h, deadline 2 h), C (1 h, deadline 2 h) all arrive at t=0. FCFS → B and C late. (EDF test comes in Phase 4.)

### Phase 3: Cloud pool
- Request → booting (boot delay) → ready → job runs → released; cap; billing from READY to RELEASED, 60 s minimum; cheapest fitting type.
- Burst policies NEVER and IMMEDIATE.
- **Check (test):** one cluster node busy for 10 h, one 1 h CPU job arrives, IMMEDIATE → job starts at t=90 s on cloud and cost = price × 1 h. NEVER → it waits 10 h.

### Phase 4: Rule sets B2–B5
- Orders FCFS/SJF/EDF, EASY backfill, DEADLINE_AWARE burst, `rules/library.py` menu.
- **Check (tests):** (1) slide example with EDF → no late jobs; (2) one backfill case: a small short job starts in the gap while a big job waits for its reservation.

### Phase 5: Metrics + runner (backend demo-ready)
- `eval/metrics.py`, `eval/runner.py` running all rule sets on the same scenario+seed.
- **Check:** `python -m eval.runner --scenario rush --seed 1` prints a table with one row per rule set. Tune scenario parameters until the rows differ in a sensible way (e.g. B5 lowest wait but highest cost, B2 zero cost but most late jobs in `deadline_heavy`). Same seed → same table.

### Phase 6: FastAPI
- Endpoints from the API contract; reuse the runner.
- **Check:** `uvicorn api.main:app --reload` from `backend/`, open `http://localhost:8000/docs`, run `POST /run` successfully.

### Phase 7: React page (the visible demo)
One page, plain CSS, no router, no state library.
- Controls: scenario dropdown, seed input, Run button, "timeline for" dropdown.
- Results table (one row per rule set, best value per column highlighted).
- Bar charts (Recharts): late %, average wait, cloud cost per rule set.
- Timeline: a simple horizontal chart of jobs over time (cluster vs cloud coloured differently), and cloud instances as bars. A plain SVG or div-based Gantt is enough; show at most a few hundred jobs.
- **Check:** pick "quiet", then "deadline_heavy": the winner changes. That is the demo.

### Phase 8 (stretch): Simple switching chooser
- Window event every 30 simulated minutes; chooser picks a rule set from queue length, deadline pressure and queue growth (a hand-written if/else).
- Switch semantics: running jobs untouched; queue re-sorted; reservations recomputed.
- Shown as an extra row "Rule chooser" in the table, and its switches marked on the timeline.

### Phase 9 (stretch): Save/replay + oracle headroom
- Snapshot state with `copy.deepcopy` at each window; replay the next hour with each rule set; oracle keeps the best.
- Report "oracle beats the best fixed rule set by X%" per scenario.

---

## Progress-check demo script (≈3 min)
All runs use seed 1.
1. Profile `balanced`, scenario `quiet` → best is **B3**: nothing is under pressure, so the cheap cluster-only rule set wins.
2. Profile `balanced`, scenario `rush` → best is **B4**: EDF + deadline-aware renting saves every deadline at a small cloud cost. → "no rule wins everywhere".
3. Stay on `rush`, change only the profile: `cost_first` → **B3**, `deadline_first` → **B5**. → what "best" means depends on what the operator values.
4. Show the timeline (B4 on rush): jobs on the cluster vs on cloud, late jobs outlined, cloud instances booting and billed; move the window slider into the rush hours.
5. Show `/docs` and the passing tests (slide example FCFS/EDF, cloud billing, backfill).
6. Next: choosers (rules, look-ahead, ML, LLM) and the attack study.

## After the progress check (not now)
Phases 8–9 if not done, look-ahead chooser, ML chooser (LightGBM on replay labels), LLM chooser via Ollama (menu-only), attack study (unsafe / warned / menu-only LLM), rule sets B6/B8, one real trace, simple statistics, dashboard polish.
