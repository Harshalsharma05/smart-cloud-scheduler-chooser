# Evaluation Prep: Smart Scheduler Chooser

A plain-language guide to what has been built for the progress check, how it works, why it was built this way, and what comes next.

---

## 1. The project in one minute

**Problem.** A computing cluster (a group of machines) receives a stream of jobs. Some jobs have deadlines. When the cluster is full, jobs wait. You can also **rent cloud machines** to run jobs sooner, but that costs money. A **scheduler** decides which job runs next, where it runs, and whether to pay for the cloud.

**Observation.** There are many scheduling *rule sets* (for example "first come first served" or "shortest job first"). **None of them is best in every situation.**
- On a quiet day, the cheapest rule is fine.
- In a rush hour with deadlines, a deadline-aware rule that rents cloud machines is better.
- What counts as "better" also depends on what the operator cares about: deadlines, speed or money.

**Idea.** Build a **chooser** that watches the situation and switches between rule sets (later: every 30 simulated minutes, using rules, ML or an LLM).

**What exists now (progress check).** Everything the chooser needs underneath it:
1. A simulator of a cluster plus cloud.
2. Four rule sets.
3. Three workload scenarios.
4. Metrics and a single score.
5. A REST API.
6. A web page showing that **the winning rule set changes with the situation**, which is the motivation for the chooser.

---

## 2. Overall architecture

```
 ┌──────────────────────────── backend (Python) ────────────────────────────┐
 │                                                                          │
 │  workloads/            sim/                     rules/        eval/      │
 │  generator.py  ──jobs──►  engine.py  ◄──RuleSet── library.py   metrics.py │
 │  scenarios.py            cluster.py              base.py      runner.py  │
 │                          cloud.py                               │        │
 │  configs/default.yaml    safety.py  ── results ──────────────► │        │
 │  (world settings)        models.py                              │        │
 │                                                                 ▼        │
 │                                             api/main.py (FastAPI)        │
 └──────────────────────────────────────────────────┬───────────────────────┘
                                                    │ JSON over HTTP
                                         ┌──────────▼──────────┐
                                         │ frontend (React +   │
                                         │ Vite + Recharts)    │
                                         │ one page            │
                                         └─────────────────────┘
```

**The flow of one "Run" click:**
1. The page sends `POST /run` with a scenario, seed, profile and "timeline for" rule set.
2. The backend **generates the same job list** for that scenario and seed.
3. It **simulates 24 hours** of that job list once per rule set (B2, B3, B4, B5).
4. It computes the **metrics and score J** for each run.
5. It returns a table, plus a detailed **timeline** for one rule set.
6. The page shows the winner banner, the table, three bar charts and the interactive timeline.

---

## 3. The simulated world

All world settings are in `backend/configs/default.yaml`. They are assumptions and easy to change.

### The cluster (machines we own: free to use, but limited)

| Node type | Count | Cores | Memory | GPUs |
|---|---|---|---|---|
| cpu | 16 | 32 | 128 GB | 0 |
| gpu | 4 | 32 | 256 GB | 4 |
| highmem | 2 | 64 | 1024 GB | 0 |

In total: 22 nodes, 704 cores and 16 GPUs.

### The cloud (machines we rent: unlimited types, but they cost money and take time to start)

| Type | Cores | Memory | GPUs | Price / hour |
|---|---|---|---|---|
| cpu_small | 8 | 32 GB | 0 | 8 |
| cpu_large | 32 | 128 GB | 0 | 32 |
| gpu_1 | 8 | 64 GB | 1 | 58 |
| highmem | 16 | 512 GB | 0 | 24 |
| cpu_xlarge | 64 | 256 GB | 0 | 64 |
| gpu_4 | 32 | 128 GB | 4 | 232 |

- **Boot delay:** 90 seconds from request to ready.
- **Cap:** at most 22 live instances at once (roughly the size of the cluster).
- **Billing:** per second from READY to RELEASED, with a 60-second minimum. You don't pay while it boots.
- **Matching:** one job per instance; the cheapest type that fits the job is chosen; the instance is released as soon as its job finishes.
- **Price unit:** 1 unit is roughly one CPU core-hour. A GPU costs about 50 times a core.

---

## 4. What was built, phase by phase

### Phase 0: Setup
- A Python virtual environment, with FastAPI, uvicorn, pydantic, PyYAML and pytest.
- A React frontend made with Vite, plus Recharts for charts.
- A git repository, with one commit per phase.

### Phase 1: Data model and workload generator
**Data model** (`sim/models.py`): plain Python dataclasses.
- `Job`:
  - Basic fields: id, arrival time, cores, memory, GPUs.
  - Runtimes: the true runtime and the estimated runtime.
  - Deadline (optional), plus owner and name.
  - Runtime state, filled in during the run: start, end, where it ran, and whether it was on the cloud.
- `Node`, `InstanceType` and `CloudInstance`, plus a loader for the YAML config.
- `owner` and `name` are unused for now. They exist so the later "attack study" won't need a model change.

**Workload generator** (`workloads/generator.py`). It makes a 24-hour list of jobs from a scenario name and a seed.

*Arrivals:*
- Arrivals are random (a Poisson process). The rate can change by hour; for example, rush hour runs from 8 to 14 h.

*Job size:*
- **CPU jobs** need 1, 2, 4, 8, 16, 32 or 64 cores, with small jobs more common.
- **GPU jobs** need 1–4 GPUs, with 8 cores and 32 GB per GPU.

*Runtimes:*
- **True runtime** follows a log-normal distribution with a median of 30 minutes, capped at 12 hours. That means many short jobs and a few very long ones, as in real clusters.
- **Estimated runtime** = true runtime × a random factor from 1.0 to 3.0, because users overestimate. **Schedulers only see the estimate**; only the engine knows the true runtime.

*Deadlines:*
- **Deadline** = arrival + slack × true runtime, given to a fraction of the jobs.

**Scenarios** (`workloads/scenarios.py`):

| Scenario | Arrival rate | GPU jobs | Jobs with a deadline | Deadline slack |
|---|---|---|---|---|
| quiet | ~15 / h all day | 15% | 20% | 3–6× runtime |
| rush | ~15 / h, but ~80 / h from 8 h to 14 h | 20% | 30% | 2–4× |
| deadline_heavy | ~40 / h all day | 20% | 50% | 2–4× |

With seed 1, that gives 343, 768 and 984 jobs.

**Check:** running the generator twice with the same seed gives byte-identical output.

### Phase 2: Simulation engine, cluster and safety checker
**Discrete-event simulation** (`sim/engine.py`). Instead of ticking every second, the simulator jumps from event to event, kept in a priority queue (`heapq`).
- The event types are `JOB_ARRIVAL`, `JOB_FINISH` and `CLOUD_READY`.
- At the same timestamp, finishes are processed first, because they free resources.
- The run **drains**: after the last arrival it keeps going until every job has finished, so no job is left unfinished.

**Cluster** (`sim/cluster.py`):
- **Best-fit placement:** a job goes on the node that will have the *fewest spare cores left* after placing it. This keeps big holes free for big jobs.
- Jobs never span more than one node.

**Safety checker** (`sim/safety.py`), run after **every** event. If any rule is broken, the run stops with an error:
- the clock never goes backwards;
- no node is ever over capacity in cores, memory or GPUs, and the free-resource bookkeeping always adds up;
- no job runs on two nodes, and no job starts twice.

**Test:** the "slide example". One node; job A (10 h), and jobs B and C (1 h each, deadline 2 h), all arriving at t = 0. With first-come-first-served, A goes first, so **B and C are late**.

### Phase 3: Cloud pool
`sim/cloud.py` covers the life of a cloud machine:
- **request → booting (90 s) → ready → the job runs → released.**
- It enforces the 22-instance cap, picks the cheapest type that fits, and computes the bill (READY to RELEASED, minimum 60 s).

**Important rule:** once a job has a cloud machine requested, it **leaves the cluster queue for good**. It can never also start on the cluster, so there is no double execution.

**Safety checker additions:**
- never more than 22 live instances, and no job gets two instances;
- each running cloud job is on exactly one READY instance;
- nothing is billed while booting;
- an instance is released only after its job is done, and is billed at least 60 s.

**Test:** one node is busy for 10 h and a 1-hour job arrives.
- With **IMMEDIATE** renting, it starts on the cloud at t = 90 s and costs exactly price × 1 h = 8 units.
- With **NEVER**, it waits 10 h.

### Phase 4: Rule sets B2 to B5
There is one `RuleSet` dataclass with three settings and **no class hierarchy**:

| ID | Order of the queue | Backfill | When to rent cloud |
|---|---|---|---|
| **B2** | FCFS: first come, first served | EASY | Never |
| **B3** | SJF: shortest *estimated* job first | EASY | Never |
| **B4** | EDF: earliest deadline first; jobs with no deadline go last | EASY | Deadline-aware |
| **B5** | FCFS | EASY | Immediately, for every job that can't start |

**One generic scheduling pass** runs after the events at each timestamp:
1. **Sort** the queue by the rule set's order.
2. **Start jobs from the front** while they fit (best-fit).
3. **EASY backfill**, when the front job doesn't fit:
   - Compute the front job's **reservation**: for each node that could ever hold it, imagine its running jobs ending in order of estimated end until the front job fits. The earliest such (node, time) is the reservation.
   - A later job may **jump ahead** ("backfill") if it fits now **and** either finishes (by estimate) before the reservation time, or runs on a different node than the reserved one.
   - So small jobs fill gaps, but **never delay the front job**.
4. **Burst** (renting cloud), for jobs still waiting:
   - **NEVER:** nothing.
   - **IMMEDIATE:** rent a machine for every waiting job (up to the cap).
   - **DEADLINE_AWARE:** only jobs with a deadline. Estimate when the job would start on the cluster:
     - for the front job, the reservation time;
     - for others, a rough "now + (estimated work ahead ÷ number of nodes that could hold it)".

     If the estimated start + estimated runtime > deadline − 10 min, rent a machine.

**Tests:**
- **EDF slide example:** EDF runs B and C first, so **no job is late**.
- **Backfill:**
  - a small 1-hour job starts immediately in the gap;
  - the big job starts exactly at its 10-hour reservation;
  - a long 20-hour job that would have delayed the big job is held back.

### Phase 5: Metrics, score J and the runner
**Metrics per run** (`eval/metrics.py`):

| Metric | Meaning |
|---|---|
| `late_pct` | % of deadline jobs that finished after their deadline |
| `avg_wait_min` | average time between arrival and start |
| `avg_bounded_slowdown` | average of max(1, (wait + runtime) / max(runtime, 10 s)). "How many times longer than its runtime did the job take?" The 10 s floor stops tiny jobs dominating. |
| `cloud_cost` | total cloud bill in units |
| `cloud_instance_hours` | total billed instance time |

**Score J** combines the three goals into one number, where **lower is better**:

```
J = w_miss · (miss_rate / miss_ref) + w_slowdown · (slowdown / slowdown_ref) + w_cost · (cost / cost_ref)
```
- Each term is divided by a **reference from the same scenario and seed**, so the three terms are on a similar scale:
  - `miss_ref` = B2's miss rate, but at least 0.05, so a near-zero value can't blow up the term;
  - `slowdown_ref` = B2's slowdown;
  - `cost_ref` = B5's cost, or 1 if it is zero.
- A consequence: under the balanced profile, **B2 always scores about 0.70** whenever its miss rate is at least 0.05. That's because B2 *is* the reference for miss and slowdown: 0.4·1 + 0.3·1 + 0.3·0. That is expected, not a bug.

**Weight profiles** (`configs/default.yaml`), which express what the operator values:

| Profile | w_miss | w_slowdown | w_cost |
|---|---|---|---|
| balanced (default) | 0.4 | 0.3 | 0.3 |
| deadline_first | 0.6 | 0.3 | 0.1 |
| cost_first | 0.2 | 0.3 | 0.5 |

**Runner** (`eval/runner.py`):
- `run_comparison(scenario, seed, rulesets, timeline_for, profile)` runs every rule set on the **same** job list and returns exactly the JSON the API sends.
- The CLI prints it as a table: `python -m eval.runner --scenario rush --seed 1 --profile balanced`.

**Tuning:** the deadline_heavy scenario was tuned so the rule sets clearly differ. The fraction of jobs with a deadline went from 70% to 50%, and slack from 1.3–2.5× to 2–4×. Only scenario parameters were changed, never the simulator logic.

### Phase 6: REST API (FastAPI)
| Endpoint | Returns |
|---|---|
| `GET /health` | `{"ok": true}` |
| `GET /scenarios` | the 3 scenarios with labels and descriptions |
| `GET /rulesets` | B2–B5 with their order, backfill and burst settings |
| `GET /profiles` | the 3 weight profiles |
| `POST /run` | results table + timeline (see `IMPLEMENTATION.md`, "API contract") |

- Every request field has a default.
- Unknown ids return a clear 400 error.
- CORS allows the Vite page at `localhost:5173`.
- Interactive docs are at `http://localhost:8000/docs`.

### Phase 7: The web page (React)
One page with plain CSS and Recharts; no router or state library.
- **Controls:** scenario, profile, seed, "timeline for", and a Run button. There is a loading state, and a clear error if the backend is down.
- **Winner banner:** "Best under balanced: B4". Ties show every tied rule set.
- **Results table:** the best value in each column is highlighted.
- **Three bar charts:** late %, average wait and cloud cost.
- **Interactive timeline** (an SVG Gantt chart):
  - **Overview strip** of the whole run; drag its box to pan.
  - **Drag to zoom** on the chart, with **Back** and **Reset zoom**; the axis adapts from hours down to 5-minute ticks.
  - **Hover tooltips**:
    - jobs: size, wait, run times, estimate, and whether the deadline was met or missed and by how much;
    - cloud instances: boot time, billed time and cost.
  - Hovering a cloud job **also highlights its instance**, and the other way round.
  - **Legend chips** with live counts: click to hide or show a category, or to highlight only late jobs.
  - Red deadline markers on late jobs, a time crosshair, and light/dark colours.

---

## 5. Results (seed 1)

### Raw metrics: the same in every profile

| Scenario | Rule set | Late % | Avg wait (min) | Slowdown | Cloud cost | Instance h |
|---|---|---|---|---|---|---|
| quiet | B2 | 1.6 | 0.2 | 1.02 | 0 | 0 |
| quiet | B3 | 0.0 | 0.2 | 1.02 | 0 | 0 |
| quiet | B4 | 0.0 | 0.2 | 1.01 | 22.9 | 0.1 |
| quiet | B5 | 0.0 | 0.0 | 1.00 | 331.0 | 1.43 |
| rush | B2 | 7.9 | 27.5 | 2.40 | 0 | 0 |
| rush | B3 | 4.2 | 16.6 | 1.37 | 0 | 0 |
| rush | B4 | 0.0 | 22.0 | 2.13 | 785.7 | 4.28 |
| rush | B5 | 0.0 | 0.1 | 1.01 | 12462.4 | 57.97 |
| deadline_heavy | B2 | 13.1 | 36.7 | 3.25 | 0 | 0 |
| deadline_heavy | B3 | 4.3 | 15.0 | 1.28 | 0 | 0 |
| deadline_heavy | B4 | 0.0 | 15.0 | 2.24 | 6740.2 | 32.14 |
| deadline_heavy | B5 | 0.0 | 0.1 | 1.01 | 16999.4 | 81.60 |

**How to read it:**
- **B2 (FCFS)** is free but the most late, because short urgent jobs wait behind long ones.
- **B3 (SJF)** has the best slowdown among the free rule sets, because short jobs go first. But it ignores deadlines, so some are still missed.
- **B4 (EDF + deadline-aware cloud)** saves *every* deadline. It rents only for jobs at risk, so its cost is **6% of B5's in rush** and 40% in deadline_heavy.
- **B5 (rent immediately)** is the fastest by far (wait ≈ 0) and also by far the most expensive.

### Score J winner (lower J is better)

| Scenario | balanced | deadline_first | cost_first |
|---|---|---|---|
| quiet | **B3** | B3 | B3 (ties on seeds 2–3) |
| rush | **B4** | **B5** | **B3** |
| deadline_heavy | B3 | B5 | B3 |

The same winners hold on seeds 1, 2 and 3.

**The main message: the best rule set depends on the situation *and* on what you value.** No single rule set wins everywhere, which is exactly why a chooser is needed.

---

## 6. Key design decisions (and why)

| Decision | Why |
|---|---|
| **Discrete-event simulation**, time in integer seconds, no wall clock | Fast: a 24 h day runs in under a second. Exact, and easy to test. |
| **One seeded `random.Random(seed)`**, passed in | Same config + seed gives identical results, so experiments are reproducible and comparisons are fair. Every rule set sees the *same* jobs. |
| **Schedulers see only the estimated runtime** | Realistic: real schedulers don't know the true runtime. Estimates are 1–3× too high, as user estimates usually are. |
| **No preemption, no migration** | Running jobs are never stopped or moved. Simpler and safer, and common in real batch clusters. |
| **Best-fit placement** for every rule set | Keeps the comparison about *ordering, backfill and renting*, not placement. |
| **EASY backfill** in all rule sets | The standard technique in real HPC schedulers. It improves utilisation without starving the front job. |
| **Reservation computed per node** | Jobs never span nodes, so "when will enough resources be free?" only makes sense per node. |
| **One job per cloud instance, cheapest fitting type, release immediately** | The simplest correct cloud model; easy to bill and to check. |
| **Bigger cloud types added** (cpu_xlarge, gpu_4) | Without them, 64-core and multi-GPU jobs could never be rented, so the burst rules would silently skip them. Now every job fits a cloud type. |
| **Safety checker after every event** | Catches simulator bugs right away (over-capacity, double starts, wrong billing) instead of giving silently wrong results. |
| **Runs drain** (simulate until every job finishes) | No unfinished-job penalty is needed, so the metrics are simple and comparable. |
| **Scheduling pass once per timestamp** | Several jobs can arrive at the same second. Scheduling after *each* event would let the first job grab the node before the others are seen. Found through the EDF slide test. |
| **Normalised miss term + weight profiles** | With a raw miss rate, the late-jobs term was tiny (0.4 × 0.04), so B3 won almost everything and lateness barely counted. Normalising puts all three terms on the same scale, and profiles make "what is best" an explicit operator choice. |
| **One `RuleSet` dataclass, plain functions** | Academic prototype: readable over clever. A new rule set is one line in `rules/library.py`. |
| **No database, auth, Docker, queues or websockets** | Not needed: a run takes under a second and is computed on request. |
| **Plain React + CSS + Recharts, SVG timeline** | One page; no framework to learn or explain. The timeline is hand-written SVG, so it is fully controllable (zoom, tooltips). |

---

## 7. Problems we hit and how we solved them

1. **Big jobs couldn't go to the cloud.** The largest cloud type had 32 cores and 1 GPU, but jobs can need 64 cores or 4 GPUs. Fix: added `cpu_xlarge` (64 cores) and `gpu_4` (4 GPUs).
2. **The EDF slide test failed.** All three jobs arrive at t = 0, but the pass ran after the *first* arrival, so job A started before B and C were even in the queue. Fix: run the scheduling pass once after all events at the same timestamp. The safety check still runs after every event.
3. **deadline_heavy didn't separate the rule sets.** B4 cost 72% of B5. Fix: tuned only the scenario (fewer and looser deadlines) until B4 cost about 40% of B5 while staying at 0% late.
4. **The score always picked B3.** Fix: the normalised miss term and the three profiles (see the design decisions above).
5. **"WinError 10013" when starting uvicorn.** A leftover server from testing was still holding port 8000. Fix: stop that process, or use `--port 8001` with `VITE_API_URL`.

---

## 8. How to run it

```
# Terminal 1: backend
cd backend
.venv\Scripts\activate
uvicorn api.main:app --reload            # API docs at http://localhost:8000/docs

# Terminal 2: frontend
cd frontend
npm run dev                              # page at http://localhost:5173

# Other useful commands (from backend/, venv active)
pytest -q                                              # 4 tests
python -m workloads.generator --scenario rush --seed 1 # see the generated jobs
python -m eval.runner --scenario rush --seed 1 --profile balanced
```

**The 4 tests** (`backend/tests/test_basic.py`):
1. Slide example with FCFS: B and C are late.
2. Slide example with EDF: no job is late.
3. Cloud: IMMEDIATE starts at 90 s and costs price × 1 h; NEVER waits 10 h.
4. EASY backfill: a short job fills the gap, the big job keeps its reservation, and a long job is held back.

---

## 9. Demo script (about 3 minutes, seed 1)

1. **balanced + quiet → B3.** Nothing is under pressure, so the cheap cluster-only rule set wins.
2. **balanced + rush → B4.** EDF plus deadline-aware renting saves every deadline at a small cloud cost. *"No rule wins everywhere."*
3. **Stay on rush, change only the profile:** cost_first → **B3**, deadline_first → **B5**. *"What 'best' means depends on what the operator values."*
4. **Timeline (B4 on rush):**
   - drag the overview to 8–14 h and zoom into a busy stretch;
   - hover an orange cloud job: its instance lights up, showing the boot time and bill;
   - click "Late" to highlight late jobs.
5. **Show `/docs` and run `pytest -q`.**
6. **Next steps:** choosers and the attack study (section 11).

---

## 10. Likely questions and short answers

**Q: Why simulate instead of using a real cluster?**
Repeatable, fast and free. We can run the *exact same* 24 hours under every rule set, which is impossible on a real system. Real schedulers are evaluated this way too (for example with traces from the Parallel Workloads Archive).

**Q: How do you know the simulator is correct?**
- The safety checker asserts the core rules after every event: capacity, no double start, billing only while live, and the clock moving forward.
- The 4 tests check hand-worked examples with known answers.
- Same seed gives identical output.

**Q: Why does B3 often score better than B4?**
B3 has a very low slowdown (short jobs first) and zero cost, while B4 pays for cloud. Under the balanced profile, B3's few late jobs cost it less than B4's cloud bill. When deadlines matter more (deadline_first), the cloud-using rules win. That is exactly the trade-off the chooser must manage.

**Q: Why does B5 win under deadline_first even though it's the most expensive?**
deadline_first gives cost only 0.1 weight. B5 has zero late jobs and near-perfect slowdown, so its high cost barely matters.

**Q: What is EASY backfill, in one sentence?**
Let a later job jump the queue if it fits now and won't delay the reserved start of the job at the front.

**Q: Why are estimates 1–3× the true runtime?**
Users overestimate so their jobs aren't killed. Using estimates also makes the scheduler realistic, since it can't see the future.

**Q: Why normalise by B2 and B5?**
To put miss rate, slowdown and cost on the same scale. B2 is the natural "do nothing special" baseline for lateness and slowdown, and B5 is the "rent everything" upper bound for cost.

**Q: Isn't J then only comparable within one scenario and seed?**
Yes, by design. J ranks rule sets *against each other on the same workload*; it is not an absolute quality number. Use the raw metrics to compare across scenarios.

**Q: Why no preemption?**
It keeps the model simple and safe. Killing or moving running jobs has real costs, and many production batch clusters don't do it either.

**Q: How long does a run take?**
Under half a second per rule set, so a full comparison of 4 rule sets is about 1–2 seconds.

**Q: What does the chooser add, if you can just pick the best rule set per scenario?**
In reality you *don't know* which scenario you're in. A day can start quiet and turn into a rush. The chooser looks at live signals (queue length, deadline pressure, queue growth) and switches rule sets during the day.

---

## 11. Limitations (be honest about these)

- **Synthetic workload only.** The arrival rates, sizes and runtime distribution are assumptions. No real trace yet.
- **Estimates are always at least the true runtime** (factor 1–3), so backfilled jobs never overrun their estimate. Real systems also have underestimates.
- **The deadline-aware projection is rough.** For jobs other than the front one, it is "work ahead ÷ nodes that could hold it", not an exact schedule.
- **Only the front job gets a reservation** (standard EASY), so other waiting jobs can be pushed back by backfilling.
- **Score J depends on B2 and B5 as references**, so it ranks rule sets within one workload and is not an absolute measure.
- **Results shown are mostly from seeds 1–3.** There are no confidence intervals or statistical tests yet.
- **Quiet ends in ties** (on seeds 2–3 several rule sets are identical) because there is no load, so nothing differentiates them. This is the intended "quiet" behaviour.
- **No determinism test in the test suite yet.** Determinism was checked by hand (identical output twice).

---

## 12. Next steps

### Stretch goals in the current plan
- **Phase 8: a simple switching chooser.**
  - Every 30 simulated minutes, a hand-written if/else picks a rule set from queue length, deadline pressure and queue growth.
  - Running jobs are untouched; the queue is re-sorted and the reservation recomputed.
  - It appears as an extra "Rule chooser" row in the table, with its switches marked on the timeline.
- **Phase 9: save/replay plus an oracle.**
  - At each 30-minute window, snapshot the state and replay the next hour under every rule set; an "oracle" keeps the best.
  - This gives the **headroom** number: "a perfect chooser would beat the best fixed rule set by X%".

### After the progress check
- **Look-ahead chooser:** short simulations of the near future to pick the rule set.
- **ML chooser:** train a model (e.g. LightGBM) on replay labels ("which rule set was best in this window?").
- **LLM chooser:** an LLM (e.g. via Ollama) that may only pick from the menu of rule sets, never write its own policy.
- **Attack study:** can a user who games their job descriptions (owner, name, estimates) trick the chooser? Compare unsafe, warned and menu-only LLM setups.
- **More rule sets** (B6, B8 from the v0.3 plan).
- **A real trace** from the Parallel Workloads Archive, instead of only synthetic jobs.
- **Statistics:** many seeds, averages with confidence intervals, significance tests.
- **Dashboard polish:** advisor screen, chooser decisions on the timeline, and exporting results.

### Small quick wins (optional, any time)
- Add a determinism test (run twice, compare the results).
- A multi-seed summary in the runner (mean ± spread over seeds 1–10).
