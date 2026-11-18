# Track 2 plan: simulator and experiments ("Cloud Bursting Rarely Pays for Demand Response in On-Premise GPU Clusters", CCGrid 2027)

Owner: Harshal. Reviewed in the Track 2 chat. Version 1, 11 Nov 2026.
Source of truth for the project: `claude/PROJECT_CONTEXT.md` v2 (sections 6, 7, 7b, 9, 10).
Companion files in this folder: `PAPER_PLAN.md`, `STARTER_PROMPTS_T2.md`, `CLAUDE_MD_AND_IMPLEMENTATION_NOTES.md`.

**How to use this file:** one phase per Claude Code session. Each phase is self-contained: it has inputs, outputs, a done-check and a safe pause point (commit after the check passes). Phases in Part A need **no Philly data**; they run on a synthetic Philly-like workload. Part B needs Track 1 outputs.

**Golden rule:** synthetic results are only for testing. **No synthetic number ever goes into the paper.**

---

## 0. Summary

| # | Phase | Needs Track 1? | Effort (h) | Must / trim |
|---|---|---|---|---|
| S0 | Housekeeping and freeze | no | 0.5 | must |
| S1 | Flex engine core (GPU pool, FCFS skip-ahead, synthetic workload, snapshots, safety) | no | 3–4 | must |
| S2 | Running-job moves (suspend/resume, power cap speed, GPU cloud, burst) | no | 4 | must |
| S3 | Power model | no | 1.5 | must |
| S4 | Grid events and paired-run harness (shed, compliance, rebound) | no | 2.5 | must |
| S5 | Predictions reader and synthetic prediction files | no | 1.5 | must |
| S6 | Single-action policies (delay-only, cap-only, burst-only), gating, cost metrics | no | 4 | must |
| S7 | Controller (mix), burst selectors, experiment runner, synthetic dry run of E1–E4 | no | 3–4 | must |
| **P1** | **Pause point: whole pipeline works on synthetic data** | | | |
| S8 | Philly loader (test window, deadlines, warm-up jobs, cluster size, event placement) | T1 | 3 | must |
| S9 | Utilisation join → `philly_job_util.parquet` | T1 + raw util CSV | 3–5 | trim (fallback: cluster average) |
| S10 | Motivation charts | T1 (+S9 optional) | 2 | must (2 panels), rest trim |
| S11 | Parameter sourcing (power, cap curve, prices, overheads, carbon) | no (any time) | 1–2 | must |
| S12 | E1 + E4 on Philly with oracle and last2 | T2 export | 3 | must |
| S13 | E2 sweeps + sensitivity table | T2 export | 2 | must (reduced grid OK) |
| S14 | E3 predictor comparison | T5 (partial with T2) | 2 | must |
| S15 | E5 carbon | T2 export | 1.5 | trim |
| S16 | Final numbers file and figure export for the paper | all above | 2 | must |

Total ≈ 40–45 h of Claude Code sessions plus review. W0 (paper LaTeX skeleton) is in `PAPER_PLAN.md` and can run any time.

**Critical path for the 24 Nov abstract:** S0–S8, S11, S12 (≈ 28 h) plus Track 1's T1 and T2 export. If that is not done, the abstract stays qualitative (see `PAPER_PLAN.md`, section 4).

---

## 1. Design decisions

Agreed in PROJECT_CONTEXT v2 unless marked **NEW**. NEW decisions follow from the agreed ones but need your OK.

### D1 (NEW) A new engine next to the frozen demo engine
- The paper simulator lives in a new package **`backend/flex/`**.
- The demo code stays untouched and frozen: `sim/`, `rules/`, `workloads/`, `eval/`, `api/`, `frontend/`. The 4 old tests must keep passing.
- Reason: the new rules conflict with the demo rules (running jobs move, no runtime estimates, GPU pool), so refactoring `sim/engine.py` would break the demo and its tests.
- We copy ideas from the demo code (heapq loop, cloud lifecycle and billing, safety checker), not import them.
- This replaces the module names suggested in PROJECT_CONTEXT section 8 (`sim/power.py` etc.). Update that section.

### D2 Cluster: a GPU pool (agreed)
- One pool of `G` GPUs. A job fits if `free_gpus >= job.gpus`. No nodes, no VCs, no fragmentation.
- `G` = the total GPUs in `philly_machines.parquet` (S8). For synthetic runs it comes from the config.

### D3 Base scheduler: FCFS with skip-ahead (agreed)
- After every event, scan the queue in submit order (ties: job_id) and start every job that fits.
- It uses no runtime estimates, so predictors only affect the controller.
- Known weakness: big jobs can starve. We accept it, state it in the paper, and list a fix under Later.

### D4 (NEW, matches Track 1) Job timing
- A job occupies its GPUs continuously for `wall_s` (first start to final end), then ends with its trace `status` (Pass / Killed / Failed).
- Attempts and retry gaps are merged. Reason: Track 1 defines `elapsed_s` and `rt_*` on wall time since first start, so the simulator must use the same clock.
- S8 reports how much this overstates GPU use (the share of jobs with `wall_s > 1.1 × run_s`).
- **Progress** is tracked in nominal seconds: `work_done`, from 0 to `wall_s`. At full speed, 1 s of wall time = 1 s of progress. Under a power cap, progress runs at speed `s < 1`. While suspended, progress stops.
- **Elapsed time for prediction lookup = `work_done`.** With the oracle this stays exactly correct under caps and suspends.
- Overheads (checkpoint, restore) add nominal seconds to the job's remaining work.

### D5 Power model (agreed form)
For each GPU:
- `P_gpu(u, cap) = min(P_idle + u·(P_max − P_idle), cap·P_max)`
- `u` = the job's mean GPU utilisation (S9; fallback = cluster average).
- Idle GPUs draw `P_idle`.

For the facility:
- `P(t) = PUE × [ Σ_GPUs P_gpu + G·H_idle + (busy GPUs)·H_busy ]`
- `H_idle` = host power per installed GPU (always on). `H_busy` = extra host power per busy GPU. The cap does not affect host power.

Rules:
- One representative GPU type, with ±20% sensitivity on `P_idle` and `P_max` (the GPU model is not confirmed; see ML_PLAN T1).
- Power is piecewise constant between simulator events, so energy is integrated exactly.

### D6 Shed baseline and paired runs (A1, agreed, made concrete)
**Snapshots:**
- One baseline run covers the whole test window and saves a snapshot (`copy.deepcopy` of the engine state) at each event start `t_s`.

**Per event:**
- Restore the snapshot twice: once with no event (the baseline continuation) and once per policy.
- Both runs see the same arrivals. Both stop injecting arrivals at `t_e + H_after`, then drain until those jobs finish. Default `H_after` = 24 h.

**Shed metrics:**
- Target: `P* = (1 − cut) × mean baseline power over [t_s, t_e)`. One flat target per event.
- Shed energy (MWh): `∫ (P_base − P_policy) dt` over the event window.
- Compliance: the fraction of event time with `P ≤ 1.01 × P*`.
- Shortfall: `max(0, max P − P*)` in kW, reported when the cut can't be reached (for example because of the idle floor).
- Rebound energy: `∫ (P_policy − P_base)⁺` over `[t_e, t_e + H_after)`. Reported, not priced (Later).

### D7 (NEW, refines A2) Cost of an event
**Formula:**
- `Cost$ = cloud bill + c_delay × ΔGPUh_good + m × c_delay × Σ_missed (gpus × wall_h)`

**Terms:**
- `ΔGPUh_good` = Σ over **Pass** jobs of `max(0, end_policy − end_base) × gpus` in hours, from the paired runs. Delay, cap, hold and boot waits are all measured the same way.
- `c_delay` default = the cloud $/GPU-hour (A2). Sensitivity: 0.5× and 2×.
- Missed deadline = a deadline job that meets its deadline in the baseline but misses it in the policy run. Penalty multiplier `m` default 1 (missing a deadline costs about one re-run). Sensitivity: m = 0 and 3.
- **Failed/Killed jobs:** their delay weight `w_fail` defaults to **0**, because delaying work that will fail or be cancelled loses no useful output. Sensitivity: `w_fail = 1` (count all delays). Cloud money spent on them is always counted, and it is pure waste.

**Headline metric:**
- `$/MWh = Cost$ / shed MWh`, only when shed ≥ 0.1 × the target energy. Otherwise "n/a" (shortfall).

**Why `w_fail` = 0:** with `w_fail` = 1 the failure head would barely change any decision, and contribution 2 (failure-aware selection) would be weak. The paper must state the choice and show the `w_fail = 1` result.

### D8 Burst and suspend semantics (A3, agreed, made concrete)
**Suspend (used by delay):**
- The GPUs are freed at once.
- The job keeps `work_done`. On resume, `restore_s` is added to its remaining work. The checkpoint is assumed instant at suspend (simplification).
- Suspended jobs resume at `t_e`, in FCFS order, under the normal skip-ahead scheduler.

**Burst a running job:**
- The on-prem GPUs are freed at once.
- A cloud instance is requested. When it is READY (after `boot_delay_s`), the job runs `remaining + migrate_s` on the cloud at full speed.
- It never comes back. The instance is released when the job ends.

**Burst a queued job:**
- Same, without `migrate_s`.

**Cloud:**
- An instance = the job's GPU count; price = `gpus × $/GPU-h`.
- Billing is per second from READY to RELEASED, with a 60 s minimum. Cap = `max_cloud_gpus` (default 1000, sensitivity).

### D9 Power cap (A4, agreed, made concrete)
**The cap:**
- A cluster-wide cap level `f ∈ {1.0, 0.9, 0.8, 0.7, 0.6}` × `P_max` applies to every on-prem GPU during the event. Uniform, so it is easy to explain.

**Speed of a capped job:**
- `r = f·P_max / P_uncapped(u)`. If `r ≥ 1`, speed is 1 (the cap doesn't bind).
- Otherwise speed `s = curve(r)`: piecewise linear through points read from published measurements (Zeus NSDI'23, Patel et al. ASPLOS'24), with a source note per point.
- Two variants: `optimistic` and `pessimistic` (sensitivity). No invented points: until S11 fills them, the config holds points marked `ASSUMPTION`.

### D10 (NEW) Gating during an event
- Between `t_s` and `t_e`, the skip-ahead scheduler starts a job only if the estimated facility power after the start stays ≤ `P*`. This applies to all policies.
- Jobs held this way count as delay (measured in `ΔGPUh_good`).
- Burst-capable policies may burst a job that is held at arrival (D11).

### D11 (NEW) Controller and policies
**Plan once at `t_s`:**
- Reach `P*` by the cheapest set of actions.
- Gating (D10) then keeps power under `P*` for the rest of the event, because power can only fall as jobs end.

**Per-job options and their estimated cost:**
- `cost_suspend(j) = c_delay·g_j·D·(1 − p_fail_j − p_cancel_j) + miss_risk(j)`, where D = event length.
- `cost_burst(j) = price·g_j·(rt_j + migrate_s) + c_delay·g_j·boot_delay·(1 − p_fail − p_cancel)`, with `rt_j` = the predictor's p50 remaining runtime (contract lookup rule).
- `kW(j)` = the job's facility power now.

**Mix controller:**
- For each cap level `f`, compute the kW saved by the cap and its slowdown cost.
- Then greedily add job actions in order of the lowest `cost/kW` until `P*` is reached. Each job takes its cheaper option (suspend or burst).
- Pick the `f` with the lowest total estimated cost.

**Single-action policies (for E1):**
- **delay-only:** suspend greedily by `cost_suspend/kW`, then gate.
- **cap-only:** the deepest cap needed, no job actions (shortfall if even 0.6 is not enough), then gate.
- **burst-only:** burst greedily by `cost_burst/kW`; arrivals during the event that gating would hold are burst instead.
- **mix:** the controller above. For held arrivals: burst if `cost_burst < cost_hold`.

**Burst selectors (for E3):**
- `none-largest`: most GPUs first, no predictor.
- `random`: seeded.
- `predictor:<name>`: last2, gbdt, lightgbm, transformer or oracle.

### D12 Deadlines (agreed)
- A seeded random 30% of test jobs get `deadline = submit + slack × wall_s`, with `slack ~ U(2, 4)`.
- This is drawn from the job itself, independent of its status. Stated as an assumption in the paper.

### D13 Events (agreed)
**Shape:**
- Duration ∈ {1, 2, 4} h; cut ∈ {10, 20, 30}%.
- Start: a random half hour between 17:00 and 20:00 America/Los_Angeles (the same timezone rule as T1).

**Placement:**
- At most one event per day.
- Only on days ≥ 2 after `TEST_START`. The event must end before `TEST_END − 1 day`.

**Count and default:**
- 10 events per (duration, cut), seeded.
- Default event for E2/E3 = 2 h, 20%.

### D14 Carbon (E5, agreed scope)
- Constant intensities (gCO2/kWh): local peak hours vs three cloud regions (low / medium / high), from a dated public source, with sensitivity.
- Local avoided = shed energy × local intensity. Cloud added = cloud GPU energy (same power model, cloud PUE) × region intensity.

### D15 Determinism and checks
- Integer seconds. Finish time = `now + ceil(remaining / speed)`. Stale finish events are dropped with a per-job version counter.
- One `random.Random(seed)` is passed in. Same config + seed → identical CSVs.
- Full safety checks run every event in tests. On Philly runs they run every `check_every` events (default 1000) to keep runs fast.

### D16 Parameters carry their source
- Every number in `configs/flex.yaml` has a comment: `# source: <paper/page or URL>, accessed <date>` or `# ASSUMPTION`.
- S11 replaces the ASSUMPTIONs that matter. Synthetic test configs may keep them.

---

## 2. Layout (new files only)

```
backend/
  configs/flex.yaml          world for the paper: pool, power, cap curve, cloud, overheads, costs, events, carbon, paths
  flex/
    models.py                FJob (+ state, work_done, segments), constants
    engine.py                GPU-pool engine: submit/start/finish, skip-ahead, snapshot, run_until, actions
    cloud.py                 per-GPU cloud pool (lifecycle, billing, cap)
    safety.py                invariants
    power.py                 power and energy
    grid.py                  event generator
    predictions.py           read predictions/<name>.csv, lookup rule
    policies.py              delay-only, cap-only, burst-only, mix (controller), selectors
    metrics.py               shed, compliance, rebound, cost, user impact, waste, carbon
    paired.py                baseline snapshots + paired event runs
    experiments.py           CLI: python -m flex.experiments e1|e2|e3|e4|e5 ...
  traces/
    synthetic.py             Philly-like synthetic jobs + synthetic oracle/last2 files
    philly.py                load philly_jobs.parquet → FJob list (test window, deadlines, warm-up flag)
    philly_util.py           utilisation join → data/processed/philly_job_util.parquet
  tests/test_flex.py         Track 2 tests (about 10 in total)
paper/figures/*.py           figure scripts (read results/*.csv, write paper/figures/out/*.pdf)
results/<exp>/*.csv          experiment outputs (git-ignored)
data/synthetic/              synthetic jobs and predictions (git-ignored)
```
Commands run from `backend/` (as today). The repo-root paths `data/`, `predictions/` and `results/` are set in `flex.yaml` (`paths:`), relative to the repo root.

---

## 3. Phases

Each phase ends with the done-check, a short report, a commit, and a stop. Starter prompts are in `STARTER_PROMPTS_T2.md`.

### S0 Housekeeping and freeze (0.5 h, no data)
- **Goal:** a clean start; the demo stays reproducible.
- **Do:**
  - Tag the current commit `progress-check-demo`.
  - Apply the CLAUDE.md section and the IMPLEMENTATION.md note (`CLAUDE_MD_AND_IMPLEMENTATION_NOTES.md`).
  - Add to `backend/requirements.txt`: numpy, pandas, pyarrow, matplotlib.
  - Check `.gitignore` has `data/`, `predictions/`, `results/`. Track 1 adds the first two; add only what is missing.
  - Create `backend/flex/__init__.py`, `backend/traces/__init__.py`, `paper/figures/`.
- **Done-check:** `pytest -q` → the 4 old tests pass; `git tag` shows `progress-check-demo`.

### S1 Flex engine core (3–4 h, no data)
- **Goal:** a correct, fast GPU-pool simulator with no grid events yet.
- **Inputs:** `configs/flex.yaml` (pool size and seed for synthetic runs).
- **Do:**
  - `flex/models.py`: `FJob(job_id: str, user, submit, gpus, wall_s, status, util, deadline, warmup)` plus runtime fields:
    - `state` ∈ {PENDING (not yet submitted), QUEUED, RUNNING, SUSPENDED, CLOUD_BOOTING, CLOUD_RUNNING, DONE}
    - `work_done`, `speed`, `version`, `first_start`, `end`, `on_cloud`, `segments` (list of `(start, end, where, speed)`)
  - `flex/engine.py`:
    - The event heap (SUBMIT, FINISH, plus kinds added later) and FCFS skip-ahead (D3).
    - `run_until(t)`, `run_to_drain()`, `snapshot()` (deepcopy), an `on_event` hook for power tracking.
    - Keep the queue as a list sorted by submit time. Avoid O(n²) work per event; report run time on 20k jobs.
  - `traces/synthetic.py`: a Philly-like seeded generator. All numbers are marked synthetic.
    - Diurnal arrivals; gpus from {1, 2, 4, 8, 16}; lognormal `wall_s` (median 1 h, cap 3 days).
    - Status mix about 70% Pass / 15% Killed / 15% Failed, with Failed jobs shorter. 50 users with different failure rates; `util ~ Beta`.
  - `flex/safety.py`:
    - GPUs used ≤ G; each job is in exactly one state.
    - A finished job has `work_done == wall_s + overheads`; no job starts twice from QUEUED; the clock never goes back.
- **Outputs:** an engine API that S2–S7 build on.
- **Done-check (tests):**
  1. A 2-GPU pool with A (2 GPUs, 10 h), B (1 GPU, 1 h), C (1 GPU, 1 h) at t=0 → A runs 0–10 h; B and C start at 10 h. Skip-ahead variant: A (1 GPU, 10 h), B (2 GPUs, 1 h), C (1 GPU, 1 h) → C starts at 0 (skips B), B starts at 10 h.
  2. A synthetic day (seed 1) runs twice with identical results, and the safety checker passes.
  - Also print the run time for 20k synthetic jobs (target < 30 s).
- **Pause:** safe after commit.

### S2 Running-job moves (4 h, no data)
- **Goal:** the actions exist as engine operations, correct and billed.
- **Do:**
  - Engine methods:
    - `suspend(job)`, `resume(job)` (D8)
    - `set_cap(f)`: changes the speed of all running on-prem jobs and reschedules their finishes via `version`.
    - `burst(job)` for running and queued jobs.
  - `flex/cloud.py` (per-GPU pricing, boot delay, billing, cap; D8).
  - Segments are recorded for every state change.
  - The safety checker learns the cloud rules (billed only READY→RELEASED, one instance per job, a burst job never returns on-prem).
  - From the S1 review:
    - Make `Engine.running` a dict `job_id → FJob` (insertion order) instead of a set, and never iterate a set of job ids. Set order of strings changes between Python processes (hash seed), which would make float power sums and greedy tie-breaks non-deterministic.
    - The `_schedule(new=job)` fast path is only valid when no queued job fits before the submit. Every operation that frees GPUs or changes what may start (finish, suspend, burst, resume, cap or gate change) must call the full `_schedule()`.
    - Make the work-conservation check real: at finish, `Σ (end − start) × speed` over the job's on-prem and cloud segments must equal `wall_s + extra_work` within `ceil` rounding (at most 1 s × speed per segment).
- **Done-check (tests):**
  1. A 1-GPU, 10 h job runs at full speed until 1 h, at speed 0.5 from 1 h to 3 h, and at full speed after (cap set at 1 h and removed at 3 h) → it ends at 11 h. The same job suspended from 1 h to 3 h with `restore_s = 0` instead ends at 12 h, and with `restore_s > 0` it ends at `12 h + restore_s`.
  2. A 2-GPU job with 1 h remaining is burst at t: `boot_delay` 300 s, `migrate_s` 120 s, price $3/GPU-h → it ends at `t + 300 + 3600 + 120`, and the bill = 2 × 3 × (3720/3600).
- **Pause:** safe after commit.

### S3 Power model (1.5 h, no data)
- **Goal:** facility power and energy for any run.
- **Do:**
  - `flex/power.py` implements D5.
  - A tracker hooked into `on_event` records `(t, P)` steps and integrates energy over any window.
  - Power parameters go into `flex.yaml`, marked ASSUMPTION until S11.
- **Done-check (test):** hand-computed power for 1 busy GPU at u=0.5 plus 3 idle GPUs, with and without a cap that binds, matches to 1e-9. On a synthetic day, energy = Σ steps (no gaps).
- **Pause:** safe after commit.

### S4 Grid events and paired runs (2.5 h, no data)
- **Goal:** events, the shed target, and the paired-run machinery.
- **Do:**
  - `flex/grid.py`: the event generator (D13), working on the synthetic window as well.
  - `flex/paired.py`: a baseline run with snapshots at event starts; `run_event(snapshot, event, policy)` and `run_base(snapshot, event)` with the horizon rule (D6).
  - `flex/metrics.py`: target `P*`, shed MWh, compliance, shortfall, rebound.
  - A trivial `policy=none` (no action, gating only) for testing.
- **Done-check (tests):**
  1. A paired run with `policy=none` and a 0% cut gives shed = 0 and identical job end times.
  2. Events are inside the window and in 17:00–20:00 local time, and the same seed gives the same events.
- **Pause:** safe after commit.

### S5 Predictions reader (1.5 h, no data)
- **Goal:** the contract lookup, used by the controller.
- **Do:**
  - `flex/predictions.py` loads `predictions/<name>.csv` into a dict `job_id → arrays` per elapsed grid point.
  - `lookup(job, elapsed)`: take the latest row with `elapsed_s ≤ elapsed`, subtract the gap from `rt_*`, floor at 60 s, and return `(p10, p50, p90, p_fail, p_cancel)`.
  - Jobs with no rows (warm-up, train period) fall back to oracle values from the job itself. Log the count.
  - Validate the columns on load.
  - `traces/synthetic.py` also writes `data/synthetic/predictions/oracle.csv` and `last2.csv` with the same rules as Track 1 (wall-based, grid `0, 60, 600, 3600, 14400, 43200`, rows only while `e < wall_s`).
- **Done-check (test):** a hand-made 2-job CSV gives the right lookups at elapsed 0, 59, 600 and 5000 s, including the floor.
- **Pause:** safe after commit.

### S6 Single-action policies and cost metrics (4 h, no data)
- **Goal:** delay-only, cap-only and burst-only produce correct $/MWh rows.
- **Do:**
  - `flex/policies.py`: gating (D10), delay-only, cap-only and burst-only (D11) using a predictor object.
  - `flex/metrics.py`:
    - Cost (D7): cloud bill, `ΔGPUh_good`, missed deadlines, $/MWh.
    - User impact: Δwait, Δbounded slowdown, missed deadlines.
    - Cloud waste: $ on Failed/Killed jobs, and $ on jobs whose remaining runtime at burst was < `boot_delay + 10 min`.
  - Every policy returns the same row shape (columns in section 4).
- **Done-check (tests):**
  1. A tiny hand case (4-GPU pool, 3 known jobs, a 1 h event, 50% cut): delay-only suspends the job with the lowest cost/kW, meets `P*`, and the cost matches a hand calculation.
  2. On the synthetic default event (2 h, 20%), all three policies run and give compliance near 100% or a reported shortfall.
- **Pause:** safe after commit.

### S7 Controller, selectors and the experiment runner (3–4 h, no data)
- **Goal:** the mix controller and the end-to-end pipeline on synthetic data.
- **Do:**
  - The mix controller and the burst selectors (D11).
  - `flex/experiments.py`:
    - CLI `python -m flex.experiments <e1|e2|e3|e4|e5> --data synthetic|philly --predictor last2 [--quick]`.
    - Writes `results/<exp>/runs.csv` (one row per event × policy × setting) and `results/<exp>/summary.csv` (median and IQR over events).
  - One figure script per experiment in `paper/figures/` that reads the CSVs. Plain matplotlib, IEEE column width 3.5 in.
- **Done-check:**
  - `python -m flex.experiments e1 --data synthetic --quick` finishes in < 10 min, and the mix's $/MWh is ≤ the best single action on at least most events. If not, report it; don't tune.
  - The e2, e3 and e4 quick runs also finish. One test: with the oracle and `w_fail = 0`, the mix never bursts a job that will Fail.
- **Pause point P1:** the whole pipeline works without Philly. Good moment to switch to Track 1.

### S8 Philly loader (3 h, needs T1)
- **Goal:** real jobs into the engine.
- **Inputs:** `data/processed/philly_jobs.parquet`, `philly_machines.parquet`, and `TEST_START`/`TEST_END` from `ml/config.py` (read the constants; don't copy them).
- **Do:**
  - `traces/philly.py`:
    - Jobs with `submit_time` in the test window become FJobs.
    - Jobs submitted earlier and still running at `TEST_START` become warm-up jobs (`warmup=True`, oracle values, excluded from metrics).
      The S1 engine can only start jobs from QUEUED. Simplest fix: start the simulation at a warm-up start (for example 7 days before `TEST_START`), submit warm-up jobs at their real `submit_time`, and exclude them from metrics. Decide in S8.
    - Drop `never_started`. Keep `censored` with their observed `wall_s` and count them.
    - Deadlines per D12.
    - `G` from the machine list.
    - Util from `philly_job_util.parquet` if present, else the cluster average (logged).
  - Events placed per D13 on the real window.
- **Done-check:**
  - A baseline run over the test window finishes; the safety checks pass every 1000 events.
  - Report: job count, warm-up count, G, the run time, the `wall_s > 1.1 × run_s` share, and simulated vs real mean wait. The last one is a sanity check, not a target.
- **Pause:** safe after commit.

### S9 Utilisation join (3–5 h, needs T1 + raw `cluster_gpu_util`)
- **Goal:** `data/processed/philly_job_util.parquet` (`job_id, gpu_util_mean`, plus `n_minutes` and `coverage`).
- **Do:**
  - `traces/philly_util.py`:
    - Read the util CSV in chunks; keep only the machines and minutes covered by test-window attempts (plus warm-up attempts).
    - For each attempt, average its `machine:gpu` columns over its minutes, skipping "NA". Combine to a job mean weighted by minutes.
  - First run on the first 2M rows, then the full file.
- **Done-check:** coverage stats (the share of jobs with ≥ 1 minute matched), the util distribution, and the run time. If coverage < 50%, stop and report; the fallback stays.
- **Trim:** if this takes more than one session, keep the cluster-average fallback and move on.

### S10 Motivation charts (2 h, needs T1; S9 optional)
- **Goal:** the figures for paper section 2.
- **Do:** `paper/figures/motivation.py`:
  - (a) GPU-hours and job counts by final status.
  - (b) A CDF of `wall_s` by status ("how quickly failed jobs die").
  - (c) The baseline facility power over one test week, with the 17–20 h windows shaded (uses S3 + S8).
  - (d) Optional: the util histogram (S9).
- **Done-check:** the PDFs exist in `paper/figures/out/`. The numbers printed in the figure captions are recorded in `results/motivation/stats.csv`.

### S11 Parameter sourcing (1–2 h, any time, mostly the review chat)
- **Goal:** replace the ASSUMPTIONs in `flex.yaml` that drive results:
  - `P_idle`/`P_max` (datasheet + a measurement paper)
  - `H_idle`/`H_busy`, PUE
  - the cap curve points (Zeus, Patel et al.)
  - cloud $/GPU-hour (a dated public price page; on-demand plus a cheaper alternative for sensitivity)
  - `boot_delay_s` (a published VM start-up study)
  - `migrate_s`/`restore_s` (a published checkpoint study, or derived from model size and bandwidth)
  - carbon intensities
- **Do:** the review chat finds and verifies the sources. Claude Code only edits `flex.yaml` and the parameter table script (`paper/figures/params_table.py` → `paper/tables/params.tex`).
- **Done-check:** no `ASSUMPTION` left on a parameter used in E1–E5, or each remaining one is listed in the paper's threats section.

### S12 E1 + E4 on Philly (3 h, needs T2 export: `oracle.csv`, `last2.csv`)
- **Goal:** the main result.
- **Do:**
  - `e1`: durations {1, 2, 4} h × cuts {10, 20, 30}% × 10 events × {delay, cap, burst, mix}.
  - The predictor is the best available realistic one (`last2` now, `transformer` at S16), plus `mix+oracle` as an upper bound.
  - `e4` reads the same runs (user impact, rebound).
  - Figures F1, F6, F9 and table T2 (PAPER_PLAN).
- **Done-check:**
  - All 450 policy runs plus 90 baseline continuations complete; the run time is reported.
  - Report every event with compliance < 99%.
  - Sanity: the shed MWh of cap-only scales with the cut until it hits a shortfall.

### S13 E2 sweeps and sensitivity (2 h, needs T2 export)
- **Goal:** when does burst win (RQ2)?
- **Do:**
  - On the default event (all 10): cloud price × {0.25, 0.5, 1, 2, 4} by `boot_delay` ∈ {60, 300, 900} s.
  - Record the winner and the mix's burst share. Figure F7.
  - Sensitivity table T4 on the default event: `P_idle`/`P_max` ±20%, PUE, cap curve optimistic/pessimistic, `c_delay` 0.5×/2×, m ∈ {0, 3}, `w_fail` = 1, `migrate_s` ×2.
- **Done-check:** the summary CSVs exist, and the figure and table scripts run.

### S14 E3 predictor comparison (2 h, needs T5; partial with T2)
- **Goal:** RQ3: does a better predictor make bursting cheaper?
- **Do:**
  - On the default event and the 4 h / 30% event: burst-only and mix × selectors {none-largest, random, last2, gbdt, lightgbm, transformer, oracle}.
  - Metrics: $/MWh, cloud $, cloud $ wasted (Failed/Killed; short remaining).
  - Figure F8. Table T3 takes Track 1's accuracy numbers (pinball/MAE, AUC) from `ml/reports/`, not recomputed here.
  - It can run first with {none, random, last2, oracle} and be re-run when T5 lands.
  - **Declared 2026-11-14, before the E3 transformer runs:** option `cost_quantile` = p50 (default) | p90. With p90 the cloud bill in `burst_cost` uses `rt_p90` instead of `rt_p50` (hold cost and the rest unchanged). The mix is run with both variants for gbdt, lightgbm and transformer (last2 has p90 = p50, so it is unchanged). Both variants are reported; neither is hidden.
  - Robustness run (2026-11-14): the default cell repeated with burst candidates restricted to jobs submitted on or after TEST_START (out-of-sample for the learned predictors). Carry-over and warm-up jobs may then only be suspended.
  - **S14c, declared 2026-11-14 after the S14b diagnostics and before any E3 re-run:** `predictions.lookup_rule` = `contract` (default; E1/E2 unchanged) | `hold_beyond_grid`. With `hold_beyond_grid`, for a non-oracle predictor and elapsed beyond the last grid point (43,200 s), the lookup returns the last row's p10/p50/p90 unchanged (no gap subtraction); between grid points the contract rule stays; the oracle stays exact; p_fail/p_cancel as before. Reason (S14b, default cell, 4,552 running jobs): 87% of running jobs are past 43,200 s (median elapsed 237 h); the mix bursts 98-100% such jobs, 99-100% of them at the 60 s floor (gap > grid p50) against 29-48% of all running jobs; 98-100% of the cloud $ goes to them although their actual remaining time is a median 35-71 h. Both rules are reported; the contract-rule E3 results stay in the paper's tables as the contract baseline.
  - **S16a, decided 2026-11-14: `hold_beyond_grid` is now the default lookup rule** for all non-oracle predictors (`flex.yaml: lookup_rule`; `contract` stays selectable with `--lookup-rule contract`). Reason: S14b/S14c, 87% of running jobs are past the 12 h (43,200 s) grid, and the contract rule pins their prediction to the 60 s floor. The old S12/S13/S14 results stay in their folders (`results/e1_contract`, `e4_contract`, `e2`, `e2b`, `t4`, `e3/runs_*.csv`) and are reported as the "contract rule" robustness results. E1/E4 and F1/F6/F9/T2 are re-run under the hold rule with last2, transformer (the realistic predictor) and the oracle.
  - w_fail = 0 is primary; w_fail = 1 is read from the same runs. Hard cell (4 h / 30%): feasible events only, cloud cap 3000, labelled as such.
- **Done-check:** the CSVs exist for every predictor file present. Missing files are listed, not faked.

### S15 E5 carbon (1.5 h, needs T2 export; trim)
- **Goal:** RQ4.
- **Do:** from the E1 runs, compute local avoided vs cloud added kgCO2 per event for 3 cloud-region intensities. Figure F10.
- **Done-check:** the CSV exists and the figure script runs.
- **Trim:** if short of time, one sentence plus one number in the discussion instead of a figure.

### S16 Final numbers and figures (2 h, needs everything above)
- **Goal:** every number in the paper comes from a file.
- **Do:**
  - Re-run E1/E3 (and E2 if time) with the transformer as the realistic predictor.
  - `paper/figures/make_numbers.py` writes `paper/numbers.tex` (LaTeX macros like `\newcommand{\BurstPerMWhDefault}{...}`) from the summary CSVs. Rebuild all figures.
- **Done-check:** `paper/` compiles; every number in the text is a macro or a figure; `git status` is clean after commit.

---

## 4. Output row shape (`results/<exp>/runs.csv`)
`exp, event_id, t_start, duration_h, cut, policy, selector, predictor, setting, target_kw, base_mean_kw, shed_mwh, compliance, shortfall_kw, rebound_mwh, cloud_usd, cloud_gpu_h, cloud_waste_fail_usd, cloud_waste_short_usd, dgpuh_good, dgpuh_all, missed_deadlines, cost_usd, usd_per_mwh, d_wait_min, d_slowdown, n_suspended, n_burst, cap_level, runtime_s, seed`

## 5. Interface contract (as used by Track 2)
- **Reads** `data/processed/philly_jobs.parquet`, `philly_attempts.parquet` (S9), `philly_machines.parquet` (S8), and `predictions/<name>.csv` (S5).
- **Writes** `data/processed/philly_job_util.parquet` (S9).
- `TEST_START`/`TEST_END` are read from `ml/config.py`.
- Elapsed and remaining are wall time since first start (Track 1 decision). The simulator's elapsed = nominal progress (D4).
- If a contract file is missing or has the wrong columns: stop and report. Never patch around it silently.

## 6. Later (revisit after the first full draft)
- Retries and attempts as separate runs (instead of D4 merging).
- A better base scheduler (an EASY-style reservation for the head job) to avoid starving big jobs.
- Per-job or per-tier power caps instead of one uniform cap.
- Re-planning during the event (resume suspended jobs early if power drops a lot).
- Pricing the rebound energy, and comparing $/MWh with published demand-response compensation.
- Hourly carbon traces instead of constants.
- Spot/preemptible cloud instances; GPU type mismatch between on-prem and cloud (speed factor).
- Bursting jobs back on-prem after the event.
- Helios trace (after 1 Dec).
- Confidence intervals by bootstrap over events (now: median and IQR).
- Using p10/p90 (risk-aware burst choice) instead of p50 only.

---

## How to run the pipeline (written at pause point P1, phase S7)

All commands from `backend/` with the venv active (`.venv\Scriptsctivate`), except the figure scripts (repo root).
Synthetic data only; no synthetic number goes into the paper. `--data philly` arrives with S8.

```
pytest -q                                                    # 4 demo tests + the flex tests
python -m traces.synthetic --predictions                     # 20k-job timing, then synthetic oracle.csv and last2.csv
python -m flex.experiments e1 --data synthetic --quick       # 3 events per cell; drop --quick for 10 per cell
python -m flex.experiments e2 --data synthetic --quick       # cloud price x boot delay on the default event
python -m flex.experiments e3 --data synthetic --quick       # burst selectors (missing prediction files are listed)
python -m flex.experiments e4 --data synthetic --quick       # reads results/e1 (runs e1 first if absent)
cd ..
backend\.venv\Scripts\python paperiguresig_e1.py       # also fig_e2.py, fig_e3.py, fig_e4.py -> paper/figures/out/*.pdf
```

- Outputs: `results/<exp>/runs.csv` (one row per event x policy x setting, columns of section 4) and
  `results/<exp>/summary.csv` (median, 25th and 75th percentile over events). `runtime_s` is one policy run;
  the baseline continuation of an event is shared by its policies.
- `--predictor <name>` reads `data/synthetic/predictions/<name>.csv` (synthetic) or `predictions/<name>.csv` (Philly).
  `e1` adds `mix` with the oracle as the upper bound when `oracle.csv` exists. `e5` is phase S15.
- Where things are: policies and the mix in `flex/policies.py`, cost metrics in `flex/metrics.py`, paired runs and the
  gate in `flex/paired.py`, events in `flex/grid.py`, parameters in `configs/flex.yaml` (cloud values are still
  `# ASSUMPTION`, replaced in S11).
