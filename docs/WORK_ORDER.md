# Work order: which phase to run when (Burst to Flex, CCGrid 2027)

Version 1, 12 Nov 2026. Written in the main planning chat. This file decides the **order of Claude Code sessions**
across the two tracks, and the rules for switching between them. It does not change what a phase does: that is in
`docs/track-1/ML_PLAN.md` (T0-T5) and `docs/track-2/SIM_PLAN.md` + `PAPER_PLAN.md` (S0-S16, W0-W8).

If this file and a track plan disagree on *what* a phase does, the track plan wins.
If they disagree on *order*, this file wins.

---

## 1. The rules (short)

1. **One Claude Code session at a time.** Never run Track 1 and Track 2 sessions together.
2. **Switch only at a pause point:** the phase's done-check passed, the work is committed, `git status` is clean.
3. **Every new session starts with that track's session-start prompt**, then the phase prompt
   (Track 1: `docs/track-1/STARTER_PROMPTS_ML.md`, Track 2: `docs/track-2/STARTER_PROMPTS_T2.md`).
   Start a fresh Claude Code session (or `/clear`) when switching tracks, so context from one track doesn't leak
   into the other.
4. **After each phase:** paste the report into that track's review chat, fix anything it asks for, then tick the
   step below (`[x]`) in the same commit or a tiny follow-up commit ("work order: step N done").
5. **Cross-track hand-offs** (marked **→** below): tell the other track's review chat in one line, for example
   "T2b done: predictions/oracle.csv and last2.csv pass the checker".
6. **If a phase is blocked** (bug, missing data, a question for the review chat): mark it `[!] blocked: <why>`,
   commit, and move to the next unblocked step of the *other* track. Come back when it is unblocked.
7. **Unattended work** (download, Kaggle training) never blocks you: start it, then run the next step that doesn't
   need it.
8. **Medium complexity.** If a phase grows beyond its plan, stop, write the extra idea into that plan's "Later"
   section, and finish the phase in its simplest correct form.

---

## 2. The order

Legend: **[T1]** Track 1 session, reviewed in the Track 1 chat. **[T2]** Track 2 session, reviewed in the Track 2 chat.
**[You]** manual or writing work (no Claude Code session needed). Hours are rough Claude Code + review time.

### Block A: set up and get the data moving (~3 h)

- [x] **1. [T1] T0 Housekeeping** (0.5-1 h): `.gitignore`, `ml/` skeleton, `ml/.venv` with Python 3.11 + CUDA torch,
      `ml/config.py` (TEST_START/TEST_END live here; Track 2 reads them).
- [x] **2. [You] Start the Philly download** (`trace-data.tar.gz`, ~1 GB, Git LFS; see ML_PLAN T1 step 1) into
      `data/raw/`. Check ~8 GB free disk. **Unattended**: carry on with step 3 while it downloads.
- [x] **3. [T2] S0 Housekeeping and freeze** (0.5 h): tag `progress-check-demo`, requirements, folders.
      **S0 applies all CLAUDE.md edits in one go**: Track 2's edits 1-4 *and* Track 1's ML section
      (`docs/track-1/CLAUDE_MD_ML_SECTION.md`), plus the IMPLEMENTATION.md note, so CLAUDE.md is edited only once.
- [x] **4. [T2] S1 Flex engine core** (3-4 h, no data needed). Do it while the download finishes.

### Block B: data and the simple predictors (~8-10 h)

- [x] **5. [T1] T1 Clean the Philly job log** (3-4 h; needs step 2 finished).
- [x] **5b. [T1] T1 fixup: machine GPU counts from util width** (G = 2,892).
      **→ Track 2 chat:** "philly_jobs / attempts / machines parquet ready", plus the T1 report numbers
      (job counts, test window, GPU memory sizes).
- [x] **6. [T1] T2 Training examples and time split** (3-4 h).
- [x] **7. [T1] T2b Export oracle + last2 and the checker** (1.5-2 h).
      **→ Track 2 chat:** "oracle.csv and last2.csv pass check_predictions.py".
- [x] **7b. [T1] T2c Re-export oracle + last2 over all jobs overlapping [SIM_START, TEST_END)** (1.5-2 h).
      **→ Track 2 chat:** "oracle.csv and last2.csv re-exported (37,262 jobs, 111,451 rows)".

### Block C: the simulator, part 1 (~10-12 h)

- [x] **8. [T2] S2 Running-job moves** (4 h).
- [x] **9. [T2] S3 Power model** (1.5 h).
- [x] **10. [T2] S4 Grid events and paired runs** (2.5 h).
- [x] **11. [T2] S5 Predictions reader** (1.5 h).

### Block D: transformer code, then train it unattended (~5-7 h of sessions)

- [x] **12. [T1] T3a Transformer code + smoke run on the GTX 1650** (4-6 h). Its report says whether full
      training fits locally (< ~1.5 h) or needs Kaggle.
- [x] **13. [T1] T3b Start the full training run** (~1 h of your time: local run, or upload to Kaggle per
      `ml/KAGGLE.md`). **Unattended**: once it is running, go straight to step 14.
      Finish T3b (download the checkpoint, print the validation numbers, commit) whenever training ends, even if that
      is between later Track 2 steps. That is the only allowed "interrupt", and it is a short session.

### Block E: the simulator, part 2 (~10 h, while the transformer trains)

- [x] **14. [T2] S6 Single-action policies and cost metrics** (4 h).
- [x] **15. [T2] S7 Controller, selectors, experiment runner** (3-4 h). **Pause point P1**: the whole pipeline works
      on synthetic data.
- [x] **16. [T2] W0 Paper LaTeX skeleton** (1.5 h). Small; fits here. (Build not yet verified: no LaTeX on this PC.)

### Block F: real results for the abstract (~6-8 h)

- [x] **17. [T2] S8 Philly loader** (3 h).
- [x] **17b. [T2] S8b Carry-over jobs** (1 h). Inserted after S8: jobs running at epoch0 become pseudo-jobs; baseline validation table.
- [x] **17c. [T2] S8c Diagnostics** (measurement only; commit bad60fa).
- [x] **17d. [T2] S8d Carry-over keeps real ids, all-jobs deltas, censored = useful, pred_coverage, tie-breaks.**
- [x] **17e. [T2] S9b G = 2,892 re-run + GPU availability** (measurement only). T1 fixed philly_machines.parquet (2,892 GPUs).
- [x] **17f. [T2] S9c Pool = GPUs reporting in cluster_gpu_util** (auto pool 2,300; installed 2,892 only logged).
- [x] **18. [You + Track 2 chat] S11 Parameter sourcing** (1-2 h): the review chat finds and verifies sources; a short
      Claude Code session only edits `configs/flex.yaml` and the parameter table script.
- [x] **19. [T2] S12 E1 + E4 on Philly with oracle and last2** (3 h). These are the numbers the abstract may use.
- [x] **19b. [T2] S12b Diagnostics of the S12 results** (measurement only: burst cap, 30% floor, mix burst predictions).
- [ ] **20. [You] W5 Abstract v1 → submit by 24 Nov AoE** (1 h, with the Track 2 chat). Only numbers that exist.
      If step 19 isn't done, submit the qualitative abstract (PAPER_PLAN section 4).

**Abstract-critical path = steps 1-11, 14, 15, 17-20.** Steps 12-13 (transformer) run alongside, but the abstract
does not depend on them.

### Block G: baselines, all predictors, remaining experiments (~10-12 h)

- [x] **21. [T1] T4 Baselines and evaluation** (3-4 h; needs T3b finished).
- [x] **22. [T1] T5 Export gbdt / lightgbm / transformer** (1.5-2 h). **→ Track 2 chat:** "all 5 prediction files
      pass the checker; E3 can run".
- [x] **23. [T2] S10 Motivation charts** (2 h).
- [x] **24. [T2] S14 E3 predictor comparison** (2 h).
- [x] **24b. [T2] S14b Lookup-rule diagnostics** (jobs past the last grid point; measurement only).
- [x] **24c. [T2] S14c E3 re-run with lookup_rule = hold_beyond_grid** (declared 2026-11-14 after S14b; contract rule kept as baseline).
- [x] **25. [T2] S13 E2 sweeps + sensitivity** (2 h; reduced grid is fine).
- [x] **25b. [T2] S13b Diagnostics of the S13 results** (migrate/restore, oracle mechanism, absolute table; measurement only).
- [x] **25c. [T2] S13c Sweeps and T4 redone under hold_beyond_grid** (contract-rule results kept in results/*_contract).
- [x] **25d. [T2] S13d Marginal value of bursting (mix-nocloud control)** (measurement only).
- [x] **26. [T2] S15 E5 carbon** (1.5 h; **trim**: one paragraph if short of time).
- [x] **27. [T2] S9 Utilisation join** (3-5 h; **trim**: skip and keep the cluster-average fallback if short of time).
      If done, re-run step 19 quickly so E1 uses real utilisation.

### Block H: finish the paper (~8-10 h)

- [x] **28a. [T2] S16a hold_beyond_grid default; E1/E4 re-run (last2, transformer, oracle); F1/F6/F8/F9/T2/T3 redone; attribution control** (brought forward part of 28).
- [x] **28. [T2] S16 Final numbers and figures** (2 h): E1/E3 re-run with the transformer as the realistic predictor;
      `paper/numbers.tex`.
- [ ] **29. [You + Track 2 chat] W6-W8** (≈8 h): methodology, evaluation text, final pass, page fit, double-blind
      check. **Submit by 1 Dec AoE.**

### Writing you can do any time (no Claude Code, with the Track 2 chat)

W1 references (check each on dblp), W2 formulation + design text, W3 figures F3/F4 (draw.io), W4 related work +
intro + threats. Good moments: while the download runs (step 2), while the transformer trains (step 13), or whenever
a phase is blocked. They need about 10 h in total and must be done before step 29.

---

## 2b. End-of-step hand-off (Claude Code does this automatically)

Two Claude Code sessions stay open in VS Code: a **Track 1 session** and a **Track 2 session**. Only one is active at
a time. So that you never have to work out what comes next, every step ends with a hand-off block.

**When a step's done-check has passed and the work is committed, Claude Code must:**
1. Tick the step in section 2 (`[x]`) and add a line to the progress log (section 4), in the same commit or a tiny
   follow-up commit ("work order: step N done").
2. Find the **next unticked, unblocked step** in section 2 (skip steps marked `[!]`; a `[You]` step is listed as
   manual work).
3. Print the report (the track's usual format), then this block **exactly**, filled in:

```
=== HAND-OFF ===
DONE: step <N> (<phase>), commit <hash>
1. REVIEW: paste the report above into the <Track 1 | Track 2> review chat. If it asks for fixes, do them in THIS
   session first (fixup commit), then continue with 2.
2. CROSS-TRACK NOTE: <only if step N has a "→" in WORK_ORDER: the one line to paste into the OTHER review chat; else "none">
3. MANUAL FIRST: <anything you must do by hand before the next step (download, Kaggle upload, writing); else "none">
4. NEXT: step <M> (<phase>) in the <Track 1 | Track 2> Claude Code session.
   Paste this into that session:
   ----
   Continue with WORK_ORDER step <M> (<phase>). Re-read CLAUDE.md (<ML track | Paper work> section),
   docs/WORK_ORDER.md and <docs/track-1/ML_PLAN.md | docs/track-2/SIM_PLAN.md (+ PAPER_PLAN.md for W phases)>,
   then run the <phase> prompt from <docs/track-1/STARTER_PROMPTS_ML.md | docs/track-2/STARTER_PROMPTS_T2.md>.
   Stop after the done-check, commit, and print the report and the hand-off block.
   ----
=== END ===
```

**Rules for the hand-off:**
- If the next step is in the same track, the "NEXT" prompt goes into the same session.
- If the next step is a `[You]` step, say so under MANUAL FIRST and give the step after it as NEXT. Steps 18, 20 and
  29 need the Track 2 review chat for writing or sourcing.
- If a step is `[!] blocked`, the hand-off names the blocker and points to the next unblocked step in the other track.
- Never start the NEXT step yourself. Always stop after printing the block.

## 3. Load check (honest)

| Block | Claude Code + review (h) | Cumulative |
|---|---|---|
| A | ~5 | 5 |
| B | ~8-10 | ~15 |
| C | ~10 | ~25 |
| D | ~5-7 | ~31 |
| E | ~10 | ~41 |
| F | ~7-9 | ~49 |
| G | ~15 | ~64 |
| H | ~10 | ~74 |
| Writing W1-W4 | ~10 | ~84 |

About **84 hours of work in total, ~50 of them before the abstract**. From 12 Nov that is roughly **4-5 h every
day until 24 Nov**, then ~5 h a day until 1 Dec. It is possible but tight. If you fall behind:

1. First trim: S15 (carbon) → one paragraph; S9 → cluster-average fallback; S13 → default event only.
2. Second trim: drop `gbdt` and keep `lightgbm` only (T4/T5), one transformer seed.
3. If still behind on 28 Nov: submit what is solid, or switch to the **poster track (15 Feb 2027, 2-4 pages)**.
   Decide this with your faculty, not alone.

Never trim: the leakage rules, the checker, the paired-run design, honest reporting.

---

## 4. Progress log

Claude Code appends one line per finished step (date, step, commit hash, one-line result).

| Date | Step | Commit | Result |
|---|---|---|---|
| 2026-11-12 | 1 (T0) | aad75af | ml/ skeleton, ml/.venv (py3.11, torch 2.14.1+cu126, CUDA OK on GTX 1650), gitignore |
| 2026-11-12 | 5 (T1) | b16a851 | 117,325 jobs cleaned to 3 parquet tables; job log is UTC (YARN check); test window 27,022 jobs, G = 2,490 |
| 2026-11-12 | 3 (S0) | 29e2414 | tag progress-check-demo (e80be3c); CLAUDE.md merged for both tracks; requirements + flex/traces/paper folders; 4 old tests pass |
| 2026-11-12 | 4 (S1) | e3b5804 | flex engine + safety + synthetic generator; 6 tests pass (4 old + 2 new); 20k synthetic jobs drain in 0.2 s |
| 2026-11-13 | 6 (T2) | 982d7b9 | ml_examples.parquet 331,896 rows (train 207,717 / val 52,373 / test 71,806); 1.85% empty history; leakage test passes |
| 2026-11-13 | 7 (T2b) | 23a90dc | oracle.csv 74,728 rows (test 71,806 + 2,922 warm-up), last2.csv 71,806 rows; both pass check_predictions.py |
| 2026-11-13 | 8 (S2) | 0a5c96c | suspend/resume/set_cap/burst + flex/cloud.py + cloud safety rules; 8 tests pass (4 old + 4 flex); 20k jobs drain in 0.6 s |
| 2026-11-13 | 9 (S3) | ce30093 | flex/power.py (D5 power, D9 cap curve, tracker, cloud energy); power/cap params ASSUMPTION in flex.yaml; 10 tests pass; 20k jobs with tracker 1.0 s |
| 2026-11-13 | 10 (S4) | ca39926 | flex/grid.py (D13 events, LA local time), metrics.py (D6 shed metrics), paired.py (snapshots, run_base/run_event/run_pair, gate D10), engine admit hook + truncate_arrivals; 12 tests pass |
| 2026-11-13 | 11 (S5) | c7d4142 | flex/predictions.py (contract lookup, oracle fallback, validation); synthetic oracle/last2 writer; real oracle.csv/last2.csv load (25,044 / 24,536 jobs); S4 events now share 10 start times; 14 tests pass |
| 2026-11-13 | 12 (T3a) | d2e2f38 | 69,702-param transformer; smoke val loss 1.775 -> 1.440; 47 s/epoch on GTX 1650 (30 epochs ~24 min) so T3b runs locally |
| 2026-11-13 | 13 (T3b) | dc3a1a2 | tf_v1 trained locally (9 epochs, best val loss 1.3243); val pinball 0.440 vs last2 1.003, val status log-loss 0.858 vs 1.206 |
| 2026-11-13 | 14 (S6) | 366607c | flex/policies.py (delay/cap/burst-only, gating hook on_hold) + D7 cost, waste and user-impact metrics + result row; cloud/cost params in flex.yaml (cloud values ASSUMPTION); 16 tests pass |
| 2026-11-13 | 15 (S7) | 6e25d7e | mix controller + selectors (flex/policies.py), flex/experiments.py (e1-e4 synthetic quick runs: e1 19 s), 4 figure scripts, gate power cache (800-GPU stress 4.7 s -> 2.3 s), on_hold rule; P1 reached; 20 tests pass |
| 2026-11-13 | 16 (W0) | d5d8c21 | paper/main.tex (IEEEtran, anonymous), 9 section files with page budgets, refs.bib empty, numbers.tex stub, README; PDF build NOT verified (no LaTeX installed) |
| 2026-11-13 | 17 (S8) | 946ca92 | traces/philly.py: 36,751 jobs (12,209 warm-up), G 2,490; baseline run 12 s; mean util 0.41 (util_default 0.5 ASSUMPTION); sim wait 0 vs real 175 min (pool has no VC limits); one paired run 1.2-1.8 s |
| 2026-11-13 | 17b (S8b) | 1ecadae | 643 carry-over jobs (2,253 GPUs); util 0.655 vs real 0.665; mean wait 0.31 vs 174.7 min (still < half, stopped); validation.csv in results/philly_baseline |
| 2026-11-13 | 17c (S8c) | bad60fa | diagnostics: burst $40k = 13,317 GPU-h of long Failed jobs billed to failure (no bug); capacity sweep; warm-up-excluded delay cost was the reason delay cost $0 |
| 2026-11-13 | 17d (S8d) | 50cd35c | carry-over keeps real ids + elapsed0; deltas include warm-up/carry; censored -> Pass; pred_coverage column; (submit, job_id) tie-breaks; 20 tests pass |
| 2026-11-13 | 18 (S11) | bae65cc | flex.yaml: p_idle 70, PUE 1.58, cloud PUE 1.09, price 3.06, boot 300 sourced; host power, migrate/restore, cap-curve points still ASSUMPTION; paper/tables/params.tex generated |
| 2026-11-13 | 7b (T2c) | ad2c5f1 | export keys = valid jobs overlapping [2017-11-08, 2017-12-16): 37,262 jobs, 111,451 rows; oracle + last2 pass checker; 4 tests pass |
| 2026-11-13 | 27 (S9) | 000da99 | philly_job_util.parquet: 37,264 of 37,296 jobs measured (99.9%), 119 s full run; job util mean 0.287 / p50 0.195, minute-weighted 0.558 (now the fallback); cap shed 0.067 -> 0.196 MWh, still compliance 0; FOUND: philly_machines.parquet GPU counts swapped vs util file (G 2490 vs 3030?) |
| 2026-11-13 | 5b (T1 fixup) | ab64a41 | philly_machines num_gpus derived from util width: 150x2, 156x4, 246x8, G = 2,892; 0 attempt violations (415 verified, 137 util-only); jobs/attempts unchanged (117,325 / 152,714, sum num_gpus 195,296) |
| 2026-11-14 | 17e (S9b) | 9da03a5 | G = 2,892 from machines table (no hard-coded pool); baseline sim util 0.563 vs real 0.573, wait 0 vs 174.7 min (pool no longer binds); capacity 0.8x -> wait 65 min, 0.7x -> 348 min; default event rerun; util file shows only ~2,297 GPUs reporting per minute in the test window (busy/available 0.72, busy > available in 1,979 min) |
| 2026-11-14 | 17f (S9c) | 663eeff | philly.pool_gpus: auto = 2,300 (median non-NA GPUs/min, test window; 132 four-GPU machines have no util rows and no attempts); baseline sim util 0.722 vs real 0.720 (of pool), wait mean 80 vs 175 min, p90 326 vs 45; default event delay now $629/MWh (w_fail 0), 20 tests pass |
| 2026-11-14 | 19 (S12) | a2afe3a | E1 Philly (pool 2,300, last2 + oracle delay/mix): 540 policy rows + 90 baselines in 1,352 s; delay is cheapest compliant policy in every cell with last2 (median 2.1-4.7 k$/MWh); mix(last2) costs a median 32x delay; mix+oracle wins 7 of 9 cells at w_fail=0; cap complies in 5 of 90 events, burst fails every 30% event; F1/F6/F9/T2 made |
| 2026-11-14 | 19b (S12b) | 5447a28 | burst hits the 1,000-GPU cloud cap in all 30 events at 30% (3000 -> compliant, 100% of target, same $/MWh order); 30%-cut events 8, 9: P* 578-604 kW < floor 618 kW (infeasible); mix(last2) bursts jobs predicted at the 60 s floor (95-98%) whose actual remaining is median 25-30 h: $46k-125k billed vs $70-150 predicted |
| 2026-11-14 | 25 (S13) | c126053 | sweeps on the 10 default-cell events: A price x boot (600 rows), B pool x price x cap (1,000 rows, 4 pools), T4 11 perturbations (550 rows); 26 min wall with 3 processes; no infeasible events in this cell; w_fail 0: mix+oracle wins every cell, w_fail 1: delay wins (mix+oracle within 1-5%) except spot price/installed pool; mix(last2) and burst never beat delay (0 of 10 events, 3 at 0.9 x auto + spot); cap 1000 vs 3000 identical for delay/mix |
| 2026-11-14 | 25b (S13b) | c3fc2f1 | burst cheaper at migrate x2 is a selection effect, not a bug (shared jobs bill +120 s each; 143 short jobs replace 64 long ones: $236k -> $16k); restore_s does not touch burst; oracle mix bursts Pass jobs with ~1 min left and suspends only Failed/Killed (w_fail 0), mix-delay median -$17 per event, sum -$1,062, cheaper in 7/10 events; no bug found |
| 2026-11-14 | 24 (S14) | c2eae8b | E3 on Philly, 3 parts x ~8 min in parallel; no learned predictor beats last2 on median cost (mix: gbdt 1.9x, lightgbm 2.3x, transformer 1.09x; 5/10 events cheaper); oracle mix 0.002x last2 (w_fail 0); p90 cuts mix cost for all three learned (transformer 101k -> 47k $/event) but not below delay; out-of-sample run same picture; ~24% of running GPUs fall back to oracle values (censored carry-over); F8, T3 written |
| 2026-11-14 | 24b (S14b) | 1060059 | 87% of running jobs are past 43,200 s (median elapsed 237 h); mix bursts 98-100% such jobs, 99-100% at the 60 s floor (gap > grid p50) vs 29-48% of all running; 98-100% of the cloud $ goes to them although actual remaining is median 35-70 h; lookup rule makes learned-predictor E3 result mostly an artifact for long jobs (rule not changed) |
| 2026-11-14 | 24c (S14c) | 10743b2 | hold_beyond_grid: default-cell mix with gbdt/lightgbm/transformer falls from 224k/257k/102k to 1.5k/1.8k/1.5k $/event (p90: 1.2-1.3k), burst 272k/286k/281k -> 191k/159k/174k; mix now cheaper than delay/last2 in only 4-6 of 10 events (p50), 7-8 (p90); last2 also changes under the rule (mix 85k -> 52k), blind and oracle rows identical; no predictor claimed best |
| 2026-11-14 | 28a (S16a) | 9484aa0 | hold_beyond_grid is the default; E1 90 events (6 infeasible, all 30% cells), 9 parts in parallel 7 min; transformer: delay wins 3 cells (10% cuts), mix-p90 6 (w_fail 1: mix-p90 9); last2: delay all 9; oracle: mix 7; mix vs delay/transformer cheaper in 35/84 events (median ratio 1.012), p90 66/84 (0.945); mix gain: delay -$188, cloud +$101 per event, 34 fewer suspensions, cap 0.6; zero-burst events still cheaper |
| 2026-11-14 | 20a (W5a) | f31a476 | drafted: title and 194-word abstract in paper/main.tex and abstract.txt; ABSTRACT_CLAIMS.md checks every number (burst/delay 121-125x at w_fail 0 slightly above "120"; "few percent cheaper" holds for mix-p90 only, p50 mix is 1.2% dearer); step 20 stays open ([You]) |
| 2026-11-14 | 20b (W5b) | 4c5c675 | abstract revised (194 words); claims file updated: (c) 30-125x, (d) p90 only with p50 1.2% dearer at w_fail 0, (e) 40-70x from results/e3 contract vs hold; caveats: "median estimate is not cheaper" and "40-70x" hold at w_fail 0 only |
| 2026-11-14 | 20c (W5c) | 4dafc99 | abstract wording fixed: mix-p90 cheaper than delay in both cost weightings (0.945 / 0.941); naive-reading penalty 10-70x (39-68x w_fail 0, 11-22x w_fail 1) |
| 2026-11-14 | 25c (S13c) | afe497b | 12 parts in parallel, ~19 min; mix-p90/transformer wins all 12 sweep-A cells and all pool x price cells (0.83-0.89 x delay, cheaper in 8-10 of 10 events) but its burst share is ~1%: gain is the cap, vanishes with the pessimistic cap curve; burst-only 18-260x delay, never wins; mix with last2 5-52x; migrate x2 selection effect gone (burst +0.06%); sanity checks 0 difference |
| 2026-11-14 | 25d (S13d) | 7464896 | mix-nocloud (max_cloud_gpus 0), 6 parts ~8 min; transformer p90: mix/nocloud 1.00 (mix cheaper in 38/84 events), nocloud/delay 0.95 -> gain is capping+selection, bursting ~0; transformer p50: bursting costs +5% (median), last2 24x; oracle: nocloud/delay 1.00, mix/nocloud 0.99 median (sum 0.85, mix cheaper 47/84) -> oracle gain is bursting, in 4 h and 30% cells |
| 2026-11-14 | 20d (W5d) | f850649 | abstract reflects S13d (199 words): delay+capping with p90 ~5% cheaper than delay, offloading adds ~nothing, bursting helps only with oracle in long deep events; caveats in ABSTRACT_CLAIMS.md (oracle gain at w_fail 0 only; deep = 4 h cells at 20-30%) |
| 2026-11-14 | 23 (S10) | 1bf3ebe | F2 (4 panels) + results/motivation/stats.csv (42 numbers, macros \Mot* in numbers.tex) + Section 2 text; Failed+Killed = 59.3% of GPU-hours, 64.3% of them end within 1 h; baseline power flat (evening/overall 0.996); 86.8% of running jobs at the 90 event starts are past the 12 h grid |
| 2026-11-14 | 21 (T4) | e12119d | test pinball: gbdt 0.419, lightgbm 0.426, transformer 0.442, last2 1.034; AUROC Failed ~0.80 for all learned; status log-loss lightgbm 0.678 vs prior 0.888; oracle perfect; 6 tests pass |
| 2026-11-14 | 22 (T5) | 487709e | gbdt/lightgbm/transformer.csv over export keys (111,451 rows each); 5 checker OKs; test metrics from CSVs match evaluate (max diff 2.5e-8) |
| 2026-11-15 | 26 (S15) | 296fcaa | E5 carbon: eGRID2023 intensities in flex.yaml; re-ran mix(transformer p90) and mix(oracle) on 90 E1 events (shed reproduced exactly), 84 feasible; median net kg CO2/MWh shed 198.1 (delay, nocloud, mix-p90), 197.1-197.4 (oracle mix); net > 0 in 100% of events; cloud adds 0.14% (transformer) / 3.2% (oracle) of avoided at us-east-2; t6_carbon.tex, e5_snippet.tex |
| 2026-11-15 | 28 (S16) | adadb53 | make_numbers.py (211 macros from results only), make_all.py (no re-simulation; F1/F2 need --with-sim), check_claims.py (8 of 9 claims OK; claims-file value 0.85 for oracle 4 h/30% is 0.845 now), check_paper.py (0 unresolved refs/cites, 2 [CITE?], 1 comment-only blind hit), T6 now % cloud emissions of avoided (mix 0.1%, oracle 2.1-3.2%), figure metadata scrubbed |
| 2026-11-15 | W3 + S16 fixes | ccde3d4 | claims 9/9 OK (oracle 4 h/30% 0.84); abstract scopes claim c to the 2 h/20% cell (199 words); F3 overview and F4 controller flow as TikZ (figures/f3_overview.tex, f4_controller.tex), not compiled; check_paper: 0 unresolved refs/cites |
| 2026-11-15 | Overleaf fixes | 7b74afd | 8 macro names had digits (TransP90/P50 -> TransPNineZero/PFiveZero), name/value asserts in make_numbers + check_paper; kappa_j(f) -> Delta P_j(f) in Section 4/F4/Alg. 1; make_overleaf_zip.py (23 files, 92 KiB, 0 missing); not compiled |
| 2026-11-15 | Overleaf round 2 | cf5bd91 | F4 dropped (repo only); eq. (4),(5) split; F3 prediction label moved; refs.bib acronyms/names braced; local build paper/build.py (Tectonic 0.15, XeTeX + TeX Gyre Termes): 5 pages, 0 overfull, 0 undefined refs; Algorithm 1 long lines split |
| 2026-11-15 | W6 (Section 5) | 8fd2807 | 05_methodology.tex (trace/split, pool + baseline check Table I, events, predictors/policies, parameters/metrics with 3 URL footnotes); params table trimmed to 14 rows; 27 new macros (Meth*, Base*, SetupInfeasibleCutPct, MethNoPredGpuShareMedianPct); PAPER_PLAN budgets + writing rules; local build 6 pages, 0 overfull |
| 2026-11-15 | W7 Part 0+1 | 05041ce | Sec. 3 cap-curve wording (assumed, anchored by patel24); missing_pred option (oracle default, uninformed = 30 d, useful) + 1 test (21 pass); robust_missing on 84 events: identical to oracle fallback (all 174 no-row jobs are censored Pass jobs with >= 247 h left); stop condition not fired (0.945 / 0.941) |
| 2026-11-15 | W7 Part 2 (Section 6) | c4c7173 | 06_evaluation.tex (6.A-6.F, about 2.0 pages incl. 3 floats: Table main = T2 + mix-nocloud + ratios, F6, F8; carbon as text); F6 legend/notes fixed; 26 new macros; build 8 pages, 0 overfull; claims 9/9 |
| 2026-11-15 | W4 Part 1 (in-sample check) | ba25665 | uninformed_submit_before option + test (22 pass); jobs submitted before 2017-11-01 masked for learned predictors: mix-p90/delay 0.956 (w_f 0) / 0.939 (w_f 1) vs 0.945 / 0.941; total 10.5% / 8.5% lower; in-sample jobs = 38% of running jobs, 25% of running GPUs at event starts; stop not fired |
