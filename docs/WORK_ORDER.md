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
      **→ Track 2 chat:** "philly_jobs / attempts / machines parquet ready", plus the T1 report numbers
      (job counts, test window, GPU memory sizes).
- [x] **6. [T1] T2 Training examples and time split** (3-4 h).
- [x] **7. [T1] T2b Export oracle + last2 and the checker** (1.5-2 h).
      **→ Track 2 chat:** "oracle.csv and last2.csv pass check_predictions.py".

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
- [ ] **16. [T2] W0 Paper LaTeX skeleton** (1.5 h). Small; fits here.

### Block F: real results for the abstract (~6-8 h)

- [ ] **17. [T2] S8 Philly loader** (3 h).
- [ ] **18. [You + Track 2 chat] S11 Parameter sourcing** (1-2 h): the review chat finds and verifies sources; a short
      Claude Code session only edits `configs/flex.yaml` and the parameter table script.
- [ ] **19. [T2] S12 E1 + E4 on Philly with oracle and last2** (3 h). These are the numbers the abstract may use.
- [ ] **20. [You] W5 Abstract v1 → submit by 24 Nov AoE** (1 h, with the Track 2 chat). Only numbers that exist.
      If step 19 isn't done, submit the qualitative abstract (PAPER_PLAN section 4).

**Abstract-critical path = steps 1-11, 14, 15, 17-20.** Steps 12-13 (transformer) run alongside, but the abstract
does not depend on them.

### Block G: baselines, all predictors, remaining experiments (~10-12 h)

- [ ] **21. [T1] T4 Baselines and evaluation** (3-4 h; needs T3b finished).
- [ ] **22. [T1] T5 Export gbdt / lightgbm / transformer** (1.5-2 h). **→ Track 2 chat:** "all 5 prediction files
      pass the checker; E3 can run".
- [ ] **23. [T2] S10 Motivation charts** (2 h).
- [ ] **24. [T2] S14 E3 predictor comparison** (2 h).
- [ ] **25. [T2] S13 E2 sweeps + sensitivity** (2 h; reduced grid is fine).
- [ ] **26. [T2] S15 E5 carbon** (1.5 h; **trim**: one paragraph if short of time).
- [ ] **27. [T2] S9 Utilisation join** (3-5 h; **trim**: skip and keep the cluster-average fallback if short of time).
      If done, re-run step 19 quickly so E1 uses real utilisation.

### Block H: finish the paper (~8-10 h)

- [ ] **28. [T2] S16 Final numbers and figures** (2 h): E1/E3 re-run with the transformer as the realistic predictor;
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
