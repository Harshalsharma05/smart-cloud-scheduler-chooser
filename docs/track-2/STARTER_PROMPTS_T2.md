# Track 2 starter prompts for Claude Code

Open Claude Code in the repo root (`smart-scheduler-chooser`).

- Paste the **session-start prompt** at the start of every new Claude Code session.
- Then paste the prompt for the phase you are doing.
- After the report, paste the report into the Track 2 review chat before starting the next phase.

---

## Session-start prompt (every new session)

```
Read CLAUDE.md (especially the "Paper work" sections), docs/track-2/SIM_PLAN.md (sections 1-2 and the
phase I give you) and, for paper phases, docs/track-2/PAPER_PLAN.md. Do not read or change ml/ except
ml/config.py (read only, for TEST_START/TEST_END).

Rules for this session:
- Do only the phase I name. Do not start the next one.
- Never change the frozen demo code: sim/, rules/, workloads/, eval/, api/, frontend/. New simulator code goes
  in backend/flex/ and backend/traces/.
- Follow the design decisions D1-D16 in SIM_PLAN.md exactly. If one is unclear or looks wrong, stop and ask
  me instead of redesigning.
- Keep it simple: plain functions and dataclasses, no class hierarchies, no new dependencies beyond
  requirements.txt, only the tests the phase's done-check asks for.
- Never invent parameter values: use the ones in configs/flex.yaml; a new one gets "# ASSUMPTION".
- I am on Windows. Give me any commands I must run myself.

At the end of the phase:
1. Run the phase's done-check and show the output.
2. Report in at most 15 lines: what you built, files changed, the check result, run times, anything
   surprising, and open questions.
3. Commit with message "Track 2 <phase>: <summary>", tick the step in docs/WORK_ORDER.md with a
   progress-log line (same or tiny follow-up commit), and stop.
```

---

## S0 Housekeeping and freeze
```
Do phase S0 from docs/track-2/SIM_PLAN.md. S0 edits CLAUDE.md once for both tracks (docs/WORK_ORDER.md step 3).
1. git tag progress-check-demo on the current commit (before any change).
2. CLAUDE.md: apply Edits 1-4 from docs/track-2/CLAUDE_MD_AND_IMPLEMENTATION_NOTES.md, and add Track 1's ML
   section (the markdown block in docs/track-1/CLAUDE_MD_ML_SECTION.md) exactly as written. Final order of
   sections: What this project is (Edit 1) / How to work / Keep it simple / Rules the demo simulator must follow
   (Edit 2) / Layout (Edit 4) / Commands / ML track (Track 1 block) / Paper work (Edit 3). Do not reword either
   track's text; only fix heading levels if needed.
3. IMPLEMENTATION.md: add the status note at the very top. README.md: replace the status line as given.
4. backend/requirements.txt: add numpy, pandas, pyarrow, matplotlib. Tell me the pip command to run.
5. .gitignore: make sure it has data/, predictions/, results/, *.pt, *.ckpt, runs/, ml/.venv/,
   ml/results/*.parquet (add only missing lines; Track 1's T0 may have added some already).
6. Create backend/flex/__init__.py, backend/traces/__init__.py and paper/figures/.gitkeep.
7. Commit docs/ too if it is still untracked (WORK_ORDER.md, track-1/, track-2/).
8. In docs/WORK_ORDER.md tick step 3 ([x]) and add a progress-log line (date, step, commit, result).
Done-check: pytest -q (the 4 old tests pass), git tag lists progress-check-demo, git status is clean, and show me
the list of "##" headings in CLAUDE.md.
```

## S1 Flex engine core
```
Do phase S1 from docs/track-2/SIM_PLAN.md: backend/configs/flex.yaml (only the sections S1 needs: paths,
pool size for synthetic runs, seed, check_every), flex/models.py, flex/engine.py (GPU pool, FCFS
skip-ahead, run_until, run_to_drain, snapshot, on_event hook), flex/safety.py and traces/synthetic.py
(Philly-like synthetic jobs; mark every number in it as synthetic).
Keep the per-event cost low (no O(n^2) scans over all jobs); full safety checks every event in tests and
every check_every events otherwise.
Done-check: the two tests in S1 (in backend/tests/test_flex.py), plus print the run time for 20k synthetic jobs.
After the commit, tick the step in docs/WORK_ORDER.md and add a progress-log line.
```

## S2 Running-job moves
```
Do phase S2 from docs/track-2/SIM_PLAN.md: engine methods suspend, resume, set_cap and burst (running and
queued jobs), flex/cloud.py (per-GPU price, boot delay, per-second billing with 60 s minimum, cap on cloud
GPUs), segments for every state change, and the extra safety rules. Use a per-job version counter so that
stale FINISH events are ignored. The cap only needs a speed function here; take it as an argument
(the real cap curve comes in S3/S11).
Done-check: the two tests in S2.
```

## S3 Power model
```
Do phase S3 from docs/track-2/SIM_PLAN.md: flex/power.py implementing D5 exactly (including the cap
binding rule and the speed curve from D9 as a piecewise-linear function of r), a power tracker on the
engine's on_event hook that records (t, P) steps and integrates energy over any window, and the power and
cap-curve parameters in flex.yaml, each marked "# ASSUMPTION - replace in S11".
Done-check: the test in S3.
```

## S4 Grid events and paired runs
```
Do phase S4 from docs/track-2/SIM_PLAN.md: flex/grid.py (event generator per D13; it must also work on
the synthetic time range, using the same local-time rule), flex/paired.py (baseline run with snapshots at
event starts; run_event and run_base with the horizon rule in D6), the shed metrics in flex/metrics.py
(target P*, shed MWh, compliance, shortfall, rebound), and a trivial policy "none" with gating (D10).
Done-check: the two tests in S4.
```

## S5 Predictions reader
```
Do phase S5 from docs/track-2/SIM_PLAN.md: flex/predictions.py (load predictions/<name>.csv, validate
columns, lookup rule exactly as in the contract: latest row with elapsed_s <= elapsed, subtract the gap,
floor 60 s; fallback to oracle values for jobs without rows, with a logged count), and synthetic
oracle.csv / last2.csv under data/synthetic/predictions/ written by traces/synthetic.py with the same
rules as Track 1 (wall-based, grid 0, 60, 600, 3600, 14400, 43200; rows only while e < wall_s; last2 =
mean wall_s of the user's last two finished jobs minus elapsed, plus the user's historical fail/cancel rates).
Done-check: the test in S5.
```

## S6 Single-action policies and cost metrics
```
Do phase S6 from docs/track-2/SIM_PLAN.md: in flex/policies.py the gating rule (D10) and the policies
delay-only, cap-only and burst-only (D11) using a predictor object from S5; in flex/metrics.py the cost
model (D7: cloud bill, dGPUh_good with w_fail, missed deadlines with multiplier m, $/MWh with the 0.1x
shed rule), user impact and cloud waste. Every policy returns one row with the columns in SIM_PLAN
section 4 (fill what exists, leave the rest empty).
Done-check: the two tests in S6.
```

## S7 Controller, selectors, experiment runner
```
Do phase S7 from docs/track-2/SIM_PLAN.md: the mix controller and burst selectors (D11),
flex/experiments.py (CLI: python -m flex.experiments <e1|e2|e3|e4|e5> --data synthetic|philly
--predictor <name> [--quick]; writes results/<exp>/runs.csv and summary.csv with median and IQR over
events), and one figure script per experiment in paper/figures/ (matplotlib, 3.5 in wide, a fixed colour
per policy) that reads the CSVs and writes paper/figures/out/<name>.pdf.
Do not tune anything to make the mix win; if it doesn't, report it.
Done-check: the e1 quick run on synthetic data in < 10 min, the e2-e4 quick runs, and the oracle test in S7.
This is pause point P1: after the commit, write a short "how to run" note at the end of SIM_PLAN.md.
```

## S8 Philly loader
```
Do phase S8 from docs/track-2/SIM_PLAN.md: traces/philly.py (test window from ml/config.py, warm-up
jobs, drop never_started, keep censored and count them, deadlines per D12, G from philly_machines.parquet,
util from philly_job_util.parquet if present, else the cluster average with a log line) and event
placement on the real window. First check that the input files exist and have the contract columns;
if not, stop and report.
Done-check: a baseline run over the test window, plus the report items listed in S8.
```

## S9 Utilisation join
```
Do phase S9 from docs/track-2/SIM_PLAN.md: traces/philly_util.py writes
data/processed/philly_job_util.parquet (job_id, gpu_util_mean, n_minutes, coverage). Read the raw
cluster_gpu_util CSV in chunks; keep only the machines and minutes that test-window and warm-up attempts
need; skip "NA". First run it on the first 2M rows and show me the coverage and run time, then the full file.
If coverage is below 50% or the full run would take more than about 1 hour, stop and report.
Done-check: coverage stats, util distribution, run time.
```

## S10 Motivation charts
```
Do phase S10 from docs/track-2/SIM_PLAN.md: paper/figures/motivation.py with panels (a) GPU-hours and
job counts by status, (b) CDF of wall_s by status, (c) one test week of baseline facility power with
17:00-20:00 local windows shaded, (d) util histogram only if philly_job_util.parquet exists. Write the
caption numbers to results/motivation/stats.csv.
Done-check: PDFs in paper/figures/out/ and the stats CSV.
```

## S11 Parameter sourcing (Claude Code part only)
```
Update backend/configs/flex.yaml with these sourced values: <paste the table from the review chat>.
Each value gets a "# source: ..., accessed ..." comment. Then write paper/figures/params_table.py, which
writes paper/tables/params.tex (T1 in PAPER_PLAN) from flex.yaml. List any parameter that is still marked
ASSUMPTION and used by E1-E5.
Done-check: the table compiles as LaTeX (paste it into a scratch doc) and the ASSUMPTION list.
```

## S12 E1 + E4 on Philly
```
Do phase S12 from docs/track-2/SIM_PLAN.md: run e1 on Philly (durations 1/2/4 h x cuts 10/20/30% x 10
events x delay/cap/burst/mix with --predictor last2, plus mix with oracle), then e4 from the same runs,
and make F1, F6, F9 and T2. Check that predictions/last2.csv and predictions/oracle.csv exist first.
Report run time, every event with compliance < 99%, and the sanity check from S12.
Done-check: as in S12.
```

## S13 E2 sweeps and sensitivity
```
Do phase S13 from docs/track-2/SIM_PLAN.md: the price x boot-delay sweep on the default event (F7) and
the sensitivity table T4. Use --predictor last2 for now. Report the winner per cell.
Done-check: as in S13.
```

## S14 E3 predictor comparison
```
Do phase S14 from docs/track-2/SIM_PLAN.md: e3 for every predictor file that exists in predictions/
(list the missing ones; never fake them), plus selectors none-largest and random. Make F8, and T3 using
the accuracy numbers from ml/reports/ (read only; tell me which file you took them from).
Done-check: as in S14.
```

## S15 E5 carbon
```
Do phase S15 from docs/track-2/SIM_PLAN.md: e5 from the e1 runs with the three cloud-region intensities
in flex.yaml, and F10.
Done-check: as in S15.
```

## S16 Final numbers and figures
```
Do phase S16 from docs/track-2/SIM_PLAN.md: re-run e1 and e3 (e2 if time allows) with --predictor
transformer as the realistic predictor, write paper/figures/make_numbers.py, which creates
paper/numbers.tex (one \newcommand per number used in the text, named in CamelCase, values from the
summary CSVs), rebuild all figures, and compile the paper.
Done-check: paper compiles; list every macro in numbers.tex with its value.
```

---

## W0 Paper skeleton (any time)
```
Do task W0 from docs/track-2/PAPER_PLAN.md: create paper/main.tex with the IEEEtran conference class,
an anonymous author block ("Anonymous Author(s)"), \hypersetup with empty pdfauthor, sections/ with one
.tex file per section from PAPER_PLAN section 1 (each holding its page budget as a comment and a
placeholder paragraph), refs.bib (empty for now), numbers.tex (stub) and paper/README.md with the build
command (latexmk -pdf main.tex, or Overleaf upload). Tell me how to install a LaTeX distribution on Windows
if latexmk is missing (MiKTeX), or to use Overleaf instead.
Done-check: main.pdf builds (or the Overleaf upload compiles) with all section headings.
```
