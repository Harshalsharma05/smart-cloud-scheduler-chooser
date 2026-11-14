# Abstract claims check (W5a, revised in W5b, W5c and W5d, 2026-11-14)

Abstract: `paper/abstract.txt` (199 words, limit about 200). Values computed from the result CSVs on 2026-11-14.
Default cell = 2 h / 20% (10 events); hold rule = `hold_beyond_grid` (default since S16a). w_fail = 0 is the primary cost.

## Remaining caveats

1. **Naive reading, last2:** E1 under the contract rule has no transformer rows. With last2 the contract rule is 55x delay vs 33x under
   hold (1.7x worse). The 10-70x claim is for the transformer mix and comes from the E3 runs (claim e).
2. The p50 mix is not part of the abstract any more; for reference it is 1.2% dearer than delay at w_fail=0 (cheaper in 35/84 events)
   and 2.4% cheaper at w_fail=1 (54/84).

3. **"bursting helps only with near-perfect runtime knowledge, in long, deep events" (claim f)** is supported at w_fail=0 only: the oracle
   mix/mix-nocloud is 0.99 overall (4 h / 30%: 0.85, 4 h / 20%: 0.91, but 2 h / 30%: 0.96, 1 h / 30%: 0.98, 4 h / 10%: 0.99). At w_fail=1 the
   oracle ratio is 1.00 overall (4 h / 30%: 0.95, 4 h / 20%: 0.99). "Deep" therefore means the 4 h cells at 20-30% cuts; a 30% cut alone is not enough.
4. **"about 5% cheaper than delay alone" (claim g)** is the median per-event ratio over 84 feasible events (0.95). In total cost it is 0.91 (w_fail 0)
   and 0.93 (w_fail 1); in the default cell 10 of 10 events are cheaper than delay. The controller is mix-nocloud (the mix with the cloud refused),
   which equals the mix with the p90 estimate to within 1% (mix/mix-nocloud = 1.00).

## Claims table

| # | Abstract claim | Value now | Source |
|---|---|---|---|
| a | Delay is the cheapest single action, roughly $3k-13k per MWh for a 2-hour, 20% cut | transformer: 3,334 (w_fail=0) to 12,941 (w_fail=1) $/MWh; last2: 3,583 to 12,728. Delay < burst in 9 of 9 cells; cap non-compliant in 9 of 9 cells | `results/e1/runs.csv` (usd_per_mwh, usd_per_mwh_wfail1, median of 10 events), `paper/tables/t2.tex` |
| b | Capping alone reaches less than half of a 20% cut | 46% of target at 2 h / 20% (42% at 1 h, 47% at 4 h); compliance 0.00 | `results/e1/runs.csv` (shed_mwh vs cut x base_mean_kw x duration), `t2.tex` ("shed 46% of target") |
| c | Bursting alone costs 30 to 125 times more (than delay, transformer) | w_fail=1: 33x cost (per-event median 30x; $/MWh 31x). w_fail=0: 123x cost (per-event median 125x; $/MWh 121x) | `results/e1/runs.csv` |
| d | The mix that budgets bursts with the transformer's 90th-percentile estimate is a few percent cheaper than delay alone in both cost weightings | holds at w_fail=0 and w_fail=1: mix p90 / delay median per-event ratio 0.945 and 0.941; cheaper in 66/84 and 75/84 feasible events (ratio of medians 0.932, 0.897) | `results/e1/runs.csv`, `results/e1/s16a_mix_vs_delay.csv` |
| d' | (removed from the abstract) p50 mix vs delay, for reference | ratio 1.012 (1.2% dearer, cheaper in 35/84) at w_fail=0; 0.976 (cheaper in 54/84) at w_fail=1 | same files |
| e | A naive reading of runtime predictions for long-running jobs makes the controller 10 to 70 times more expensive | transformer mix cost per event, default cell, contract vs hold: w_fail=0: 101,658 / 1,492 = 68x (p50), 47,178 / 1,199 = 39x (p90); w_fail=1: 104,216 / 4,791 = 22x (p50), 50,612 / 4,554 = 11x (p90) | contract: `results/e3/runs_default.csv`; hold: `results/e3/runs_default_hold.csv` (same 10 events) |
| f | Offloading adds almost nothing to the delay+capping gain (transformer p90); bursting helps only with near-perfect runtime knowledge, in long, deep events | mix / mix-nocloud, median per-event cost ratio, 84 feasible events: transformer p90 = 1.00 at w_fail 0 and 1 (0.998-1.000 per cell; mix cheaper in 38/84 events). Oracle = 0.99 overall at w_fail 0 (1.00 at w_fail 1); 4 h / 30% 0.85, 4 h / 20% 0.91 (w_fail 0). Transformer p50 = 1.05 (bursting costs more); last2 = 22 | `results/e1nc/runs_d1.csv`, `runs_d2.csv`, `runs_d4.csv` (mix-nocloud), `results/e1/runs.csv` (mix), summary `results/e1nc/s13d_cells.csv`, `paper/tables/t5.tex` |
| g | A controller combining delay and capping (mix-nocloud), guided by the transformer's p90 estimate, is about 5% cheaper than delay alone in both cost weightings | mix-nocloud / delay, transformer p90, median per-event ratio, 84 events: 0.949 (w_fail 0), 0.946 (w_fail 1); cheaper in 59/84 events (w_fail 0), 16 equal. Default cell 0.86-0.87, 10/10 events cheaper | `results/e1nc/runs_*.csv`, `results/e1/runs.csv` (delay), `results/e1nc/s13d_cells.csv`, `paper/tables/t5.tex` |
| h | "We discuss when offloading is worth using" | discussion, no numeric claim | (S13 sweeps and winner maps support the discussion, not checked here) |
