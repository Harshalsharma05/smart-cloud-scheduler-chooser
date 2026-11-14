# Abstract claims check (W5a, 2026-11-14)

Abstract: `paper/abstract.txt` (194 words, limit about 200). Values computed from the result CSVs on 2026-11-14.
The abstract text was NOT edited. Default cell = 2 h / 20% (10 events); hold rule = `hold_beyond_grid` (default since S16a).

## Mismatches and caveats (read first)

1. **"30 to 120 times more"** (burst vs delay): w_fail=1 gives 31-33x (supported), but w_fail=0 gives 121-125x, slightly above "120".
   Within "roughly" but the upper end is 121-125, not 120. Suggest "30 to 125" or "about 30 to 120" if you want it exact.
2. **"reading runtime predictions naively ... makes it far worse"** is supported for the transformer (E3 contract vs hold: 68x / 39x
   in mix cost) but NOT shown for last2 in E1: contract 55x delay vs hold 33x delay (only 1.7x worse). E1 under the contract rule has no
   transformer rows (`results/e1_contract` has last2 and oracle only), so the transformer comparison comes from the S14/S14c E3 runs,
   where mix/transformer is compared with its own hold run.
3. **"a few percent cheaper than delay alone"** holds for the p90 variant only (5.5-5.9% median per event, cheaper in 66/84 and 75/84
   events). The p50 mix with the transformer is 1.2% DEARER at w_fail=0 (ratio 1.012, cheaper in 35/84 events). Delay still wins 3 of 9
   cells on $/MWh at w_fail=0 (the 10% cuts). The abstract's "uses the transformer's uncertainty" therefore means the p90 variant.
4. **"We derive guidelines for when offloading is worth using"** is not a number from these files (it rests on the S13 sweeps/winner
   maps, `results/e2*`, `paper/figures/out/f7*.pdf`); not checked here.

## Claims table

| # | Abstract claim | Value now | Source |
|---|---|---|---|
| a | Delay costs roughly $3k-13k per MWh, 2 h / 20% | transformer: 3,334 (w_fail=0) to 12,941 (w_fail=1); last2: 3,583 to 12,728 $/MWh | `results/e1/runs.csv` (usd_per_mwh, usd_per_mwh_wfail1, median of 10 events), `paper/tables/t2.tex` |
| b | Capping alone reaches less than half of a 20% cut | 46% of target at 2 h / 20% (other 20% cells: 42% at 1 h, 47% at 4 h); compliance 0.00 | `results/e1/runs.csv` (shed_mwh vs cut x base_mean_kw x duration), `t2.tex` ("shed 46% of target") |
| c | Burst alone costs 30 to 120 times more than delay (transformer) | cost: 123x at w_fail=0 (per-event median 125x), 33x at w_fail=1 (30x); $/MWh: 121x and 31x. Delay < burst in 9 of 9 cells | `results/e1/runs.csv` |
| d | Mix with the transformer's uncertainty (p90) a few percent cheaper than delay | median per-event ratio 0.945 (w_fail=0), 0.941 (w_fail=1); cheaper in 66/84 and 75/84 feasible events (ratio of medians 0.932, 0.897). Mix p50: 1.012 (35/84) and 0.976 (54/84) | `results/e1/runs.csv`, `results/e1/s16a_mix_vs_delay.csv` |
| e | Naive reading of predictions for long jobs makes it far worse | last2 mix/delay, default cell, w_fail=0: contract 55x vs hold 33x (w_fail=1: 16x vs 10x); all events: 32x vs 22x. Transformer mix cost per event, default cell: contract 101,658 (p50) / 47,178 (p90) vs hold 1,492 / 1,199 (68x / 39x), against delay/transformer 1,418 | `results/e1_contract/runs.csv`, `results/e1/runs.csv`, `results/e3/runs_default.csv` (contract), `results/e3/runs_default_hold.csv` (hold) |
| - | Delay is the cheapest single action | delay < burst in 9/9 cells; cap non-compliant (median compliance < 0.99) in 9/9 cells | `results/e1/runs.csv` |
