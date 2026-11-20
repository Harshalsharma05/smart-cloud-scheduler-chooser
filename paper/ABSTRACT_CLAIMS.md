# Abstract claims check (rewritten in W11, step 36, 2026-11-20, for the two-trace abstract; earlier versions in git history)

Abstract: `paper/abstract.txt` (= `abstract_easychair.txt` = the abstract in `main.tex`), 200 words. Every value below is recomputed
by `python paper/figures/check_claims.py` from `results/` (via `make_numbers.compute()`); the script fails if a value or the abstract
wording drifts. Default cell = 2 h / 20% (10 events per trace). Hold rule (`hold_beyond_grid`) on both traces. w_f = 0 is the primary cost.
Philly: 84 feasible events (6 infeasible excluded). Acme (Seren): 90 events, none infeasible.

## Remaining caveats

1. **Claim b uses w_f = 0** ("when delaying failed work costs nothing"). At w_f = 1 the default-cell medians are 30x (Philly) and 9.3x (Acme).
2. **Claim c ("at most one of 174 events")** is for the transformer predictor: Philly 0 of 84 at both weightings; Acme 0 of 90 at w_f = 0 and
   1 of 90 at w_f = 1. With the oracle, burst-only beats delay in 2 (w_f = 0) and 4 (w_f = 1) of the 90 Acme events, never on Philly.
3. **Claim d ("a cheaper cloud does not change this")**: Philly at the spot price $0.43/GPU-h 18x delay (default cell, $/MWh ratio of medians,
   E2); Acme at $1.50/GPU-h 6.65x (transformer) and 4.55x (oracle), per-event median, default cell; c_delay stays at the base price in both.
4. **Claim f (Philly 5-16%)** holds only under the default cap curve (pessimistic curve: default-cell ratio 1.00, cheaper in 6 of 10) and at
   w_f = 0; in-sample masking moves each ratio by at most 0.029.
5. **Claim g ("not at all on Acme")** is the w_f = 0 median over 90 events: 1.000 (gbdt, lightgbm) to 1.011 (transformer); at w_f = 1 the
   learned mixes are 0.985-1.000 (about 1.5% at most). "Exact runtimes would save 17%" is the oracle mix, median ratio 0.827 (64 of 90 cheaper).
6. Acme cloud price: $2.745/A100-h (p4d.24xlarge listing on 2026-11-19), not the older ~$4.10; at $4.10 burst-only is 18.1x delay.

## Claims table

| # | Abstract claim | Value now | Source |
|---|---|---|---|
| a | Delay is the cheapest action that meets the cut on both traces | Philly: delay < burst (median $/MWh, transformer) in 9 of 9 cells; cap non-compliant in 9 of 9 cells. Acme: delay < burst in 9 of 9 cells; cap compliant in 1 of 90 events | `results/e1/runs.csv`, `results/acme_e1/runs.csv` |
| b | bursting alone costs a median of 125 (Philly) and 12 (Acme) times as much as delay per event when delaying failed work costs nothing | per-event median, default cell, transformer, w_f = 0: Philly 125, Acme 12.1 (w_f = 1: 30 and 9.3) | `results/e1/runs.csv`, `results/acme_e1/runs.csv` |
| c | With the transformer predictor it beat delay in at most one of 174 events | burst-only cheaper than delay-only: Philly 0 / 84 (w_f 0 and 1), Acme 0 / 90 (w_f 0) and 1 / 90 (w_f 1); 84 + 90 = 174 events | same |
| d | a cheaper cloud does not change this | Philly spot $0.43: 18x; Acme $1.50: 6.65x (transformer), 4.55x (oracle) | `results/e2`, `results/acme_e1/runs.csv` (setting price=1.5) |
| e | the cloud bill covers a job's whole remaining runtime, so long jobs make bursting much worse | Acme burst-only (transformer): jobs with > 1 h left are 35% of burst jobs and 99% of the cloud $; Philly: 86.8% of running jobs at event starts have run > 12 h; burst/delay 125x (Philly) vs 12x (Acme) | `results/acme_e1/bursts.csv`, `results/motivation/stats.csv` |
| f | The predictor-guided mix beats delay by 5–16% on Philly under our default cap curve | mix-p90 / delay-only per-event median, 84 events, w_f 0: transformer 0.945, gbdt 0.839, lightgbm 0.870; pessimistic cap curve 1.00; masking shift 0.029 | `results/e1`, `results/e1_learned`, `results/t4`, `results/robust_insample` |
| g | not at all on Acme, where exact runtimes would save 17% | Acme mix-p90 / delay-only, 90 events, w_f 0: learned 1.000-1.011; oracle 0.827 (17% cheaper) | `results/acme_e1/runs.csv` |
| h | Reading predictions naively for long-running jobs makes the controller 11 to 68 times more expensive | Philly transformer mix, default cell, contract vs hold: 68x (p50, w_f 0), 39x (p90, w_f 0), 22x (p50, w_f 1), 11x (p90, w_f 1) | `results/e3/runs_default.csv`, `results/e3/runs_default_hold.csv` |
