# SA1: Acme (Seren) simulator run (WORK_ORDER step 35)

Same engine, policies, metrics and paired-run design as Philly, through a dataset switch: `python -m flex.experiments acme_e1 --data acme`
(loader `backend/traces/acme.py`, config section `acme:` in `backend/configs/flex.yaml`, report `python -m traces.acme_report`).
Results: `results/acme_baseline/` and `results/acme_e1/` (runs, summary, events, bursts, fallback, report_*.csv). One seed. No paper text changed (step 36);
83 `Acme*` macros added to `paper/numbers.tex` (all 470 existing macros unchanged; `check_claims.py` 9 OK).

## Set-up (decisions of 19 Nov; what was found when applying them)

| item | value | source / note |
|---|---|---|
| Data | Seren GPU jobs from `acme_jobs.parquet`; 231,876 jobs simulated (175,233 test, 56,089 warm-up, 554 carry-over, none truncated); 5,865 jobs with wall_s <= 0 dropped | window from `ml/config.py` ACME_* (read, not copied): sim start 2023-06-22, events in [06-29, 07-30) China time |
| Useful work (M1) | waste = FAILED / NODE_FAIL / TIMEOUT (test jobs: 53,354 waste, 121,879 useful); CANCELLED and PREEMPTED useful; controller P(useful) = 1 - p_fail | `World.p_useful = "fail_only"` (Philly default unchanged) |
| Pool | G = 2,288 A100 | Hu et al. NSDI'24 |
| GPU power | P_idle 63 W, P_max 400 W | Seren GPU-power CDF idle floor; NVIDIA A100 datasheet (SXM, Max TDP 400 W), https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf, accessed 2026-11-19 |
| Utilisation | one cluster-wide u = **0.612**, calibrated so the simulated mean GPU power over the test window is 169.1 W (CDF mean 168.6 W; the CDF is August 2023, the window July) | per-job utilisation is not in the job trace |
| Host power | h_idle 115 W, h_busy 163 W per GPU (arithmetic in section 1) | GPU-node server-power CDF |
| PUE, cap curve, overheads, deadlines, cost weights | as Philly (cap curve stays ASSUMPTION) | |
| Cloud price | **$2.745 per A100-hour** = AWS p4d.24xlarge on-demand us-east-1 $21.957642/h / 8, instances.vantage.sh, page updated 2026-11-19 | **not the expected ~$4.10**: that matches the older $32.77/h listing. Today's listing is used (same site and method as the Philly V100 price); $4.10 is run as a default-cell sensitivity. c_delay = cloud price as in Philly |
| Cloud limit | 1,000 GPUs; no running job at any event start was larger | |
| Events | full grid: 1/2/4 h x 10/20/30%, 10 shared start days (seed 1), 17:00-19:30 China time; 90 events, **0 infeasible** (floor 643 kW, P* >= 736 kW) | |
| Predictions | `predictions/acme_{oracle,last2,gbdt,lightgbm,transformer}.csv`, hold-beyond-grid (never for the oracle); **0 jobs without rows** for every predictor | |
| Runs | per event: cap; delay, burst, mix (p50), mix-p90, mix-nocloud (p90) x 5 predictors; default cell also mix (oracle), mix-p90 (transformer), burst (oracle, transformer) at $1.50 and $4.10; 2,420 rows | two processes, 19 + 25 min |

**Philly unchanged.** The first default-cell Philly event (d2h_c20_0: cap, delay, burst, mix, mix-p90, mix-nocloud with the transformer)
was re-run before and after the changes: all columns identical, and equal to the stored `results/e1` / `e1nc` rows (difference 0.0).
Backend tests: 23 pass. Code changes that touch shared files have Philly-neutral defaults: optional `grid.tz` (Asia/Shanghai for Acme),
`World.p_useful`, `--data acme` / `--cells`, and `user_impact` adds `info["n_done_pruned"]` (0 unless set).
**Speed-up (Acme only):** before each event start the snapshot drops arrivals after the event horizon and the jobs already done
(identical in every run, so they add 0 to every metric; their count is kept for the user-impact means). Checked on one 4 h / 30% event
with 5 policies: every field of every row identical with and without it; 16.8 -> 4.2 s per delay run.

## Does "delay is cheapest, bursting rarely pays" hold on Acme?

**Bursting rarely pays: yes, clearly.** Burst-only costs a median 12.1x delay-only per event in the default cell (9.3x at w_f = 1) and 11.9x over
all 90 events (Philly: 125x / 30x and 111x / 23x). With every learned predictor burst-only is cheaper than delay in 0 of 90 events at w_f = 0
(transformer 1 of 90 at w_f = 1); even with exact runtimes (oracle) in 2 of 90. A cheaper cloud does not change this: at $1.50/GPU-h burst-only is
still 6.7x (transformer) and 4.5x (oracle, cheaper in 1 of 10) delay; at $4.10 it is 18x. The cloud bill again comes from a few long jobs:
jobs with more than 1 h left are 36% of the jobs burst-only moves (transformer) but 99% of its cloud dollars.
**Delay is the cheapest compliant single action in every cell** (cap complies in 1 of 90 events; its median $/MWh is only defined because it
sheds more than 10% of the target).

**What differs from Philly.**
1. *The gap is ten times smaller.* Acme jobs are short (A1: median wall time 2.3 min against 22 min; > 12 h jobs hold 52% of the running GPUs at 17-20 h against 87%),
   so a burst job's remaining runtime, and with it the cloud bill, is much shorter.
2. *The predictor-guided mix gains nothing over delay.* mix-p90/delay per event: median 1.011 (transformer, cheaper in 28 of 90), 1.000 (gbdt,
   45 of 90) and 1.000 (lightgbm, 42 of 90) at w_f = 0, against 0.945 / 0.839 / 0.870 on Philly. With the p50 cloud bill the mix costs 2.1-3.8x
   delay: it bursts jobs predicted to end soon that then run for more than an hour (98% of its cloud dollars with the transformer). Exact runtimes
   still help: the oracle mix is a median 0.827 of delay (64 of 90 events), so the room is there but the learned predictions do not find it.
   Offloading inside the mix adds nothing for the learned predictors either (mix-p90/mix-nocloud 1.000-1.003).
3. *w_f matters much less.* Delay-only costs 4,365 $/MWh at w_f = 0 and 5,050 at w_f = 1 (Philly 3,334 and 12,941), because CANCELLED and
   PREEMPTED count as useful work here (M1), so most of the delayed work is useful in both weightings.
4. *No infeasible events* (the A100 idle floor is about half the event-time power), and no job above the cloud limit was running at any event start.
5. `last2` is a poor guide here: the mix with last2 costs 9-12x delay (A2 already found last2's status and runtime values poor on Acme).

Caveats: one trace month, one seed; per-job utilisation unknown (one u for all jobs, calibrated to an August CDF); host power from CDF ends
(the p10/p90 reading would give 126 / 80 W); waits are 0 in the simulation against a 4.0 min mean in the trace (the pool never binds, as on Philly).


## 1. Baseline check (test window) and power calibration

| metric | real | simulated |
|---|---|---|
| busy_gpus_mean | 1,177 | 1,177 |
| busy_gpus_p95 | 1,766 | 1,760 |
| utilisation_mean | 0.514 | 0.514 |
| wait_mean_min | 3.999 | 0.000 |
| wait_p50_min | 0.033 | 0.000 |
| wait_p90_min | 4.900 | 0.000 |
| frac_wait_gt_60s | 0.219 | 0.000 |

- Pool G = 2,288 A100 GPUs. Simulated and real busy GPUs agree (the pool never binds, so simulated waits are 0, as on Philly).
- Calibration: simulated busy share 0.514; one cluster-wide u = (169 - 63) / ((400 - 63) x 0.514) = **0.612** (used: 0.612); simulated mean GPU power 169.1 W against the Seren CDF mean 168.6 W (August 2023 CDF; the test window is July). Facility mean 1,330 kW; all-idle floor 643 kW.
- Host power from the GPU-node server-power CDF (8 GPUs per node): low end p1 = 1,421 W = idle host + 8 x 63 W, so idle host = 917 W and h_idle = 114.6 W per GPU; high end p99 = 5,418 W = idle host + 8 x (400 W + h_busy), so h_busy = (5,418 - 917 - 3,200) / 8 = 162.7 W per GPU. Config: 115 / 163 W. (With p10 / p90 instead: 126 / 80 W; not run.)
- Events: 90 (10 start days x 9 cells), start 17:00-19:30 China time; infeasible (P* below the all-idle floor): 0 .
- Jobs without prediction rows (observed-value fallback), per predictor: {'gbdt': 0, 'last2': 0, 'lightgbm': 0, 'oracle': 0, 'transformer': 0}.

## 2. Per cell

### 2a. Median $/MWh per cell (w_f = 0; n/a = shed below 10% of target)

| policy/predictor | 1h/10% | 1h/20% | 1h/30% | 2h/10% | 2h/20% | 2h/30% | 4h/10% | 4h/20% | 4h/30% |
|---|---|---|---|---|---|---|---|---|---|
| cap | 4,064 | 4,119 | 4,119 | 5,369 | 5,595 | 5,595 | 8,164 | 9,864 | 9,864 |
| delay/oracle | 3,357 | 3,965 | 4,299 | 3,587 | 4,055 | 4,313 | 4,327 | 4,616 | 4,734 |
| burst/oracle | 30,445 | 68,492 | 71,765 | 31,488 | 34,409 | 43,736 | 22,895 | 21,682 | 28,176 |
| mix/oracle | 2,802 | 3,734 | 4,321 | 2,435 | 3,750 | 4,310 | 2,558 | 3,825 | 4,162 |
| mix-p90/oracle | 2,802 | 3,734 | 4,321 | 2,435 | 3,750 | 4,310 | 2,558 | 3,825 | 4,162 |
| mix-nocloud/oracle | 3,438 | 4,053 | 4,442 | 4,124 | 4,489 | 4,567 | 6,263 | 5,522 | 5,434 |
| delay/last2 | 4,448 | 4,507 | 4,495 | 4,396 | 4,551 | 4,613 | 4,860 | 4,954 | 4,827 |
| burst/last2 | 139,038 | 137,761 | 116,830 | 70,308 | 72,006 | 63,697 | 35,710 | 37,948 | 34,056 |
| mix/last2 | 76,624 | 107,885 | 87,132 | 51,222 | 50,552 | 50,825 | 31,600 | 32,544 | 28,302 |
| mix-p90/last2 | 76,624 | 107,885 | 87,132 | 51,222 | 50,552 | 50,825 | 31,600 | 32,544 | 28,302 |
| mix-nocloud/last2 | 4,640 | 4,549 | 4,569 | 4,667 | 4,761 | 4,718 | 5,575 | 5,443 | 4,885 |
| delay/gbdt | 4,209 | 4,395 | 4,516 | 4,298 | 4,422 | 4,578 | 7,149 | 5,218 | 4,893 |
| burst/gbdt | 145,972 | 129,981 | 143,717 | 81,333 | 71,153 | 78,403 | 47,760 | 38,703 | 41,527 |
| mix/gbdt | 33,925 | 27,313 | 12,096 | 25,178 | 16,103 | 11,565 | 15,960 | 17,140 | 15,863 |
| mix-p90/gbdt | 4,593 | 4,690 | 4,643 | 4,597 | 4,482 | 4,604 | 5,702 | 5,591 | 5,031 |
| mix-nocloud/gbdt | 3,691 | 4,335 | 4,519 | 3,908 | 4,458 | 4,598 | 6,381 | 5,105 | 4,872 |
| delay/lightgbm | 4,365 | 4,415 | 4,516 | 4,573 | 4,438 | 4,543 | 7,096 | 4,805 | 4,852 |
| burst/lightgbm | 166,752 | 170,334 | 164,071 | 93,069 | 85,295 | 82,458 | 46,550 | 42,914 | 41,425 |
| mix/lightgbm | 45,819 | 29,599 | 19,185 | 25,314 | 20,812 | 16,553 | 16,579 | 17,267 | 12,867 |
| mix-p90/lightgbm | 4,735 | 4,703 | 4,754 | 4,451 | 4,641 | 4,658 | 5,384 | 5,088 | 5,078 |
| mix-nocloud/lightgbm | 3,816 | 4,255 | 4,534 | 3,839 | 4,475 | 4,554 | 6,205 | 4,811 | 4,880 |
| delay/transformer | 4,188 | 4,353 | 4,436 | 4,135 | 4,365 | 4,468 | 5,047 | 5,430 | 4,812 |
| burst/transformer | 117,509 | 90,289 | 126,171 | 57,876 | 49,908 | 70,169 | 30,723 | 27,706 | 38,903 |
| mix/transformer | 16,386 | 12,572 | 8,305 | 13,735 | 10,002 | 8,998 | 15,373 | 13,180 | 11,441 |
| mix-p90/transformer | 4,158 | 4,323 | 4,547 | 4,275 | 4,533 | 4,576 | 5,841 | 5,062 | 4,765 |
| mix-nocloud/transformer | 3,717 | 3,988 | 4,436 | 3,900 | 4,441 | 4,483 | 6,044 | 5,323 | 4,868 |

### 2b. Median $/MWh per cell, w_f = 1

| policy/predictor | 1h/10% | 1h/20% | 1h/30% | 2h/10% | 2h/20% | 2h/30% | 4h/10% | 4h/20% | 4h/30% |
|---|---|---|---|---|---|---|---|---|---|
| cap | 6,213 | 6,213 | 6,213 | 7,331 | 8,794 | 8,794 | 10,688 | 14,938 | 14,938 |
| delay/oracle | 5,094 | 5,018 | 4,867 | 5,417 | 5,079 | 5,113 | 7,205 | 5,916 | 5,836 |
| burst/oracle | 30,627 | 68,512 | 71,841 | 31,625 | 34,419 | 43,828 | 23,091 | 21,708 | 28,237 |
| mix/oracle | 5,215 | 5,200 | 4,935 | 5,384 | 5,121 | 5,001 | 6,336 | 5,632 | 6,231 |
| mix-p90/oracle | 5,215 | 5,200 | 4,935 | 5,384 | 5,121 | 5,001 | 6,336 | 5,632 | 6,231 |
| mix-nocloud/oracle | 6,299 | 5,624 | 5,137 | 8,744 | 6,536 | 5,527 | 11,964 | 7,734 | 7,477 |
| delay/last2 | 5,312 | 5,401 | 4,996 | 5,066 | 5,618 | 5,153 | 5,665 | 6,900 | 5,623 |
| burst/last2 | 139,082 | 137,845 | 116,886 | 70,329 | 72,176 | 63,807 | 35,721 | 38,080 | 34,211 |
| mix/last2 | 77,088 | 108,256 | 88,383 | 51,432 | 50,660 | 52,015 | 31,784 | 34,016 | 29,506 |
| mix-p90/last2 | 77,088 | 108,256 | 88,383 | 51,432 | 50,660 | 52,015 | 31,784 | 34,016 | 29,506 |
| mix-nocloud/last2 | 5,054 | 5,335 | 4,991 | 5,040 | 5,954 | 5,141 | 6,173 | 7,444 | 5,894 |
| delay/gbdt | 5,272 | 5,349 | 4,982 | 5,586 | 5,615 | 5,095 | 9,845 | 6,361 | 5,587 |
| burst/gbdt | 146,033 | 130,168 | 143,917 | 81,380 | 71,278 | 78,520 | 47,811 | 40,575 | 41,689 |
| mix/gbdt | 34,591 | 28,113 | 12,740 | 25,739 | 16,513 | 12,146 | 16,792 | 18,505 | 16,471 |
| mix-p90/gbdt | 5,530 | 5,746 | 5,263 | 5,256 | 5,088 | 5,091 | 6,894 | 6,643 | 5,497 |
| mix-nocloud/gbdt | 5,052 | 5,215 | 4,994 | 5,013 | 5,353 | 5,115 | 8,020 | 6,182 | 5,585 |
| delay/lightgbm | 5,206 | 5,203 | 4,938 | 5,498 | 5,393 | 5,095 | 9,939 | 6,423 | 5,495 |
| burst/lightgbm | 166,876 | 170,425 | 164,144 | 93,187 | 85,340 | 82,495 | 46,600 | 43,090 | 41,544 |
| mix/lightgbm | 47,460 | 30,398 | 19,893 | 25,787 | 21,058 | 16,805 | 17,251 | 18,591 | 13,387 |
| mix-p90/lightgbm | 5,975 | 5,436 | 5,372 | 5,181 | 5,274 | 5,273 | 7,025 | 6,330 | 5,506 |
| mix-nocloud/lightgbm | 4,834 | 5,280 | 5,012 | 5,308 | 5,530 | 5,115 | 8,964 | 6,454 | 5,623 |
| delay/transformer | 5,145 | 4,946 | 5,034 | 5,399 | 5,050 | 5,093 | 6,932 | 7,006 | 5,527 |
| burst/transformer | 117,626 | 90,300 | 126,195 | 57,988 | 50,029 | 70,263 | 30,818 | 27,826 | 38,959 |
| mix/transformer | 17,974 | 13,246 | 8,798 | 15,122 | 10,820 | 9,556 | 15,639 | 13,440 | 12,074 |
| mix-p90/transformer | 5,034 | 5,106 | 5,023 | 5,079 | 5,191 | 5,014 | 7,959 | 6,331 | 5,404 |
| mix-nocloud/transformer | 4,707 | 4,893 | 4,895 | 4,926 | 5,210 | 5,070 | 8,138 | 7,387 | 5,606 |

### 2c. Median compliance per cell (selected rows)

| policy/predictor | 1h/10% | 1h/20% | 1h/30% | 2h/10% | 2h/20% | 2h/30% | 4h/10% | 4h/20% | 4h/30% |
|---|---|---|---|---|---|---|---|---|---|
| cap | 0.02 | 0.00 | 0.00 | 0.15 | 0.00 | 0.00 | 0.48 | 0.00 | 0.00 |
| delay/transformer | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| burst/transformer | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| mix/transformer | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| mix-p90/transformer | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| mix-nocloud/transformer | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| burst/oracle | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| mix/oracle | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Share of events with compliance >= 0.99, all feasible events: cap 0.01, delay/transformer 1.00, burst/transformer 1.00, burst/oracle 1.00, mix-p90/transformer 1.00, mix-nocloud/transformer 1.00

### 2d. Per-event cost ratios (median over events; 'cheaper' = events where the first is cheaper)

| w_f | predictor | events | mix-p90/delay median | mix-p90 cheaper | sum ratio | mix(p50)/delay median | mix(p50) cheaper | mix-p90/mix-nocloud median | mix-p90 cheaper than nocloud | burst/delay median |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | oracle | all | 0.827 | 64 of 90 | 0.746 | 0.827 | 64 of 90 | 0.743 | 84 of 90 | 7.465 |
| 0 | oracle | 2h/20% | 0.781 | 7 of 10 | 0.674 | 0.781 | 7 of 10 | 0.849 | 9 of 10 | 8.265 |
| 0 | last2 | all | 12.161 | 0 of 90 | 10.428 | 12.161 | 0 of 90 | 12.165 | 0 of 90 | 15.642 |
| 0 | last2 | 2h/20% | 12.860 | 0 of 10 | 12.468 | 12.860 | 0 of 10 | 12.950 | 0 of 10 | 17.675 |
| 0 | gbdt | all | 1.000 | 45 of 90 | 0.970 | 2.813 | 1 of 90 | 1.000 | 45 of 90 | 13.780 |
| 0 | gbdt | 2h/20% | 0.989 | 5 of 10 | 1.054 | 3.287 | 0 of 10 | 0.991 | 6 of 10 | 15.273 |
| 0 | lightgbm | all | 1.000 | 42 of 90 | 0.956 | 3.770 | 2 of 90 | 1.001 | 38 of 90 | 15.349 |
| 0 | lightgbm | 2h/20% | 1.013 | 4 of 10 | 1.140 | 3.716 | 0 of 10 | 1.011 | 4 of 10 | 20.538 |
| 0 | transformer | all | 1.011 | 28 of 90 | 1.222 | 2.149 | 3 of 90 | 1.003 | 36 of 90 | 11.950 |
| 0 | transformer | 2h/20% | 1.039 | 2 of 10 | 1.217 | 2.351 | 0 of 10 | 0.991 | 6 of 10 | 12.115 |
| 1 | oracle | all | 0.963 | 57 of 90 | 0.893 | 0.963 | 57 of 90 | 0.851 | 84 of 90 | 5.115 |
| 1 | oracle | 2h/20% | 0.850 | 7 of 10 | 0.820 | 0.850 | 7 of 10 | 0.884 | 9 of 10 | 5.993 |
| 1 | last2 | all | 8.779 | 0 of 90 | 8.689 | 8.779 | 0 of 90 | 9.501 | 0 of 90 | 12.233 |
| 1 | last2 | 2h/20% | 9.344 | 0 of 10 | 10.178 | 9.344 | 0 of 10 | 9.292 | 0 of 10 | 14.262 |
| 1 | gbdt | all | 0.985 | 54 of 90 | 0.920 | 2.467 | 3 of 90 | 0.990 | 55 of 90 | 11.518 |
| 1 | gbdt | 2h/20% | 0.968 | 6 of 10 | 1.015 | 2.891 | 0 of 10 | 0.981 | 7 of 10 | 12.204 |
| 1 | lightgbm | all | 0.990 | 50 of 90 | 0.899 | 3.235 | 4 of 90 | 0.990 | 54 of 90 | 12.605 |
| 1 | lightgbm | 2h/20% | 0.983 | 6 of 10 | 1.091 | 3.059 | 0 of 10 | 0.987 | 6 of 10 | 16.623 |
| 1 | transformer | all | 1.000 | 40 of 90 | 1.144 | 1.931 | 11 of 90 | 1.000 | 40 of 90 | 9.941 |
| 1 | transformer | 2h/20% | 1.000 | 5 of 10 | 1.153 | 2.109 | 1 of 10 | 0.979 | 6 of 10 | 9.337 |

### 2e. mix-p90 / delay per cell, w_f = 0 (median ratio, cheaper in X of N)

| cell | oracle | last2 | gbdt | lightgbm | transformer | burst/delay (transformer) | burst/delay (oracle) |
|---|---|---|---|---|---|---|---|
| 1h/10% | 0.754 (6/10) | 19.429 (0/10) | 0.925 (5/10) | 0.837 (6/10) | 0.915 (5/10) | 29.6 | 11.5 |
| 1h/20% | 0.868 (7/10) | 23.353 (0/10) | 0.991 (5/10) | 1.003 (4/10) | 1.000 (3/10) | 24.0 | 16.4 |
| 1h/30% | 1.004 (3/10) | 23.027 (0/10) | 1.052 (1/10) | 1.010 (1/10) | 1.004 (2/10) | 25.2 | 19.7 |
| 2h/10% | 0.601 (8/10) | 9.938 (0/10) | 0.820 (6/10) | 0.849 (7/10) | 0.943 (5/10) | 15.2 | 8.4 |
| 2h/20% | 0.781 (7/10) | 12.860 (0/10) | 0.989 (5/10) | 1.013 (4/10) | 1.039 (2/10) | 12.1 | 8.3 |
| 2h/30% | 0.944 (7/10) | 12.516 (0/10) | 1.000 (5/10) | 1.040 (2/10) | 1.004 (3/10) | 13.9 | 11.1 |
| 4h/10% | 0.416 (9/10) | 5.523 (0/10) | 0.860 (6/10) | 0.791 (7/10) | 1.057 (3/10) | 6.1 | 5.1 |
| 4h/20% | 0.759 (9/10) | 7.836 (0/10) | 0.931 (6/10) | 0.964 (7/10) | 1.065 (2/10) | 5.4 | 4.5 |
| 4h/30% | 0.911 (8/10) | 6.305 (0/10) | 0.995 (6/10) | 1.007 (4/10) | 1.015 (3/10) | 6.8 | 5.7 |

### 2f. Cheapest policy per cell by median $/MWh (w_f = 0)

| cell | predictor | cheapest (median $/MWh, w_f 0) | $/MWh |
|---|---|---|---|
| 1h/10% | transformer | mix-nocloud/transformer | 3,717 |
| 1h/10% | oracle | mix-p90/oracle | 2,802 |
| 1h/20% | transformer | mix-nocloud/transformer | 3,988 |
| 1h/20% | oracle | mix-p90/oracle | 3,734 |
| 1h/30% | transformer | cap | 4,119 |
| 1h/30% | oracle | cap | 4,119 |
| 2h/10% | transformer | mix-nocloud/transformer | 3,900 |
| 2h/10% | oracle | mix-p90/oracle | 2,435 |
| 2h/20% | transformer | delay/transformer | 4,365 |
| 2h/20% | oracle | mix-p90/oracle | 3,750 |
| 2h/30% | transformer | delay/transformer | 4,468 |
| 2h/30% | oracle | mix-p90/oracle | 4,310 |
| 4h/10% | transformer | delay/transformer | 5,047 |
| 4h/10% | oracle | mix-p90/oracle | 2,558 |
| 4h/20% | transformer | mix-p90/transformer | 5,062 |
| 4h/20% | oracle | mix-p90/oracle | 3,825 |
| 4h/30% | transformer | mix-p90/transformer | 4,765 |
| 4h/30% | oracle | mix-p90/oracle | 4,162 |

### 2g. Cloud price, default cell (c_delay stays at 2.745 $/GPU-h, as in Philly E2)

| w_f | run | vs | median ratio | cheaper | median $/MWh |
|---|---|---|---|---|---|
| 0 | burst @1.5/transformer | delay/transformer | 6.655 | 0 of 10 | 27,394 |
| 0 | burst @1.5/oracle | delay/oracle | 4.549 | 1 of 10 | 18,939 |
| 0 | mix @1.5/oracle | delay/oracle | 0.730 | 7 of 10 | 3,114 |
| 0 | mix-p90 @1.5/transformer | delay/transformer | 1.010 | 3 of 10 | 4,452 |
| 0 | burst @4.1/transformer | delay/transformer | 18.057 | 0 of 10 | 74,412 |
| 0 | burst @4.1/oracle | delay/oracle | 12.310 | 0 of 10 | 51,246 |
| 0 | mix @4.1/oracle | delay/oracle | 0.836 | 7 of 10 | 3,859 |
| 0 | mix-p90 @4.1/transformer | delay/transformer | 1.020 | 2 of 10 | 4,551 |
| 0 | burst/transformer (at 2.745) | delay/transformer | 12.115 | 0 of 10 | 49,908 |
| 0 | burst/oracle (at 2.745) | delay/oracle | 8.265 | 0 of 10 | 34,409 |
| 0 | mix/oracle (at 2.745) | delay/oracle | 0.781 | 7 of 10 | 3,750 |
| 0 | mix-p90/transformer (at 2.745) | delay/transformer | 1.039 | 2 of 10 | 4,533 |
| 1 | burst @1.5/transformer | delay/transformer | 5.130 | 0 of 10 | 27,515 |
| 1 | burst @1.5/oracle | delay/oracle | 3.298 | 1 of 10 | 18,949 |
| 1 | mix @1.5/oracle | delay/oracle | 0.828 | 8 of 10 | 4,715 |
| 1 | mix-p90 @1.5/transformer | delay/transformer | 0.995 | 5 of 10 | 5,289 |
| 1 | burst @4.1/transformer | delay/transformer | 13.916 | 0 of 10 | 74,533 |
| 1 | burst @4.1/oracle | delay/oracle | 8.927 | 0 of 10 | 51,256 |
| 1 | mix @4.1/oracle | delay/oracle | 0.883 | 6 of 10 | 5,193 |
| 1 | mix-p90 @4.1/transformer | delay/transformer | 1.000 | 3 of 10 | 5,278 |
| 1 | burst/transformer (at 2.745) | delay/transformer | 9.337 | 0 of 10 | 50,029 |
| 1 | burst/oracle (at 2.745) | delay/oracle | 5.993 | 1 of 10 | 34,419 |
| 1 | mix/oracle (at 2.745) | delay/oracle | 0.850 | 7 of 10 | 5,121 |
| 1 | mix-p90/transformer (at 2.745) | delay/transformer | 1.000 | 5 of 10 | 5,191 |

## 3. What gets burst (all events, main settings)

| policy/predictor | events with bursts | burst jobs | median remaining (min) | median GPUs | % Failed (waste) jobs | jobs by type | cloud $ by type |
|---|---|---|---|---|---|---|---|
| burst/gbdt | 90 | 17,461 | 9.0 | 1.0 | 23.9 | Eval 14964, Other 1732, Debug 229, MLLM 223, SFT 159, Pretrain 154 | Pretrain 59%, Other 15%, Eval 14% |
| burst/last2 | 90 | 11,525 | 18.4 | 1.0 | 23.1 | Eval 9803, Other 1118, MLLM 176, SFT 161, Pretrain 155, Debug 112 | Pretrain 61%, Other 13%, Eval 13% |
| burst/lightgbm | 90 | 16,634 | 9.2 | 1.0 | 23.9 | Eval 14265, Other 1622, MLLM 224, Debug 215, Pretrain 158, SFT 150 | Pretrain 59%, Eval 16%, Other 13% |
| burst/oracle | 90 | 23,203 | 6.8 | 1.0 | 27.8 | Eval 20475, Other 1863, Debug 306, MLLM 279, Pretrain 141, SFT 139 | Pretrain 56%, MLLM 22%, Other 8% |
| burst/transformer | 90 | 18,563 | 8.4 | 1.0 | 26.1 | Eval 16577, Other 1260, Debug 248, MLLM 200, SFT 140, Pretrain 138 | Pretrain 56%, MLLM 17%, Other 13% |
| mix-p90/gbdt | 89 | 13,594 | 5.4 | 1.0 | 31.8 | Eval 12653, Other 699, Debug 169, MLLM 47, SFT 16, Pretrain 10 | Eval 76%, Other 10%, Pretrain 7% |
| mix-p90/last2 | 90 | 8,630 | 44.0 | 1.0 | 11.8 | Eval 6939, Other 1038, Debug 195, SFT 176, MLLM 166, Pretrain 116 | Pretrain 60%, Other 13%, Eval 13% |
| mix-p90/lightgbm | 88 | 13,251 | 6.1 | 1.0 | 30.5 | Eval 12269, Other 705, Debug 211, MLLM 38, SFT 18, Pretrain 10 | Eval 84%, Other 7%, Pretrain 4% |
| mix-p90/oracle | 84 | 15,564 | 6.9 | 1.0 | 0.4 | Eval 13993, Other 1104, Debug 249, MLLM 143, Pretrain 47, SFT 28 | Eval 44%, Pretrain 24%, Other 13% |
| mix-p90/transformer | 84 | 17,077 | 5.3 | 1.0 | 34.6 | Eval 16229, Other 639, Debug 169, MLLM 27, SFT 10, Pretrain 3 | Eval 60%, Other 31%, MLLM 7% |
| mix/gbdt | 90 | 16,789 | 7.8 | 1.0 | 24.6 | Eval 14849, Other 1334, Debug 316, MLLM 156, Pretrain 67, SFT 67 | Pretrain 55%, Eval 15%, Other 12% |
| mix/last2 | 90 | 8,630 | 44.0 | 1.0 | 11.8 | Eval 6939, Other 1038, Debug 195, SFT 176, MLLM 166, Pretrain 116 | Pretrain 60%, Other 13%, Eval 13% |
| mix/lightgbm | 90 | 15,586 | 8.0 | 1.0 | 24.9 | Eval 13822, Other 1237, Debug 232, MLLM 153, Pretrain 75, SFT 67 | Pretrain 64%, Eval 12%, Other 9% |
| mix/oracle | 84 | 15,564 | 6.9 | 1.0 | 0.4 | Eval 13993, Other 1104, Debug 249, MLLM 143, Pretrain 47, SFT 28 | Eval 44%, Pretrain 24%, Other 13% |
| mix/transformer | 90 | 19,177 | 7.2 | 1.0 | 26.8 | Eval 17461, Other 1207, Debug 250, MLLM 148, Pretrain 60, SFT 51 | Pretrain 41%, MLLM 21%, Eval 19% |

Burst-capable policies that never burst: none.

Running GPUs at the 10 event starts: median 1,274 (min 695, max 1,785); GPUs held by jobs larger than the cloud limit (1,000 GPUs): median 0, max 0 (in 0 of 10 starts; jobs: 0). Such jobs can be delayed or capped but not burst.

| start (China time) | running_jobs | running_gpus | gpus_above_cloud_limit | base_mean_kw |
|---|---|---|---|---|
| 07-03 19:30 | 610 | 1,700 | 0 | 1,522 |
| 07-04 18:30 | 112 | 1,785 | 0 | 1,679 |
| 07-05 17:30 | 44 | 695 | 0 | 1,053 |
| 07-09 17:00 | 53 | 744 | 0 | 1,065 |
| 07-15 18:30 | 28 | 1,334 | 0 | 1,390 |
| 07-16 17:00 | 37 | 1,353 | 0 | 1,286 |
| 07-19 18:30 | 54 | 1,175 | 0 | 1,331 |
| 07-21 18:30 | 50 | 1,292 | 0 | 1,430 |
| 07-25 19:00 | 128 | 1,256 | 0 | 1,289 |
| 07-26 17:00 | 64 | 984 | 0 | 1,280 |

## 4. Side by side with Philly

Philly values are the paper macros (`paper/numbers.tex`); Acme values computed here the same way.

| quantity | Philly | Acme (Seren) |
|---|---|---|
| Delay-only, default cell, median $/MWh (w_f 0 / 1), transformer | 3,334 / 12,941 | 4,365 / 5,050 |
| Cap-only, default cell, median compliance | 0.00 | 0.00 |
| Burst/delay per event, default cell, median (w_f 0 / 1), transformer | 125 / 30 | 12.1 / 9.3 |
| Burst/delay per event, all feasible events (w_f 0 / 1), transformer | 111 / 23 | 11.9 / 9.9 |
| mix-p90/delay, transformer, median (cheaper in X of N), w_f 0 | 0.945 (66 of 84) | 1.011 (28 of 90) |
| mix-p90/delay, gbdt, w_f 0 | 0.839 (72 of 84) | 1.000 (45 of 90) |
| mix-p90/delay, lightgbm, w_f 0 | 0.870 (71 of 84) | 1.000 (42 of 90) |
| mix-p90/delay, transformer, w_f 1 | 0.941 (75 of 84) | 1.000 (40 of 90) |
| mix-p90/mix-nocloud, transformer, median (cheaper), w_f 0 | 1.00 (38) | 1.003 (36 of 90) |
| Infeasible events (P* below the idle floor) | 6 | 0 |

