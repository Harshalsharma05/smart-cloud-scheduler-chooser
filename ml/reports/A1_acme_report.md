# A1: AcmeTrace job characterisation (with Philly alongside)

Source: AcmeTrace job traces (github.com/InternLM/AcmeTrace, CC-BY-4.0; Hu et al., "Characterization of Large Language Model Development in the Datacenter", NSDI 2024). Only the ~109 MB job traces were downloaded; the 80 GB utilisation data on Hugging Face was not.
Philly numbers are computed here from `data/processed/philly_jobs.parquet` (read only). All Acme tables use GPU jobs only (`num_gpus >= 1`).

## 1. Loading decisions

- Files: `trace_seren.csv` (818,327 rows) and `trace_kalos.csv` (62,413 rows). CPU-only jobs (`gpu_num = 0`) dropped from the parquet: Seren 367,737, Kalos 42,506. `data/processed/acme_jobs.parquet`: 470,497 GPU jobs.
- Columns as in `philly_jobs.parquet` (job_id, user, vc, submit_time, first_start_time, end_time, wall_s, num_gpus, status, censored, never_started) plus `cluster`, `type`, `raw_state`, `node_num`. `num_attempts` and `run_s` do not exist in Acme and are omitted.
- `vc` = cluster name: the raw `queue` column is the queue wait in seconds, not a queue name.
- `wall_s` = `end_time - first_start_time` (= the raw `duration`). Jobs with `wall_s <= 0` are kept, as in Philly.
- State map: COMPLETED -> Pass; CANCELLED -> Killed; FAILED, TIMEOUT, NODE_FAIL -> Failed (as requested). **PREEMPTED -> Killed**: the request did not cover it, but 43,838 Seren jobs have it (about 5% of rows). It is an interruption by the scheduler, not a failure of the job, so it goes with Killed; the raw state is kept in `raw_state`. RUNNING (3 Kalos jobs, no end time) gets `status` = null and `censored` = true.
- Duplicate job ids: none. No job starts before it is submitted and none ends before it starts.

## 2. Time zone

- Seren strings end in `+08:00` (China Standard Time, no daylight saving). Kalos strings end in `+00:00` (UTC). Every string is parsed with its own offset into UTC epoch seconds, so both clusters are on one clock.
- Check that Kalos' UTC label is real and not a mislabelled local time: the share of submissions by hour of day (below) peaks at 01-10 UTC for Kalos, which is 09-18 China time, the same working-day shape Seren shows in China time (10-19, low at 07). So both clusters are China-time workloads; "local time" for the 17:00-20:00 analysis means UTC+8.

Share of GPU-job submissions by hour of day (%):

| hour | seren (UTC hour) | seren (China hour) | kalos (UTC hour) | kalos (China hour) |
|---|---|---|---|---|
| 0.0 | 2.6 | 3.4 | 3.9 | 1.3 |
| 1.0 | 4.2 | 3.6 | 8.6 | 1.1 |
| 2.0 | 5.4 | 5.3 | 4.9 | 0.6 |
| 3.0 | 6.4 | 2.0 | 2.5 | 0.5 |
| 4.0 | 3.8 | 1.9 | 2.5 | 0.3 |
| 5.0 | 4.5 | 1.0 | 1.9 | 0.1 |
| 6.0 | 5.7 | 0.8 | 7.5 | 1.9 |
| 7.0 | 6.1 | 1.0 | 7.7 | 1.4 |
| 8.0 | 7.4 | 2.6 | 11.3 | 3.9 |
| 9.0 | 6.7 | 4.2 | 8.6 | 8.6 |
| 10.0 | 5.1 | 5.4 | 5.2 | 4.9 |
| 11.0 | 6.1 | 6.4 | 3.4 | 2.5 |
| 12.0 | 4.6 | 3.8 | 4.9 | 2.5 |
| 13.0 | 4.6 | 4.5 | 9.2 | 1.9 |
| 14.0 | 3.9 | 5.7 | 8.5 | 7.5 |
| 15.0 | 3.9 | 6.1 | 2.0 | 7.7 |
| 16.0 | 3.4 | 7.4 | 1.3 | 11.3 |
| 17.0 | 3.6 | 6.7 | 1.1 | 8.6 |
| 18.0 | 5.3 | 5.1 | 0.6 | 5.2 |
| 19.0 | 2.0 | 6.1 | 0.5 | 3.4 |
| 20.0 | 1.9 | 4.6 | 0.3 | 4.9 |
| 21.0 | 1.0 | 4.6 | 0.1 | 9.2 |
| 22.0 | 0.8 | 3.9 | 1.9 | 8.5 |
| 23.0 | 1.0 | 3.9 | 1.4 | 2.0 |

## 3. GPU jobs per cluster and month (by submit time, UTC)

| month | kalos | seren |
|---|---|---|
| 2023-02 | 0 | 6 |
| 2023-03 | 0 | 3,990 |
| 2023-04 | 0 | 19,804 |
| 2023-05 | 153 | 70,651 |
| 2023-06 | 10,251 | 181,692 |
| 2023-07 | 8,933 | 135,016 |
| 2023-08 | 570 | 39,431 |

Trace ranges (submit time, UTC): Seren 2023-02-28 16:18:22 to 2023-08-16 15:59:42; Kalos 2023-05-17 11:00:58 to 2023-08-16 15:18:02. Philly: Aug-Dec 2017 (117,325 jobs, T1 report).

## 4. State and type shares, by jobs and by GPU-hours

GPU-hours = `wall_s * num_gpus / 3600` (unfinished jobs count 0).

### Seren - mapped status
| status | jobs | gpu_hours | % jobs | % GPU-hours |
|---|---|---|---|---|
| Killed | 89,754 | 3,827,553.06 | 19.92 | 68.03 |
| Pass | 207,770 | 1,195,053.39 | 46.11 | 21.24 |
| Failed | 153,066 | 603,913.41 | 33.97 | 10.73 |

### Seren - raw state
| raw state | jobs | gpu_hours | % jobs | % GPU-hours |
|---|---|---|---|---|
| CANCELLED | 45,942 | 3,780,605.50 | 10.20 | 67.19 |
| COMPLETED | 207,770 | 1,195,053.39 | 46.11 | 21.24 |
| NODE_FAIL | 174 | 340,391.75 | 0.04 | 6.05 |
| FAILED | 152,838 | 235,993.35 | 33.92 | 4.19 |
| PREEMPTED | 43,812 | 46,947.55 | 9.72 | 0.83 |
| TIMEOUT | 54 | 27,528.31 | 0.01 | 0.49 |

### Seren - job type
| type | jobs | gpu_hours | % jobs | % GPU-hours |
|---|---|---|---|---|
| Pretrain | 4,026 | 3,910,073.44 | 0.89 | 69.49 |
| MLLM | 13,288 | 484,702.73 | 2.95 | 8.61 |
| Other | 65,624 | 425,135.82 | 14.56 | 7.56 |
| SFT | 66,971 | 359,117.45 | 14.86 | 6.38 |
| Eval | 292,212 | 351,149.40 | 64.85 | 6.24 |
| Debug | 8,469 | 96,341.03 | 1.88 | 1.71 |

### Kalos - mapped status
| status | jobs | gpu_hours | % jobs | % GPU-hours |
|---|---|---|---|---|
| Killed | 1,240 | 890,899.27 | 6.23 | 61.76 |
| Pass | 10,762 | 459,861.89 | 54.06 | 31.88 |
| Failed | 7,902 | 91,643.39 | 39.69 | 6.35 |
| (unfinished) | 3 | 0.00 | 0.02 | 0.00 |

### Kalos - raw state
| raw state | jobs | gpu_hours | % jobs | % GPU-hours |
|---|---|---|---|---|
| CANCELLED | 1,240 | 890,899.27 | 6.23 | 61.76 |
| COMPLETED | 10,762 | 459,861.89 | 54.06 | 31.88 |
| FAILED | 7,902 | 91,643.39 | 39.69 | 6.35 |
| RUNNING | 3 | 0.00 | 0.02 | 0.00 |

### Kalos - job type
| type | jobs | gpu_hours | % jobs | % GPU-hours |
|---|---|---|---|---|
| Pretrain | 641 | 1,378,184.13 | 3.22 | 95.55 |
| Other | 250 | 36,952.77 | 1.26 | 2.56 |
| Debug | 537 | 20,422.80 | 2.70 | 1.42 |
| SFT | 8 | 3,487.18 | 0.04 | 0.24 |
| Eval | 18,471 | 3,357.67 | 92.79 | 0.23 |

### Philly - status
| status | jobs | gpu_hours | % jobs | % GPU-hours |
|---|---|---|---|---|
| Failed | 17,617 | 1,137,889.37 | 15.73 | 33.78 |
| Pass | 83,151 | 1,116,547.60 | 74.25 | 33.15 |
| Killed | 11,225 | 1,114,168.99 | 10.02 | 33.08 |

## 5. Wall-time percentiles by status (minutes; finished jobs with wall_s > 0)

### Seren
| status | jobs | p10 | p50 | p90 | p99 | mean |
|---|---|---|---|---|---|---|
| all | 444,725 | 0.13 | 2.32 | 22.03 | 428.38 | 26.94 |
| Failed | 148,987 | 0.08 | 0.38 | 4.40 | 57.73 | 9.80 |
| Killed | 88,360 | 0.08 | 0.55 | 44.57 | 1,147.29 | 61.92 |
| Pass | 207,378 | 1.58 | 5.00 | 32.82 | 315.68 | 24.36 |

### Kalos
| status | jobs | p10 | p50 | p90 | p99 | mean |
|---|---|---|---|---|---|---|
| all | 19,902 | 0.15 | 0.80 | 15.88 | 164.44 | 17.35 |
| Failed | 7,900 | 0.10 | 0.23 | 0.60 | 6.52 | 2.21 |
| Killed | 1,240 | 0.72 | 6.63 | 137.64 | 2,657.67 | 135.25 |
| Pass | 10,762 | 0.55 | 2.15 | 20.22 | 90.73 | 14.89 |

### Philly
| status | jobs | p10 | p50 | p90 | p99 | mean |
|---|---|---|---|---|---|---|
| all | 111,793 | 0.93 | 21.78 | 537.17 | 11,143.37 | 551.98 |
| Failed | 17,545 | 0.38 | 21.43 | 2,159.61 | 24,664.48 | 1,192.43 |
| Killed | 11,103 | 2.97 | 81.45 | 3,852.92 | 34,331.27 | 1,901.78 |
| Pass | 83,145 | 1.02 | 19.68 | 213.22 | 4,633.29 | 236.58 |

## 6. Share of GPU-hours used by Failed + Killed jobs

| dataset | GPU-hours | % Failed | % Killed | % Failed+Killed |
|---|---|---|---|---|
| seren | 5,626,519.9 | 10.7 | 68.0 | 78.8 |
| kalos | 1,442,404.6 | 6.4 | 61.8 | 68.1 |
| philly | 3,368,606.0 | 33.8 | 33.1 | 66.9 |

Sensitivity (Seren): without PREEMPTED the Failed+Killed share is 77.9%.

## 7. Running work at 17:00-19:30 local time (30-min slots, 10 random days per trace)

Days drawn with `random.Random(0)` from each trace's fully observed range (Seren 2023-03-02..08-15, Kalos 2023-05-18..08-15, Philly 2017-10-03..12-14), weekends included. Slots: 17:00, 17:30, ..., 19:30 (6 per day, 60 per trace). Local time: China (UTC+8) for Acme; America/Los_Angeles for Philly, the time zone used for the grid events in the paper. A job is running at t if start <= t < end (Philly: first start to end, so the gaps between attempts count as running). "Long" = already running for more than 12 h at t.

| cluster | slots | mean_running_jobs | mean_running_gpus | % running GPUs held by >12 h jobs | % running jobs >12 h |
|---|---|---|---|---|---|
| seren | 60 | 60.6 | 1,562.5 | 51.6 | 28.6 |
| kalos | 60 | 2.4 | 811.0 | 56.6 | 46.2 |
| philly | 60 | 570.7 | 1,849.0 | 87.4 | 85.2 |

Per-day share of running GPUs held by >12 h jobs (%), mean over the 6 slots of each day:

- seren: 2023-03-12 0, 2023-05-07 91, 2023-05-18 14, 2023-06-01 26, 2023-06-08 76, 2023-06-13 73, 2023-06-17 61, 2023-07-02 71, 2023-07-04 56, 2023-07-10 40
- kalos: 2023-05-23 0, 2023-06-20 99, 2023-06-25 0, 2023-07-02 46, 2023-07-06 0, 2023-07-08 0, 2023-07-10 76, 2023-07-18 100, 2023-07-19 100, 2023-07-22 100
- philly: 2017-10-08 87, 2017-11-05 93, 2017-11-10 89, 2017-11-17 83, 2017-11-21 78, 2017-11-23 95, 2017-11-25 97, 2017-12-03 94, 2017-12-04 68, 2017-12-07 91

Acme's share swings between 0% and 100% from day to day because a handful of pretraining jobs decide it (Kalos has only about 2 jobs running at these hours, so its per-day value is one or two jobs). Philly stays high every day.

## 8. Largest jobs (GPUs per job)

### Seren: maximum 1,024 GPUs
| threshold | jobs | % GPU-hours |
|---|---|---|
| >= 8 GPUs | 144,229 | 97.6 |
| >= 64 GPUs | 4,940 | 80.7 |
| >= 128 GPUs | 2,902 | 71.3 |
| >= 256 GPUs | 1,914 | 56.4 |
| >= 512 GPUs | 994 | 35.9 |
| >= 1024 GPUs | 192 | 12.6 |

Five largest jobs:

| job_id | type | raw_state | num_gpus | wall_s |
|---|---|---|---|---|
| 5949007 | Debug | CANCELLED | 1,024 | 649 |
| 5949024 | Debug | CANCELLED | 1,024 | 201 |
| 5949030 | Debug | CANCELLED | 1,024 | 558 |
| 5949042 | Debug | CANCELLED | 1,024 | 119 |
| 5949048 | Debug | CANCELLED | 1,024 | 532 |

### Kalos: maximum 1,024 GPUs
| threshold | jobs | % GPU-hours |
|---|---|---|
| >= 8 GPUs | 2,318 | 99.9 |
| >= 64 GPUs | 1,065 | 98.2 |
| >= 128 GPUs | 928 | 96.6 |
| >= 256 GPUs | 860 | 96.4 |
| >= 512 GPUs | 557 | 62.8 |
| >= 1024 GPUs | 207 | 34.7 |

Five largest jobs:

| job_id | type | raw_state | num_gpus | wall_s |
|---|---|---|---|---|
| dlck0l3icrefzi53 | Pretrain | CANCELLED | 1,024 | 4,158 |
| dlc1x57pc4oa9a6h | Pretrain | COMPLETED | 1,024 | 19,342 |
| dlc12gspacg8vy1v | Pretrain | CANCELLED | 1,024 | 405 |
| dlcqjcyyscpw901x | Pretrain | CANCELLED | 1,024 | 349 |
| dlcirsjyc07qzhgo | Pretrain | CANCELLED | 1,024 | 7,383 |

### Philly: maximum 128 GPUs
| threshold | jobs | % GPU-hours |
|---|---|---|
| >= 8 GPUs | 6,928 | 62.3 |
| >= 64 GPUs | 39 | 0.2 |
| >= 128 GPUs | 11 | 0.1 |
| >= 256 GPUs | 0 | 0.0 |
| >= 512 GPUs | 0 | 0.0 |
| >= 1024 GPUs | 0 | 0.0 |

## 9. GPU power files in the repo

The repo has no raw power samples (those are in the 80 GB Hugging Face data, not downloaded). `data/utilization/util_pkl/gpu_power_{seren,kalos}.pkl` and `server_power.pkl` hold only **CDF curves** (1000 points, power in W against CDF %), used for the paper's Figure on power. The notebook says they use **August 2023** data. They give the distribution of GPU power over all GPU samples, not power conditioned on idle or busy, so an exact idle and a busy mean cannot be computed from the repo. The low end of the curve shows the idle floor and the A100 TDP of 400 W (marked in the notebook) is the busy ceiling.

| series | min | p1 | p10 | p25 | p50 | p75 | p90 | p99 | max | mean (approx, from CDF) |
|---|---|---|---|---|---|---|---|---|---|---|
| Seren GPU power (W) | 15.4 | 60.0 | 63.0 | 66.2 | 96.5 | 290.2 | 395.1 | 436.6 | 558.0 | 168.6 |
| Kalos GPU power (W) | 55.7 | 60.2 | 63.0 | 66.4 | 101.4 | 371.8 | 403.6 | 439.7 | 599.7 | 191.6 |
| GPU node server power (W) | 0.0 | 1,421.0 | 1,512.0 | 1,692.5 | 2,520.0 | 4,284.0 | 4,851.0 | 5,418.3 | 6,554.0 | 2,930.4 |
| CPU node server power (W) | 468.0 | 522.0 | 522.0 | 540.0 | 546.0 | 648.0 | 678.0 | 732.1 | 960.0 | 580.9 |

Reading (approximate, from the CDFs): the lowest quarter of GPU samples sit at 60-66 W, which is the **idle floor** (about 63 W); the median is 97-101 W; the 90th percentile is 395-404 W, close to the A100 TDP of 400 W, which is the **busy level**. The mean over all samples is about 169 W (Seren) and 192 W (Kalos). These are the means over all samples, not idle-only or busy-only means. Philly has no power data at all.

## 10. Headline comparison

|  | GPU jobs | GPU-hours | % GPU-h Failed+Killed | median wall (min) | p99 wall (h) | max GPUs | mean running jobs 17-20h | mean running GPUs 17-20h | % GPUs held by >12h jobs |
|---|---|---|---|---|---|---|---|---|---|
| philly | 111,993 | 3,368,606.0 | 66.9 | 21.8 | 185.7 | 128 | 570.7 | 1,849.0 | 87.4 |
| seren | 450,590 | 5,626,519.9 | 78.8 | 2.3 | 7.1 | 1,024 | 60.6 | 1,562.5 | 51.6 |
| kalos | 19,907 | 1,442,404.6 | 68.1 | 0.8 | 2.7 | 1,024 | 2.4 | 811.0 | 56.6 |

## 11. Proposed time split (not implemented)

GPU jobs per week (submit time, UTC, week starting Monday):

| week | kalos | seren |
|---|---|---|
| 2023-02-27 | 0 | 395 |
| 2023-03-06 | 0 | 563 |
| 2023-03-13 | 0 | 1,228 |
| 2023-03-20 | 0 | 1,016 |
| 2023-03-27 | 0 | 848 |
| 2023-04-03 | 0 | 1,101 |
| 2023-04-10 | 0 | 2,502 |
| 2023-04-17 | 0 | 7,069 |
| 2023-04-24 | 0 | 9,078 |
| 2023-05-01 | 0 | 5,700 |
| 2023-05-08 | 0 | 17,684 |
| 2023-05-15 | 50 | 10,371 |
| 2023-05-22 | 63 | 19,103 |
| 2023-05-29 | 442 | 35,357 |
| 2023-06-05 | 775 | 18,664 |
| 2023-06-12 | 1,131 | 25,552 |
| 2023-06-19 | 1,956 | 47,265 |
| 2023-06-26 | 6,565 | 100,757 |
| 2023-07-03 | 5,386 | 37,252 |
| 2023-07-10 | 1,639 | 9,667 |
| 2023-07-17 | 1,102 | 34,767 |
| 2023-07-24 | 215 | 21,153 |
| 2023-07-31 | 288 | 19,040 |
| 2023-08-07 | 170 | 15,851 |
| 2023-08-14 | 125 | 8,607 |

Proposal (by submit time, China time UTC+8, half-open, the same shape as Philly's split):

- **train**: submit < 2023-06-01  (Seren Mar-May)
- **val**: 2023-06-01 <= submit < 2023-06-15  (2 weeks)
- **test**: 2023-06-15 <= submit < 2023-07-16  (31 days, as long as Philly's test window)
- Not used: submit >= 2023-07-16 (late July and August; the repo's power curves are August data, but they are curves, not time series).
- The simulation would start 7 days before the test window, on 2023-06-08, as in Philly (`SIM_START`), so carry-over and val-period jobs get predictions.

Jobs per split under this proposal:

| cluster | split | jobs | jobs_wall_gt_10min | users | censored |
|---|---|---|---|---|---|
| kalos | train | 153 | 49 | 6 | 0 |
| kalos | val | 1,682 | 637 | 7 | 0 |
| kalos | test | 15,861 | 1,591 | 8 | 0 |
| kalos | after | 2,211 | 356 | 6 | 3 |
| seren | train | 93,900 | 16,263 | 82 | 0 |
| seren | val | 44,171 | 8,196 | 62 | 0 |
| seren | test | 212,190 | 34,045 | 64 | 0 |
| seren | after | 100,329 | 24,528 | 85 | 0 |

Points for the decision (these are not made here):
1. **Seren only for the ML and simulation, Kalos as a second case.** Kalos starts on 2023-05-17, has only about 20,000 GPU jobs, and 96% of its GPU-hours are pretraining. A pooled table would need `cluster` as a feature (there is no `vc`).
2. **The time split is uneven in volume.** June 26-July 2 alone has about 100,000 Seren jobs (89% Eval), so the test window is much larger than train; moving `train_end` later (to 2023-06-15) would balance train and test at the cost of a 2-week val set in July.
3. **Most Acme jobs are very short** (Seren median 2.3 min; 82% of Seren GPU jobs run 600 s or less), so the grid 0, 60, 600, 3600, 14400, 43200 s gives most jobs only 1-2 rows, and the 60 s floor on remaining runtime matters much more than in Philly.
4. `PREEMPTED` (9.7% of Seren GPU jobs) is mapped to Killed here; a model for failure risk would treat it separately or drop it.
