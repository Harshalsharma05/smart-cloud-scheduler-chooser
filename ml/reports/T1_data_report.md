# T1 data report (Philly job log)

## Counts
- Jobs: 117,325 (README says 117,325)
- Attempts in raw log: 159,351; kept: 152,714
- Users: 325; virtual clusters: 15

## Status
| status | jobs |
|---|---|
| Pass | 84358 |
| Failed | 18633 |
| Killed | 14334 |

## Dropped / special cases
- Never started (no valid attempt): 5,256 (4.48%)
- Censored (still running at trace end): 194 (0.17%)
- Attempts dropped, missing start: 136 (0.09% of raw attempts)
- Attempts dropped, missing end (not the last attempt): 6,501
- Last attempts kept with missing end (censored): 194
- Attempts with end < start (kept as is): 4
- Jobs with >1 valid attempt: 15,711; of those, attempts use different GPU counts in 540 jobs (`num_gpus` = max)

## Timestamps
- `cluster_job_log` times are naive `YYYY-MM-DD HH:MM:SS` strings that are **UTC** (evidence below); used as UTC epoch seconds. Missing = JSON null.
- Missing/unparseable job-log values: 6,831
- `cluster_gpu_util` (and the other util CSVs) carry `PDT`/`PST` labels and must be converted using the label. The DST-change hour (5 Nov 2017) has 660 malformed rows with 16 values instead of 8 in the first 2M rows of `cluster_gpu_util`; Track 2's join must handle them.
- `end_time` and `wall_s` are null for censored jobs (end unknown).

## Timezone evidence 1 (primary): YARN job IDs
Job IDs are `application_<RM start, ms since epoch (UTC)>_<seq>`. For each resource-manager start with >= 1000 jobs, the table compares that UTC start with the submit string of its first (lowest-seq) job. Differences near 0 h mean the strings are UTC.

| cluster_start_utc | jobs | first_seq | first_submit_string | diff_hours |
|---|---|---|---|---|
| 2017-08-30 22:21:16 | 6290 | 129 | 2017-08-31 05:48:35 | 7.46 |
| 2017-09-28 22:41:12 | 39264 | 1 | 2017-09-28 22:41:23 | 0.0 |
| 2017-10-18 22:09:18 | 52922 | 1 | 2017-10-18 22:11:57 | 0.04 |
| 2017-11-24 13:04:44 | 2786 | 1 | 2017-11-24 13:12:16 | 0.13 |
| 2017-11-29 00:43:04 | 3781 | 1 | 2017-11-29 00:44:56 | 0.03 |
| 2017-12-05 02:19:42 | 6699 | 1 | 2017-12-05 02:19:54 | 0.0 |
| 2017-12-11 20:54:20 | 3066 | 1 | 2017-12-11 20:56:05 | 0.03 |
| 2017-12-18 20:03:26 | 2464 | 1 | 2017-12-18 21:29:56 | 1.44 |

- 6 of 8 rows have a first job within 0.13 h of the RM start. The 2017-08-30 row is not comparable: its lowest seq is 129, not 1, so the first jobs are missing from the log. The 2017-12-18 row (1.44 h) is a plain idle gap before the first job; a timezone error would show 7-8 h.

## Distributions (started, non-censored jobs)
- `wall_s` (s): median 1,304, p90 32,204
- `run_s` (s): median 1,218, p90 31,163
- `num_gpus`: median 1, p90 4
- `num_gpus` histogram: 0: 72, 1: 96,855, 2: 2,526, 4: 5,519, 8: 5,739, 12: 12, 16: 857, 20: 1, 24: 41, 32: 212, 48: 2, 64: 28, 128: 11

## Jobs per week (by submit time, UTC, week starting Monday)

| week | jobs |
|---|---|
| 2017-08-07 | 5 |
| 2017-08-14 | 11 |
| 2017-08-21 | 8 |
| 2017-08-28 | 22 |
| 2017-09-04 | 63 |
| 2017-09-11 | 96 |
| 2017-09-18 | 1,133 |
| 2017-09-25 | 7,482 |
| 2017-10-02 | 13,593 |
| 2017-10-09 | 15,926 |
| 2017-10-16 | 18,622 |
| 2017-10-23 | 10,825 |
| 2017-10-30 | 5,783 |
| 2017-11-06 | 9,891 |
| 2017-11-13 | 10,022 |
| 2017-11-20 | 5,675 |
| 2017-11-27 | 5,015 |
| 2017-12-04 | 6,988 |
| 2017-12-11 | 3,701 |
| 2017-12-18 | 2,464 |

## Split check (half-open on submit_time, UTC date bounds from `ml/config.py`)
- train (< 2017-11-01): 69,529
- val [2017-11-01, 2017-11-15): 18,229
- test [2017-11-15, 2017-12-16): 27,022
- after test window (not used): 2,545
- Test-window jobs censored: 7 of 27,022 (0.03%); never started: 2,457

## Machines
`num_gpus` is **derived from the `cluster_gpu_util` width** (most frequent count of non-empty value columns per machine; DST-hour rows with more than 8 values ignored). The raw `cluster_machine_list` GPU count is inconsistent with the util file and with the attempts, so it is not used. `gpu_mem_gb` is kept as listed.

### Machines by derived GPU count and listed GPU memory
| num_gpus | gpu_mem_gb | machines | total_gpus |
|---|---|---|---|
| 2 | 24 | 150 | 300 |
| 4 | 12 | 83 | 332 |
| 4 | 24 | 73 | 292 |
| 8 | 12 | 238 | 1904 |
| 8 | 24 | 8 | 64 |

- Machines: 552; **total GPUs (pool size G): 2,892**. G is a maximum: machines go offline over the trace, so the usable pool is at times smaller.

### Listed vs derived GPU counts (rows = listed count, `num_gpus` = derived)
| listed_gpus | num_gpus | machines |
|---|---|---|
| 2 | 4 | 83 |
| 2 | 8 | 238 |
| 8 | 2 | 150 |
| 8 | 4 | 73 |
| 8 | 8 | 8 |

- The listed count equals the derived count on only 8 of 552 machines.

### Verification by attempts
- For every machine with attempts, max GPU index + 1 in `philly_attempts` <= derived `num_gpus`: **0 violations**.
- Machines verified by attempts: 415; relying on util width alone: 137 (no attempts in the job log: 137).
| num_gpus | machines | verified_by_attempts | util_width_only |
|---|---|---|---|
| 2 | 150 | 148 | 2 |
| 4 | 156 | 24 | 132 |
| 8 | 246 | 243 | 3 |

- 4-GPU machines: 156 in total, 24 verified by attempts, 132 not.

- The pairing of GPU memory with GPU count is **unverified**: the paper describes only 2-GPU and 8-GPU server types, but the util width shows 4-GPU machines too. GPU model is not named in the trace.

## Machine-ID check (attempt `detail.ip` vs machine list `machineId`)
- Distinct attempt machines: 415; found in machine list: **100.00%**
- Examples from attempts: ['m0', 'm1', 'm10']
- Examples from machine list: ['m0', 'm1', 'm10']
- Attempt machines missing from list: []

## Timezone evidence 2 (secondary): GPU utilisation around attempt starts
- First 2,000,000 rows of `cluster_gpu_util` (19 machines; the file is sorted by machine, then time). Its PST/PDT labels were converted exactly.
- Sampled 50 single-machine attempts that run >= 10 min; mean util on the attempt's GPUs 10 min before vs after start:

| time assumption | n | mean util before | mean util after | % attempts with after > before |
|---|---|---|---|---|
| job log as UTC (our choice) | 50 | 7.9 | 24.8 | 74% |
| control: starts shifted by +8 h | 49 | 29.1 | 28.3 | 27% |
- Note: an earlier run that read the strings as `America/Los_Angeles` showed no rise (28.3 to 27.4), and a shift sweep peaked at -7/-8 h, i.e. UTC.

## Notes for Track 2
- Censored jobs have null `end_time` and `wall_s`.
- 76 started jobs have `num_gpus = 0` (attempt without GPU detail); 4 attempts have `end < start`. Jobs with `wall_s <= 0`: 22. T2 excludes jobs with `wall_s <= 0` or `num_gpus = 0`.
- Test-window never-started jobs: 2,457 of 27,022.
- Total GPU pool G = 2,892 (derived from util width; a maximum).
