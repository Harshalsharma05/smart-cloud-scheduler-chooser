# T2 examples report

- Example table: **331,896 rows** from 109,356 jobs (started, non-censored, `wall_s > 0`, `num_gpus > 0`, submitted before 2017-12-16; 109,428 started non-censored jobs before that date, so 72 excluded).
- `split` is by `submit_time`; jobs submitted in a gap or after the test window get no split (none expected: the bounds are contiguous).

## Rows per split and grid point

| split | e=0 | e=60 | e=600 | e=3600 | e=14400 | e=43200 | total |
|---|---|---|---|---|---|---|---|
| train | 67,199 | 60,629 | 46,219 | 18,792 | 9,258 | 5,620 | 207,717 |
| val | 17,621 | 15,347 | 9,640 | 5,533 | 2,800 | 1,432 | 52,373 |
| test | 24,536 | 20,435 | 12,143 | 8,269 | 3,994 | 2,429 | 71,806 |
| all | 109,356 | 96,411 | 68,002 | 32,594 | 16,052 | 9,481 | 331,896 |

## History

- Rows with empty history: 1.85% overall; train 2.53%, val 1.19%, test 0.37%
- Median history length: 32 (mean 30.0, max 32, cap H = 32)
- Rows with a full history (H = 32): 90.4%

## Notes

- `t_obs = submit_time` at e = 0, `first_start_time + e` otherwise; history = same user's jobs with `end_time <= t_obs`, newest last, at most 32, never the job itself.
- `hist_idx` are row indices into `philly_jobs.parquet`. History jobs are started, non-censored jobs with `wall_s > 0`.
- `submit_hour` / `submit_dow` are from the job's `submit_time` in UTC (the job log is UTC).
- Targets: `remaining_s = wall_s - elapsed_s` (models predict `log1p`), `status`. Floor of 60 s is applied only at export (T2b).
