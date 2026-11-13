"""SIM_PLAN S9: per-job mean GPU utilisation from the raw Philly cluster_gpu_util CSV.

Writes data/processed/philly_job_util.parquet: job_id, gpu_util_mean (fraction 0-1, mean over the job's GPU-minutes
inside [SIM_START, TEST_END)), n_minutes (attempt-minutes with at least one measured value, summed over attempts),
coverage (n_minutes / attempt-minutes inside the window). Only jobs with n_minutes >= 1 are written.

Format notes (T1 report, checked here):
- File is ordered by machine, then minute. One row = time label (PST/PDT), machine, then one value per GPU,
  then a trailing comma. "NA" values are skipped.
- The number of values per machine is taken from the util file itself (first row of the machine). It is the
  opposite of philly_machines.parquet (321 machines listed with 2 GPUs have 8 columns and attempts using gpu7; the 231
  listed with 8 have 2 columns). Reported to the user; philly_machines.parquet is not changed here.
- Time labels are converted with their own label: PDT = UTC-7, PST = UTC-8.
- A row whose value count differs from the machine's GPU count (the 660 rows of the 5 Nov 2017 DST hour have 16)
  is skipped and counted.
- A minute row at t covers [t, t + 60 s). An attempt uses the rows that overlap [start_time, end_time).

Run from backend/:  python -m traces.philly_util [--max-rows 2000000] [--out path]
"""
import argparse
import calendar
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from traces.philly import ROOT, ml_config, utc_epoch

RAW = ROOT / "data" / "raw" / "philly" / "trace-data" / "cluster_gpu_util"
OFFSET_S = {"PDT": 7 * 3600, "PST": 8 * 3600}


def minute_index(label, w0, n_min, cache):
    """Index of the minute in [w0, w0 + n_min * 60) for a 'YYYY-MM-DD HH:MM:SS PDT' label, else -1 (cached)."""
    i = cache.get(label)
    if i is None:
        naive = calendar.timegm(datetime.strptime(label[:19], "%Y-%m-%d %H:%M:%S").timetuple())
        t = naive + OFFSET_S[label[20:23]]
        i = (t - w0) // 60 if w0 <= t < w0 + n_min * 60 else -1
        cache[label] = i
    return i


def load_needed(w0, w1):
    """Attempts overlapping [w0, w1): (job_id, start, end, [(machine, gpu index), ...]); plus the machines table."""
    att = pd.read_parquet(ROOT / "data" / "processed" / "philly_attempts.parquet")
    jobs = pd.read_parquet(ROOT / "data" / "processed" / "philly_jobs.parquet", columns=["end_time"])
    att["end_time"] = att["end_time"].fillna(int(jobs["end_time"].max()))      # unfinished attempt: trace end
    att = att[(att["start_time"] < w1) & (att["end_time"] > w0)]
    out = []
    for r in att.itertuples(index=False):
        gpus = []
        for g in r.gpu_ids.split(";"):
            m, _, k = g.partition(":gpu")
            if k.isdigit():
                gpus.append((m, int(k)))
        out.append((r.job_id, int(r.start_time), int(r.end_time), gpus))
    return out, len(att)


def read_util(path, needed_machines, w0, n_min, max_rows=None):
    """Streams the CSV; returns ({machine: float32 array [n_min, n_gpus]}, stats)."""
    arr, width = {}, {}
    cache, st = {}, dict(rows=0, rows_kept=0, malformed=0, malformed_machines=set(), unknown_label=0, t_min=None, t_max=None)
    t0 = time.perf_counter()
    with open(path, "r", buffering=1 << 22) as f:
        next(f)
        for line in f:
            st["rows"] += 1
            if max_rows and st["rows"] > max_rows:
                st["rows"] -= 1
                break
            if st["rows"] % 10_000_000 == 0:
                print(f"  {st['rows'] / 1e6:.0f}M rows, {time.perf_counter() - t0:.0f} s", flush=True)
            label, mach, rest = line.split(",", 2)
            if mach not in needed_machines:
                continue
            n = rest.count(",")                                   # values per row (every row ends with a comma)
            if mach not in width:
                width[mach] = n
                arr[mach] = np.full((n_min, n), np.nan, dtype=np.float32)
            if n != width[mach]:                                  # DST-hour rows with 16 values (counted over the whole file)
                st["malformed"] += 1
                st["malformed_machines"].add(mach)
                continue
            i = minute_index(label, w0, n_min, cache)
            if i < 0:
                continue
            arr[mach][i] = [np.nan if v == "NA" else float(v) for v in rest.split(",")[:-1]]
            st["rows_kept"] += 1
    st["read_s"] = time.perf_counter() - t0
    return arr, st


def job_means(attempts, arr, w0, w1, n_min):
    """Per job: GPU-minute weighted mean util, attempt-minutes matched and expected."""
    acc = {}                                                    # job -> [sum, count, matched minutes, expected minutes]
    for job_id, s, e, gpus in attempts:
        lo, hi = max(s, w0), min(e, w1)
        i0, i1 = (lo - w0) // 60, max(-(-(hi - w0) // 60), (lo - w0) // 60 + 1)
        i1 = min(i1, n_min)
        tot, cnt, seen = 0.0, 0, np.zeros(max(0, i1 - i0), dtype=bool)
        by_machine = {}
        for m, k in gpus:
            by_machine.setdefault(m, []).append(k)
        for m, ks in by_machine.items():
            if m not in arr:
                continue
            ks = [k for k in ks if k < arr[m].shape[1]]
            if not ks:
                continue
            blk = arr[m][i0:i1][:, ks]
            ok = ~np.isnan(blk)
            tot += float(blk[ok].sum())
            cnt += int(ok.sum())
            seen |= ok.any(axis=1)
        a = acc.setdefault(job_id, [0.0, 0, 0, 0])
        a[0] += tot
        a[1] += cnt
        a[2] += int(seen.sum())
        a[3] += max(1, i1 - i0)
    rows = [(j, a[0] / a[1] / 100.0 if a[1] else np.nan, a[2], a[2] / a[3]) for j, a in acc.items()]
    return pd.DataFrame(rows, columns=["job_id", "gpu_util_mean", "n_minutes", "coverage"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=None, help="read only the first N rows (trial run)")
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "philly_job_util.parquet"))
    args = ap.parse_args()
    mc = ml_config()
    w0, w1 = utc_epoch(mc.SIM_START), utc_epoch(mc.TEST_END)
    n_min = (w1 - w0) // 60
    attempts, n_att = load_needed(w0, w1)
    need = {m for *_, gpus in attempts for m, _ in gpus}
    print(f"window {mc.SIM_START} .. {mc.TEST_END} UTC ({n_min} minutes): {n_att} attempts, {len(need)} machines")
    arr, st = read_util(RAW, need, w0, n_min, args.max_rows)
    seen_m = {m for m, a in arr.items() if not np.isnan(a).all()}
    print(f"rows read {st['rows']:,}, kept {st['rows_kept']:,}, malformed (skipped) {st['malformed']} "
          f"on {len(st['malformed_machines'])} machines, read time {st['read_s']:.0f} s; machines with data in window: {len(seen_m)}")
    widths = pd.Series({m: a.shape[1] for m, a in arr.items()}).value_counts().to_dict()
    print(f"GPU columns per machine in the util file: {widths} machines by width")
    if args.max_rows:                                            # trial run: judge only attempts on machines that were read
        attempts = [(j, s, e, g) for j, s, e, g in attempts if g and all(m in arr for m, _ in g)]
        print(f"trial run: {len(attempts)} attempts lie entirely on the machines read")
    df = job_means(attempts, arr, w0, w1, n_min)
    matched = df[df["n_minutes"] >= 1]
    print(f"jobs with attempts in window: {len(df)}; with >= 1 matched minute: {len(matched)} ({len(matched) / max(1, len(df)):.1%}); "
          f"mean coverage of matched jobs {matched['coverage'].mean():.2f}")
    u = matched["gpu_util_mean"]
    if len(u) == 0:
        return
    print(f"gpu_util_mean (fraction): mean {u.mean():.3f}, p10 {u.quantile(.1):.3f}, p50 {u.quantile(.5):.3f}, p90 {u.quantile(.9):.3f}; "
          f"minute-weighted mean {np.average(u, weights=matched['n_minutes']):.3f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    matched.to_parquet(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
