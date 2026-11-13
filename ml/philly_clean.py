"""T1: clean the Philly job log into the shared parquet tables and write the data report.

Run from the repo root:  ml\\.venv\\Scripts\\python ml\\philly_clean.py
"""
import csv
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

TRACE_DIR = config.DATA_RAW / "philly" / "trace-data"
UTIL_ROWS = 2_000_000


# ---------------------------------------------------------------- timestamps
def to_epoch(strings, stats=None):
    """Job-log strings (naive, UTC; None allowed) -> UTC epoch seconds (float, NaN if missing)."""
    dt = pd.to_datetime(pd.Series(strings, dtype="object"), format="%Y-%m-%d %H:%M:%S", errors="coerce")
    present = dt.notna()
    if stats is not None:
        stats["unparseable_or_missing"] = stats.get("unparseable_or_missing", 0) + int((~present).sum())
    secs = (dt - pd.Timestamp(0)).dt.total_seconds()
    return secs.where(present)


def yarn_check(raw_jobs, min_jobs=1000):
    """Primary timezone evidence. Job ids are application_<RM start, ms since epoch (UTC)>_<seq>.
    For each RM start with >= min_jobs jobs: compare its UTC start with the submit string of its first (lowest seq) job."""
    rows = {}
    for j in raw_jobs:
        _, ms, seq = str(j["jobid"]).split("_")
        rows.setdefault(ms, []).append((int(seq), j["submitted_time"]))
    out = []
    for ms, lst in rows.items():
        if len(lst) < min_jobs:
            continue
        seq, sub = min(lst)
        start = pd.Timestamp(int(ms), unit="ms")
        diff_h = (pd.Timestamp(sub) - start).total_seconds() / 3600
        out.append({"cluster_start_utc": str(start.floor("s")), "jobs": len(lst), "first_seq": seq,
                    "first_submit_string": sub, "diff_hours": round(diff_h, 2)})
    return sorted(out, key=lambda r: r["cluster_start_utc"])


# --------------------------------------------------------------------- jobs
def build_tables(raw_jobs):
    """raw_jobs: list of dicts from cluster_job_log -> (jobs_df, attempts_df, stats)."""
    stats = {}
    jobs = pd.DataFrame(
        {
            "job_id": [str(j["jobid"]) for j in raw_jobs],
            "user": [j["user"] for j in raw_jobs],
            "vc": [j["vc"] for j in raw_jobs],
            "status": [j["status"] for j in raw_jobs],
            "sub_str": [j["submitted_time"] for j in raw_jobs],
        }
    )
    jobs["submit_time"] = to_epoch(jobs["sub_str"], stats)

    # flatten attempts in their raw order
    rows = []
    for ji, j in enumerate(raw_jobs):
        n = len(j["attempts"])
        for k, a in enumerate(j["attempts"]):
            det = a.get("detail") or []
            rows.append(
                (
                    ji, k, k == n - 1, a.get("start_time"), a.get("end_time"),
                    ";".join(d["ip"] for d in det),
                    ";".join(f"{d['ip']}:{g}" for d in det for g in d["gpus"]),
                    sum(len(d["gpus"]) for d in det),
                )
            )
    att = pd.DataFrame(rows, columns=["ji", "k", "is_last", "s", "e", "machine", "gpu_ids", "ngpu"])
    att["start_time"] = to_epoch(att["s"], stats)
    att["end_time"] = to_epoch(att["e"], stats)
    stats["raw_attempts"] = len(att)
    stats["raw_jobs"] = len(jobs)

    no_start = att["start_time"].isna()
    no_end = att["end_time"].isna()
    censored_att = ~no_start & no_end & att["is_last"]  # still running at trace end
    keep = ~no_start & (~no_end | censored_att)
    stats["attempts_dropped_missing_start"] = int(no_start.sum())
    stats["attempts_dropped_missing_end"] = int((~no_start & no_end & ~att["is_last"]).sum())
    stats["attempts_censored_last_end_missing"] = int(censored_att.sum())
    stats["attempts_end_before_start"] = int((att["end_time"] < att["start_time"]).sum())
    att = att[keep].copy()
    stats["attempts_kept"] = len(att)
    att["attempt"] = att.groupby("ji").cumcount()
    att["job_id"] = jobs["job_id"].values[att["ji"].values]

    g = att.groupby("ji")
    agg = pd.DataFrame(
        {
            "first_start_time": g["start_time"].min(),
            "last_end": g["end_time"].max(),
            "num_attempts": g.size(),
            "num_gpus": g["ngpu"].max(),
        }
    )
    att["dur"] = att["end_time"] - att["start_time"]
    agg["run_s"] = att.groupby("ji")["dur"].sum()
    agg["censored"] = att[censored_att[keep].values].groupby("ji").size().reindex(agg.index).fillna(0) > 0
    stats["jobs_attempts_differ_in_gpus"] = int((g["ngpu"].nunique() > 1).sum())
    stats["jobs_with_multiple_attempts"] = int((agg["num_attempts"] > 1).sum())

    jobs = jobs.join(agg)
    jobs["never_started"] = jobs["first_start_time"].isna()
    jobs["censored"] = jobs["censored"].fillna(False).astype(bool)
    jobs["end_time"] = jobs["last_end"].where(~jobs["censored"])  # unknown for censored jobs
    jobs["wall_s"] = jobs["end_time"] - jobs["first_start_time"]
    jobs["num_attempts"] = jobs["num_attempts"].fillna(0).astype(int)
    jobs["num_gpus"] = jobs["num_gpus"].fillna(0).astype(int)
    for c in ["submit_time", "first_start_time", "end_time", "wall_s", "run_s"]:
        jobs[c] = jobs[c].round().astype("Int64")
    cols = ["job_id", "user", "vc", "submit_time", "first_start_time", "end_time", "wall_s", "run_s",
            "num_gpus", "num_attempts", "status", "censored", "never_started"]
    jobs_out = jobs[cols].reset_index(drop=True)

    att_out = att[["job_id", "attempt", "start_time", "end_time", "machine", "gpu_ids"]].copy()
    att_out["start_time"] = att_out["start_time"].round().astype("Int64")
    att_out["end_time"] = att_out["end_time"].round().astype("Int64")
    return jobs_out, att_out.reset_index(drop=True), stats


def util_widths(path):
    """Per machine: most frequent number of non-empty value fields in cluster_gpu_util (the 8 value columns).

    "NA" counts as present, empty does not. Rows with more than 8 values (DST-hour glitch) are ignored.
    Returns (machine_id -> width, number of malformed rows ignored)."""
    counts, bad = {}, 0
    with open(path, newline="") as f:
        rd = csv.reader(f)
        next(rd)
        for row in rd:
            if len(row) < 3:
                continue
            if any(v != "" for v in row[10:]):
                bad += 1
                continue
            w = sum(v != "" for v in row[2:10])
            c = counts.setdefault(row[1].strip(), {})
            c[w] = c.get(w, 0) + 1
    return {m: max(c, key=c.get) for m, c in counts.items()}, bad


def load_machines(path, widths):
    """Machine table. num_gpus comes from the util-file width (the raw count column is inconsistent);
    gpu_mem_gb is kept as listed. Also returns the listed counts for the report."""
    m = pd.read_csv(path, skipinitialspace=True)
    m.columns = ["machine_id", "num_gpus", "gpu_mem_gb"]
    m["machine_id"] = m["machine_id"].astype(str).str.strip()
    m["listed_gpus"] = m["num_gpus"].astype(int)
    m["num_gpus"] = m["machine_id"].map(widths)
    missing = m[m["num_gpus"].isna()]["machine_id"].tolist()
    if missing:
        raise SystemExit(f"{len(missing)} machines have no util rows, e.g. {missing[:3]}; stopping")
    m["num_gpus"] = m["num_gpus"].astype(int)
    m["gpu_mem_gb"] = m["gpu_mem_gb"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    return m


def attempt_gpu_max(attempts):
    """machine_id -> highest GPU index + 1 seen in attempts (gpu_ids are 'machine:gpuN' pairs)."""
    pairs = pd.Series(";".join(attempts["gpu_ids"].dropna()).split(";"))
    pairs = pairs[pairs != ""].str.extract(r"^(.+):gpu(\d+)$").dropna()
    pairs[1] = pairs[1].astype(int) + 1
    return pairs.groupby(0)[1].max().to_dict()


def check_gpu_counts(machines, attempts):
    """Every machine's attempt-seen GPU count must be <= derived num_gpus. Returns a stats dict."""
    seen = attempt_gpu_max(attempts)
    derived = dict(zip(machines["machine_id"], machines["num_gpus"]))
    viol = [m for m, k in seen.items() if k > derived.get(m, 0)]
    return {"violations": len(viol), "violation_examples": viol[:3], "verified_by_attempts": len(seen),
            "util_width_only": len(machines) - len(seen), "seen": seen}


# ------------------------------------------------------------- sanity checks
def machine_id_check(attempts, machines):
    used = set(";".join(attempts["machine"].dropna()).split(";")) - {""}
    known = set(machines["machine_id"])
    return {
        "n_attempt_machines": len(used),
        "pct_found": 100.0 * len(used & known) / max(len(used), 1),
        "attempt_examples": sorted(used)[:3],
        "list_examples": sorted(known)[:3],
        "missing_examples": sorted(used - known)[:3],
    }


def timezone_check(attempts, n_attempts=50, window=600):
    """Compare GPU util in the 10 min before vs after attempt starts, using the util CSV's own PST/PDT stamps."""
    # a few rows in the DST-change hour have 16 values instead of 8; keep only the first 8
    names = ["time", "machine"] + [f"gpu{i}" for i in range(8)]
    rows = []
    with open(TRACE_DIR / "cluster_gpu_util", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in itertools.islice(reader, UTIL_ROWS):
            row = row[:10] + [""] * (10 - len(row))
            rows.append(row)
    u = pd.DataFrame(rows, columns=names)
    for c in names[2:]:
        u[c] = pd.to_numeric(u[c], errors="coerce")  # "NA" and blanks -> NaN
    t = pd.to_datetime(u["time"].str[:19]).astype("datetime64[s]")
    off = np.where(u["time"].str.endswith("PDT"), 7, 8) * 3600
    u["t"] = (t - pd.Timestamp(0)).dt.total_seconds().astype("int64") + off
    series = {}
    for m, grp in u.groupby("machine"):
        grp = grp.sort_values("t")
        series[m] = (grp["t"].to_numpy(), {c: grp[c].to_numpy(dtype=float) for c in names[2:10]})
    t_lo, t_hi = u["t"].min(), u["t"].max()

    cand = attempts[attempts["start_time"].notna() & attempts["end_time"].notna()].copy()
    cand = cand[~cand["machine"].str.contains(";") & cand["machine"].isin(series.keys())]
    cand = cand[(cand["end_time"] - cand["start_time"] >= window)
                & (cand["start_time"] >= t_lo + window) & (cand["start_time"] <= t_hi - window)]
    rng = random.Random(config.SEED)
    idx = list(cand.index)
    rng.shuffle(idx)
    cand = cand.loc[idx[:n_attempts]]

    def mean_util(m, gpus, a, b):
        ts, cols = series[m]
        lo, hi = np.searchsorted(ts, [a, b])
        vals = [cols[g][lo:hi] for g in gpus if g in cols]
        vals = np.concatenate(vals) if vals else np.array([])
        return np.nanmean(vals) if np.isfinite(vals).any() else np.nan

    out = {"n_candidates": len(cand), "rows_read": len(u), "machines_in_util_sample": len(series)}
    for label, shift in [("utc", 0), ("control_plus8h", 8 * 3600)]:
        diffs = []
        for r in cand.itertuples():
            gpus = [x.split(":")[1] for x in r.gpu_ids.split(";") if x]
            s = r.start_time + shift
            b, a = mean_util(r.machine, gpus, s - window, s), mean_util(r.machine, gpus, s, s + window)
            if np.isfinite(a) and np.isfinite(b):
                diffs.append((b, a))
        d = np.array(diffs)
        out[label] = {
            "n": len(d),
            "mean_before": float(d[:, 0].mean()) if len(d) else None,
            "mean_after": float(d[:, 1].mean()) if len(d) else None,
            "pct_after_gt_before": float(100 * (d[:, 1] > d[:, 0]).mean()) if len(d) else None,
        }
    return out


# -------------------------------------------------------------------- report
def md_table(df):
    """Tiny DataFrame -> markdown table (avoids the tabulate dependency)."""
    cols = [df.index.name or ""] * (df.index.name is not None) + list(df.columns)
    rows = [([str(i)] if df.index.name is not None else []) + [str(v) for v in r] for i, r in zip(df.index, df.itertuples(index=False))]
    return "\n".join(["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(r) + " |" for r in rows])


def pctl(s):
    s = s.dropna().astype(float)
    return f"median {s.median():,.0f}, p90 {s.quantile(0.9):,.0f}"


def utc_epoch(date_str):
    return int(pd.Timestamp(date_str, tz="UTC").timestamp())


def machine_report(machines, gc):
    """Report lines for the machines section (derived vs listed GPU counts)."""
    m = machines.copy()
    m["verified"] = m["machine_id"].isin(gc["seen"])
    g = m.groupby(["num_gpus", "gpu_mem_gb"]).agg(machines=("machine_id", "size")).reset_index()
    g["total_gpus"] = g["machines"] * g["num_gpus"]
    lc = m.groupby(["listed_gpus", "num_gpus"]).agg(machines=("machine_id", "size")).reset_index()
    by = m.groupby("num_gpus").agg(machines=("machine_id", "size"), verified_by_attempts=("verified", "sum")).reset_index()
    by["util_width_only"] = by["machines"] - by["verified_by_attempts"]
    n_match = int((m["listed_gpus"] == m["num_gpus"]).sum())
    noatt = m[~m["verified"]]
    four = by[by["num_gpus"] == 4].iloc[0]
    return ["## Machines",
            "`num_gpus` is **derived from the `cluster_gpu_util` width** (most frequent count of non-empty value columns per machine; "
            "DST-hour rows with more than 8 values ignored). The raw `cluster_machine_list` GPU count is inconsistent with the util file and with the attempts, so it is not used. "
            "`gpu_mem_gb` is kept as listed.", "",
            "### Machines by derived GPU count and listed GPU memory", md_table(g), "",
            f"- Machines: {len(m)}; **total GPUs (pool size G): {int(m['num_gpus'].sum()):,}**. "
            "G is a maximum: machines go offline over the trace, so the usable pool is at times smaller.", "",
            "### Listed vs derived GPU counts (rows = listed count, `num_gpus` = derived)", md_table(lc), "",
            f"- The listed count equals the derived count on only {n_match} of {len(m)} machines.", "",
            "### Verification by attempts",
            f"- For every machine with attempts, max GPU index + 1 in `philly_attempts` <= derived `num_gpus`: "
            f"**{gc['violations']} violations**.",
            f"- Machines verified by attempts: {gc['verified_by_attempts']}; relying on util width alone: {gc['util_width_only']} "
            f"(no attempts in the job log: {len(noatt)}).",
            md_table(by), "",
            f"- 4-GPU machines: {int(four['machines'])} in total, {int(four['verified_by_attempts'])} verified by attempts, {int(four['util_width_only'])} not.", "",
            "- The pairing of GPU memory with GPU count is **unverified**: the paper describes only 2-GPU and 8-GPU server types, "
            "but the util width shows 4-GPU machines too. GPU model is not named in the trace.", ""]


def write_report(jobs, attempts, machines, stats, id_check, yarn, util_check, gpu_check, path):
    n = len(jobs)
    L = ["# T1 data report (Philly job log)", ""]
    L += ["## Counts", f"- Jobs: {n:,} (README says 117,325)",
          f"- Attempts in raw log: {stats['raw_attempts']:,}; kept: {stats['attempts_kept']:,}",
          f"- Users: {jobs['user'].nunique():,}; virtual clusters: {jobs['vc'].nunique()}", ""]
    L += ["## Status", md_table(jobs["status"].value_counts().rename_axis("status").to_frame("jobs")), ""]
    L += ["## Dropped / special cases",
          f"- Never started (no valid attempt): {jobs['never_started'].sum():,} ({100*jobs['never_started'].mean():.2f}%)",
          f"- Censored (still running at trace end): {jobs['censored'].sum():,} ({100*jobs['censored'].mean():.2f}%)",
          f"- Attempts dropped, missing start: {stats['attempts_dropped_missing_start']:,} "
          f"({100*stats['attempts_dropped_missing_start']/stats['raw_attempts']:.2f}% of raw attempts)",
          f"- Attempts dropped, missing end (not the last attempt): {stats['attempts_dropped_missing_end']:,}",
          f"- Last attempts kept with missing end (censored): {stats['attempts_censored_last_end_missing']:,}",
          f"- Attempts with end < start (kept as is): {stats['attempts_end_before_start']:,}",
          f"- Jobs with >1 valid attempt: {stats['jobs_with_multiple_attempts']:,}; of those, attempts use "
          f"different GPU counts in {stats['jobs_attempts_differ_in_gpus']:,} jobs (`num_gpus` = max)", ""]
    L += ["## Timestamps",
          "- `cluster_job_log` times are naive `YYYY-MM-DD HH:MM:SS` strings that are **UTC** (evidence below); used as UTC epoch seconds. Missing = JSON null.",
          f"- Missing/unparseable job-log values: {stats['unparseable_or_missing']:,}",
          "- `cluster_gpu_util` (and the other util CSVs) carry `PDT`/`PST` labels and must be converted using the label. "
          "The DST-change hour (5 Nov 2017) has 660 malformed rows with 16 values instead of 8 in the first 2M rows of `cluster_gpu_util`; Track 2's join must handle them.",
          "- `end_time` and `wall_s` are null for censored jobs (end unknown).", ""]
    L += ["## Timezone evidence 1 (primary): YARN job IDs",
          "Job IDs are `application_<RM start, ms since epoch (UTC)>_<seq>`. For each resource-manager start with >= 1000 jobs, "
          "the table compares that UTC start with the submit string of its first (lowest-seq) job. Differences near 0 h mean the strings are UTC.", "",
          md_table(pd.DataFrame(yarn)), "",
          "- 6 of 8 rows have a first job within 0.13 h of the RM start. The 2017-08-30 row is not comparable: its lowest seq is 129, not 1, "
          "so the first jobs are missing from the log. The 2017-12-18 row (1.44 h) is a plain idle gap before the first job; a timezone error would show 7-8 h.", ""]
    started = jobs[~jobs["never_started"] & ~jobs["censored"]]
    L += ["## Distributions (started, non-censored jobs)",
          f"- `wall_s` (s): {pctl(started['wall_s'])}",
          f"- `run_s` (s): {pctl(started['run_s'])}",
          f"- `num_gpus`: {pctl(started['num_gpus'])}",
          "- `num_gpus` histogram: " + ", ".join(f"{k}: {v:,}" for k, v in started['num_gpus'].value_counts().sort_index().items()), ""]
    wk = pd.to_datetime(jobs["submit_time"].astype(float), unit="s", utc=True).dt.tz_localize(None).dt.to_period("W-SUN").dt.start_time
    wc = wk.value_counts().sort_index()
    L += ["## Jobs per week (by submit time, UTC, week starting Monday)", "", "| week | jobs |", "|---|---|"]
    L += [f"| {w.date()} | {c:,} |" for w, c in wc.items()]
    L += [""]
    st = jobs["submit_time"].astype(float)
    def cnt(a, b):
        return int(((st >= utc_epoch(a)) & (st < utc_epoch(b))).sum())
    L += ["## Split check (half-open on submit_time, UTC date bounds from `ml/config.py`)",
          f"- train (< {config.TRAIN_END}): {int((st < utc_epoch(config.TRAIN_END)).sum()):,}",
          f"- val [{config.VAL_START}, {config.VAL_END}): {cnt(config.VAL_START, config.VAL_END):,}",
          f"- test [{config.TEST_START}, {config.TEST_END}): {cnt(config.TEST_START, config.TEST_END):,}",
          f"- after test window (not used): {int((st >= utc_epoch(config.TEST_END)).sum()):,}"]
    tw = jobs[(st >= utc_epoch(config.TEST_START)) & (st < utc_epoch(config.TEST_END))]
    L += [f"- Test-window jobs censored: {int(tw['censored'].sum()):,} of {len(tw):,} ({100*tw['censored'].mean():.2f}%); "
          f"never started: {int(tw['never_started'].sum()):,}", ""]
    L += machine_report(machines, gpu_check)
    L += ["## Machine-ID check (attempt `detail.ip` vs machine list `machineId`)",
          f"- Distinct attempt machines: {id_check['n_attempt_machines']}; found in machine list: **{id_check['pct_found']:.2f}%**",
          f"- Examples from attempts: {id_check['attempt_examples']}",
          f"- Examples from machine list: {id_check['list_examples']}",
          f"- Attempt machines missing from list: {id_check['missing_examples']}", ""]
    L += ["## Timezone evidence 2 (secondary): GPU utilisation around attempt starts",
          f"- First {util_check['rows_read']:,} rows of `cluster_gpu_util` ({util_check['machines_in_util_sample']} machines; the file is sorted by machine, then time). "
          "Its PST/PDT labels were converted exactly.",
          f"- Sampled {util_check['n_candidates']} single-machine attempts that run >= 10 min; mean util on the attempt's GPUs 10 min before vs after start:",
          "", "| time assumption | n | mean util before | mean util after | % attempts with after > before |", "|---|---|---|---|---|"]
    for k, lab in [("utc", "job log as UTC (our choice)"), ("control_plus8h", "control: starts shifted by +8 h")]:
        r = util_check[k]
        L.append(f"| {lab} | {r['n']} | {r['mean_before']:.1f} | {r['mean_after']:.1f} | {r['pct_after_gt_before']:.0f}% |"
                 if r["n"] else f"| {lab} | 0 | - | - | - |")
    L += ["- Note: an earlier run that read the strings as `America/Los_Angeles` showed no rise (28.3 to 27.4), and a shift sweep peaked at -7/-8 h, i.e. UTC.", ""]
    started0 = jobs[~jobs["never_started"]]
    L += ["## Notes for Track 2",
          "- Censored jobs have null `end_time` and `wall_s`.",
          f"- {int((started0['num_gpus'] == 0).sum())} started jobs have `num_gpus = 0` (attempt without GPU detail); "
          f"{stats['attempts_end_before_start']} attempts have `end < start`. "
          f"Jobs with `wall_s <= 0`: {int((started0['wall_s'].dropna() <= 0).sum())}. T2 excludes jobs with `wall_s <= 0` or `num_gpus = 0`.",
          f"- Test-window never-started jobs: {int(tw['never_started'].sum()):,} of {len(tw):,}.",
          f"- Total GPU pool G = {int(machines['num_gpus'].sum()):,} (derived from util width; a maximum).", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


def main():
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    raw = json.load(open(TRACE_DIR / "cluster_job_log"))
    print("raw timestamp examples:", [(raw[0]["submitted_time"], raw[0]["attempts"][0]["start_time"],
                                       raw[0]["attempts"][0]["end_time"])],
          "| missing end example:", next((a["end_time"] for j in raw for a in j["attempts"] if a["end_time"] is None), None))
    jobs, attempts, stats = build_tables(raw)
    # to_epoch stats were accumulated inside build_tables
    widths, bad_rows = util_widths(TRACE_DIR / "cluster_gpu_util")
    print(f"util widths for {len(widths)} machines; malformed rows ignored: {bad_rows:,}")
    machines = load_machines(TRACE_DIR / "cluster_machine_list", widths)
    jobs.to_parquet(config.DATA_PROCESSED / "philly_jobs.parquet", index=False)
    attempts.to_parquet(config.DATA_PROCESSED / "philly_attempts.parquet", index=False)
    machines[["machine_id", "num_gpus", "gpu_mem_gb"]].to_parquet(config.DATA_PROCESSED / "philly_machines.parquet", index=False)
    print(f"jobs {len(jobs):,}  attempts {len(attempts):,}  machines {len(machines)}")
    gpu_check = check_gpu_counts(machines, attempts)
    print({k: v for k, v in gpu_check.items() if k != "seen"}, f"| G = {int(machines['num_gpus'].sum()):,}")
    if gpu_check["violations"] > 0:
        print("GPU COUNT VIOLATIONS - stopping.")
        sys.exit(1)
    id_check = machine_id_check(attempts, machines)
    print("machine-id check:", id_check)
    if id_check["pct_found"] < 99.0:
        print("MACHINE IDS DO NOT MATCH - stopping (no mapping invented).")
        sys.exit(1)
    yarn = yarn_check(raw)
    print("YARN check:", *yarn, sep="\n  ")
    util_check = timezone_check(attempts)
    print("timezone check:", util_check)
    write_report(jobs, attempts, machines, stats, id_check, yarn, util_check, gpu_check,
                 config.REPORTS / "T1_data_report.md")
    print("wrote", config.REPORTS / "T1_data_report.md")


if __name__ == "__main__":
    main()
