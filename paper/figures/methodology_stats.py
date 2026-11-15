"""W6: setup numbers for Section 5 that come from the trace loader (no simulation): job counts by kind, carry-over jobs,
censored jobs, jobs without prediction rows, and the event dates. Writes results/methodology/stats.csv (name, value, note).

Run from the repo root: python paper/figures/methodology_stats.py   (about 15 s; needs data/processed and predictions/)
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import pandas as pd  # noqa: E402

from flex.grid import to_local  # noqa: E402
from flex.models import load_config  # noqa: E402
from traces.philly import load_philly  # noqa: E402


def main():
    cfg = load_config()
    L = load_philly(cfg)
    r = L["report"]
    ids = set(pd.read_csv(ROOT / "predictions" / "transformer.csv", usecols=["job_id"], dtype=str)["job_id"])
    no_rows = [j for j in L["jobs"] if str(j.job_id) not in ids]
    epoch0 = L["epoch0"]
    starts = sorted({e.t_start for e in L["events"]})
    local = [to_local(epoch0 + t) for t in starts]
    rows = [
        ("sim_jobs", len(L["jobs"]), "jobs in the simulation (warm-up + test + carry-over pseudo-jobs)"),
        ("warmup_jobs", r["warmup_jobs"], "jobs submitted in the 7-day warm-up before the test window"),
        ("carry_jobs", r["carry_jobs"], "jobs already running at the simulation start (enter at t=0 with their elapsed time)"),
        ("carry_gpus", r["carry_gpus"], "GPUs held by carry-over jobs"),
        ("censored_jobs", r["censored_kept"], "jobs with no observed end in the log (kept, status treated as Pass)"),
        ("no_pred_jobs", len(no_rows), "simulated jobs without any prediction row (fall back to true values)"),
        ("no_pred_carry_jobs", sum(j.carry for j in no_rows), "of which carry-over jobs"),
        ("no_pred_gpus", sum(j.gpus for j in no_rows), "GPUs of jobs without prediction rows"),
        ("event_days", len(starts), "distinct event start times (one per local day), shared by all cells"),
        ("event_first_date", local[0].strftime("%Y-%m-%d"), "first event day (America/Los_Angeles)"),
        ("event_last_date", local[-1].strftime("%Y-%m-%d"), "last event day (America/Los_Angeles)"),
        ("event_first_start_hhmm", min(t.strftime("%H:%M") for t in local), "earliest local start time"),
        ("event_last_start_hhmm", max(t.strftime("%H:%M") for t in local), "latest local start time"),
    ]
    out = ROOT / "results" / "methodology"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "value", "note"])
        w.writerows(rows)
    for n, v, _ in rows:
        print(f"{n:24s} {v}")
    print("wrote", (out / "stats.csv").relative_to(ROOT))


if __name__ == "__main__":
    main()
