"""Step 36 (W11): trace facts for the paper's Acme sentences, written to results/acme_motivation/stats.csv (read by
paper/figures/make_numbers.py). Run from backend/: python -m traces.acme_motivation

The running-work shares use the A1 method (ml/acme_load.py, reimplemented here, not imported): 10 days drawn with
random.Random(0) from each trace's fully observed range, 30-min slots 17:00-19:30 local time (China for Acme,
America/Los_Angeles for Philly); a job runs at t if start <= t < end; "long" = running for more than 12 h at t.
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd

from flex.models import load_config
from traces.acme import acme_config, load_acme

ROOT = Path(__file__).resolve().parents[2]
LONG_S = 12 * 3600
RANGES = {"seren": ("2023-03-02", "2023-08-15", "Asia/Shanghai"), "kalos": ("2023-05-18", "2023-08-15", "Asia/Shanghai"),
          "philly": ("2017-10-03", "2017-12-14", "America/Los_Angeles")}


def running_at_slots(df, first, last, tz, seed=0, n=10):
    days = sorted(random.Random(seed).sample(list(pd.date_range(first, last, freq="D").date), n))
    start = df["first_start_time"].astype("float64").to_numpy()
    end = df["end_time"].astype("float64").fillna(np.inf).to_numpy()
    gpus = df["num_gpus"].to_numpy().astype("float64")
    jobs = run_gpus = long_gpus = 0.0
    for d in days:
        for k in range(6):
            ts = (pd.Timestamp(d, tz=tz) + pd.Timedelta(hours=17, minutes=30 * k)).timestamp()
            run = (start <= ts) & (end > ts)
            jobs += run.sum()
            run_gpus += gpus[run].sum()
            long_gpus += gpus[run & (ts - start > LONG_S)].sum()
    return jobs / (6 * n), long_gpus / run_gpus


def wall_median_min(df):
    f = df[~df["censored"] & (df["wall_s"].astype("float64") > 0)]
    return float(f["wall_s"].astype("float64").median() / 60)


def main():
    acme = pd.read_parquet(ROOT / "data" / "processed" / "acme_jobs.parquet")
    acme = acme[~acme["never_started"] & (acme["num_gpus"] >= 1)]
    ph = pd.read_parquet(ROOT / "data" / "processed" / "philly_jobs.parquet")
    ph = ph[~ph["never_started"] & (ph["num_gpus"] >= 1)]
    sets = {"seren": acme[acme["cluster"] == "seren"], "kalos": acme[acme["cluster"] == "kalos"], "philly": ph}
    S = {}
    for name, df in sets.items():
        mean_jobs, long_share = running_at_slots(df, *RANGES[name])
        S[f"{name}_evening_running_jobs_mean"] = mean_jobs
        S[f"{name}_evening_long_gpu_share_pct"] = 100 * long_share
        S[f"{name}_wall_median_min"] = wall_median_min(df)
    s = sets["seren"]
    gh = s["wall_s"].astype("float64").fillna(0).clip(lower=0) * s["num_gpus"] / 3600
    S["seren_gpuh_share_ge256_pct"] = 100 * gh[s["num_gpus"] >= 256].sum() / gh.sum()
    S["seren_max_gpus"] = int(s["num_gpus"].max())
    L = load_acme(acme_config(load_config()))
    r = L["report"]
    S.update(sim_jobs=r["jobs"], test_jobs=r["test_jobs"], warmup_jobs=r["warmup_jobs"], carry_jobs=r["carry_jobs"],
             carry_gpus=r["carry_gpus"], wall_nonpositive_dropped=r["wall_nonpositive_dropped"],
             test_waste_jobs=r["test_status"].get("Failed", 0), test_useful_jobs=r["test_status"].get("Pass", 0))
    out = ROOT / "results" / "acme_motivation"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"name": list(S), "value": list(S.values())})
    df.to_csv(out / "stats.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
