"""S9b measurement only: GPUs reporting a value (not "NA") per minute in the test window, from cluster_gpu_util.

Width per machine = num_gpus in philly_machines.parquet; rows with another value count (DST hour) are ignored.
Writes results/philly_baseline/gpu_availability.csv (hourly means). Run from backend/:  python -m traces.gpu_availability
"""
import numpy as np
import pandas as pd

from traces.philly import ROOT, _busy_per_minute, load_philly, ml_config, utc_epoch
from traces.philly_util import RAW, minute_index


def main():
    mc = ml_config()
    w0, w1 = utc_epoch(mc.TEST_START), utc_epoch(mc.TEST_END)
    n_min = (w1 - w0) // 60
    width = pd.read_parquet(ROOT / "data" / "processed" / "philly_machines.parquet").set_index("machine_id")["num_gpus"].to_dict()
    avail, cache, skipped = np.zeros(n_min), {}, 0
    with open(RAW, "r", buffering=1 << 22) as f:
        next(f)
        for line in f:
            label, mach, rest = line.split(",", 2)
            w = width.get(mach)
            if w is None:
                continue
            if rest.count(",") != w:
                skipped += 1
                continue
            i = minute_index(label, w0, n_min, cache)
            if i >= 0:
                avail[i] += w - rest.count("NA")
    L = load_philly()
    a = L["all_started"]
    end = a["end_time"].fillna(L["last_end"]).astype(float)
    busy = _busy_per_minute(a["first_start_time"].astype(float), end, a["num_gpus"].to_numpy(), w0, w1)
    share = busy / np.maximum(avail, 1)
    print(f"G (machines table) = {sum(width.values())}; malformed rows skipped {skipped}; test window {n_min} minutes")
    print(f"available GPUs per minute: mean {avail.mean():.0f}, p5 {np.percentile(avail, 5):.0f}, p50 {np.percentile(avail, 50):.0f}, "
          f"min {avail.min():.0f}, max {avail.max():.0f}; minutes with 0 available: {int((avail == 0).sum())}")
    print(f"real busy GPUs: mean {busy.mean():.0f}; busy / available: mean of ratios {share.mean():.3f}, "
          f"mean busy / mean available {busy.mean() / avail.mean():.3f}, p95 {np.percentile(share, 95):.3f}, max {share.max():.3f}")
    print(f"minutes with busy > available: {int((busy > avail).sum())}; busy > 0.95 x available: {int((busy > 0.95 * avail).sum())}")
    h = n_min // 60
    df = pd.DataFrame(dict(hour=np.arange(h), t_utc=pd.to_datetime(w0 + np.arange(h) * 3600, unit="s"),
                           available_gpus=avail[:h * 60].reshape(h, 60).mean(1).round(1),
                           busy_gpus=busy[:h * 60].reshape(h, 60).mean(1).round(1)))
    df["busy_share"] = (df["busy_gpus"] / df["available_gpus"].clip(lower=1)).round(4)
    out = ROOT / "results" / "philly_baseline" / "gpu_availability.csv"
    df.to_csv(out, index=False)
    print("wrote", out, len(df), "rows")


if __name__ == "__main__":
    main()
