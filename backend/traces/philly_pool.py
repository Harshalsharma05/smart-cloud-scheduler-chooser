"""SIM_PLAN S9c: the Philly pool = GPUs that report a non-NA utilisation value (not the installed G).

pool_gpus (auto) = median over the minutes of the test window of the number of non-NA GPU values per minute, summed over
all machines (width per machine = num_gpus in philly_machines.parquet; rows with another value count, i.e. the DST hour,
are ignored). It is derived from the trace, not tuned on waits. Cached in data/processed/philly_pool.json.
Run from backend/:  python -m traces.philly_pool   (prints the availability findings)
"""
import json

import numpy as np
import pandas as pd

from traces.philly import ROOT, ml_config, utc_epoch
from traces.philly_util import RAW, minute_index

POOL_JSON = ROOT / "data" / "processed" / "philly_pool.json"


def measure():
    """One pass over the util file. Returns dict with per-minute availability over [SIM_START, TEST_END) and per-machine stats."""
    mc = ml_config()
    s0, ts, w1 = utc_epoch(mc.SIM_START), utc_epoch(mc.TEST_START), utc_epoch(mc.TEST_END)
    n_min, test_off = (w1 - s0) // 60, (ts - s0) // 60
    width = pd.read_parquet(ROOT / "data" / "processed" / "philly_machines.parquet").set_index("machine_id")["num_gpus"].to_dict()
    avail, cache, skipped = np.zeros(n_min), {}, 0
    msum, mrows = dict.fromkeys(width, 0), dict.fromkeys(width, 0)
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
            i = minute_index(label, s0, n_min, cache)
            if i < 0:
                continue
            a = w - rest.count("NA")
            avail[i] += a
            if i >= test_off:
                msum[mach] += a
                mrows[mach] += 1
    return dict(avail=avail, test_off=test_off, width=width, msum=msum, mrows=mrows, skipped=skipped, s0=s0)


def load_pool():
    """{'pool_gpus', 'installed_gpus', ...} from the cache; computed (about 2 min) when missing or for other window dates."""
    mc = ml_config()
    key = [mc.SIM_START, mc.TEST_START, mc.TEST_END]
    if POOL_JSON.exists():
        d = json.loads(POOL_JSON.read_text())
        if d.get("window") == key:
            return d
    m = measure()
    test = m["avail"][m["test_off"]:]
    d = dict(window=key, pool_gpus=int(round(float(np.median(test)))), installed_gpus=int(sum(m["width"].values())),
             test_mean=float(test.mean()), test_min=float(test.min()), test_max=float(test.max()))
    POOL_JSON.write_text(json.dumps(d, indent=1))
    return d


def main():
    m = measure()
    avail, off, width = m["avail"], m["test_off"], m["width"]
    days = avail[: (len(avail) // 1440) * 1440].reshape(-1, 1440).mean(1)
    print(f"window days {len(days)} (from SIM_START); malformed rows skipped {m['skipped']}")
    print("per-day mean available GPUs, first 10 days:", days[:10].round(0).astype(int).tolist())
    print(f"all days: mean {days.mean():.0f}, min {days.min():.0f}, max {days.max():.0f}; whole-window per-minute mean {avail.mean():.0f}")
    load_pool.__globals__["POOL_JSON"].unlink(missing_ok=True)
    d = load_pool()
    print("pool cache:", d)
    att = pd.read_parquet(ROOT / "data" / "processed" / "philly_attempts.parquet", columns=["machine", "start_time", "end_time"])
    ts, w1 = utc_epoch(ml_config().TEST_START), utc_epoch(ml_config().TEST_END)
    inwin = att[(att["start_time"] < w1) & (att["end_time"].fillna(10**12) > ts)]
    used_any = {x for s in att["machine"] for x in s.split(";")}
    used_win = {x for s in inwin["machine"] for x in s.split(";")}
    rows = []
    for mach, w in width.items():
        r = m["mrows"][mach]
        rows.append(dict(machine=mach, width=w, minutes=r, mean_reporting=m["msum"][mach] / r if r else 0.0,
                         attempts_any=mach in used_any, attempts_test=mach in used_win))
    df = pd.DataFrame(rows)
    df["kind"] = np.where(df.minutes == 0, "no rows", np.where(df.mean_reporting == 0, "never reports",
                  np.where(df.mean_reporting < df.width - 0.5, "partial", "full")))
    print("\nmachines by width and reporting kind (test window):")
    print(pd.crosstab(df.width, df.kind).to_string())
    print("\nmachines that never report (or have no rows), by width x attempts in the whole trace / in the test window:")
    nr = df[df.kind.isin(["never reports", "no rows"])]
    print(nr.groupby(["width", "attempts_any", "attempts_test"]).size().to_string())
    print("\nall machines by width x attempts in the whole trace:")
    print(df.groupby(["width", "attempts_any"]).size().to_string())
    print("\nwidth-4 machines without attempts (whole trace):", int(((df.width == 4) & ~df.attempts_any).sum()),
          "| of them never reporting:", int(((df.width == 4) & ~df.attempts_any & df.kind.isin(['never reports', 'no rows'])).sum()))
    part = df[df.kind == "partial"]
    print(f"partial machines: {len(part)}, reporting {part.mean_reporting.sum():.0f} of {part.width.sum()} GPUs")
    print(f"GPUs reporting (sum of per-machine means) {df.mean_reporting.sum():.0f} of installed {df.width.sum()}")
    df.to_csv(ROOT / "results" / "philly_baseline" / "machine_reporting.csv", index=False)


if __name__ == "__main__":
    main()
