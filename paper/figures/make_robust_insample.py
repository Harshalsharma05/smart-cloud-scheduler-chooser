"""W4 leakage check: non-oracle predictors return uninformed values (30 d, p_fail = p_cancel = 0) for jobs submitted before
2017-11-01 (training period). Transformer, hold_beyond_grid, all feasible E1 events: delay, mix p90, mix-nocloud p90, compared with
the main runs. Reads results/robust_insample/runs_ins_*.csv and share.csv. Writes results/robust_insample/summary.csv.
python paper/figures/make_robust_insample.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_robust_missing import COST, pick, read, stats  # noqa: E402
from style import RESULTS  # noqa: E402

OUT = RESULTS / "robust_insample"


def main():
    e1 = read(RESULTS / "e1" / "runs.csv")
    e1 = e1[e1["setting"] != "attribution"]
    nc = read(RESULTS / "e1nc" / "runs_d*.csv")
    ins = read(OUT / "runs_ins_*.csv")
    variants = {
        "main": dict(delay=pick(e1, "delay", []), mix=pick(e1, "mix", ["cq=p90"]), nocloud=pick(nc, "mix", ["nocloud", "cq=p90"])),
        "insample_uninformed": dict(delay=pick(ins, "delay", ["uninformed<"], ["cq="]), mix=pick(ins, "mix", ["cq=p90"], ["nocloud"]),
                                    nocloud=pick(ins, "mix", ["nocloud", "cq=p90"])),
    }
    rows = []
    for v, d in variants.items():
        for w, col in COST.items():
            for who in ("mix", "nocloud"):
                rows.append(dict(variant=v, wfail=w, policy=who, **stats(d[who], d["delay"], col)))
    for w, col in COST.items():
        rows.append(dict(variant="insample_over_main", wfail=w, policy="delay", **stats(variants["insample_uninformed"]["delay"], variants["main"]["delay"], col)))
    s = pd.DataFrame(rows)
    sh = pd.read_csv(OUT / "share.csv")
    s = pd.concat([s, pd.DataFrame([dict(variant="share", policy="running_jobs_pct_median", median_ratio=float(np.median(sh.insample_jobs_pct))),
                                    dict(variant="share", policy="running_gpus_pct_median", median_ratio=float(np.median(sh.insample_gpus_pct)))])])
    s.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.round(3).to_string(index=False))
    u = s[(s.variant == "insample_uninformed") & (s.policy == "mix")].set_index("wfail")["median_ratio"]
    print("STOP CONDITION (mix-p90 / delay per-event median >= 0.98 at either w_fail):", bool((u >= 0.98).any()), u.round(4).to_dict())


if __name__ == "__main__":
    main()
