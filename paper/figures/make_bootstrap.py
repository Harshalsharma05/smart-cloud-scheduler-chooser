"""W9: bootstrap 95% confidence interval of the median per-event ratio mix-p90 / delay-only, per learned predictor and w_f.
Resampling unit = event day: the 90 events are 10 start days x 9 cells, so a resample draws 10 days with replacement and keeps
all (feasible) events of each drawn day together. 2,000 resamples, numpy default_rng(1). Writes results/bootstrap/ci.csv.
python paper/figures/make_bootstrap.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_learned_all import by, read  # noqa: E402
from make_robust_missing import COST  # noqa: E402
from style import RESULTS  # noqa: E402

N_BOOT, SEED = 2000, 1


def day_of(event_id):
    return event_id.rsplit("_", 1)[1]                      # "d2h_c20_3" -> "3": the k-th shared start time


def main():
    e1 = read(RESULTS / "e1" / "runs.csv")
    lr = read(RESULTS / "e1_learned" / "runs_*.csv")
    pairs = {"transformer": (by(e1, "delay", "transformer", ""), by(e1, "mix", "transformer", "cq=p90")),
             "gbdt": (by(lr, "delay", "gbdt", ""), by(lr, "mix", "gbdt", "cq=p90")),
             "lightgbm": (by(lr, "delay", "lightgbm", ""), by(lr, "mix", "lightgbm", "cq=p90"))}
    rng = np.random.default_rng(SEED)
    days = sorted({day_of(e) for e in pairs["transformer"][0].index})
    draws = rng.integers(0, len(days), size=(N_BOOT, len(days)))   # the same day draws for every predictor and w_f
    rows = []
    for p, (dl, mx) in pairs.items():
        for w, col in COST.items():
            x = pd.concat([mx[col].rename("m"), dl[col].rename("d")], axis=1).dropna()
            r = (x["m"] / x["d"]).rename("r").to_frame()
            r["day"] = [day_of(e) for e in r.index]
            per_day = {d: r.loc[r.day == d, "r"].to_numpy() for d in days}
            meds = np.array([np.median(np.concatenate([per_day[days[i]] for i in draw])) for draw in draws])
            lo, hi = np.percentile(meds, [2.5, 97.5])
            rows.append(dict(predictor=p, wfail=w, median=float(r["r"].median()), ci_lo=float(lo), ci_hi=float(hi),
                             events=len(r), days=len(days), resamples=N_BOOT))
    s = pd.DataFrame(rows)
    out = RESULTS / "bootstrap"
    out.mkdir(parents=True, exist_ok=True)
    s.to_csv(out / "ci.csv", index=False)
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
