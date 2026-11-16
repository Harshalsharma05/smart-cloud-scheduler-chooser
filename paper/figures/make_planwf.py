"""W9b: planning that knows w_f (plan_wf = match: q'_j = q_j + w_f (1 - q_j); at w_f = 1 every job counts as useful in the plan).
Run at w_f = 1 on all feasible E1 events (hold rule, p90 bill, same seed): delay-only and mix-p90 with transformer, gbdt, lightgbm.
Per predictor: per-event median ratio mix-p90 / no-predictor delay-only (results/e1_blind) and mix-p90 / same-predictor delay-only
(both plan_wf = match), events cheaper, bootstrap 95% CI over the 10 event days (same method as make_bootstrap.py). Also the
default-planning (plan_wf = zero) values for comparison. All costs at w_f = 1. Writes results/robust_planwf/summary.csv.
python paper/figures/make_planwf.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_bootstrap import N_BOOT, SEED, day_of  # noqa: E402
from make_learned_all import by, read  # noqa: E402
from style import RESULTS  # noqa: E402

OUT = RESULTS / "robust_planwf"
COL = "cost_usd_wfail1"


def boot_ci(ratio, draws, days):
    per_day = {d: ratio[[day_of(e) == d for e in ratio.index]].to_numpy() for d in days}
    meds = np.array([np.median(np.concatenate([per_day[days[i]] for i in draw])) for draw in draws])
    return np.percentile(meds, [2.5, 97.5])


def compare(a, b):
    x = pd.concat([a[COL].rename("a"), b[COL].rename("b")], axis=1).dropna()
    return x["a"] / x["b"], int((x["a"] < x["b"] - 1e-9).sum()), len(x)


def main():
    pw = read(OUT / "runs_*.csv")
    blind = by(read(RESULTS / "e1_blind" / "runs_*.csv"), "delay", "blind", "blind")
    e1 = read(RESULTS / "e1" / "runs.csv")
    lr = read(RESULTS / "e1_learned" / "runs_*.csv")
    tag = "plan_wf=match;w_fail=1.0"
    zero = {"transformer": (by(e1, "delay", "transformer", ""), by(e1, "mix", "transformer", "cq=p90")),
            "gbdt": (by(lr, "delay", "gbdt", ""), by(lr, "mix", "gbdt", "cq=p90")),
            "lightgbm": (by(lr, "delay", "lightgbm", ""), by(lr, "mix", "lightgbm", "cq=p90"))}
    days = sorted({day_of(e) for e in blind.index})
    draws = np.random.default_rng(SEED).integers(0, len(days), size=(N_BOOT, len(days)))
    rows = []
    for p in ("transformer", "gbdt", "lightgbm"):
        variants = {"match": (by(pw, "delay", p, tag), by(pw, "mix", p, tag + ";cq=p90")), "zero": zero[p]}
        for v, (dl, mx) in variants.items():
            for ref_name, ref in (("blind_delay", blind), ("own_delay", dl)):
                r, cheaper, n = compare(mx, ref)
                lo, hi = boot_ci(r, draws, days)
                rows.append(dict(predictor=p, plan_wf=v, reference=ref_name, median=float(r.median()), ci_lo=float(lo), ci_hi=float(hi),
                                 cheaper=cheaper, events=n))
        r, cheaper, n = compare(variants["match"][0], blind)
        rows.append(dict(predictor=p, plan_wf="match", reference="delay_vs_blind", median=float(r.median()), ci_lo=np.nan, ci_hi=np.nan,
                         cheaper=cheaper, events=n))
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
