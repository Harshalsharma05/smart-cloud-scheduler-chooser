"""W8: mix-p90 / delay-only per event for gbdt and lightgbm (each with its own delay-only) on all feasible E1 events,
next to the transformer. Reads results/e1_learned/runs_*.csv and results/e1/runs.csv; checks that the new gbdt/lightgbm
delay rows reproduce the E1 attribution rows of the default cell. Writes results/e1_learned/summary.csv.
python paper/figures/make_learned.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_robust_missing import COST, read, stats  # noqa: E402
from style import RESULTS  # noqa: E402

OUT = RESULTS / "e1_learned"


def by(d, policy, pred, setting):
    s = d[(d["policy"] == policy) & (d["predictor"] == pred) & (d["setting"] == setting)]
    assert s["event_id"].is_unique, (policy, pred, setting)
    return s.set_index("event_id")


def main():
    e1 = read(RESULTS / "e1" / "runs.csv")
    new = read(OUT / "runs_*.csv")
    for p in ("gbdt", "lightgbm"):                          # reproduction check against the stored attribution rows
        a, b = by(new, "delay", p, ""), by(e1, "delay", p, "attribution")
        diff = (a["cost_usd"].reindex(b.index) - b["cost_usd"]).abs().max()
        print(f"check delay/{p}: {len(b)} default-cell events, max |cost - stored| = {diff:.3g}")
        assert diff < 1e-6
    sets = {"transformer": (by(e1, "mix", "transformer", "cq=p90"), by(e1, "delay", "transformer", "")),
            "gbdt": (by(new, "mix", "gbdt", "cq=p90"), by(new, "delay", "gbdt", "")),
            "lightgbm": (by(new, "mix", "lightgbm", "cq=p90"), by(new, "delay", "lightgbm", ""))}
    rows = [dict(predictor=p, wfail=w, **stats(m, d, col)) for p, (m, d) in sets.items() for w, col in COST.items()]
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "summary.csv", index=False)
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
