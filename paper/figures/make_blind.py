"""W9: delay-only without a predictor ("blind": q = 1 for every job, no deadline term) on all feasible E1 events, compared with
delay-only and the p90 mix of each learned predictor and with the oracle. Per-event median ratios, events cheaper, w_f 0 and 1.
Reads results/e1_blind/runs_*.csv, results/e1/runs.csv, results/e1_learned/runs_*.csv. Writes results/e1_blind/summary.csv.
python paper/figures/make_blind.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_learned_all import by, read  # noqa: E402
from make_robust_missing import COST, stats  # noqa: E402
from style import RESULTS  # noqa: E402


def main():
    e1 = read(RESULTS / "e1" / "runs.csv")
    lr = read(RESULTS / "e1_learned" / "runs_*.csv")
    blind = by(read(RESULTS / "e1_blind" / "runs_*.csv"), "delay", "blind", "blind")
    others = {"transformer delay": by(e1, "delay", "transformer", ""), "gbdt delay": by(lr, "delay", "gbdt", ""),
              "lightgbm delay": by(lr, "delay", "lightgbm", ""), "oracle delay": by(e1, "delay", "oracle", ""),
              "transformer mix-p90": by(e1, "mix", "transformer", "cq=p90"), "gbdt mix-p90": by(lr, "mix", "gbdt", "cq=p90"),
              "lightgbm mix-p90": by(lr, "mix", "lightgbm", "cq=p90"), "oracle mix": by(e1, "mix", "oracle", "")}
    rows = [dict(policy=k, wfail=w, **stats(v, blind, col)) for k, v in others.items() for w, col in COST.items()]
    s = pd.DataFrame(rows)                       # ratio = policy / blind delay-only
    s.to_csv(RESULTS / "e1_blind" / "summary.csv", index=False)
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
