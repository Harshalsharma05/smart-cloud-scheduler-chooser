"""T4 test: metric functions on a 5-row toy example."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate as E  # noqa: E402


def test_metrics_toy():
    rem = np.array([100, 200, 400, 1000, 5000])
    status = np.array([0, 1, 0, 2, 1])  # Pass, Failed, Pass, Killed, Failed
    perfect = pd.DataFrame({"rt_p10_s": rem, "rt_p50_s": rem, "rt_p90_s": rem,
                            "p_fail": (status == 1).astype(float), "p_cancel": (status == 2).astype(float)})
    m = E.all_metrics(rem, status, perfect)
    assert m["pinball"] == 0 and m["mae_p50_min"] == 0 and m["coverage_p10_p90"] == 1.0
    assert m["logloss"] < 1e-12 and m["auroc_failed"] == 1.0 and m["brier_pfail"] == 0.0

    # wide interval covers everything; p50 off by 2 minutes on every row
    wide = perfect.assign(rt_p10_s=60, rt_p90_s=10_000, rt_p50_s=rem + 120)
    assert E.coverage(rem, wide["rt_p10_s"], wide["rt_p90_s"]) == 1.0
    assert E.mae_p50_min(rem, wide["rt_p50_s"]) == 2.0

    # uninformative status: uniform 1/3 gives log-loss ln 3; constant p_fail gives AUROC 0.5; Brier (0.4 - y)^2
    assert abs(E.logloss3(status, np.full(5, 1 / 3), np.full(5, 1 / 3)) - np.log(3)) < 1e-12
    assert E.auroc_failed(status, np.full(5, 0.4)) == 0.5
    assert abs(E.brier_pfail(status, np.full(5, 0.4)) - (3 * 0.16 + 2 * 0.36) / 5) < 1e-12

    # pinball: predicting q50 one log-unit too high costs 0.5 * 1 / 3 per quantile-avg when only p50 is off
    y = np.array([1000.0])
    q = np.expm1(np.log1p(y) + 1.0)
    p = E.pinball_log(y, y, q, y)
    assert abs(p - 0.5 / 3) < 1e-9
