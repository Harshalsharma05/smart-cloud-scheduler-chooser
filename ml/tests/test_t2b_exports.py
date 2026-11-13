"""T2b test: oracle = wall_s - e (floored); last2 on a 3-job toy user."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import examples  # noqa: E402
import export_simple as ex_s  # noqa: E402

BOUNDS = {"train_end": 1000, "val_start": 1000, "val_end": 2000, "test_start": 2000, "test_end": 1_000_000}


def toy():
    # user u: three finished jobs (walls 100, 300, 1000), then job D submitted later; user w has no history
    rows = []
    for jid, user, sub, wall, st in [("A", "u", 0, 100, "Pass"), ("B", "u", 200, 300, "Failed"), ("C", "u", 600, 1000, "Pass"),
                                     ("D", "u", 2000, 5000, "Killed"), ("E", "w", 2100, 3000, "Pass")]:
        rows.append(dict(job_id=jid, user=user, vc="v", submit_time=sub, first_start_time=sub + 10, end_time=sub + 10 + wall,
                         wall_s=wall, run_s=wall, num_gpus=1, num_attempts=1, status=st, censored=False, never_started=False))
    jobs = pd.DataFrame(rows).astype({c: "Int64" for c in ["submit_time", "first_start_time", "end_time", "wall_s", "run_s"]})
    att = pd.DataFrame({"job_id": jobs["job_id"], "attempt": 0, "start_time": jobs["first_start_time"]})
    return jobs, att


def test_oracle_and_last2():
    jobs, att = toy()
    ex = examples.build_examples(jobs, att, bounds=BOUNDS)
    test = ex[ex["split"] == "test"]

    o = ex_s.oracle_frame(test).set_index(["job_id", "elapsed_s"])
    assert o.loc[("D", 0), "rt_p50_s"] == 5000 and o.loc[("D", 3600), "rt_p50_s"] == 1400
    assert o.loc[("D", 3600), "p_cancel"] == 1 and o.loc[("D", 0), "p_fail"] == 0
    assert o.loc[("D", 600), "rt_p10_s"] == o.loc[("D", 600), "rt_p90_s"] == 4400
    assert ex_s.oracle_frame(ex[ex.job_id == "A"]).rt_p50_s.iloc[0] == 100
    assert (ex_s.oracle_frame(ex).rt_p50_s >= 60).all()

    l = ex_s.last2_frame(test, fallback_wall=777.0, fallback_fail=0.2, fallback_cancel=0.1).set_index(["job_id", "elapsed_s"])
    # D: last two finished jobs are C (1000) and B (300) -> mean 650; remaining 650 - e, floored at 60
    assert l.loc[("D", 0), "rt_p50_s"] == 650 and l.loc[("D", 600), "rt_p50_s"] == 60
    assert abs(l.loc[("D", 0), "p_fail"] - 1 / 3) < 1e-12 and l.loc[("D", 0), "p_cancel"] == 0  # B failed, of A, B, C
    # E has no history -> train fallbacks
    assert l.loc[("E", 0), "rt_p50_s"] == 777 and l.loc[("E", 0), "p_fail"] == 0.2 and l.loc[("E", 0), "p_cancel"] == 0.1
