"""T2 test: no leakage in the example table (synthetic jobs)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import examples  # noqa: E402

BOUNDS = {"train_end": 4000, "val_start": 4000, "val_end": 6000, "test_start": 6000, "test_end": 8000}


def make_jobs(n=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        submit = int(rng.integers(0, 9000))
        wait = int(rng.integers(0, 300))
        wall = int(rng.choice([30, 90, 700, 5000, 20000, 60000]))
        rows.append(dict(job_id=f"j{i}", user=f"u{rng.integers(0, 6)}", vc=f"v{rng.integers(0, 2)}",
                         submit_time=submit, first_start_time=submit + wait, end_time=submit + wait + wall,
                         wall_s=wall, run_s=wall, num_gpus=int(rng.choice([1, 2, 4])), num_attempts=1,
                         status=str(rng.choice(["Pass", "Failed", "Killed"])), censored=False, never_started=False))
    jobs = pd.DataFrame(rows)
    jobs.loc[3, ["censored", "end_time", "wall_s"]] = [True, pd.NA, pd.NA]  # a censored job
    jobs.loc[5, ["never_started", "first_start_time", "end_time", "wall_s"]] = [True, pd.NA, pd.NA, pd.NA]
    jobs = jobs.astype({c: "Int64" for c in ["submit_time", "first_start_time", "end_time", "wall_s", "run_s"]})
    attempts = pd.DataFrame({"job_id": jobs["job_id"], "attempt": 0, "start_time": jobs["first_start_time"]}).dropna()
    return jobs, attempts


def test_no_leakage():
    jobs, attempts = make_jobs()
    ex = examples.build_examples(jobs, attempts, bounds=BOUNDS)
    assert len(ex) > 300
    end = jobs["end_time"].astype("float64").to_numpy()
    for r in ex.itertuples():
        # history jobs ended by t_obs, at most H, never the job itself
        h = np.asarray(r.hist_idx)
        assert len(h) <= config.H
        assert (end[h] <= r.t_obs).all()
        assert r.job_row not in h
        assert jobs.loc[r.job_row, "job_id"] == r.job_id
        # grid rows exist only while the job runs; t_obs rule
        assert r.elapsed_s in config.GRID and r.elapsed_s < r.wall_s
        j = jobs.loc[r.job_row]
        assert r.t_obs == (j.submit_time if r.elapsed_s == 0 else j.first_start_time + r.elapsed_s)
        # history ordered oldest -> newest
        assert (np.diff(end[h]) >= 0).all()
    # every example job has all grid points below its wall_s
    for jid, g in ex.groupby("job_id"):
        assert sorted(g.elapsed_s) == [e for e in config.GRID if e < g.wall_s.iloc[0]]
    # splits by submit_time; nothing after test_end; censored / never-started jobs absent
    assert (ex.loc[ex.split.isin(["train", "val"]), "submit_time"] < BOUNDS["test_start"]).all()
    assert (ex.loc[ex.split == "test", "submit_time"] >= BOUNDS["test_start"]).all()
    assert (ex["submit_time"] < BOUNDS["test_end"]).all()
    assert not ex["job_id"].isin(["j3", "j5"]).any()
