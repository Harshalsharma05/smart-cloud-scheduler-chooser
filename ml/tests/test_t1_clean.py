"""T1 test: 4 hand-written jobs cover the attempt edge cases."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import philly_clean as pc  # noqa: E402

RAW = json.loads("""
[
 {"jobid": "j_two", "user": "u1", "vc": "v1", "status": "Failed", "submitted_time": "2017-10-09 06:00:00",
  "attempts": [
   {"start_time": "2017-10-09 06:10:00", "end_time": "2017-10-09 06:20:00", "detail": [{"ip": "m1", "gpus": ["gpu0"]}]},
   {"start_time": "2017-10-09 06:30:00", "end_time": "2017-10-09 07:00:00", "detail": [{"ip": "m2", "gpus": ["gpu0", "gpu1"]}]}]},
 {"jobid": "j_none", "user": "u1", "vc": "v1", "status": "Killed", "submitted_time": "2017-10-09 06:00:00", "attempts": []},
 {"jobid": "j_running", "user": "u2", "vc": "v2", "status": "Pass", "submitted_time": "2017-12-22 06:00:00",
  "attempts": [{"start_time": "2017-12-22 06:05:00", "end_time": null, "detail": [{"ip": "m3", "gpus": ["gpu1"]}]}]},
 {"jobid": "j_nostart", "user": "u2", "vc": "v2", "status": "Pass", "submitted_time": "2017-10-09 06:00:00",
  "attempts": [{"start_time": null, "end_time": "2017-10-09 06:30:00", "detail": [{"ip": "m4", "gpus": ["gpu0"]}]},
               {"start_time": "2017-10-09 06:40:00", "end_time": "2017-10-09 06:50:00", "detail": [{"ip": "m4", "gpus": ["gpu1"]}]}]}
]
""")


def test_four_cases():
    jobs, att, _ = pc.build_tables(RAW)
    j = jobs.set_index("job_id")

    two = j.loc["j_two"]  # 2 attempts
    assert two.submit_time == 1507528800  # 2017-10-09 06:00:00 UTC (strings are UTC, no tz shift)
    assert two.num_attempts == 2 and two.num_gpus == 2 and not two.censored and not two.never_started
    assert two.wall_s == 50 * 60  # 06:10 -> 07:00
    assert two.run_s == 40 * 60  # 10 min + 30 min
    assert two.first_start_time - two.submit_time == 10 * 60
    a = att[att.job_id == "j_two"].sort_values("attempt")
    assert list(a.machine) == ["m1", "m2"] and list(a.gpu_ids) == ["m1:gpu0", "m2:gpu0;m2:gpu1"]

    none = j.loc["j_none"]  # no attempts
    assert none.never_started and none.num_attempts == 0 and none.num_gpus == 0
    assert none.isna()["first_start_time"] and none.isna()["wall_s"]

    run = j.loc["j_running"]  # last end missing = still running
    assert run.censored and not run.never_started and run.num_attempts == 1
    assert run.isna()["end_time"] and run.isna()["wall_s"]
    assert len(att[att.job_id == "j_running"]) == 1

    ns = j.loc["j_nostart"]  # first attempt has no start: dropped
    assert ns.num_attempts == 1 and not ns.never_started
    assert ns.first_start_time - ns.submit_time == 40 * 60 and ns.wall_s == 10 * 60
    assert len(att[att.job_id == "j_nostart"]) == 1

    assert len(jobs) == 4 and all(isinstance(x, str) for x in jobs.job_id)


def test_util_widths(tmp_path):
    """2-wide and 8-wide machines are read; the malformed 16-wide row is ignored; 'NA' counts as present."""
    hdr = "time,machineId," + ",".join(f"gpu{i}_util" for i in range(8)) + "\n"
    rows = ["t1,a,1.0,2.0,,,,,,,", "t2,a,1.0,NA,,,,,,,",
            "t3,b," + ",".join(["5.0"] * 8) + ",", "t4,b," + ",".join(["5.0"] * 14) + ","]
    f = tmp_path / "util.csv"
    f.write_text(hdr + "\n".join(rows) + "\n")
    w, bad = pc.util_widths(f)
    assert w == {"a": 2, "b": 8} and bad == 1
