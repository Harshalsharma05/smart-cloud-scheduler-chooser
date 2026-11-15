"""Tests for the paper engine (backend/flex)."""
import datetime as dt
import random

from flex.cloud import Cloud, CloudParams, total_cost
from flex.metrics import COLUMNS, CostParams
from flex.engine import Engine
from flex.grid import Event, epoch_from_iso, generate_events, local_epoch, local_hour, to_local
from flex.models import FJob, load_config
from flex import paired
from flex.paired import Gate, baseline_snapshots, policy_none, run_pair
from flex.power import PowerParams, PowerTracker, make_speed_fn, params_from_config, power_kw, start_delta_kw
from flex.policies import World, run_policy, world_from_config
from flex.predictions import Predictions, load_predictions, oracle_values
from traces.synthetic import generate, write_predictions

H = 3600


def job(jid, gpus, hours, submit=0):
    return FJob(job_id=jid, user="u", submit=submit, gpus=gpus, wall_s=int(hours * H), status="Pass")


def starts(eng):
    return {j.job_id: j.first_start for j in eng.jobs.values()}


def test_fcfs_and_skip_ahead():
    # FCFS: A fills the pool, B and C wait until A ends.
    eng = Engine([job("A", 2, 10), job("B", 1, 1), job("C", 1, 1)], G=2, check_every=1)
    eng.run_to_drain()
    assert starts(eng) == {"A": 0, "B": 10 * H, "C": 10 * H}
    # Skip-ahead: B (2 GPUs) does not fit, so C (1 GPU) jumps ahead.
    eng = Engine([job("A", 1, 10), job("B", 2, 1), job("C", 1, 1)], G=2, check_every=1)
    eng.run_to_drain()
    assert starts(eng) == {"A": 0, "B": 10 * H, "C": 0}


def test_synthetic_day_is_deterministic_and_safe():
    cfg = load_config()
    runs = []
    for _ in range(2):
        jobs = generate(random.Random(cfg["seed"]), days=1)
        eng = Engine(jobs, cfg["synthetic"]["pool_gpus"], check_every=1)
        eng.run_to_drain()
        runs.append([(j.job_id, j.first_start, j.end) for j in eng.jobs.values()])
    assert len(runs[0]) > 100 and runs[0] == runs[1]
    assert all(end is not None for _, _, end in runs[0])


def test_cap_and_suspend_timing():
    half = lambda j, f: 0.5          # speed function for the test (the real curve comes in S3/S11)
    # Cap 0.5 from 1 h to 3 h: 1 h + 2 h at half speed (1 h of work) + 8 h left -> ends at 11 h.
    eng = Engine([job("A", 1, 10)], G=1, check_every=1)
    eng.run_until(1 * H)
    eng.set_cap(0.5, half)
    eng.run_until(3 * H)
    eng.set_cap(1.0)
    eng.run_to_drain()
    assert eng.jobs["A"].end == 11 * H
    # Suspended from 1 h to 3 h: ends at 12 h, plus restore_s if given.
    for restore in (0, 90):
        eng = Engine([job("A", 1, 10)], G=1, check_every=1)
        eng.run_until(1 * H)
        eng.suspend(eng.jobs["A"])
        eng.run_until(3 * H)
        eng.resume(eng.jobs["A"], restore_s=restore)
        eng.run_to_drain()
        assert eng.jobs["A"].end == 12 * H + restore


def test_burst_running_job_is_billed_from_ready():
    params = CloudParams(price_per_gpu_h=3.0, boot_delay_s=300, migrate_s=120, min_billed_s=60, max_gpus=1000)
    eng = Engine([job("A", 2, 2)], G=2, check_every=1, cloud=Cloud(params))
    eng.run_until(1 * H)             # 1 h of work remains
    assert eng.burst(eng.jobs["A"])
    assert eng.free == 2             # on-prem GPUs freed at once
    eng.run_to_drain()
    assert eng.jobs["A"].end == 1 * H + 300 + 3600 + 120
    assert abs(total_cost(eng.cloud, eng.now) - 2 * 3.0 * 3720 / 3600) < 1e-9


def test_power_model_and_tracker():
    # Hand calculation. 4 GPUs, one busy job (1 GPU, u = 0.5), 3 idle. P_idle 50, P_max 250, H_idle 100, H_busy 50, PUE 1.3.
    pp = PowerParams(p_idle_w=50, p_max_w=250, h_idle_w=100, h_busy_w=50, pue=1.3, cloud_pue=1.2)
    curve = [(0.3, 0.55), (0.5, 0.80), (0.7, 0.93), (1.0, 1.0)]
    j = FJob(job_id="A", user="u", submit=0, gpus=1, wall_s=10 * H, status="Pass", util=0.5)
    eng = Engine([j], G=4, check_every=1)
    tracker = PowerTracker(pp)
    tracker.attach(eng)
    eng.run_until(1)
    # no cap: GPU 50 + 0.5*200 = 150; idle 3*50 = 150; host 4*100 + 1*50 = 450 -> 750 W * 1.3 = 975 W
    assert abs(power_kw(eng, pp) - 0.975) < 1e-9
    assert abs(tracker.power_at(1) - 0.975) < 1e-9
    # cap f = 0.5 binds (125 W < 150 W): (125 + 150 + 450) * 1.3 = 942.5 W; speed from r = 125/150
    eng.run_until(H)
    eng.set_cap(0.5, make_speed_fn(pp, curve))
    assert abs(power_kw(eng, pp) - 0.9425) < 1e-9
    r = 125 / 150                                      # lies on the 0.7 -> 1.0 segment of the curve
    assert abs(j.speed - (0.93 + (1.0 - 0.93) * (r - 0.7) / 0.3)) < 1e-12
    # energy over [0, 2 h]: 1 h at 0.975 kW, then 1 h at 0.9425 kW (cap 0.5 from 1 h to 2 h)
    eng.run_until(2 * H)
    eng.set_cap(1.0)
    assert abs(tracker.energy_kwh(0, 2 * H) - (0.975 + 0.9425)) < 1e-9
    # cap f = 0.8 does not bind (200 W > 150 W): power as with no cap, speed 1
    eng.set_cap(0.8)
    assert abs(power_kw(eng, pp) - 0.975) < 1e-9 and j.speed == 1.0


def test_energy_is_sum_of_steps_on_a_synthetic_day():
    cfg = load_config()
    pp = params_from_config(cfg)
    jobs = generate(random.Random(cfg["seed"]), days=1)
    eng = Engine(jobs, cfg["synthetic"]["pool_gpus"], check_every=1)
    tracker = PowerTracker(pp)
    tracker.attach(eng)
    eng.run_to_drain()
    eng.run_until(eng.now + 3600)                    # clock moves past the last event: one more step is recorded
    steps = tracker.steps
    assert steps[0][0] == 0 and all(a[0] < b[0] for a, b in zip(steps, steps[1:]))
    by_hand = sum(p * (t1 - t0) for (t0, p), (t1, _) in zip(steps, steps[1:])) / 3600.0
    assert abs(tracker.energy_kwh(0, steps[-1][0]) - by_hand) < 1e-9
    # splitting the window at any point gives the same total (no gaps, no double counting)
    mid = eng.now // 3
    assert abs(tracker.energy_kwh(0, mid) + tracker.energy_kwh(mid, steps[-1][0]) - by_hand) < 1e-9
    # final state is an empty cluster: idle floor only
    assert abs(steps[-1][1] - power_kw(eng, pp)) < 1e-12


def test_paired_none_policy_with_zero_cut_changes_nothing():
    cfg = load_config()
    pp = params_from_config(cfg)
    epoch0 = epoch_from_iso(cfg["synthetic"]["epoch0_utc"])
    jobs = generate(random.Random(cfg["seed"]), days=4)
    base_engine = Engine(jobs, cfg["synthetic"]["pool_gpus"], check_every=1)
    t_s = local_epoch(dt.date(2017, 11, 17), 17) - epoch0
    event = Event("test", t_s, t_s + 2 * H, 2, 0.0)
    (_, snap), = list(baseline_snapshots(base_engine, [t_s]))
    metrics, base, pol = run_pair(snap, event, policy_none, pp, cfg["grid"]["h_after_h"] * H, restore_s=0)
    assert abs(metrics.shed_mwh) < 1e-12 and metrics.rebound_mwh == 0.0
    assert base.end_times() == pol.end_times() and all(e is not None for e in pol.end_times().values())
    # the same snapshot with a real cut and the gate on must change something and still finish every job
    event20 = Event("test", t_s, t_s + 2 * H, 2, 0.2)
    m20, _, pol20 = run_pair(snap, event20, policy_none, pp, cfg["grid"]["h_after_h"] * H, restore_s=0)
    assert m20.shed_mwh > 0 and all(e is not None for e in pol20.end_times().values())


def test_events_are_in_window_in_local_evening_and_seeded():
    cfg = load_config()
    g = cfg["grid"]
    epoch0 = epoch_from_iso(cfg["synthetic"]["epoch0_utc"])
    t_lo, t_hi = 0, 25 * 86400

    def make(seed):
        return generate_events(random.Random(seed), epoch0, t_lo, t_hi, g)

    events = make(1)
    assert len(events) == len(g["durations_h"]) * len(g["cuts"]) * g["events_per_cell"]
    for e in events:
        assert t_lo + g["skip_first_days"] * 86400 <= e.t_start and e.t_end <= t_hi - g["end_margin_days"] * 86400
        assert e.t_end - e.t_start == e.duration_h * 3600
        h = local_hour(epoch0 + e.t_start)
        assert g["start_hour_min"] <= h < g["start_hour_max"] and h * 2 == int(h * 2)   # a half-hour slot
    days = {}                                        # one start per local day; every cell has the same 10 starts
    for e in events:
        days.setdefault((e.duration_h, e.cut), []).append((to_local(epoch0 + e.t_start).date(), e.t_start))
    assert len(days) == 9 and all(len(v) == g["events_per_cell"] for v in days.values())
    assert all(len({d for d, _ in v}) == g["events_per_cell"] for v in days.values())
    assert len({tuple(v) for v in days.values()}) == 1
    assert make(1) == events and make(2) != events


def test_prediction_lookup_rule_and_floor(tmp_path):
    # Hand-made file: 001 runs long (wall 20000 s), 002 is short (wall 700 s). Rows exist only while running.
    csv = tmp_path / "toy.csv"
    csv.write_text(
        "job_id,elapsed_s,rt_p10_s,rt_p50_s,rt_p90_s,p_fail,p_cancel\n"
        "001,0,9000,10000,11000,0.1,0.2\n"
        "001,60,8940,9940,10940,0.1,0.2\n"
        "001,600,8400,9400,10400,0.1,0.2\n"
        "001,3600,5400,6400,7400,0.3,0.1\n"
        "002,0,600,700,800,0.0,0.0\n"
        "002,60,540,640,740,0.0,0.0\n"
        "002,600,60,100,200,0.0,0.0\n")
    p = load_predictions(csv)
    assert set(p.rows) == {"001", "002"}                       # ids stay strings ("001" is not 1)
    a = FJob("001", "u", 0, 1, 20000, "Pass")
    b = FJob("002", "u", 0, 1, 700, "Pass")
    assert p.lookup(a, 0) == (9000, 10000, 11000, 0.1, 0.2)
    assert p.lookup(a, 59) == (8941, 9941, 10941, 0.1, 0.2)    # row 0, gap 59
    assert p.lookup(a, 600) == (8400, 9400, 10400, 0.1, 0.2)   # exact grid point
    assert p.lookup(a, 5000) == (4000, 5000, 6000, 0.3, 0.1)   # row 3600, gap 1400
    assert p.lookup(b, 5000) == (60, 60, 60, 0.0, 0.0)         # row 600, gap 4400: all three floored at 60
    assert p.lookup(b, 659) == (60, 60, 141, 0.0, 0.0)         # p10 1 and p50 41 are floored to 60; p90 200 - 59 = 141
    # a job without rows falls back to oracle values from the job itself, and is counted
    c = FJob("003", "u", 0, 1, 5000, "Failed")
    assert p.lookup(c, 1000) == (4000, 4000, 4000, 1.0, 0.0)
    assert p.lookup(c, 4990) == (60, 60, 60, 1.0, 0.0)
    assert p.fallback_jobs == {"003"} and p.log_fallback_count() == 1
    # wrong columns are rejected
    bad = tmp_path / "bad.csv"
    bad.write_text("job_id,elapsed_s,rt_p50_s\n001,0,5\n")
    try:
        load_predictions(bad)
        assert False, "should reject wrong columns"
    except ValueError:
        pass


def test_synthetic_prediction_files_follow_the_contract(tmp_path):
    cfg = load_config()
    jobs = generate(random.Random(cfg["seed"]), days=3)
    eng = Engine(jobs, cfg["synthetic"]["pool_gpus"], check_every=0)
    eng.run_to_drain()
    write_predictions(list(eng.jobs.values()), tmp_path)
    orc, l2 = load_predictions(tmp_path / "oracle.csv"), load_predictions(tmp_path / "last2.csv")
    assert set(orc.rows) == set(l2.rows) == {j.job_id for j in jobs}
    for j in list(eng.jobs.values())[:50]:
        grid, vals = orc.rows[j.job_id]
        assert grid == [e for e in (0, 60, 600, 3600, 14400, 43200) if e < j.wall_s]   # rows only while running
        assert grid == l2.rows[j.job_id][0]
        assert orc.lookup(j, 0) == oracle_values(j, 0)


def test_delay_only_hand_case():
    # 4 GPUs, no host power and PUE 1, so a busy GPU draws 50 + u (250 - 50) W and an idle one 50 W.
    pp = PowerParams(p_idle_w=50, p_max_w=250, h_idle_w=0, h_busy_w=0, pue=1.0, cloud_pue=1.0)
    cp = CloudParams(price_per_gpu_h=3.0, boot_delay_s=300, migrate_s=120, min_billed_s=60, max_gpus=100)
    world = World(pp, make_speed_fn(pp, [(0.3, 0.5), (1.0, 1.0)]), [1.0, 0.9, 0.8, 0.7, 0.6], cp,
                  CostParams(c_delay=3.0, m=1, w_fail=0), Predictions("oracle", {}), 24 * H, restore_s=0)
    # A: Pass, u=1 (250 W, saves 200 W); B: Pass, u=0.5 (150 W, saves 100 W); C: Failed, u=1 (saves 200 W). All 10 h.
    jobs = [FJob("A", "u", 0, 1, 10 * H, "Pass", util=1.0), FJob("B", "u", 0, 1, 10 * H, "Pass", util=0.5),
            FJob("C", "u", 0, 1, 10 * H, "Failed", util=1.0)]
    eng = Engine(jobs, G=4, check_every=1, cloud=Cloud(cp))
    event = Event("hand", 1 * H, 2 * H, 1, 0.5)                  # 1 h event, 50% cut
    (_, snap), = list(baseline_snapshots(eng, [event.t_start]))
    row, base, pol = run_policy(snap, event, "delay", world, "oracle", return_runs=True)
    # Base power 250 + 150 + 250 + 50 = 700 W, so P* = 350 W. Cost per kW: C 0 (a Failed job loses nothing),
    # A 3 / 0.2 = 15, B 3 / 0.1 = 30. Suspend C (-200 W -> 500 W), then A (-200 W -> 300 W <= 350 W): B keeps running.
    assert abs(row["target_kw"] - 0.35) < 1e-12 and row["n_suspended"] == 2
    assert pol.engine.jobs["C"].n_suspends == 1 and pol.engine.jobs["A"].n_suspends == 1
    assert pol.engine.jobs["B"].n_suspends == 0
    assert row["compliance"] == 1.0 and row["shortfall_kw"] == 0.0
    assert abs(row["shed_mwh"] - 0.0004) < 1e-12                 # (700 - 300) W for 1 h
    # A is delayed by 1 h (1 GPU-h of Pass work), C by 1 h but Failed jobs have weight 0: cost = 3 USD x 1 GPU-h.
    assert abs(row["dgpuh_good"] - 1.0) < 1e-9 and abs(row["dgpuh_all"] - 2.0) < 1e-9
    assert abs(row["cost_usd"] - 3.0) < 1e-9 and row["cloud_usd"] == 0.0 and row["missed_deadlines"] == 0
    assert abs(row["usd_per_mwh"] - 3.0 / 0.0004) < 1e-6
    assert pol.engine.jobs["A"].end == base.engine.jobs["A"].end + H


def test_three_policies_run_on_the_synthetic_default_event():
    cfg = load_config()
    epoch0 = epoch_from_iso(cfg["synthetic"]["epoch0_utc"])
    jobs = generate(random.Random(cfg["seed"]), days=6)
    for j in jobs:                                              # deadlines for 30% of jobs (the real rule comes in S8)
        if j.job_id[-1] in "012":
            j.deadline = j.submit + 3 * j.wall_s
    world = world_from_config(cfg, Predictions("oracle", {}))
    eng = Engine(jobs, cfg["synthetic"]["pool_gpus"], check_every=1000, cloud=Cloud(world.cloud))
    t_s = local_epoch(dt.date(2017, 11, 18), 17) - epoch0
    event = Event("default", t_s, t_s + 2 * H, 2, 0.2)          # the default event: 2 h, 20%
    (_, snap), = list(baseline_snapshots(eng, [t_s]))
    rows = {name: run_policy(snap, event, name, world, "oracle") for name in ("delay", "cap", "burst")}
    for name, row in rows.items():
        assert list(row) == COLUMNS and row["policy"] == name
        assert row["compliance"] >= 0.95 or row["shortfall_kw"] > 0, (name, row)   # meets P* or reports a shortfall
        assert row["shed_mwh"] > 0
    assert rows["cap"]["cap_level"] is not None and rows["cap"]["n_burst"] == 0 and rows["cap"]["n_suspended"] == 0
    assert rows["burst"]["n_burst"] > 0 and rows["burst"]["cloud_usd"] > 0 and rows["burst"]["n_suspended"] == 0
    assert rows["delay"]["n_suspended"] > 0 and rows["delay"]["cloud_usd"] == 0
    print({k: (round(v["compliance"], 3), round(v["shortfall_kw"], 2), v["usd_per_mwh"]) for k, v in rows.items()})


def test_on_hold_fires_once_for_a_job_that_fits_but_the_gate_refuses():
    # 2 GPUs, no host power, PUE 1: a busy GPU at u = 1 draws 250 W, an idle one 50 W.
    pp = PowerParams(p_idle_w=50, p_max_w=250, h_idle_w=0, h_busy_w=0, pue=1.0, cloud_pue=1.0)
    cp = CloudParams(price_per_gpu_h=3.0, boot_delay_s=300, migrate_s=120, min_billed_s=60, max_gpus=100)
    jobs = [FJob("A", "u", 0, 1, 10 * H, "Pass", util=1.0), FJob("C", "u", 0, 1, 5400, "Pass", util=1.0),
            FJob("B", "u", 4200, 1, 2 * H, "Pass", util=1.0)]
    eng = Engine(jobs, G=2, check_every=1, cloud=Cloud(cp))
    event = Event("hold", 1 * H, 2 * H, 1, 0.5)          # base power is 500 W all event long, so P* = 250 W
    (_, snap), = list(baseline_snapshots(eng, [event.t_start]))
    calls = []

    def hold_burst(e, ev, p_star, params):               # no action at t_s; burst whatever the gate holds
        e.on_hold = lambda job: (calls.append((job.job_id, e.now)), e.burst(job))

    _, base, pol = run_pair(snap, event, hold_burst, pp, 24 * H, restore_s=0)
    # B arrives at 4200 s with the pool full (A and C): it does not fit, so it is not burst then.
    # C ends at 5400 s, B now fits (1 GPU free) but the gate refuses (300 W + 200 W > 250 W): burst once, at 5400 s.
    assert calls == [("B", 5400)]
    inst = pol.engine.cloud.instances
    assert [(i.job_id, i.requested) for i in inst] == [("B", 5400)] and pol.engine.jobs["B"].on_cloud
    assert not base.engine.jobs["B"].on_cloud


def test_policy_kw_is_the_marginal_saving():
    pp = PowerParams(p_idle_w=50, p_max_w=250, h_idle_w=100, h_busy_w=50, pue=1.3, cloud_pue=1.2)
    for f in (1.0, 0.5):
        jobs = [FJob("A", "u", 0, 2, 10 * H, "Pass", util=0.8), FJob("B", "u", 0, 1, 10 * H, "Pass", util=0.4)]
        eng = Engine(jobs, G=4, check_every=1)
        eng.run_until(1)
        eng.set_cap(f, make_speed_fn(pp, [(0.3, 0.5), (1.0, 1.0)]))
        before = power_kw(eng, pp)
        saving = start_delta_kw(pp, eng.jobs["A"], eng.cap_f)
        eng.suspend(eng.jobs["A"])
        # pue x gpus x (gpu_power(u, f) - p_idle + h_busy) / 1000, and it equals the real drop in facility power
        expect = 1.3 * 2 * (min(50 + 0.8 * 200, f * 250) - 50 + 50) / 1000
        assert abs(saving - expect) < 1e-12 and abs((before - power_kw(eng, pp)) - saving) < 1e-12


def test_gate_power_cache_gives_identical_results(monkeypatch):
    cfg = load_config()
    pp = params_from_config(cfg)
    epoch0 = epoch_from_iso(cfg["synthetic"]["epoch0_utc"])
    jobs = generate(random.Random(cfg["seed"]), days=6)
    eng = Engine(jobs, 16, check_every=1000)                     # a small pool keeps a long queue
    t_s = local_epoch(dt.date(2017, 11, 19), 17) - epoch0
    event = Event("stress", t_s, t_s + 2 * H, 2, 0.2)
    (_, snap), = list(baseline_snapshots(eng, [t_s]))
    assert len(snap.queue) > 20                                  # the gate has something to hold
    out = {}
    for cache in (True, False):
        monkeypatch.setattr(paired, "make_gate", lambda e, p, t, ps, cache=cache: Gate(e, p, t, ps, cache))
        _, _, pol = run_pair(snap, event, policy_none, pp, 24 * H, restore_s=0)
        out[cache] = (pol.end_times(), pol.tracker.steps)
    assert out[True] == out[False]


def test_oracle_mix_never_bursts_a_job_that_will_fail():
    cfg = load_config()
    epoch0 = epoch_from_iso(cfg["synthetic"]["epoch0_utc"])
    jobs = generate(random.Random(cfg["seed"]), days=6)           # no deadlines: they could make a failing job worth saving
    world = world_from_config(cfg, Predictions("oracle", {}))     # no rows: every lookup is the oracle value
    assert world.cost.w_fail == 0
    eng = Engine(jobs, cfg["synthetic"]["pool_gpus"], check_every=1000, cloud=Cloud(world.cloud))
    starts = [local_epoch(dt.date(2017, 11, d), 17) - epoch0 for d in (18, 19)]
    n_burst = 0
    for t_s, snap in baseline_snapshots(eng, starts):
        for ev in (Event("a", t_s, t_s + 2 * H, 2, 0.2), Event("b", t_s, t_s + 4 * H, 4, 0.3)):
            _, _, pol = run_policy(snap, ev, "mix", world, "oracle", return_runs=True)
            insts = pol.engine.cloud.instances
            n_burst += len(insts)
            assert all(pol.engine.jobs[i.job_id].status == "Pass" for i in insts)
    assert n_burst > 0                                           # the check is not vacuous


def test_missing_prediction_fallback_default_is_oracle():
    """W7: a job without rows uses its oracle values by default; missing_pred="uninformed" gives long, useful values
    (except for the oracle predictor itself)."""
    job = FJob("X", "u", 0, 1, 5000, "Failed")
    pr = Predictions("transformer", {})
    assert pr.missing_pred == "oracle"
    assert pr.lookup(job, 1000) == oracle_values(job, 1000)
    pr.missing_pred = "uninformed"
    assert pr.lookup(job, 1000) == (30 * 86400, 30 * 86400, 30 * 86400, 0.0, 0.0)
    orc = Predictions("oracle", {})
    orc.missing_pred = "uninformed"
    assert orc.lookup(job, 1000) == oracle_values(job, 1000)
