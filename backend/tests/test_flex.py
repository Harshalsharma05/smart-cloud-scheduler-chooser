"""Tests for the paper engine (backend/flex)."""
import datetime as dt
import random

from flex.cloud import Cloud, CloudParams, total_cost
from flex.engine import Engine
from flex.grid import Event, epoch_from_iso, generate_events, local_epoch, local_hour, to_local
from flex.models import FJob, load_config
from flex.paired import baseline_snapshots, policy_none, run_pair
from flex.power import PowerParams, PowerTracker, make_speed_fn, params_from_config, power_kw
from traces.synthetic import generate

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
    cells = {}
    for e in events:                                 # at most one event per local day within a cell
        cells.setdefault((e.duration_h, e.cut), []).append(to_local(epoch0 + e.t_start).date())
    assert all(len(days) == len(set(days)) == g["events_per_cell"] for days in cells.values())
    assert make(1) == events and make(2) != events
