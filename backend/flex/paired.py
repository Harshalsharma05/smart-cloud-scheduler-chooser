"""Paired runs for one grid event (SIM_PLAN D6): a baseline continuation and a policy run from the same snapshot.

Both runs see the same arrivals: arrivals stop at t_e + H_after (Engine.truncate_arrivals) and the run then drains.
"""
import copy
from dataclasses import dataclass

from .metrics import shed_metrics, target_kw
from .models import SUSPENDED
from .power import PowerTracker, power_kw, start_delta_kw


def baseline_snapshots(engine, starts):
    """Run the baseline engine forward and yield (t_s, snapshot) at each distinct event start, in time order.
    A generator, so the caller can finish one event before the next snapshot is taken (memory)."""
    for t_s in sorted(set(starts)):
        engine.run_until(t_s)
        yield t_s, engine.snapshot()


@dataclass
class RunResult:
    engine: object
    tracker: PowerTracker

    def end_times(self):
        return {jid: j.end for jid, j in sorted(self.engine.jobs.items())}


def _restore(snapshot, params, t_stop):
    eng = copy.deepcopy(snapshot)
    eng.truncate_arrivals(t_stop)
    tracker = PowerTracker(params)      # snapshots drop the hook, so every copy gets a fresh tracker
    tracker.attach(eng)
    return eng, tracker


def run_base(snapshot, event, params, h_after_s):
    """Baseline continuation: no event, no gate."""
    eng, tracker = _restore(snapshot, params, event.t_end + h_after_s)
    eng.run_to_drain()
    return RunResult(eng, tracker)


def make_gate(eng, params, t_end, p_star):
    """D10: while now < t_end a job may start only if power after the start stays <= P*. Bound to `eng`."""
    def admit(job):
        return eng.now >= t_end or power_kw(eng, params) + start_delta_kw(params, job, eng.cap_f) <= p_star
    return admit


def policy_none(eng, event, p_star, params):
    """Take no action; only the gate (installed by run_event) acts."""


def run_event(snapshot, event, policy, params, p_star, h_after_s, restore_s):
    """Policy run. The gate is on during [t_s, t_e) unless the cut is 0 (no curtailment requested).
    At t_e the cap is lifted, suspended jobs resume (FCFS order, restore_s each) and the gate turns off."""
    eng, tracker = _restore(snapshot, params, event.t_end + h_after_s)
    assert eng.now == event.t_start, "snapshot must be taken at the event start"
    if event.cut > 0:
        eng.admit = make_gate(eng, params, event.t_end, p_star)
    policy(eng, event, p_star, params)
    eng.run_until(event.t_end)
    eng.admit = None
    if eng.cap_f < 1.0:
        eng.set_cap(1.0)
    for j in sorted((j for j in eng.jobs.values() if j.state == SUSPENDED), key=lambda j: (j.submit, j.job_id)):
        eng.resume(j, restore_s)
    eng.schedule()
    eng.run_to_drain()
    return RunResult(eng, tracker)


def run_pair(snapshot, event, policy, params, h_after_s, restore_s):
    """Baseline continuation + policy run from one snapshot. Returns (ShedMetrics, base RunResult, policy RunResult)."""
    base = run_base(snapshot, event, params, h_after_s)
    p_star = target_kw(base.tracker, event.t_start, event.t_end, event.cut)
    pol = run_event(snapshot, event, policy, params, p_star, h_after_s, restore_s)
    return shed_metrics(base.tracker, pol.tracker, event, h_after_s), base, pol
