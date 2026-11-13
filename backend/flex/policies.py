"""Single-action policies for one grid event (SIM_PLAN D10-D11): delay-only, cap-only and burst-only.

A policy is called once at t_s as policy(eng, event, p_star, params, world=world). The gate (D10) is installed by
paired.run_event before the policy runs. A 0% cut means no curtailment, so every policy then does nothing.
Policies read predictions (never wall_s or status) through world.predictor.
"""
from dataclasses import dataclass
from functools import partial

from .cloud import CloudParams
from .cloud import params_from_config as cloud_params_from_config
from .metrics import CostParams, cost_params_from_config, event_row
from .models import RUNNING
from .paired import policy_none, run_pair
from .power import PowerParams, curve_from_config, make_speed_fn, power_kw, start_delta_kw
from .power import params_from_config as power_params_from_config


@dataclass
class World:
    params: PowerParams
    speed_fn: object             # speed_fn(job, f) from the cap curve
    cap_levels: list
    cloud: CloudParams
    cost: CostParams
    predictor: object            # flex.predictions.Predictions
    h_after_s: int
    restore_s: int


def world_from_config(cfg, predictor, curve_variant=None):
    params = power_params_from_config(cfg)
    return World(params, make_speed_fn(params, curve_from_config(cfg, curve_variant)), list(cfg["cap_levels"]),
                 cloud_params_from_config(cfg), cost_params_from_config(cfg), predictor,
                 int(cfg["grid"]["h_after_h"] * 3600), int(cfg["cloud"]["restore_s"]))


# ---- per-job estimates (predictions only) ----

def progress(eng, job):
    """Nominal progress of a running job right now (= elapsed time for the prediction lookup)."""
    return job.work_done + job.speed * (eng.now - job.mark)


def _predicted(eng, world, job):
    elapsed = progress(eng, job)
    _, p50, _, p_fail, p_cancel = world.predictor.lookup(job, elapsed)
    return elapsed, p50, max(0.0, 1.0 - p_fail - p_cancel)      # elapsed, predicted remaining, P(useful output)


def cost_suspend(eng, world, job, event):
    """Estimated cost of holding `job` for the whole event (D11): delay of useful work plus deadline risk."""
    elapsed, rt, useful = _predicted(eng, world, job)
    d_s = event.t_end - event.t_start
    c = world.cost.c_delay * job.gpus * (d_s / 3600.0) * useful
    if job.deadline is not None:                    # meets the deadline if left alone, misses it if delayed
        end_now = eng.now + rt
        if end_now <= job.deadline < end_now + d_s + world.restore_s:
            c += world.cost.m * world.cost.c_delay * job.gpus * (elapsed + rt) / 3600.0
    return c


def cost_burst(eng, world, job, event):
    """Estimated cost of moving `job` to the cloud (D11): cloud bill plus the boot wait."""
    _, rt, useful = _predicted(eng, world, job)
    cp = world.cloud
    return (cp.price_per_gpu_h * job.gpus * (rt + cp.migrate_s) / 3600.0
            + world.cost.c_delay * job.gpus * cp.boot_delay_s / 3600.0 * useful)


def _ranked(eng, world, event, cost_fn):
    """Running jobs ordered by cost per kW saved, cheapest first (planned once, at t_s)."""
    out = []
    for jid in sorted(eng.running):
        job = eng.jobs[jid]
        kw = start_delta_kw(world.params, job, eng.cap_f)
        if kw > 0:
            out.append((cost_fn(eng, world, job, event) / kw, jid, job))
    out.sort(key=lambda x: (x[0], x[1]))
    return [job for _, _, job in out]


# ---- policies ----

def delay_only(eng, event, p_star, params, *, world):
    """Suspend running jobs by cost_suspend / kW until power <= P*; the gate then holds new starts."""
    if event.cut == 0:
        return
    for job in _ranked(eng, world, event, cost_suspend):
        if power_kw(eng, params) <= p_star:
            break
        eng.suspend(job)


def _power_under_cap(eng, params, f):
    saved, eng.cap_f = eng.cap_f, f
    try:
        return power_kw(eng, params)
    finally:
        eng.cap_f = saved


def cap_only(eng, event, p_star, params, *, world):
    """The mildest cap level whose power is <= P*; the deepest level if none is enough (shortfall)."""
    if event.cut == 0:
        return
    levels = sorted(world.cap_levels, reverse=True)
    chosen = levels[-1]
    for f in levels:
        if _power_under_cap(eng, params, f) <= p_star:
            chosen = f
            break
    if chosen < 1.0:
        eng.set_cap(chosen, world.speed_fn)
    eng.info["cap_level"] = chosen


def burst_only(eng, event, p_star, params, *, world):
    """Burst running jobs by cost_burst / kW until power <= P*; arrivals the gate would hold are burst too."""
    if event.cut == 0:
        return
    for job in _ranked(eng, world, event, cost_burst):
        if power_kw(eng, params) <= p_star:
            break
        eng.burst(job)                      # False when the cloud GPU cap blocks it: try the next job
    eng.on_hold = eng.burst


POLICIES = {"none": policy_none, "delay": delay_only, "cap": cap_only, "burst": burst_only}


def run_policy(snapshot, event, name, world, predictor_name="", return_runs=False):
    """Paired run of one policy on one event; returns one result row (metrics.COLUMNS)."""
    policy = policy_none if name == "none" else partial(POLICIES[name], world=world)
    shed, base, pol = run_pair(snapshot, event, policy, world.params, world.h_after_s, world.restore_s)
    row = event_row(event, shed, base.engine, pol.engine, world.cost, world.cloud.boot_delay_s, name, predictor_name)
    return (row, base, pol) if return_runs else row
