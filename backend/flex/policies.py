"""Policies for one grid event (SIM_PLAN D10-D11): delay-only, cap-only, burst-only and the mix controller.

A policy is called once at t_s as policy(eng, event, p_star, params, world=world, ...). The gate (D10) is installed by
paired.run_event before the policy runs. A 0% cut means no curtailment, so every policy then does nothing.
Policies read predictions (never wall_s or status) through world.predictor.

Selectors (which jobs to burst, E3): "predictor" ranks by estimated cost per kW with world.predictor;
"none-largest" (most GPUs first) and "random" (seeded) use no predictor. Without a predictor the estimates are
blind: the job is assumed to outlast the event and to produce useful output.
"""
import random
from dataclasses import dataclass
from functools import partial

from .cloud import CloudParams
from .cloud import params_from_config as cloud_params_from_config
from .metrics import CostParams, cost_params_from_config, event_row
from .models import RUNNING
from .paired import policy_none, run_base, run_event
from .metrics import shed_metrics, target_kw
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
    cost_quantile: str = "p50"           # S14: runtime quantile in the burst cloud bill ("p50" default, "p90" variant)
    burst_min_submit: int | None = None  # S14 robustness: only jobs submitted at or after this simulated time may be burst


def world_from_config(cfg, predictor, curve_variant=None):
    params = power_params_from_config(cfg)
    return World(params, make_speed_fn(params, curve_from_config(cfg, curve_variant)), list(cfg["cap_levels"]),
                 cloud_params_from_config(cfg), cost_params_from_config(cfg), predictor,
                 int(cfg["grid"]["h_after_h"] * 3600), int(cfg["cloud"]["restore_s"]))


# ---- per-job estimates (predictions only) ----

def progress(eng, job):
    """Nominal progress of a job right now plus elapsed0 (= elapsed time for the prediction lookup)."""
    if job.state == RUNNING:
        return job.elapsed0 + job.work_done + job.speed * (eng.now - job.mark)
    return job.elapsed0 + job.work_done


def _info(eng, world, selector, job, event, quantile="p50"):
    """(elapsed, predicted remaining runtime in s (p50, or p90 if asked), P(useful output)). Blind selectors assume the job outlasts the event."""
    elapsed = progress(eng, job)
    if selector == "predictor":
        _, p50, p90, p_fail, p_cancel = world.predictor.lookup(job, elapsed)
        return elapsed, (p90 if quantile == "p90" else p50), max(0.0, 1.0 - p_fail - p_cancel)
    return elapsed, max(1, event.t_end - eng.now), 1.0


def hold_cost(eng, world, selector, job, event, d_s):
    """Estimated cost of holding `job` for d_s seconds (D11): delay of useful work plus deadline risk."""
    elapsed, rt, useful = _info(eng, world, selector, job, event)
    c = world.cost.c_delay * job.gpus * (d_s / 3600.0) * useful
    if selector == "predictor" and job.deadline is not None:   # meets the deadline if left alone, misses it if held
        end_now = eng.now + rt
        if end_now <= job.deadline < end_now + d_s + world.restore_s:
            c += world.cost.m * world.cost.c_delay * job.gpus * (elapsed + rt) / 3600.0
    return c


def burst_cost(eng, world, selector, job, event, running=True):
    """Estimated cost of moving `job` to the cloud (D11): cloud bill plus the boot wait (migrate_s only if it is running)."""
    _, rt, useful = _info(eng, world, selector, job, event, world.cost_quantile)
    cp = world.cloud
    migrate = cp.migrate_s if running else 0
    return (cp.price_per_gpu_h * job.gpus * (rt + migrate) / 3600.0
            + world.cost.c_delay * job.gpus * cp.boot_delay_s / 3600.0 * useful)


def can_burst(world, job):
    """False only in the S14 robustness run, for jobs submitted before world.burst_min_submit."""
    return world.burst_min_submit is None or tie_key(job)[0] >= world.burst_min_submit


def event_rng(seed, event):
    return random.Random(f"{seed}:{event.event_id}:{event.t_start}")


def tie_key(job):
    """Tie-break key (submit time, job_id); carry-over jobs use their trace submit time."""
    return (job.submit if job.submit_real is None else job.submit_real, job.job_id)


def _order(jobs, selector, rng, cost_per_kw):
    """Candidates in the order the selector wants them acted on. cost_per_kw(job) is used by "predictor" only."""
    if selector == "none-largest":
        return sorted(jobs, key=lambda j: (-j.gpus,) + tie_key(j))
    if selector == "random":
        keyed = [(rng.random(), tie_key(j), j) for j in sorted(jobs, key=tie_key)]
        return [j for _, _, j in sorted(keyed, key=lambda x: x[:2])]
    return sorted(jobs, key=lambda j: (cost_per_kw(j),) + tie_key(j))


def _running_jobs(eng):
    return sorted((eng.jobs[jid] for jid in eng.running), key=tie_key)


def _power_under_cap(eng, params, f):
    saved, eng.cap_f = eng.cap_f, f
    try:
        return power_kw(eng, params)
    finally:
        eng.cap_f = saved


# ---- policies ----

def delay_only(eng, event, p_star, params, *, world, selector="predictor", rng=None):
    """Suspend running jobs by cost_suspend / kW until power <= P*; the gate then holds new starts."""
    if event.cut == 0:
        return
    d_s = event.t_end - event.t_start
    kw = lambda j: start_delta_kw(params, j, eng.cap_f)
    jobs = [j for j in _running_jobs(eng) if kw(j) > 0]
    for job in _order(jobs, "predictor", None, lambda j: hold_cost(eng, world, "predictor", j, event, d_s) / kw(j)):
        if power_kw(eng, params) <= p_star:
            break
        eng.suspend(job)


def cap_only(eng, event, p_star, params, *, world, selector="predictor", rng=None):
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


def burst_only(eng, event, p_star, params, *, world, selector="predictor", rng=None):
    """Burst running jobs in selector order (predictor: by cost_burst / kW) until power <= P*.
    Every job the gate refuses although it fits the free GPUs is burst too (eng.on_hold)."""
    if event.cut == 0:
        return
    kw = lambda j: start_delta_kw(params, j, eng.cap_f)
    jobs = [j for j in _running_jobs(eng) if kw(j) > 0 and can_burst(world, j)]
    for job in _order(jobs, selector, rng, lambda j: burst_cost(eng, world, selector, j, event) / kw(j)):
        if power_kw(eng, params) <= p_star:
            break
        eng.burst(job)                      # False when the cloud GPU cap blocks it: try the next job
    eng.on_hold = eng.burst if world.burst_min_submit is None else (lambda job: can_burst(world, job) and eng.burst(job))


def mix(eng, event, p_star, params, *, world, selector="predictor", rng=None):
    """D11 mix controller. For every cap level f: the cap's slowdown cost, then job actions in cost/kW order
    (each job takes its cheaper option, suspend or burst) until P* is reached. Pick the f with the lowest total
    estimated cost among the plans that reach P* (if none does, the lowest-cost plan). Held jobs are burst only
    if cost_burst < cost_hold."""
    if event.cut == 0:
        return
    d_s = event.t_end - event.t_start
    jobs = _running_jobs(eng)
    useful = {j.job_id: _info(eng, world, selector, j, event)[2] for j in jobs}
    s_cost = {j.job_id: hold_cost(eng, world, selector, j, event, d_s) for j in jobs}
    b_cost = {j.job_id: burst_cost(eng, world, selector, j, event) for j in jobs}
    plans = []
    for f in sorted(world.cap_levels, reverse=True):
        cap_cost = {j.job_id: world.cost.c_delay * j.gpus * (d_s / 3600.0) * (1.0 - world.speed_fn(j, f)) * useful[j.job_id]
                    for j in jobs}
        total, p = sum(cap_cost[j.job_id] for j in jobs), _power_under_cap(eng, params, f)
        kw = {j.job_id: start_delta_kw(params, j, f) for j in jobs}
        option = {j.job_id: ("burst" if can_burst(world, j) and b_cost[j.job_id] < s_cost[j.job_id] else "suspend") for j in jobs}
        cost = {i: (b_cost[i] if option[i] == "burst" else s_cost[i]) for i in option}
        cands = [j for j in jobs if kw[j.job_id] > 0]
        actions = []
        for job in _order(cands, selector, rng, lambda j: cost[j.job_id] / kw[j.job_id]):
            if p <= p_star:
                break
            p -= kw[job.job_id]
            total += cost[job.job_id] - cap_cost[job.job_id]
            actions.append((job, option[job.job_id]))
        plans.append((p > p_star, total, -f, f, actions))
    _, _, _, f, actions = min(plans, key=lambda x: x[:3])
    if f < 1.0:
        eng.set_cap(f, world.speed_fn)
    eng.info["cap_level"] = f
    for job, opt in actions:
        if opt == "suspend" or not eng.burst(job):      # a blocked burst (cloud GPU cap) falls back to a suspend
            eng.suspend(job)

    def on_hold(job):
        d_rem = event.t_end - eng.now
        if d_rem > 0 and can_burst(world, job) and (burst_cost(eng, world, selector, job, event, running=False)
                          < hold_cost(eng, world, selector, job, event, d_rem)):
            eng.burst(job)
    eng.on_hold = on_hold


POLICIES = {"none": policy_none, "delay": delay_only, "cap": cap_only, "burst": burst_only, "mix": mix}
USES_PREDICTOR = {"delay", "burst", "mix"}       # for the row's predictor / selector columns
USES_SELECTOR = {"burst", "mix"}


def run_policy(snapshot, event, name, world, predictor_name="", selector="predictor", seed=0, base=None,
               return_runs=False):
    """Paired run of one policy on one event; returns one result row (metrics.COLUMNS).
    Pass `base` (from paired.run_base) to share one baseline continuation between policies of the same event."""
    if base is None:
        base = run_base(snapshot, event, world.params, world.h_after_s)
    p_star = target_kw(base.tracker, event.t_start, event.t_end, event.cut)
    policy = policy_none if name == "none" else partial(POLICIES[name], world=world, selector=selector,
                                                        rng=event_rng(seed, event))
    pol = run_event(snapshot, event, policy, world.params, p_star, world.h_after_s, world.restore_s, world.cloud)
    shed = shed_metrics(base.tracker, pol.tracker, event, world.h_after_s)
    row = event_row(event, shed, base.engine, pol.engine, world.cost, world.cloud.boot_delay_s, name,
                    predictor_name if name in USES_PREDICTOR else "")
    if name in USES_PREDICTOR and selector == "predictor":     # share of running GPUs whose job has a real prediction row
        run = [snapshot.jobs[i] for i in snapshot.running]
        row["pred_coverage"] = sum(j.gpus for j in run if j.job_id in world.predictor.rows) / max(1, sum(j.gpus for j in run))
    if name in USES_SELECTOR:
        row["selector"] = f"predictor:{predictor_name}" if selector == "predictor" else selector
    return (row, base, pol) if return_runs else row
