"""Shed metrics for one grid event (SIM_PLAN D6). Power in kW, energy in MWh, times in simulated seconds."""
from dataclasses import dataclass

from . import cloud as cloudlib


@dataclass
class ShedMetrics:
    target_kw: float
    base_mean_kw: float
    shed_mwh: float
    compliance: float       # fraction of event time with P <= 1.01 P*
    shortfall_kw: float     # max(0, max P - P*) over the event
    rebound_mwh: float      # integral of (P_policy - P_base)+ over [t_e, t_e + H_after)


def target_kw(base_tracker, t_s, t_e, cut):
    """P* = (1 - cut) x mean baseline power over [t_s, t_e): one flat target per event."""
    return (1.0 - cut) * base_tracker.mean_kw(t_s, t_e)


def shed_mwh(base_tracker, pol_tracker, t_s, t_e):
    return (base_tracker.energy_kwh(t_s, t_e) - pol_tracker.energy_kwh(t_s, t_e)) / 1000.0


def compliance(pol_tracker, t_s, t_e, p_star, tol=1.01):
    """Fraction of [t_s, t_e) during which power is at most tol x P*."""
    ok = sum(dt for dt, p in pol_tracker.pieces(t_s, t_e) if p <= tol * p_star)
    return ok / (t_e - t_s)


def shortfall_kw(pol_tracker, t_s, t_e, p_star):
    return max(0.0, max(p for _, p in pol_tracker.pieces(t_s, t_e)) - p_star)


def rebound_mwh(base_tracker, pol_tracker, t0, t1):
    """Integral of the positive part of (P_policy - P_base) over [t0, t1)."""
    times = {t0, t1}
    for tr in (base_tracker, pol_tracker):
        times.update(t for t, _ in tr.steps if t0 < t < t1)
    times = sorted(times)
    total = 0.0
    for a, b in zip(times, times[1:]):
        d = pol_tracker.power_at(a) - base_tracker.power_at(a)
        if d > 0:
            total += d * (b - a)
    return total / 3600.0 / 1000.0


def shed_metrics(base_tracker, pol_tracker, event, h_after_s):
    """All D6 metrics for one event, given the baseline and policy power trackers."""
    p_star = target_kw(base_tracker, event.t_start, event.t_end, event.cut)
    return ShedMetrics(
        target_kw=p_star,
        base_mean_kw=base_tracker.mean_kw(event.t_start, event.t_end),
        shed_mwh=shed_mwh(base_tracker, pol_tracker, event.t_start, event.t_end),
        compliance=compliance(pol_tracker, event.t_start, event.t_end, p_star),
        shortfall_kw=shortfall_kw(pol_tracker, event.t_start, event.t_end, p_star),
        rebound_mwh=rebound_mwh(base_tracker, pol_tracker, event.t_end, event.t_end + h_after_s),
    )


# ---- cost of an event (SIM_PLAN D7), cloud waste and user impact ----

COLUMNS = ["exp", "event_id", "t_start", "duration_h", "cut", "policy", "selector", "predictor", "setting", "target_kw",
           "base_mean_kw", "shed_mwh", "compliance", "shortfall_kw", "rebound_mwh", "cloud_usd", "cloud_gpu_h",
           "cloud_waste_fail_usd", "cloud_waste_short_usd", "dgpuh_good", "dgpuh_all", "missed_deadlines", "cost_usd",
           "usd_per_mwh", "cost_usd_wfail1", "usd_per_mwh_wfail1", "compliance_ok", "feasible", "d_wait_min", "d_slowdown", "n_suspended", "n_burst", "cap_level", "pred_coverage", "runtime_s", "seed"]
TAU_S = 10                   # bounded-slowdown threshold (standard definition)
SHORT_EXTRA_S = 600          # a burst is "short" if the remaining runtime was < boot_delay + 10 min


@dataclass
class CostParams:
    c_delay: float           # USD per GPU-hour of delay (default: the cloud price)
    m: float                 # missed-deadline multiplier
    w_fail: float            # delay weight of Failed/Killed jobs


def cost_params_from_config(cfg) -> CostParams:
    return CostParams(cfg["cost"]["c_delay_factor"] * cfg["cloud"]["price_per_gpu_h"], cfg["cost"]["m"], cfg["cost"]["w_fail"])


def _measured_jobs(base_eng, pol_eng):
    """(base job, policy job) for every finished job present in both runs (warm-up and carry-over included), in job_id order."""
    for jid in sorted(base_eng.jobs):
        jb, jp = base_eng.jobs[jid], pol_eng.jobs.get(jid)
        if jp is not None and jb.end is not None and jp.end is not None:
            yield jb, jp


def delay_gpuh(base_eng, pol_eng):
    """(dGPUh_good, dGPUh_all): sum of max(0, end_policy - end_base) x gpus in hours; good = Pass jobs only."""
    good = total = 0.0
    for jb, jp in _measured_jobs(base_eng, pol_eng):
        d = max(0, jp.end - jb.end) * jb.gpus / 3600.0
        total += d
        if jb.status == "Pass":
            good += d
    return good, total


def missed_deadlines(base_eng, pol_eng):
    """(count, GPU-hours of wall time) of deadline jobs that meet the deadline in the baseline but miss it in the policy run."""
    n, gpuh = 0, 0.0
    for jb, jp in _measured_jobs(base_eng, pol_eng):
        if jb.deadline is not None and jb.end <= jb.deadline < jp.end:
            n += 1
            gpuh += jb.gpus * jb.wall_s / 3600.0
    return n, gpuh


def cloud_terms(pol_eng, boot_delay_s):
    """Cloud bill, GPU-hours, and the two kinds of waste. Waste classes do not overlap: Failed/Killed jobs first,
    then Pass jobs whose remaining work at burst was < boot_delay + 10 min."""
    cl = pol_eng.cloud
    if cl is None:
        return dict(cloud_usd=0.0, cloud_gpu_h=0.0, cloud_waste_fail_usd=0.0, cloud_waste_short_usd=0.0, n_burst=0)
    usd = gpu_h = fail = short = 0.0
    for inst in cl.instances:
        job = pol_eng.jobs[inst.job_id]
        c = cloudlib.instance_cost(cl.params, inst, pol_eng.now)
        usd += c
        if inst.ready is not None:
            gpu_h += inst.gpus * ((inst.released if inst.released is not None else pol_eng.now) - inst.ready) / 3600.0
        if job.status != "Pass":
            fail += c
        elif job.burst_rem is not None and job.burst_rem < boot_delay_s + SHORT_EXTRA_S:
            short += c
    return dict(cloud_usd=usd, cloud_gpu_h=gpu_h, cloud_waste_fail_usd=fail, cloud_waste_short_usd=short,
                n_burst=len(cl.instances))


def _first_run(job):
    return next((s for s, _, w, _ in job.segments if w in ("onprem", "cloud")), None)


def user_impact(base_eng, pol_eng):
    """(mean change in wait in minutes, mean change in bounded slowdown) over measured jobs."""
    dw = ds = 0.0
    n = 0
    for jb, jp in _measured_jobs(base_eng, pol_eng):
        wb, wp = _first_run(jb) - jb.submit, _first_run(jp) - jp.submit
        sb = max(1.0, (jb.end - jb.submit) / max(jb.wall_s, TAU_S))
        sp = max(1.0, (jp.end - jp.submit) / max(jp.wall_s, TAU_S))
        dw += wp - wb
        ds += sp - sb
        n += 1
    return (dw / n / 60.0, ds / n) if n else (0.0, 0.0)


def event_row(event, shed, base_eng, pol_eng, cost, boot_delay_s, policy, predictor):
    """One result row (SIM_PLAN section 4). Columns that do not exist yet stay None."""
    cl = cloud_terms(pol_eng, boot_delay_s)
    good, total = delay_gpuh(base_eng, pol_eng)
    n_miss, miss_gpuh = missed_deadlines(base_eng, pol_eng)
    d_good = good + cost.w_fail * (total - good)          # Failed/Killed delay counts with weight w_fail
    cost_usd = cl["cloud_usd"] + cost.c_delay * d_good + cost.m * cost.c_delay * miss_gpuh
    target_mwh = event.cut * shed.base_mean_kw * (event.t_end - event.t_start) / 3600.0 / 1000.0
    ok = shed.shed_mwh >= 0.1 * target_mwh and shed.shed_mwh > 0
    usd_per_mwh = cost_usd / shed.shed_mwh if ok else None
    cost_wfail1 = cl["cloud_usd"] + cost.c_delay * total + cost.m * cost.c_delay * miss_gpuh     # same run, w_fail = 1
    d_wait, d_slow = user_impact(base_eng, pol_eng)
    row = dict.fromkeys(COLUMNS)
    row.update(
        event_id=event.event_id, t_start=event.t_start, duration_h=event.duration_h, cut=event.cut, policy=policy,
        predictor=predictor, target_kw=shed.target_kw, base_mean_kw=shed.base_mean_kw, shed_mwh=shed.shed_mwh,
        compliance=shed.compliance, shortfall_kw=shed.shortfall_kw, rebound_mwh=shed.rebound_mwh,
        cloud_usd=cl["cloud_usd"], cloud_gpu_h=cl["cloud_gpu_h"], cloud_waste_fail_usd=cl["cloud_waste_fail_usd"],
        cloud_waste_short_usd=cl["cloud_waste_short_usd"], dgpuh_good=good, dgpuh_all=total, missed_deadlines=n_miss,
        cost_usd=cost_usd, usd_per_mwh=usd_per_mwh, cost_usd_wfail1=cost_wfail1,
        usd_per_mwh_wfail1=cost_wfail1 / shed.shed_mwh if ok else None, compliance_ok=bool(shed.compliance >= 0.99),
        d_wait_min=d_wait, d_slowdown=d_slow,
        n_suspended=sum(j.n_suspends > 0 for j in pol_eng.jobs.values()), n_burst=cl["n_burst"],
        cap_level=pol_eng.info.get("cap_level"))
    return row
