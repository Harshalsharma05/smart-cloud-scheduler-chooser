"""Shed metrics for one grid event (SIM_PLAN D6). Power in kW, energy in MWh, times in simulated seconds."""
from dataclasses import dataclass


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
