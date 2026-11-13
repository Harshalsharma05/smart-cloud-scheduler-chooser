"""Facility power model (SIM_PLAN D5) and the power-cap speed curve (D9).

Per GPU:   P_gpu(u, f) = min(P_idle + u (P_max - P_idle), f P_max)      (busy, on-prem)
Facility:  P = PUE x [ sum_busy gpus P_gpu + idle P_idle + G H_idle + busy H_busy ]     (kW = W / 1000)
Cloud GPUs are not on-prem power; their energy (for E5) is `cloud_energy_kwh`.
Power is piecewise constant between engine events, so the tracker integrates energy exactly.
"""
import bisect
from dataclasses import dataclass


@dataclass
class PowerParams:
    p_idle_w: float     # one GPU, idle
    p_max_w: float      # one GPU at full load
    h_idle_w: float     # host power per installed GPU, always on
    h_busy_w: float     # extra host power per busy GPU
    pue: float
    cloud_pue: float    # used only by cloud_energy_kwh (E5)


def params_from_config(cfg) -> PowerParams:
    p = cfg["power"]
    return PowerParams(p["p_idle_w"], p["p_max_w"], p["h_idle_w"], p["h_busy_w"], p["pue"], p["cloud_pue"])


def curve_from_config(cfg, variant=None):
    """Cap curve points [(r, speed), ...] sorted by r; variant = 'optimistic' or 'pessimistic'."""
    variant = variant or cfg["cap_curve"]["variant"]
    return sorted((float(r), float(s)) for r, s in cfg["cap_curve"][variant])


def gpu_power_w(params, util, f=1.0):
    """One busy GPU at mean utilisation `util` under cap level f (f = 1: no cap)."""
    return min(params.p_idle_w + util * (params.p_max_w - params.p_idle_w), f * params.p_max_w)


def power_kw(engine, params):
    """Facility power now, from the on-prem state of the engine (cloud jobs are not counted)."""
    busy_w, busy_gpus = 0.0, 0
    p_idle, span, cap_w = params.p_idle_w, params.p_max_w - params.p_idle_w, engine.cap_f * params.p_max_w
    jobs = engine.jobs
    for jid in sorted(engine.running):          # fixed order: the float sum must not depend on set order
        j = jobs[jid]
        busy_w += j.gpus * min(p_idle + j.util * span, cap_w)        # same as gpu_power_w, inlined for speed
        busy_gpus += j.gpus
    idle_gpus = engine.G - busy_gpus
    total_w = busy_w + idle_gpus * params.p_idle_w + engine.G * params.h_idle_w + busy_gpus * params.h_busy_w
    return params.pue * total_w / 1000.0


def start_delta_kw(params, job, f=1.0):
    """Extra facility power (kW) if `job` starts now: its GPUs go from idle to busy, plus host busy power (D10)."""
    return params.pue * job.gpus * (gpu_power_w(params, job.util, f) - params.p_idle_w + params.h_busy_w) / 1000.0


def interp(curve, r):
    """Piecewise-linear speed at ratio r; clamped to the end points."""
    if r <= curve[0][0]:
        return curve[0][1]
    for (r0, s0), (r1, s1) in zip(curve, curve[1:]):
        if r <= r1:
            return s0 + (s1 - s0) * (r - r0) / (r1 - r0)
    return curve[-1][1]


def make_speed_fn(params, curve):
    """speed_fn(job, f) for Engine.set_cap. r = f P_max / P_uncapped(u); r >= 1 means the cap does not bind."""
    def speed_fn(job, f):
        r = f * params.p_max_w / gpu_power_w(params, job.util)
        return 1.0 if r >= 1.0 else interp(curve, r)
    return speed_fn


class PowerTracker:
    """Records (t, kW) steps. Attach to an engine; the engine calls it after every event,
    after run_until moves the clock, and after every suspend/resume/set_cap/burst."""

    def __init__(self, params):
        self.params = params
        self.steps = []                          # (t, kW); the value holds until the next step
        self._last_mut = None

    def attach(self, engine):
        engine.on_event = self.record
        self.record(engine, engine.now)

    def record(self, engine, t):
        if self.steps:
            assert t >= self.steps[-1][0], "power steps must not go back in time"
            if engine.mut == self._last_mut:     # power state unchanged (e.g. an arrival that only queued): no new step
                return
        self._last_mut = engine.mut
        p = power_kw(engine, self.params)
        if self.steps and self.steps[-1][0] == t:    # several changes at one instant: keep the last
            self.steps[-1] = (t, p)
            return
        self.steps.append((t, p))

    def power_at(self, t):
        i = bisect.bisect_right(self.steps, (t, float("inf"))) - 1
        if i < 0:
            raise ValueError(f"no power recorded at or before t={t}")
        return self.steps[i][1]

    def pieces(self, t0, t1):
        """(dt, kW) pieces covering [t0, t1]; the last step is held up to t1."""
        if t1 <= t0:
            return []
        i = max(bisect.bisect_right(self.steps, (t0, float("inf"))) - 1, 0)
        if self.steps[i][0] > t0:
            raise ValueError(f"no power recorded at t={t0}")
        out, n = [], len(self.steps)
        while i < n:
            start = max(self.steps[i][0], t0)
            end = min(self.steps[i + 1][0], t1) if i + 1 < n else t1
            if end > start:
                out.append((end - start, self.steps[i][1]))
            if end >= t1:
                break
            i += 1
        return out

    def energy_kwh(self, t0, t1):
        """Exact integral of power over [t0, t1]."""
        return sum(dt * p for dt, p in self.pieces(t0, t1)) / 3600.0

    def mean_kw(self, t0, t1):
        return self.energy_kwh(t0, t1) * 3600.0 / (t1 - t0)


def cloud_energy_kwh(engine, params, now=None):
    """Energy of the cloud GPUs (for E5), from READY to RELEASED (or `now`): GPU power at util, no cap,
    plus the busy-host share, times the cloud PUE. Not part of on-prem power."""
    now = engine.now if now is None else now
    total_w_s = 0.0
    for inst in engine.cloud.instances:
        if inst.ready is None:
            continue
        end = inst.released if inst.released is not None else now
        j = engine.jobs[inst.job_id]
        total_w_s += inst.gpus * (gpu_power_w(params, j.util) + params.h_busy_w) * (end - inst.ready)
    return params.cloud_pue * total_w_s / 3.6e6
