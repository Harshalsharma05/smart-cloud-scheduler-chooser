"""F1 (teaser): facility power over the default event (first 2 h, 20% event) for the baseline, delay, cap, burst and mix
(last2 predictor), with the target P*. Re-runs one event from backend/ code. Run from the repo root."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import copy  # noqa: E402

from flex.cloud import Cloud  # noqa: E402
from flex.engine import Engine  # noqa: E402
from flex.models import load_config  # noqa: E402
from flex.paired import baseline_snapshots, run_base  # noqa: E402
from flex.policies import run_policy, world_from_config  # noqa: E402
from flex.predictions import load_predictions  # noqa: E402
from flex.metrics import target_kw  # noqa: E402
from style import COLORS, new_figure, save  # noqa: E402
from traces.philly import load_philly  # noqa: E402

cfg = load_config()
L = load_philly(cfg)
world = world_from_config(cfg, load_predictions(ROOT / cfg["paths"]["predictions"] / "last2.csv"))
ev = [e for e in L["events"] if e.duration_h == 2 and abs(e.cut - 0.2) < 1e-9][0]
eng = Engine(copy.deepcopy(L["jobs"]), L["G"], check_every=cfg["check_every"], cloud=Cloud(world.cloud))
(_, snap), = baseline_snapshots(eng, [ev.t_start])
base = run_base(snap, ev, world.params, world.h_after_s)
p_star = target_kw(base.tracker, ev.t_start, ev.t_end, ev.cut)
t0, t1 = ev.t_start, ev.t_end + 4 * 3600


def steps(tracker):
    xs, ys, t = [], [], t0
    for dt, p in tracker.pieces(t0, t1):
        xs += [t, t + dt]
        ys += [p, p]
        t += dt
    return [(x - ev.t_start) / 3600 for x in xs], ys


fig, ax = new_figure(2.3)
x, y = steps(base.tracker)
ax.plot(x, y, color="0.5", lw=1, label="baseline")
for name in ("delay", "cap", "burst", "mix"):
    _, _, pol = run_policy(snap, ev, name, world, "last2", "predictor", cfg["seed"], base, return_runs=True)
    x, y = steps(pol.tracker)
    ax.plot(x, y, color=COLORS[name], lw=0.9, label=name)
ax.axhline(p_star, color="k", ls=":", lw=0.8, label="target P*")
ax.axvspan(0, ev.duration_h, color="k", alpha=0.05, lw=0)
ax.set_xlabel("hours since event start")
ax.set_ylabel("facility power (kW)")
ax.legend(ncol=3, frameon=False, loc="lower right")
save(fig, "teaser_power")
