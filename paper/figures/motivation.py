"""S10: figure F2 (Section 2, full page width) and the numbers behind it.

Four panels: (a) job and GPU-hour shares by status, (b) CDF of wall time by status, (c) one test week of baseline facility power
with the 17:00-20:00 America/Los_Angeles windows shaded, (d) histogram of per-job mean GPU utilisation.
Everything is computed for the test window (jobs submitted in [TEST_START, TEST_END)) from the same simulator input as the experiments
(pool auto, hold rule not involved: this is the baseline run without events). The test window lies entirely in PST (UTC-8): DST ended
on 5 Nov 2017, before TEST_START, so the 17:00-20:00 window is a fixed UTC offset here.

Writes results/motivation/stats.csv (name, value, note), paper/figures/out/f2_motivation.{pdf,png} and a macro block in paper/numbers.tex
(between the markers "% BEGIN motivation" and "% END motivation"; everything outside the block is kept).
Run from the repo root: python paper/figures/motivation.py   (about 4 minutes: one baseline pass over the trace)
"""
import copy
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from flex.engine import Engine  # noqa: E402
from flex.models import load_config  # noqa: E402
from flex.paired import baseline_snapshots  # noqa: E402
from flex.policies import progress  # noqa: E402
from flex.power import PowerTracker, params_from_config  # noqa: E402
from style import WIDTH_IN, plt, save  # noqa: E402
from traces.philly import load_philly, ml_config, utc_epoch  # noqa: E402

STATUSES = ["Pass", "Killed", "Failed"]
# one fixed colour, hatch and line style per status (distinct grey level and pattern for grey-print)
STYLE = {"Pass": dict(color="#9ecae9", hatch="", ls="-"), "Killed": dict(color="#fdae61", hatch="..", ls="--"),
         "Failed": dict(color="#b2182b", hatch="//", ls=":")}
LINEC = {"Pass": "#2171b5", "Killed": "#d94801", "Failed": "#b2182b"}      # line colour per status (darker than the bar fills)
GRID_MAX_S = 43200
PT_OFFSET_S = -8 * 3600
OUT = ROOT / "paper" / "figures" / "out"
STATS = ROOT / "results" / "motivation" / "stats.csv"
WEEK_START = "2017-11-27"          # a Monday, no holiday, inside the test window


def pt_midnight(date_str):
    """UTC epoch of 00:00 America/Los_Angeles (PST) on a date inside the test window."""
    return utc_epoch(date_str) - PT_OFFSET_S


def main():
    cfg = load_config()
    L = load_philly(cfg)
    mc = ml_config()
    jobs, events, G = L["jobs"], L["events"], L["G"]
    epoch0 = utc_epoch(mc.SIM_START)
    t_lo, t_hi = utc_epoch(mc.TEST_START) - epoch0, utc_epoch(mc.TEST_END) - epoch0
    test = [j for j in jobs if not j.warmup and not j.carry]
    df = pd.DataFrame([dict(job_id=j.job_id, status=j.status, gpus=j.gpus, wall_s=j.wall_s) for j in test])
    df["gpu_h"] = df["gpus"] * df["wall_s"] / 3600.0
    stats = []

    def add(name, value, note=""):
        stats.append(dict(name=name, value=value, note=note))

    add("test_jobs", len(df), "jobs submitted in the test window (simulator input, no warm-up or carry-over jobs)")
    add("test_pool_gpus", G, "pool G used by the simulator (auto)")
    add("test_job_gpus_requested", int(df["gpus"].sum()), "sum of GPUs requested over all test jobs")
    add("test_gpu_hours", df["gpu_h"].sum(), "sum of gpus x wall_s / 3600")
    for s in STATUSES:
        d = df[df["status"] == s]
        add(f"jobs_{s.lower()}", len(d), "job count")
        add(f"jobshare_{s.lower()}_pct", 100 * len(d) / len(df), "share of jobs, %")
        add(f"gpuh_{s.lower()}", d["gpu_h"].sum(), "GPU-hours")
        add(f"gpuhshare_{s.lower()}_pct", 100 * d["gpu_h"].sum() / df["gpu_h"].sum(), "share of GPU-hours, %")
        add(f"wall_median_{s.lower()}_min", d["wall_s"].median() / 60, "median wall time, minutes")
        add(f"wall_p90_{s.lower()}_min", d["wall_s"].quantile(0.9) / 60, "90th percentile wall time, minutes")
        add(f"within1h_{s.lower()}_pct", 100 * (d["wall_s"] <= 3600).mean(), "share of this status ending within 1 h of first start, %")
    fk = df[df["status"] != "Pass"]
    add("within1h_failkill_pct", 100 * (fk["wall_s"] <= 3600).mean(), "share of Failed+Killed jobs ending within 1 h, %")
    add("gpuhshare_failkill_pct", 100 * fk["gpu_h"].sum() / df["gpu_h"].sum(), "share of GPU-hours of Failed+Killed jobs, %")

    # utilisation (philly_job_util.parquet, jobs of the test window only)
    ut = pd.read_parquet(ROOT / "data" / "processed" / "philly_job_util.parquet")
    ut = ut[ut["job_id"].isin(set(df["job_id"]))]
    u = ut["gpu_util_mean"].to_numpy()
    mean_job, mean_min = float(u.mean()), float(np.average(u, weights=ut["n_minutes"]))
    add("util_jobs_with_data", len(ut), "test jobs with a measured utilisation")
    add("util_mean_jobweighted_pct", 100 * mean_job, "mean of per-job mean GPU utilisation, %")
    add("util_mean_minuteweighted_pct", 100 * mean_min, "per-job mean utilisation weighted by measured minutes, %")
    add("util_median_pct", 100 * float(np.median(u)), "median per-job mean utilisation, %")

    # baseline run: power series and snapshots at the 90 event starts
    eng = Engine(copy.deepcopy(jobs), G, check_every=cfg["check_every"])
    tracker = PowerTracker(params_from_config(cfg))
    tracker.attach(eng)
    starts = sorted({e.t_start for e in events})
    n_ev = {t: sum(e.t_start == t for e in events) for t in starts}
    run_jobs = run_gpus = beyond_jobs = beyond_gpus = 0
    per_start = []
    for t_s, snap in baseline_snapshots(eng, starts):
        running = [snap.jobs[i] for i in snap.running]
        el = np.array([progress(snap, j) for j in running])
        g = np.array([j.gpus for j in running])
        b = el > GRID_MAX_S
        w = n_ev[t_s]
        run_jobs += w * len(running)
        run_gpus += w * int(g.sum())
        beyond_jobs += w * int(b.sum())
        beyond_gpus += w * int(g[b].sum())
        per_start.append(100 * b.mean())
    eng.run_to_drain()
    add("event_starts", len(events), "E1 events (90), starts with the baseline engine state")
    add("running_jobs_at_starts_mean", run_jobs / len(events), "mean number of running jobs at an event start")
    add("running_beyond_grid_jobs_pct", 100 * beyond_jobs / run_jobs, "running jobs with elapsed > 43,200 s at the event starts, % (pooled over the 90 events)")
    add("running_beyond_grid_gpus_pct", 100 * beyond_gpus / run_gpus, "running GPUs with elapsed > 43,200 s at the event starts, %")
    add("running_beyond_grid_jobs_median_pct", float(np.median(per_start)), "same, median over event starts, %")

    steps = np.array(tracker.steps)                       # (t, kW)
    mean_kw = tracker.mean_kw(t_lo, t_hi)
    sel = steps[(steps[:, 0] >= t_lo) & (steps[:, 0] < t_hi)]
    day0 = int(np.ceil((pt_midnight(mc.TEST_START) - epoch0) / 86400.0))
    days = int(((t_hi - (pt_midnight(mc.TEST_START) - epoch0)) // 86400))
    start_pt = pt_midnight(mc.TEST_START) - epoch0
    ev_e = ev_t = 0.0
    n_days = 0
    for d in range(days):
        a, b = start_pt + d * 86400 + 17 * 3600, start_pt + d * 86400 + 20 * 3600
        if a >= t_lo and b <= t_hi:
            ev_e += tracker.energy_kwh(a, b)
            ev_t += 3.0
            n_days += 1
    eve_kw = ev_e / ev_t
    add("power_mean_kw", mean_kw, "mean baseline facility power over the test window, kW")
    add("power_peak_kw", float(sel[:, 1].max()), "highest baseline facility power step in the test window, kW")
    add("power_evening_mean_kw", eve_kw, f"mean over the 17:00-20:00 PT windows ({n_days} days), kW")
    add("power_evening_over_overall", eve_kw / mean_kw, "evening mean / overall mean")
    add("power_idle_floor_kw", params_from_config(cfg).pue * G * (params_from_config(cfg).p_idle_w + cfg["power"]["h_idle_w"]) / 1000.0,
        "all-GPU-idle floor of the pool, kW")
    add("week_start_pt", WEEK_START, "start of the week shown in panel (c)")

    # ---------------- figure
    fig = plt.figure(figsize=(2 * WIDTH_IN + 0.2, 5.2))
    gs = fig.add_gridspec(2, 2, left=0.105, right=0.99, top=0.95, bottom=0.17, hspace=0.62, wspace=0.22)
    ax_a, ax_b, ax_c, ax_d = (fig.add_subplot(gs[i, j]) for i in (0, 1) for j in (0, 1))
    # (a) shares by status
    rows = [("jobs", "job count"), ("gpu_h", "GPU-hours")]
    for k, (col, lab) in enumerate(rows):
        left = 0.0
        tot = df[col].sum() if col == "gpu_h" else len(df)
        for s in STATUSES:
            v = (df[df["status"] == s]["gpu_h"].sum() if col == "gpu_h" else int((df["status"] == s).sum())) / tot * 100
            ax_a.barh(k, v, left=left, height=0.55, color=STYLE[s]["color"], hatch=STYLE[s]["hatch"], edgecolor="black", lw=0.5)
            if v >= 6:
                ax_a.text(left + v / 2, k, f"{v:.0f}%", ha="center", va="center", fontsize=6.5,
                          bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
            left += v
    ax_a.set_yticks([0, 1], [r[1] for r in rows])
    ax_a.set_xlim(0, 100)
    ax_a.set_xlabel("share of test-window total (%)")
    ax_a.set_title("(a) jobs and GPU-hours by status")
    ax_a.invert_yaxis()
    # (b) CDF of wall time
    for s in STATUSES:
        w = np.sort(df[df["status"] == s]["wall_s"].to_numpy() / 60.0)
        w = np.maximum(w, 1 / 60)
        ax_b.plot(w, np.arange(1, len(w) + 1) / len(w), color=LINEC[s], ls=STYLE[s]["ls"], lw=1.3,
                  label=s)
    ax_b.set_xscale("log")
    ax_b.set_xlabel("wall time (minutes, log scale)")
    ax_b.set_ylabel("cumulative share of jobs")
    ax_b.axvline(60, color="0.5", lw=0.6)
    ax_b.set_title("(b) wall time by status")
    ax_b.grid(lw=0.3, color="0.85")
    # (c) one week of baseline power
    w0 = pt_midnight(WEEK_START) - epoch0
    pieces = tracker.pieces(w0, w0 + 7 * 86400)
    xs, ys, t = [], [], w0
    for dt, p in pieces:
        xs += [(t - w0) / 86400.0, (t + dt - w0) / 86400.0]
        ys += [p / 1000.0] * 2
        t += dt
    ax_c.plot(xs, ys, color="black", lw=0.7)
    for d in range(7):
        ax_c.axvspan(d + 17 / 24, d + 20 / 24, color="#fdae61", alpha=0.45, lw=0)
    ax_c.set_xlim(0, 7)
    ax_c.set_xticks(np.arange(7) + 0.5, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax_c.set_ylabel("baseline facility power (MW)")
    ax_c.set_title("(c) one test week; shaded: 17:00-20:00 PT")
    ax_c.grid(axis="y", lw=0.3, color="0.85")
    # (d) utilisation histogram
    ax_d.hist(100 * u, bins=np.linspace(0, 100, 26), color="#bdbdbd", edgecolor="black", lw=0.4)
    ax_d.axvline(100 * mean_min, color="black", ls="--", lw=1.1)
    ax_d.text(100 * mean_min + 1.5, ax_d.get_ylim()[1] * 0.93, f"minute-weighted\nmean {100 * mean_min:.0f}%", fontsize=6.5, va="top")
    ax_d.set_xlabel("per-job mean GPU utilisation (%)")
    ax_d.set_ylabel("jobs")
    ax_d.set_title("(d) GPU utilisation per job")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=STYLE[s]["color"], hatch=STYLE[s]["hatch"], edgecolor="black", lw=0.5, label=s) for s in STATUSES]
    handles += [Line2D([0], [0], color="black", ls=STYLE[s]["ls"], lw=1.1, label=f"{s} (CDF)") for s in STATUSES]
    fig.legend(handles=handles[:3], loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.3, 0.0))
    fig.legend(handles=[Line2D([0], [0], color=LINEC[s], ls=STYLE[s]["ls"], lw=1.3, label=s) for s in STATUSES] +
               [Patch(facecolor="#fdae61", alpha=0.45, label="17:00-20:00 PT")],
               loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.74, 0.0))
    fig.text(0.3, 0.075, "status (panel a)", ha="center", fontsize=6.5)
    fig.text(0.74, 0.075, "line style (panel b), shading (panel c)", ha="center", fontsize=6.5)
    path = save(fig, "f2_motivation")
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)

    # ---------------- stats and macros
    S = pd.DataFrame(stats)
    STATS.parent.mkdir(parents=True, exist_ok=True)
    S.to_csv(STATS, index=False)
    print(S.to_string(index=False))
    write_macros(S)


def camel(name):
    """stats name -> CamelCase macro name made of letters only (digits are spelled out)."""
    digits = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
    parts = re.split(r"[_]", name)
    return "Mot" + "".join((lambda q: q[:1].upper() + q[1:])("".join(digits.get(c, c) for c in p)) for p in parts)


def fmt(name, v):
    if isinstance(v, str):
        try:
            v = float(v)                     # stats.csv read back as text
        except ValueError:
            return v
    if name.endswith("_pct") or name.endswith("_min") or "_mean_kw" in name or name.endswith("_kw") or name.endswith("_over_overall"):
        return f"{v:.3f}" if name.endswith("_over_overall") else f"{v:.0f}" if abs(v) >= 100 else f"{v:.1f}"
    if float(v).is_integer():
        return f"{int(v):,}".replace(",", "{,}")
    return f"{v:,.0f}".replace(",", "{,}")


def write_macros(S):
    B = chr(92)
    lines = ["% BEGIN motivation (written by paper/figures/motivation.py from results/motivation/stats.csv; do not edit)"]
    for r in S.itertuples():
        lines.append(f"{B}newcommand{{{B}{camel(r.name)}}}{{{fmt(r.name, r.value)}}}")
    lines.append("% END motivation")
    block = "\n".join(lines)
    p = ROOT / "paper" / "numbers.tex"
    s = p.read_text(encoding="utf8")
    if "% BEGIN motivation" in s:
        s = re.sub(r"% BEGIN motivation.*?% END motivation", lambda m: block, s, flags=re.S)
    else:
        s = s.rstrip("\n") + "\n\n" + block + "\n"
    p.write_text(s, encoding="utf8")
    print("wrote", p, len(S), "macros")


if __name__ == "__main__":
    main()
