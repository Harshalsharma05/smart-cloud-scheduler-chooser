"""F6 (E1 main): $/MWh shed vs event length, one panel per cut, a line per policy (median, IQR band). Run from the repo root.

Usage: python paper/figures/fig_e1.py [wfail1]   (default: primary cost, w_fail = 0 as in flex.yaml)
A (cell, policy) whose median compliance is below 99% is not a $/MWh point: it is listed under the panel as
"shed x% of target" (median over its events), so a non-compliant policy never appears as a cost winner.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import COLORS, WIDTH_IN, load_runs, plt, save  # noqa: E402

col = "usd_per_mwh_wfail1" if "wfail1" in sys.argv else "usd_per_mwh"
df = load_runs("e1")
cuts = sorted(df["cut"].unique())
durs = sorted(df["duration_h"].unique())
who = [w for w in COLORS if w in set(df["who"])]
fig, axes = plt.subplots(1, len(cuts), figsize=(WIDTH_IN * 2, 2.9), sharey=True, constrained_layout=True)
for ax, cut in zip(np.atleast_1d(axes), cuts):
    notes = {}                                   # non-compliant policy -> shed % of target per event length
    for w in who:
        xs, med, lo, hi = [], [], [], []
        for d in durs:
            s = df[(df["who"] == w) & (df["cut"] == cut) & (df["duration_h"] == d)]
            if s.empty:
                continue
            if s["compliance"].median() < 0.99:
                notes.setdefault(w, {})[d] = f"{s['shed_pct'].median():.0f}"
                continue
            v = s[col].dropna()
            if v.empty:
                continue
            xs.append(d); med.append(v.median() / 1000); lo.append(v.quantile(.25) / 1000); hi.append(v.quantile(.75) / 1000)
        if xs:
            ax.plot(xs, med, marker="o", ms=2.5, lw=1, color=COLORS[w], label=w, ls="--" if "oracle" in w else "-")
            ax.fill_between(xs, lo, hi, color=COLORS[w], alpha=0.15, lw=0)
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_xticks(durs, [f"{d:g}" for d in durs])
    ax.set_xlabel("event length (h)")
    ax.set_title(f"cut {round(cut * 100)}%")
    if notes:                                    # one line per non-compliant policy: shed % of target at each event length
        head = "not compliant, shed % of target at " + " / ".join(f"{d:g}" for d in durs) + " h:"
        body = [f"{w}: " + " / ".join(v.get(d, "ok") for d in durs) for w, v in notes.items()]
        ax.text(0.02, -0.30, "\n".join([head] + body), transform=ax.transAxes, fontsize=5.5, va="top")
np.atleast_1d(axes)[0].set_ylabel("cost per MWh shed (k$/MWh)")
h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside lower center", ncol=len(l), frameon=False)
save(fig, "e1_usd_per_mwh" + ("_wfail1" if "wfail1" in sys.argv else ""))
