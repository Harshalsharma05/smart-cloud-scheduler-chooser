"""E1: median $/MWh shed by (duration, cut) cell and policy, bars with the IQR. Run from the repo root."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import COLORS, RESULTS, label_policy, new_figure, save  # noqa: E402

df = pd.read_csv(RESULTS / "e1" / "summary.csv", dtype={"selector": str, "predictor": str, "setting": str}).fillna({"selector": "", "predictor": "", "setting": ""})
df["who"] = df.apply(label_policy, axis=1)
cells = sorted(df[["duration_h", "cut"]].drop_duplicates().itertuples(index=False), key=lambda c: (c[1], c[0]))
who = [w for w in COLORS if w in set(df["who"])]
fig, ax = new_figure(2.4)
width = 0.8 / len(who)
for i, w in enumerate(who):
    sub = df[df["who"] == w].set_index(["duration_h", "cut"])
    med = np.array([sub["usd_per_mwh_median"].get(c, np.nan) for c in cells], dtype=float) / 1000
    lo = np.array([sub["usd_per_mwh_q25"].get(c, np.nan) for c in cells], dtype=float) / 1000
    hi = np.array([sub["usd_per_mwh_q75"].get(c, np.nan) for c in cells], dtype=float) / 1000
    x = np.arange(len(cells)) + (i - (len(who) - 1) / 2) * width
    ax.bar(x, med, width, color=COLORS[w], label=w, yerr=[np.nan_to_num(med - lo), np.nan_to_num(hi - med)],
           error_kw={"lw": 0.5, "capsize": 0.8})
ax.set_xticks(range(len(cells)), [f"{int(d)} h\n{round(c * 100)}%" for d, c in cells])
ax.set_ylabel("cost per MWh shed (k$/MWh)")
ax.legend(ncol=3, frameon=False)
save(fig, "e1_usd_per_mwh")
