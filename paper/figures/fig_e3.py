"""E3: median $/MWh and cloud money wasted on Failed/Killed jobs by burst selector (burst-only and mix)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import COLORS, RESULTS, WIDTH_IN, plt, save  # noqa: E402

df = pd.read_csv(RESULTS / "e3" / "summary.csv", dtype={"selector": str, "predictor": str, "setting": str}).fillna({"selector": "", "predictor": "", "setting": ""})
event = df[(df["duration_h"] == 2) & (abs(df["cut"] - 0.2) < 1e-9)]          # the default event
order = ["none-largest", "random"] + sorted(s for s in event["selector"].unique() if s.startswith("predictor:"))
fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.3), constrained_layout=True, sharey=False)
for ax, col, title in ((axes[0], "usd_per_mwh_median", "k$/MWh shed"), (axes[1], "cloud_waste_fail_usd_median", "cloud $ on Failed/Killed")):
    for i, pol in enumerate(("burst", "mix")):
        sub = event[event["policy"] == pol].set_index("selector")
        vals = np.array([sub[col].get(s, np.nan) for s in order], dtype=float)
        vals = vals / 1000 if col.startswith("usd") else vals
        ax.barh(np.arange(len(order)) + (i - 0.5) * 0.38, vals, 0.38, color=COLORS[pol], label=pol)
    ax.set_title(title)
    ax.set_yticks(range(len(order)), [s.replace("predictor:", "") for s in order] if ax is axes[0] else [])
axes[0].legend(frameon=False)
save(fig, "e3_selectors")
