"""E4: user impact (change in wait, change in bounded slowdown) and rebound energy by policy, median over events."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import COLORS, RESULTS, WIDTH_IN, label_policy, plt, save  # noqa: E402

df = pd.read_csv(RESULTS / "e4" / "runs.csv", dtype={"selector": str, "predictor": str, "setting": str}).fillna({"selector": "", "predictor": "", "setting": ""})
df["who"] = df.apply(label_policy, axis=1)
g = df.groupby("who")[["d_wait_min", "d_slowdown", "rebound_mwh"]].median()
who = [w for w in COLORS if w in g.index]
fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN, 2.0), constrained_layout=True)
for ax, col, title in zip(axes, g.columns, ("change in wait (min)", "change in slowdown", "rebound (MWh)")):
    ax.bar(range(len(who)), [g.loc[w, col] for w in who], color=[COLORS[w] for w in who])
    ax.set_title(title)
    ax.set_xticks(range(len(who)), [w.replace("mix+oracle", "mix+or.") for w in who], rotation=60, ha="right")
save(fig, "e4_user_impact")
