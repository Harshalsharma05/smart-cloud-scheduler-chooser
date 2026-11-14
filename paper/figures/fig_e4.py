"""F9 (E4): user impact (change in wait, missed deadlines) and rebound energy by policy, median over events."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import COLORS, WIDTH_IN, load_runs, plt, save  # noqa: E402

df = load_runs("e4")
g = df.groupby("who")[["d_wait_min", "missed_deadlines", "rebound_mwh"]].median()
who = [w for w in COLORS if w in g.index]
fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN, 2.0), constrained_layout=True)
for ax, col, title in zip(axes, g.columns, ("change in wait (min)", "missed deadlines", "rebound (MWh)")):
    ax.bar(range(len(who)), [g.loc[w, col] for w in who], color=[COLORS[w] for w in who])
    ax.set_title(title)
    ax.set_xticks(range(len(who)), [w.replace("mix+oracle", "mix+or.").replace("delay+oracle", "del.+or.") for w in who], rotation=60, ha="right")
save(fig, "e4_user_impact")
