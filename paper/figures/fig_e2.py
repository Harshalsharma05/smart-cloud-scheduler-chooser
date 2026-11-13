"""E2: median $/MWh on the default event as the cloud price changes, one panel line per policy and boot delay."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import COLORS, RESULTS, new_figure, save  # noqa: E402

df = pd.read_csv(RESULTS / "e2" / "summary.csv", dtype={"selector": str, "predictor": str, "setting": str}).fillna({"selector": "", "predictor": "", "setting": ""})
df[["price_x", "boot_s"]] = df["setting"].str.extract(r"price_x([\d.]+)_boot(\d+)").astype(float)
fig, ax = new_figure(2.3)
styles = {60: ":", 300: "-", 900: "--"}
for pol in ("delay", "cap", "burst", "mix"):
    for boot, ls in styles.items():
        sub = df[(df["policy"] == pol) & (df["boot_s"] == boot)].sort_values("price_x")
        if len(sub) == 0 or (pol in ("delay", "cap") and boot != 300):
            continue                              # delay and cap do not depend on the cloud: draw them once
        ax.plot(sub["price_x"], sub["usd_per_mwh_median"] / 1000, ls, color=COLORS[pol], marker="o", ms=2, lw=0.9,
                label=pol if boot == 300 else f"{pol}, boot {int(boot)} s")
ax.set_xscale("log", base=2)
ax.set_xlabel("cloud price (x default)")
ax.set_ylabel("cost per MWh shed (k$/MWh)")
ax.legend(frameon=False, ncol=2)
save(fig, "e2_price_boot")
