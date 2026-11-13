"""Shared look for the paper figures: IEEE column width, one fixed colour per policy."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "out"
WIDTH_IN = 3.5

COLORS = {"delay": "#4c78a8", "cap": "#f58518", "burst": "#54a24b", "mix": "#b279a2", "mix+oracle": "#e45756"}
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False})


def new_figure(height_in=2.3):
    return plt.subplots(figsize=(WIDTH_IN, height_in), constrained_layout=True)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.pdf"
    fig.savefig(path)
    print("wrote", path)
    return path


def label_policy(row):
    """Policy name for colours and legends; the oracle upper bound of the mix is its own entry."""
    return "mix+oracle" if row["policy"] == "mix" and row["predictor"] == "oracle" else row["policy"]
