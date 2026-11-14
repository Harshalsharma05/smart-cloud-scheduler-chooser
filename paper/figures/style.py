"""Shared look for the paper figures: IEEE column width, one fixed colour per policy."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "out"
WIDTH_IN = 3.5

COLORS = {"delay": "#4c78a8", "cap": "#f58518", "burst": "#54a24b", "mix": "#b279a2", "mix+oracle": "#e45756", "delay+oracle": "#9ecae9"}
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
    return f"{row['policy']}+oracle" if row["policy"] in ("mix", "delay") and row["predictor"] == "oracle" else row["policy"]


def load_runs(exp="e1"):
    """Runs of one experiment with a 'who' label and shed_pct = shed MWh as % of the event target (cut x baseline energy)."""
    import pandas as pd
    df = pd.read_csv(RESULTS / exp / "runs.csv", dtype={"selector": str, "predictor": str, "setting": str})
    df = df.fillna({"selector": "", "predictor": "", "setting": ""})
    df["who"] = df.apply(label_policy, axis=1)
    if "base_mean_kw" in df:
        df["shed_pct"] = 100.0 * df["shed_mwh"] / (df["cut"] * df["base_mean_kw"] * df["duration_h"] / 1000.0)
    return df
