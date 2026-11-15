"""Shared look for the paper figures: IEEE column width, one fixed colour per policy."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "out"
WIDTH_IN = 3.5

COLORS = {"delay": "#4c78a8", "cap": "#f58518", "burst": "#54a24b", "mix": "#b279a2", "mix-p90": "#7a4f8a", "mix+oracle": "#e45756", "delay+oracle": "#9ecae9", "burst+oracle": "#a1d99b"}
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


def label_policy(row):
    """Policy name for the S13 figures (E2 sweeps): the oracle upper bound of the mix / delay is its own entry."""
    return f"{row['policy']}+oracle" if row["policy"] in ("mix", "delay") and row["predictor"] == "oracle" else row["policy"]


def label_run(row):
    """Who a run belongs to (S16a): the realistic predictor (transformer) and cap are the plain policy names; the oracle is
    '<policy>+oracle'; last2 is '<policy> (last2)'; the p90 cloud-bill variant of the mix is 'mix-p90'; attribution controls are 'attr:...'."""
    pol, pred, st = row["policy"], row["predictor"], row["setting"]
    if st == "attribution":
        return f"attr:{pol}/{pred}"
    if pol == "cap" or pred in ("", "transformer"):
        return "mix-p90" if "cq=p90" in st else pol
    return f"{pol}+oracle" if pred == "oracle" else f"{pol} ({pred})"


def load_runs(exp="e1"):
    """Runs of one experiment (infeasible events dropped: P* below the all-idle floor) with a 'who' label (label_run) and
    shed_pct = shed MWh as % of the event target (cut x baseline energy)."""
    import pandas as pd
    df = pd.read_csv(RESULTS / exp / "runs.csv", dtype={"selector": str, "predictor": str, "setting": str})
    df = df.fillna({"selector": "", "predictor": "", "setting": ""})
    if "feasible" in df:
        df = df[df["feasible"].astype(str) != "False"]
    df["who"] = df.apply(label_run, axis=1)
    if "base_mean_kw" in df:
        df["shed_pct"] = 100.0 * df["shed_mwh"] / (df["cut"] * df["base_mean_kw"] * df["duration_h"] / 1000.0)
    return df


# ---- double-blind: no tool names or URLs in figure metadata (every figure script imports this module) ----
import matplotlib.figure as _mfig  # noqa: E402

_orig_savefig = _mfig.Figure.savefig


def _savefig_blind(self, fname, *args, **kw):
    meta = dict(kw.pop("metadata", None) or {})
    suffix = str(fname).lower().rsplit(".", 1)[-1]
    if suffix == "pdf":
        meta.update(Creator=None, Producer=None, Author=None, Title=None)
    elif suffix == "png":
        meta.update(Software=None)
    return _orig_savefig(self, fname, *args, metadata=meta, **kw)


_mfig.Figure.savefig = _savefig_blind
