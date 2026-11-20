"""S14 (E3): merge the e3 runs, print the report numbers, write F8 (selector comparison) and T3 (accuracy next to the decision metric).

Reads results/e3/runs_{default,hard,oos}.csv (python -m flex.experiments e3 --part ...) and ml/results/predictor_metrics.csv
(read only). Writes results/e3/{runs,summary,paired}.csv, paper/figures/out/f8_selectors.pdf and paper/tables/t3.tex.
Infeasible events (P* below the all-idle floor) are excluded everywhere. $/MWh is NaN when the policy shed < 10% of the target,
so medians over $/MWh skip those events (n_na is reported). w_fail = 0 is the primary cost, w_fail = 1 is the same runs post hoc.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from style import COLORS, RESULTS, ROOT, WIDTH_IN, plt, save  # noqa: E402

from flex.experiments import summarise  # noqa: E402

PARTS = ["default", "hard", "oos"]
CELL_OF = {"default": "2 h / 20%", "hard": "4 h / 30% (cloud cap 3000)", "oos": "2 h / 20%, burst only jobs submitted in the test window"}
PREDS = ["last2", "gbdt", "lightgbm", "transformer", "oracle"]
LEARNED = ["gbdt", "lightgbm", "transformer"]


def load(suffix=""):
    frames = []
    for p in PARTS:
        f = RESULTS / "e3" / f"runs_{p}{suffix}.csv"
        if f.exists():
            d = pd.read_csv(f, dtype={"selector": str, "predictor": str, "setting": str})
            d["part"] = p
            frames.append(d)
    df = pd.concat(frames, ignore_index=True).fillna({"selector": "", "predictor": "", "setting": ""})
    df["variant"] = np.where(df["setting"].str.contains("cq=p90"), "p90", "p50")
    sel = np.where(df["selector"].str.startswith("predictor:"), df["selector"].str[len("predictor:"):], df["selector"])
    df["sel"] = np.where(df["policy"] == "delay", df["predictor"], sel)
    df["who"] = df["policy"] + np.where(df["variant"] == "p90", "-p90", "") + "/" + df["sel"]
    df["wasted"] = df["cloud_waste_fail_usd"] + df["cloud_waste_short_usd"]
    df["useful_usd"] = df["cloud_usd"] - df["wasted"]
    return df


def med(df, who, part, col):
    s = df[(df["who"] == who) & (df["part"] == part)][col]
    return s.median() if len(s) else np.nan


def ranking(df, part):
    d = df[(df["part"] == part) & (df["feasible"])]
    t = d.groupby("who").agg(n=("event_id", "size"), usd_mwh=("usd_per_mwh", "median"), n_na=("usd_per_mwh", lambda s: int(s.isna().sum())),
                             cost_w0=("cost_usd", "median"), cost_w1=("cost_usd_wfail1", "median"), cloud=("cloud_usd", "median"),
                             waste_fail=("cloud_waste_fail_usd", "median"), waste_short=("cloud_waste_short_usd", "median"),
                             shed=("shed_mwh", "median"), compl_ok=("compliance_ok", "sum"), n_burst=("n_burst", "median"),
                             cov=("pred_coverage", "median")).sort_values("cost_w0")
    return t


def paired(df):
    """Each learned predictor vs last2 (and vs oracle) on the same event: events cheaper and the median cost ratio."""
    rows = []
    d = df[df["feasible"]]
    for part in PARTS:
        for pol, var in (("burst", "p50"), ("mix", "p50"), ("mix", "p90")):
            for col, lab in (("cost_usd", "w_fail=0"), ("cost_usd_wfail1", "w_fail=1")):
                sub = d[(d["part"] == part) & (d["policy"] == pol) & (d["variant"] == var)]
                if sub.empty:
                    continue
                w = sub.pivot_table(index="event_id", columns="sel", values=col)
                for a in LEARNED + ["oracle"]:
                    for b in ("last2", "oracle"):
                        if a == b or a not in w or b not in w or (pol == "mix" and var == "p90" and b == "oracle"):
                            continue
                        x = w[[a, b]].dropna()
                        if x.empty:
                            continue
                        rows.append(dict(part=part, policy=f"{pol}-{var}", cost=lab, predictor=a, vs=b, n=len(x),
                                         cheaper=int((x[a] < x[b] - 1e-9).sum()), equal=int((abs(x[a] - x[b]) <= 1e-9).sum()),
                                         median_ratio=float((x[a] / x[b].where(x[b] > 0)).median())))
    return pd.DataFrame(rows)


def f8_single(d, rules, order, y, name):
    """W11 (page budget): panel (a) of F8 only, one column wide (cost per MWh by selector, contract faded, hold solid)."""
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.15), constrained_layout=True)
    h = 0.8 / (2 * len(rules))
    for i, pol in enumerate(("burst", "mix")):
        for k, rule in enumerate(rules):
            v = [d[(d["rule"] == rule) & (d["policy"] == pol) & (d["sel"] == s)]["usd_per_mwh"].median() / 1000 for s in order]
            fade = rule == "contract" and len(rules) > 1
            ax.barh(y - 0.4 + h * (2 * i + k + 0.5), v, h, color=COLORS[pol], alpha=0.4 if fade else 1.0, edgecolor=COLORS[pol], lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("median cost, k$/MWh shed", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="x", lw=0.3, color="0.8")
    ax.set_axisbelow(True)
    names = {"last2": "last-two-jobs", "gbdt": "gradient-boosted trees", "lightgbm": "LightGBM"}
    ax.set_yticks(y, [names.get(o, o) for o in order])
    ax.invert_yaxis()
    handles = [Patch(color=COLORS["burst"], label="burst"), Patch(color=COLORS["mix"], label="mix")]
    if len(rules) > 1:
        handles += [Patch(facecolor="0.6", alpha=0.4, edgecolor="black", label="contract (faded)"), Patch(facecolor="0.4", edgecolor="black", label="hold (solid)")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.4, 1.0), frameon=False, fontsize=6, ncol=4,
              handlelength=1.2, columnspacing=0.8)
    path = save(fig, name)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def f8(df, name="f8_selectors", single=False):
    """F8 selector comparison, default cell, p50 variants. df may have a 'rule' column (contract / hold): the contract rule is drawn
    as faded, outlined bars next to the solid hold bars. Grey-print safe: stack components differ in grey level and hatch."""
    from matplotlib.patches import Patch
    d = df[(df["part"] == "default") & (df["feasible"]) & (df["variant"] == "p50")]
    if "rule" not in d:
        d = d.assign(rule="hold")
    rules = [r for r in ("contract", "hold") if r in set(d["rule"])]
    order = ["none-largest", "random"] + PREDS
    y = np.arange(len(order))
    if single:
        return f8_single(d, rules, order, y, name)
    fig, axes = plt.subplots(1, 3, figsize=(2 * WIDTH_IN, 2.6), sharey=True)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.90, bottom=0.34, wspace=0.08)
    h = 0.8 / (2 * len(rules))
    for i, pol in enumerate(("burst", "mix")):
        for k, rule in enumerate(rules):
            v = [d[(d["rule"] == rule) & (d["policy"] == pol) & (d["sel"] == s)]["usd_per_mwh"].median() / 1000 for s in order]
            fade = rule == "contract" and len(rules) > 1
            axes[0].barh(y - 0.4 + h * (2 * i + k + 0.5), v, h, color=COLORS[pol], alpha=0.4 if fade else 1.0,
                         edgecolor=COLORS[pol], lw=0.6)
    axes[0].set_xscale("log")
    axes[0].set_title("(a) cost, k$/MWh shed")
    comp = [("useful_usd", "useful", "#d9d9d9", ""), ("cloud_waste_short_usd", "short remaining", "#8c8c8c", ".."),
            ("cloud_waste_fail_usd", "Failed/Killed", "#303030", "//")]
    for ax, pol, title in ((axes[1], "burst", "(b) burst: cloud k$/event"), (axes[2], "mix", "(c) mix: cloud k$/event")):
        for k, rule in enumerate(rules):
            left = np.zeros(len(order))
            for col, lab, c, hatch in comp:
                v = np.array([d[(d["rule"] == rule) & (d["policy"] == pol) & (d["sel"] == s)][col].mean() for s in order]) / 1000
                fade = rule == "contract" and len(rules) > 1
                ax.barh(y - 0.2 + 0.4 * k, v, 0.38, left=left, color=c, hatch=hatch, alpha=0.45 if fade else 1.0, edgecolor="black", lw=0.4)
                left += v
        ax.set_title(title)
    for ax in axes:
        ax.tick_params(labelsize=6.5)
        ax.grid(axis="x", lw=0.3, color="0.8")
        ax.set_axisbelow(True)
    names = {"last2": "last-two-jobs", "gbdt": "gradient-" + chr(10) + "boosted trees", "lightgbm": "LightGBM"}   # W9: text names
    axes[0].set_yticks(y, [names.get(o, o) for o in order])
    axes[0].invert_yaxis()
    handles = [Patch(color=COLORS["burst"], label="burst"), Patch(color=COLORS["mix"], label="mix")]
    if len(rules) > 1:
        handles += [Patch(facecolor="0.6", alpha=0.4, edgecolor="black", label="contract rule (faded)"), Patch(facecolor="0.4", edgecolor="black", label="hold rule (solid)")]
    handles += [Patch(facecolor=c, hatch=hh, edgecolor="black", lw=0.4, label=f"cloud $: {lab}") for _, lab, c, hh in comp]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=6.5, handleheight=1.4, labelspacing=0.6, bbox_to_anchor=(0.5, 0.0))
    path = save(fig, name)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def t3(df, metrics):
    m = metrics[metrics["elapsed_s"].astype(str) == "all"].set_index("predictor")
    d = df[(df["part"] == "default") & (df["feasible"])]
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Predictor & MAE$_{p50}$ (min) & Pinball & AUROC$_{fail}$ & mix k\$/MWh & burst k\$/MWh & mix cost (\$/event) \\", r"\midrule"]
    for p in PREDS:
        r = m.loc[p] if p in m.index else None
        acc = [f"{r['mae_p50_min']:.1f}", f"{r['pinball']:.3f}", f"{r['auroc_failed']:.3f}"] if r is not None else ["--"] * 3
        mixv = d[(d["who"] == f"mix/{p}")]["usd_per_mwh"].median() / 1000
        burv = d[(d["who"] == f"burst/{p}")]["usd_per_mwh"].median() / 1000
        cost = d[(d["who"] == f"mix/{p}")]["cost_usd"].median()
        lines.append(f"{p} & " + " & ".join(acc) + f" & {mixv:.1f} & {burv:.0f} & {cost:,.0f} \\\\".replace(",", "{,}"))
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    out = ROOT / "paper" / "tables" / "t3.tex"
    out.write_text("\n".join(lines), encoding="utf8")
    print("wrote", out)


def main():
    df = load()
    df.to_csv(RESULTS / "e3" / "runs.csv", index=False)
    summarise(df).to_csv(RESULTS / "e3" / "summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
    print("rows:", df.groupby("part").size().to_dict(), "| infeasible events:", df[~df["feasible"]].groupby("part")["event_id"].nunique().to_dict())
    for part in PARTS:
        if (df["part"] == part).any():
            print(f"\n== {part}: {CELL_OF[part]} (feasible events: {df[(df['part'] == part) & df['feasible']]['event_id'].nunique()}) ==")
            print(ranking(df, part).round(2).to_string())
    pr = paired(df)
    pr.to_csv(RESULTS / "e3" / "paired.csv", index=False)
    print("\n== paired vs last2 / oracle (w_fail=0 and 1) ==")
    print(pr.round(3).to_string(index=False))
    d = df[(df["part"] == "default") & df["feasible"]]
    print("\npred_coverage (share of running GPUs with a real prediction row) per predictor, default cell: "
          + str(d[d["predictor"].isin(PREDS)].groupby("predictor")["pred_coverage"].agg(["min", "median", "max"]).round(3).to_dict("index")))
    f8(df)
    t3(df, pd.read_csv(ROOT / "ml" / "results" / "predictor_metrics.csv"))


if __name__ == "__main__":
    main()
