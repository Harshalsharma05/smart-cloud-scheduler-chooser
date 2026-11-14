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


def load():
    frames = []
    for p in PARTS:
        f = RESULTS / "e3" / f"runs_{p}.csv"
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


def f8(df):
    d = df[(df["part"] == "default") & (df["feasible"]) & (df["variant"] == "p50")]
    order = ["none-largest", "random"] + PREDS
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN, 2.4), constrained_layout=True, sharey=True)
    y = np.arange(len(order))
    for pol, off in (("burst", -0.2), ("mix", 0.2)):
        v = [d[(d["policy"] == pol) & (d["sel"] == s)]["usd_per_mwh"].median() / 1000 for s in order]
        axes[0].barh(y + off, v, 0.4, color=COLORS[pol], label=pol)
    axes[0].set_xscale("log")
    axes[0].set_title("k$/MWh shed (median)")
    axes[0].legend(frameon=False, loc="lower right")
    comp = [("useful_usd", "useful", "#7f7f7f"), ("cloud_waste_short_usd", "short remaining", "#f2cf5b"), ("cloud_waste_fail_usd", "Failed/Killed", "#d1495b")]
    for ax, pol in ((axes[1], "burst"), (axes[2], "mix")):
        left = np.zeros(len(order))
        for col, lab, c in comp:
            v = np.array([d[(d["policy"] == pol) & (d["sel"] == s)][col].mean() for s in order]) / 1000
            ax.barh(y, v, 0.6, left=left, color=c, label=lab)
            left += v
        ax.set_title(f"{pol}: cloud k$/event")
    axes[2].legend(frameon=False, loc="lower right")
    axes[0].set_yticks(y, order)
    axes[0].invert_yaxis()
    save(fig, "f8_selectors")


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
