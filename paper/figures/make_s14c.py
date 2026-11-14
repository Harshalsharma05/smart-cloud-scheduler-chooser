"""S14c: E3 with the contract lookup rule vs hold_beyond_grid, side by side. Redoes F8 (contract rule faded) and T3 (hold columns added).

Reads results/e3/runs_{default,hard,oos}[_hold].csv, results/s14b/{,hold/}jobs.csv, ml/results/predictor_metrics.csv (read only).
Writes results/e3/compare_rules.csv, paper/figures/out/f8_selectors.pdf, paper/tables/t3.tex.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_s14 import LEARNED, PARTS, PREDS, load  # noqa: E402
from style import COLORS, RESULTS, ROOT, WIDTH_IN, plt, save  # noqa: E402

LAST_GRID = 43200


def both():
    a, b = load(""), load("_hold")
    a["rule"], b["rule"] = "contract", "hold"
    d = pd.concat([a, b], ignore_index=True)
    return d[d["feasible"]]


def past_grid(rule):
    f = RESULTS / "s14b" / ("hold" if rule == "hold" else "") / "jobs.csv"
    j = pd.read_csv(f)
    out = {}
    for p, q in (("gbdt", "p50"), ("lightgbm", "p50"), ("transformer", "p50"), ("transformer", "p90")):
        d = j[(j["predictor"] == p) & (j[f"burst_{q}"] == True)]  # noqa: E712
        out[(p, q)] = float((d["elapsed"] > LAST_GRID).mean())
    return out


def compare(d):
    rows = []
    pg = {r: past_grid(r) for r in ("contract", "hold")}
    for part in PARTS:
        for rule in ("contract", "hold"):
            x = d[(d["part"] == part) & (d["rule"] == rule)]
            ref = lambda pol: x[(x["policy"] == pol) & (x["variant"] == "p50") & (x["sel"] == "last2")].set_index("event_id")["cost_usd"]
            dl = x[(x["policy"] == "delay") & (x["sel"] == "last2")].set_index("event_id")["cost_usd"]
            for pol, var in (("burst", "p50"), ("mix", "p50"), ("mix", "p90")):
                for p in LEARNED:
                    s = x[(x["policy"] == pol) & (x["variant"] == var) & (x["sel"] == p)].set_index("event_id")
                    if s.empty:
                        continue
                    c = s["cost_usd"]
                    r = ref(pol).reindex(c.index)
                    rows.append(dict(part=part, rule=rule, policy=f"{pol}-{var}", predictor=p, n=len(c), median_cost=c.median(), median_cost_w1=s["cost_usd_wfail1"].median(),
                                     cheaper_than_last2=int((c < r - 1e-9).sum()), cheaper_than_delay_last2=int((c < dl.reindex(c.index) - 1e-9).sum()),
                                     median_cloud=s["cloud_usd"].median(), median_burst_jobs=s["n_burst"].median(),
                                     burst_past_grid=pg[rule].get((p, var), np.nan) if part == "default" and pol == "mix" else np.nan))
    return pd.DataFrame(rows)


def f8(d):
    x = d[(d["part"] == "default") & (d["variant"] == "p50")]
    order = ["none-largest", "random"] + PREDS
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN, 2.5), constrained_layout=True, sharey=True)
    y = np.arange(len(order))
    for pol, off in (("burst", -0.3), ("mix", 0.1)):
        for k, rule in enumerate(("contract", "hold")):
            v = [x[(x["rule"] == rule) & (x["policy"] == pol) & (x["sel"] == s)]["usd_per_mwh"].median() / 1000 for s in order]
            axes[0].barh(y + off + 0.2 * k, v, 0.2, color=COLORS[pol], alpha=0.35 if rule == "contract" else 1.0, label=f"{pol}" if rule == "hold" else None)
    axes[0].set_xscale("log")
    axes[0].set_title("k$/MWh shed (median)")
    axes[0].legend(frameon=False, loc="lower right")
    comp = [("useful_usd", "useful", "#7f7f7f"), ("cloud_waste_short_usd", "short remaining", "#f2cf5b"), ("cloud_waste_fail_usd", "Failed/Killed", "#d1495b")]
    for ax, pol in ((axes[1], "burst"), (axes[2], "mix")):
        for k, rule in enumerate(("contract", "hold")):
            left = np.zeros(len(order))
            for col, lab, c in comp:
                v = np.array([x[(x["rule"] == rule) & (x["policy"] == pol) & (x["sel"] == s)][col].mean() for s in order]) / 1000
                ax.barh(y - 0.2 + 0.4 * k, v, 0.38, left=left, color=c, alpha=0.35 if rule == "contract" else 1.0, label=lab if (k == 1) else None)
                left += v
        ax.set_title(f"{pol}: cloud k$/event")
    axes[2].legend(frameon=False, loc="lower right")
    axes[0].set_yticks(y, order)
    axes[0].invert_yaxis()
    save(fig, "f8_selectors")


def t3(d, metrics):
    m = metrics[metrics["elapsed_s"].astype(str) == "all"].set_index("predictor")
    x = d[d["part"] == "default"]
    lines = [r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
             r"Predictor & MAE$_{p50}$ & Pinball & AUROC$_{fail}$ & \multicolumn{2}{c}{mix k\$/MWh} & \multicolumn{2}{c}{burst k\$/MWh} & \multicolumn{2}{c}{mix \$/event} \\",
             r" & (min) & & & contract & hold & contract & hold & contract & hold \\", r"\midrule"]
    for p in PREDS:
        r = m.loc[p]
        cells = []
        for pol, div, fmt in (("mix", 1000, "{:.1f}"), ("burst", 1000, "{:.0f}"), ("mix", None, "{:,.0f}")):
            for rule in ("contract", "hold"):
                s = x[(x["rule"] == rule) & (x["who"] == f"{pol}/{p}")]
                v = s["usd_per_mwh"].median() / 1000 if div else s["cost_usd"].median()
                cells.append(fmt.format(v).replace(",", "{,}"))
        lines.append(f"{p} & {r['mae_p50_min']:.1f} & {r['pinball']:.3f} & {r['auroc_failed']:.3f} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    out = ROOT / "paper" / "tables" / "t3.tex"
    out.write_text("\n".join(lines), encoding="utf8")
    print("wrote", out)


def main():
    d = both()
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
    ref = d[d["sel"].isin(["last2", "oracle"]) | d["selector"].isin(["none-largest", "random"])]
    k = ["part", "event_id", "who", "variant"]
    w = ref.pivot_table(index=k, columns="rule", values="cost_usd")
    print("last2 / oracle / blind rows, hold minus contract cost: max abs", float((w["hold"] - w["contract"]).abs().max()), "over", len(w), "rows")
    c = compare(d)
    c.to_csv(RESULTS / "e3" / "compare_rules.csv", index=False)
    print(c.round(3).to_string(index=False))
    f8(d)
    t3(d, pd.read_csv(ROOT / "ml" / "results" / "predictor_metrics.csv"))


if __name__ == "__main__":
    main()
