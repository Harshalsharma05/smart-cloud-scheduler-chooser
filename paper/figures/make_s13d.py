"""S13d: the marginal value of bursting. mix-nocloud = the same mix controller with the cloud switched off (max_cloud_gpus = 0).

Reads results/e1/runs.csv (mix, delay, cap; hold rule), results/e1nc/runs_*.csv (or results/s13d copies), results/e2b/runs_*.csv and
results/e2nc/runs_*.csv. Infeasible events (P* below the all-idle floor) are dropped. Per event: cost_usd (w_fail 0) and
cost_usd_wfail1 (w_fail 1) from the same run. Two ratios: mix / mix-nocloud = the value of bursting (< 1: bursting helps);
mix-nocloud / delay = the value of capping and job selection (< 1: they help). Writes paper/tables/t5.tex and
paper/figures/out/f_nocloud.pdf (+ PNG). Run from the repo root: python paper/figures/make_s13d.py"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import COLORS, RESULTS, ROOT, WIDTH_IN, plt, save  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)
V = {"transformer p50": ("transformer", ""), "transformer p90": ("transformer", "cq=p90"), "oracle": ("oracle", ""), "last2": ("last2", "")}
COSTS = {"w0": "cost_usd", "w1": "cost_usd_wfail1"}
MWH = {"w0": "usd_per_mwh", "w1": "usd_per_mwh_wfail1"}


def read(pattern):
    d = pd.concat([pd.read_csv(p, dtype={"selector": str, "predictor": str, "setting": str}) for p in sorted(glob.glob(str(pattern)))], ignore_index=True)
    d = d.fillna({"predictor": "", "setting": ""})
    return d[d["feasible"].astype(str) != "False"]


def pick(d, policy, pred, setting):
    s = d[(d["policy"] == policy) & (d["predictor"] == pred) & (d["setting"] == setting)]
    return s.set_index("event_id")


def tidy(mix_rows, nc_rows, delay_rows, extra_cols=()):
    cols = ["duration_h", "cut", "n_burst", "cloud_usd", "compliance", *COSTS.values(), *MWH.values(), *extra_cols]
    t = mix_rows[cols].add_prefix("mix_")
    t = t.rename(columns={"mix_duration_h": "duration_h", "mix_cut": "cut"})
    for pre, r in (("nc", nc_rows), ("delay", delay_rows)):
        t = t.join(r[["compliance", *COSTS.values(), *MWH.values()]].add_prefix(pre + "_"), how="inner")
    return t


def stats(t, w):
    c, m = COSTS[w], MWH[w]
    a, b, d = t[f"mix_{c}"], t[f"nc_{c}"], t[f"delay_{c}"]
    return dict(n=len(t), delay=t[f"delay_{m}"].median(), nocloud=t[f"nc_{m}"].median(), mix=t[f"mix_{m}"].median(),
                mix_lt_nc=int((a < b - 1e-9).sum()), mix_eq_nc=int((abs(a - b) <= 1e-9).sum()), mix_over_nc=float((a / b.where(b > 0)).median()),
                nc_over_delay=float((b / d.where(d > 0)).median()), mix_over_delay=float((a / d.where(d > 0)).median()),
                burst_jobs=t["mix_n_burst"].median(), cloud_usd=t["mix_cloud_usd"].median(), nc_compl=t["nc_compliance"].median(), mix_compl=t["mix_compliance"].median())


def main():
    e1 = read(RESULTS / "e1" / "runs.csv")
    nc = read(RESULTS / "e1nc" / "runs_*.csv")
    print(f"E1: {e1['event_id'].nunique()} feasible events; mix-nocloud rows: {len(nc)}")
    cap = e1[e1["policy"] == "cap"]
    capc = cap.groupby(["duration_h", "cut"])["compliance"].median()
    print("cap-only median compliance per cell (compliant needs >= 0.99):", {k: round(v, 2) for k, v in capc.items()})
    res = {}
    for name, (pred, cq) in V.items():
        mix = pick(e1, "mix", pred, cq)
        n = pick(nc, "mix", pred, "nocloud" + (";" + cq if cq else ""))
        dl = pick(e1, "delay", pred, "")
        res[name] = tidy(mix, n, dl)
    cells = sorted(set(zip(res["oracle"]["duration_h"], res["oracle"]["cut"])))
    rows = []
    for name, t in res.items():
        for w in COSTS:
            for (d, c), g in t.groupby(["duration_h", "cut"]):
                rows.append(dict(variant=name, w=w, duration_h=d, cut=c, **stats(g, w)))
            rows.append(dict(variant=name, w=w, duration_h="all", cut="all", **stats(t, w)))
    R = pd.DataFrame(rows)
    R.round(3).to_csv(RESULTS / "e1nc" / "s13d_cells.csv", index=False)
    for name in V:
        for w in COSTS:
            print(f"\n== E1 cells, {name}, {w}: median $/MWh delay / mix-nocloud / mix; events mix < nocloud; median ratios; burst jobs; cloud $")
            print(R[(R["variant"] == name) & (R["w"] == w)].drop(columns=["variant", "w"]).round(3).to_string(index=False))

    # (2) default cell, pool x price
    B = read(RESULTS / "e2b" / "runs_*.csv")
    N = read(RESULTS / "e2nc" / "runs_*.csv")
    for D in (B, N):
        s = D["setting"].str.extract(r"pool=(\d+);price=([\d.]+);boot=(\d+);cap=(\d+)").astype(float)
        D[["pool", "price", "boot", "cap"]] = s.values
    B = B[(B["cap"] == 1000) & (B["boot"] == 300) & B["price"].isin([0.43, 3.06])]
    out2 = []
    for (pool, price), _ in N.groupby(["pool", "price"]):
        for name, (pred, cq) in (("transformer p90", ("transformer", "cq=p90")), ("oracle", ("oracle", ""))):
            sel = lambda D, pol, st: D[(D["pool"] == pool) & (D["price"] == price) & (D["policy"] == pol) & (D["predictor"] == pred) & (D["setting"].str.contains(st, regex=False) if st else ~D["setting"].str.contains("cq=p90"))].set_index("event_id")
            mix, dl = sel(B, "mix", cq), sel(B, "delay", "")
            ncr = sel(N, "mix", cq)
            t = tidy(mix, ncr, dl)
            for w in COSTS:
                out2.append(dict(variant=name, w=w, pool=int(pool), price=price, **stats(t, w)))
    R2 = pd.DataFrame(out2)
    R2.round(3).to_csv(RESULTS / "e2nc" / "s13d_pool_price.csv", index=False)
    print("\n== default cell, pool x price (transformer p90 and oracle)")
    print(R2.round(3).to_string(index=False))

    # sanity: mix-nocloud never bursts; compare with mix when the mix itself bursts nothing
    print("\nsanity: max n_burst of mix-nocloud rows:", int(nc["n_burst"].max()), "| max cloud $ of mix-nocloud rows:", float(nc["cloud_usd"].max()))
    z = pd.concat([t[t["mix_n_burst"] == 0][["mix_cost_usd", "nc_cost_usd"]] for t in res.values()])
    print(f"sanity: events where the mix bursts nothing: {len(z)}; max |mix cost - mix-nocloud cost| there = {(z['mix_cost_usd'] - z['nc_cost_usd']).abs().max():.6f}")

    # T5: default cell, plus ratios over all 90 events
    lines = ["% T5 (S13d). Ratios = median over events of the per-event cost ratio (mix/nc: value of bursting; nc/delay: value of capping and selection); the $/MWh columns are cell medians, so the two need not agree.",
             r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r" & & \multicolumn{3}{c}{Default cell, \$/MWh} & \multicolumn{2}{c}{Default cell} & \multicolumn{2}{c}{All events} \\",
             r"Predictor & $w_f$ & delay & mix-nocloud & mix & mix/nc & nc/delay & mix/nc & nc/delay \\", r"\midrule"]
    for name in V:
        for w in COSTS:
            a = R[(R["variant"] == name) & (R["w"] == w) & (R["duration_h"] == 2) & (R["cut"].astype(str).str.startswith("0.2"))].iloc[0]
            al = R[(R["variant"] == name) & (R["w"] == w) & (R["duration_h"] == "all")].iloc[0]
            lines.append(f"{name} & {w[1]} & {a['delay']:,.0f} & {a['nocloud']:,.0f} & {a['mix']:,.0f} & {a['mix_over_nc']:.2f} & {a['nc_over_delay']:.2f} & "
                         f"{al['mix_over_nc']:.2f} & {al['nc_over_delay']:.2f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (ROOT / "paper" / "tables" / "t5.tex").write_text("\n".join(lines), encoding="utf8")
    print("wrote paper/tables/t5.tex")

    # figure: bars per cell
    fig, axes = plt.subplots(1, 2, figsize=(2 * WIDTH_IN, 2.5), sharey=False)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.36, wspace=0.18)
    x = np.arange(len(cells))
    for ax, name, title in ((axes[0], "transformer p90", "(a) transformer, p90 budget"), (axes[1], "oracle", "(b) oracle")):
        for k, (col, lab, color, hatch) in enumerate((("delay", "delay", COLORS["delay"], ""), ("nocloud", "mix-nocloud", "#cdb7d6", "//"), ("mix", "mix", COLORS["mix"], ""))):
            v = [R[(R["variant"] == name) & (R["w"] == "w0") & (R["duration_h"] == d) & (R["cut"] == c)][col].iloc[0] / 1000 for d, c in cells]
            ax.bar(x + (k - 1) * 0.27, v, 0.27, color=color, hatch=hatch, edgecolor="black", lw=0.4, label=lab)
        ax.set_xticks(x, [f"{int(d)} h\n{round(c * 100)}%" for d, c in cells])
        ax.tick_params(labelsize=6)
        ax.set_title(title)
        ax.grid(axis="y", lw=0.3, color="0.8")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("median cost per MWh shed (k$/MWh)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.text(0.5, 0.14, "event length / cut", ha="center", fontsize=7)
    path = save(fig, "f_nocloud")
    fig.savefig(path.with_suffix(".png"), dpi=200)


if __name__ == "__main__":
    main()
