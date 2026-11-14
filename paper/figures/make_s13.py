"""S13c analysis (hold_beyond_grid lookup rule): sweep A (price x boot), sweep B (pool x price, pool x cap x w_fail), T4 sensitivity,
F7 / F7b winner maps. Reads results/e2/runs_*.csv, results/e2b/runs_*.csv + baseline_*.csv, results/t4/runs_*.csv. The contract-rule
results of S13 are in results/*_contract and results/s13_contract. Infeasible events (P* below the all-idle floor) are excluded
from every median and counted. Winner = lowest median $/MWh among policies whose median compliance is >= 0.99, within one predictor
set: transformer {delay, burst, mix, mix-p90}, last2 {delay, mix, burst on the default setting}, oracle {delay, burst, mix}.
Run from the repo root: python paper/figures/make_s13.py"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import RESULTS, ROOT, WIDTH_IN, label_run, plt, save  # noqa: E402

OUT = RESULTS / "s13"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
SETS = {"tf": ["delay", "burst", "mix", "mix-p90"], "l2": ["delay (last2)", "mix (last2)", "burst (last2)"],
        "or": ["delay+oracle", "burst+oracle", "mix+oracle"]}
ALL = SETS["tf"] + SETS["l2"] + SETS["or"]
W = {"usd_per_mwh": "w0", "usd_per_mwh_wfail1": "w1"}
COLORS = {"delay": "#4c78a8", "burst": "#54a24b", "mix": "#b279a2", "mix-p90": "#7a4f8a", "-": "#dddddd"}


def load(paths):
    df = pd.concat([pd.read_csv(p, dtype={"selector": str, "predictor": str, "setting": str}) for p in sorted(glob.glob(str(paths)))], ignore_index=True)
    df = df.fillna({"predictor": "", "setting": ""})
    df["who"] = df.apply(label_run, axis=1)
    s = df["setting"].str.extract(r"pool=(\d+);price=([\d.]+);boot=(\d+);cap=(\d+)").astype(float)
    df[["pool", "price", "boot", "cap"]] = s.values
    df["burst_share"] = df["n_burst"] / (df["n_burst"] + df["n_suspended"]).replace(0, np.nan)
    return df


def cell_stats(g, col):
    """Per policy over the feasible events of a cell: median $/MWh, median compliance, compliant?"""
    out = {}
    for w, s in g.groupby("who"):
        s = s[s["feasible"]]
        if s.empty:
            continue
        out[w] = dict(med=s[col].median(), compl=s["compliance"].median(), n=len(s), burst_share=s["burst_share"].median(),
                      ok=s["compliance"].median() >= 0.99)
    return out


def winner(stats, names):
    ok = {k: stats[k]["med"] for k in names if k in stats and stats[k]["ok"] and not np.isnan(stats[k]["med"])}
    return min(ok, key=ok.get) if ok else None


def paired_counts(g, a, b, col):
    """Over feasible events: events where a is cheaper than b, and the median ratio a / b."""
    piv = g[g["feasible"]].pivot_table(index="event_id", columns="who", values=col, aggfunc="first")
    if a not in piv or b not in piv:
        return np.nan, np.nan, 0
    d = piv[[a, b]].dropna()
    return int((d[a] < d[b]).sum()), float((d[a] / d[b].replace(0, np.nan)).median()), len(d)


def winner_table(df, keys, col):
    rows = []
    for k, g in df.groupby(keys):
        st = cell_stats(g, col)
        inf = int((~g.groupby("event_id")["feasible"].first()).sum())
        r = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        r["infeasible_events"] = inf
        for tag, names in SETS.items():
            w = winner(st, names)
            r[f"win_{tag}"] = w or "-"
            r[f"win_{tag}_usd"] = st[w]["med"] if w else np.nan
        for w in ALL:
            r[w] = st[w]["med"] if w in st else np.nan
            r[f"{w}_compl"] = st[w]["compl"] if w in st else np.nan
        r["p90_lt_delay"], r["p90/delay"], r["n_paired"] = paired_counts(g, "mix-p90", "delay", col)
        r["mix_lt_delay"], r["mix/delay"], _ = paired_counts(g, "mix", "delay", col)
        r["burst_lt_delay"], r["burst/delay"], _ = paired_counts(g, "burst", "delay", col)
        r["mixor_lt_delayor"], r["mixor/delayor"], _ = paired_counts(g, "mix+oracle", "delay+oracle", col)
        r["mix_burst_share"] = st["mix"]["burst_share"] if "mix" in st else np.nan
        r["mixp90_burst_share"] = st["mix-p90"]["burst_share"] if "mix-p90" in st else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def winner_map(tab, row_key, col_key, name, title, row_fmt, col_fmt):
    """Colour = transformer-set winner; text: winner, k$/MWh, events where mix-p90 is cheaper than delay, oracle-set winner.
    A thick black outline marks the cells where the median mix-p90 / transformer is below the median delay / transformer."""
    rows, cols = sorted(tab[row_key].unique()), sorted(tab[col_key].unique())
    fig, ax = plt.subplots(figsize=(WIDTH_IN * 1.3, 2.6), constrained_layout=True)
    for i, rv in enumerate(rows):
        for j, cv in enumerate(cols):
            r = tab[(tab[row_key] == rv) & (tab[col_key] == cv)]
            if r.empty:
                continue
            r = r.iloc[0]
            win = r["win_tf"]
            beats = bool(r["mix-p90"] < r["delay"]) if not (np.isnan(r["mix-p90"]) or np.isnan(r["delay"])) else False
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=COLORS[win], alpha=0.85, edgecolor="black" if beats else "white", lw=1.6 if beats else 0.5))
            txt = "infeasible" if win == "-" else f"{win}\n{r['win_tf_usd'] / 1000:.1f}k\np90<d {int(r['p90_lt_delay'])}/{int(r['n_paired'])}\nor: {r['win_or'].replace('+oracle', '')}"
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=4.5)
    ax.set_xlim(0, len(cols))
    ax.set_ylim(0, len(rows))
    ax.set_xticks(np.arange(len(cols)) + 0.5, [col_fmt(c) for c in cols])
    ax.set_yticks(np.arange(len(rows)) + 0.5, [row_fmt(r) for r in rows])
    ax.set_title(title, fontsize=6.5)
    path = save(fig, name)
    fig.savefig(path.with_suffix('.png'), dpi=200)


def ratio_table(df, keys, col="usd_per_mwh"):
    """Per setting: median over events of the cost ratio and the events where the first policy is cheaper."""
    rows = []
    for k, g in df.groupby(keys):
        r = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        for a, b in (("mix", "delay"), ("mix-p90", "delay"), ("burst", "delay"), ("mix+oracle", "delay+oracle"), ("burst+oracle", "delay+oracle"),
                     ("mix (last2)", "delay (last2)")):
            n, med, tot = paired_counts(g, a, b, col)
            r[f"{a}/{b}"], r[f"{a}<{b}"] = med, n
        r["events"] = tot
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    A = load(RESULTS / "e2" / "runs_*.csv")
    A = A[A["cap"] == 1000]
    ev_inf = A.groupby("event_id")["feasible"].first()
    print(f"== Sweep A: {A['event_id'].nunique()} events, infeasible (P* < floor): {int((~ev_inf).sum())} -> {sorted(ev_inf[~ev_inf].index)}")
    for col, tag in W.items():
        t = winner_table(A, ["boot", "price"], col)
        t.round(3).to_csv(OUT / f"sweepA_winners_{tag}.csv", index=False)
        print(f"\nSweep A winners ({tag}), median $/MWh over feasible events, compliant policies only:")
        print(t[["boot", "price", "win_tf", "win_tf_usd", "win_or", "win_l2", "p90_lt_delay", "p90/delay", "mix_lt_delay", "mix/delay", "burst/delay",
                 "mixor_lt_delayor", "mixor/delayor", "mix_burst_share", "mixp90_burst_share"]].round(3).to_string(index=False))
        winner_map(t, "boot", "price", f"f7_winner_price_boot_{tag}", f"winner (transformer), price x boot ({tag}), pool auto, hold rule", lambda b: f"boot {int(b)} s", lambda p: f"${p:g}/h")
    rt = ratio_table(A, ["boot", "price"])
    rt.round(3).to_csv(OUT / "sweepA_ratios.csv", index=False)
    print("\nSweep A cost ratios (w_fail 0; median over events; '<' = events where the first is cheaper):")
    print(rt.round(3).to_string(index=False))

    B = load(RESULTS / "e2b" / "runs_*.csv")
    base = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(RESULTS / "e2b" / "baseline_*.csv")))])
    print("\n== Sweep B: baseline wait per pool")
    print(base.sort_values("pool_gpus", ascending=False).to_string(index=False))
    inf = B.groupby(["pool", "event_id"])["feasible"].first().groupby("pool").apply(lambda s: int((~s).sum()))
    print("infeasible events per pool (of 10):", inf.to_dict())
    pm = B[(B["boot"] == 300) & (B["cap"] == 1000)]
    for col, tag in W.items():
        t = winner_table(pm, ["pool", "price"], col)
        t.round(3).to_csv(OUT / f"sweepB_pool_price_{tag}.csv", index=False)
        print(f"\nSweep B winner map, pool x price ({tag}), cap 1000, boot 300:")
        print(t[["pool", "price", "infeasible_events", "win_tf", "win_tf_usd", "win_or", "p90_lt_delay", "p90/delay", "mix_lt_delay", "mix/delay", "burst/delay",
                 "mixor_lt_delayor", "mixor/delayor"]].round(3).to_string(index=False))
        winner_map(t, "pool", "price", f"f7b_winner_pool_price_{tag}", f"winner (transformer), pool x price ({tag}), hold rule", lambda p: f"{int(p)} GPUs", lambda p: f"${p:g}/h")
    sat = B[np.isclose(B["price"], 3.06) & (B["boot"] == 300)]
    for col, tag in W.items():
        t = winner_table(sat, ["pool", "cap"], col)
        t.round(3).to_csv(OUT / f"sweepB_pool_cap_{tag}.csv", index=False)
        print(f"\nSweep B saturation, pool x cloud cap ({tag}), price 3.06:")
        print(t[["pool", "cap", "infeasible_events", "win_tf", "win_tf_usd", "win_or", "delay", "burst", "mix", "mix-p90", "delay+oracle", "mix+oracle",
                 "burst_compl", "mix_compl", "mix-p90_compl"]].round(2).to_string(index=False))
    rb = ratio_table(B[B["boot"] == 300], ["pool", "cap", "price"])
    rb.round(3).to_csv(OUT / "sweepB_ratios.csv", index=False)
    print("\nSweep B cost ratios:")
    print(rb.round(3).to_string(index=False))

    # sanity 1: sweep A and sweep B (auto pool) agree on the shared settings
    a = A[(A["boot"] == 300) & (A["price"].isin([0.43, 1.5, 3.06, 6.12]))].set_index(["event_id", "who", "price"])["usd_per_mwh"]
    b = B[(B["pool"] == 2300) & (B["boot"] == 300) & (B["cap"] == 1000)].set_index(["event_id", "who", "price"])["usd_per_mwh"]
    j = pd.concat([a.rename("A"), b.rename("B")], axis=1).dropna()
    print(f"\nsanity 1: sweep A vs sweep B (auto pool) on {len(j)} shared rows, max abs difference in $/MWh = {(j['A'] - j['B']).abs().max():.6f}")
    # sanity 2: the default setting of sweep A reproduces the E1 rows (same events, same policies)
    e1 = load(RESULTS / "e1" / "runs.csv")
    e1 = e1[(e1["duration_h"] == 2) & (e1["cut"].sub(0.2).abs() < 1e-9) & (e1["setting"] != "attribution") & (e1["policy"] != "cap")]
    d0 = A[np.isclose(A["price"], 3.06) & (A["boot"] == 300)].set_index(["event_id", "who"])["cost_usd"]
    e1c = e1.set_index(["event_id", "who"])["cost_usd"]
    j = pd.concat([d0.rename("A"), e1c.rename("E1")], axis=1).dropna()
    print(f"sanity 2: sweep A default setting vs E1 on {len(j)} shared rows, max abs difference in cost = {(j['A'] - j['E1']).abs().max():.6f}")
    # sanity 3: delay rows do not change with the cloud setting
    dd = A[A["policy"] == "delay"].groupby(["event_id", "who"])["cost_usd"].agg(lambda s: s.max() - s.min()).max()
    print(f"sanity 3: delay cost spread across cloud settings (must be 0): {dd:.6f}")

    T = load(RESULTS / "t4" / "runs_*.csv")
    T["pert"] = T["setting"].str.replace("pert=", "", regex=False).str.replace(";cq=p90", "", regex=False)
    d0 = A[np.isclose(A["price"], 3.06) & (A["boot"] == 300)].copy()
    d0["pert"] = "default"
    T = pd.concat([d0, T], ignore_index=True)
    print("\n== T4: infeasible events per perturbation:", T.groupby("pert").apply(lambda g: int((~g.groupby("event_id")["feasible"].first()).sum())).to_dict())
    lines = {}
    for col, tag in W.items():
        rows = []
        for pert, g in T.groupby("pert", sort=False):
            st = cell_stats(g, col)
            pc = paired_counts(g, "mix-p90", "delay", col)
            rows.append(dict(perturbation=pert, **{w: st[w]["med"] if w in st else np.nan for w in ALL},
                             **{f"{w}_ok": st[w]["ok"] if w in st else False for w in ALL}, win_tf=winner(st, SETS["tf"]) or "-",
                             win_or=winner(st, SETS["or"]) or "-", p90_lt_delay=pc[0], p90_over_delay=pc[1]))
        t = pd.DataFrame(rows)
        t.round(3).to_csv(OUT / f"t4_{tag}.csv", index=False)
        print(f"\nT4 median $/MWh ({tag}), feasible events:")
        print(t[["perturbation"] + SETS["tf"] + ["delay+oracle", "mix+oracle", "win_tf", "win_or", "p90_lt_delay", "p90_over_delay"]].round(2).to_string(index=False))
        lines[tag] = t
    t = lines["w0"]
    cols = ["delay", "burst", "mix", "mix-p90", "delay+oracle", "mix+oracle"]
    tex = [r"\begin{tabular}{lrrrrrr}", r"\toprule", r"Perturbation & delay & burst & mix & mix-p90 & delay+or. & mix+or. \\", r"\midrule"]
    for i in range(len(t)):
        name = t.loc[i, "perturbation"].replace("x", r"$\times$").replace("_", r"\_")
        tex.append(name + " & " + " & ".join(f"{t.loc[i, w]:,.0f}" for w in cols) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper" / "tables" / "t4.tex").write_text("\n".join(tex) + "\n")
    print("wrote paper/tables/t4.tex")


if __name__ == "__main__":
    main()
