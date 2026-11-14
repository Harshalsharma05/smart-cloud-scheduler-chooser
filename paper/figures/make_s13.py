"""S13 analysis: sweep A (price x boot), sweep B (pool x price, pool x cap x w_fail), T4 sensitivity, F7 winner maps.
Reads results/e2/runs_A.csv, results/e2b/runs_*.csv + baseline_*.csv, results/t4/runs_*.csv. Infeasible events (P* below the
all-idle floor) are excluded from every median and counted. Run from the repo root: python paper/figures/make_s13.py"""
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import RESULTS, ROOT, WIDTH_IN, label_policy, plt, save  # noqa: E402

OUT = RESULTS / "s13"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
POLICIES = ["delay", "delay+oracle", "burst", "mix", "mix+oracle"]
W = {"usd_per_mwh": "w0", "usd_per_mwh_wfail1": "w1"}


def label(r):
    if r["policy"] in ("mix", "delay") and r["predictor"] == "oracle":
        return f"{r['policy']}+oracle"
    return r["policy"]


def load(paths):
    df = pd.concat([pd.read_csv(p, dtype={"selector": str, "predictor": str, "setting": str}) for p in sorted(glob.glob(str(paths)))], ignore_index=True)
    df["predictor"] = df["predictor"].fillna("")
    df["who"] = df.apply(label, axis=1)
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


def winner(stats):
    ok = {k: v["med"] for k, v in stats.items() if v["ok"] and not np.isnan(v["med"])}
    return min(ok, key=ok.get) if ok else None


def winner_table(df, keys, col):
    rows = []
    for k, g in df.groupby(keys):
        st = cell_stats(g, col)
        n_inf = int(g.drop_duplicates("event_id").pipe(lambda d: (~g.groupby("event_id")["feasible"].first()).sum()))
        win = winner(st)
        r = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        r.update(infeasible_events=n_inf, winner=win or "-", winner_usd=st[win]["med"] if win else np.nan)
        for w in POLICIES:
            r[f"{w}"] = st[w]["med"] if w in st else np.nan
            r[f"{w}_compl"] = st[w]["compl"] if w in st else np.nan
        r["mix_burst_share"] = st["mix"]["burst_share"] if "mix" in st else np.nan
        r["mixor_burst_share"] = st["mix+oracle"]["burst_share"] if "mix+oracle" in st else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def winner_map(tab, row_key, col_key, name, title, row_fmt, col_fmt):
    rows, cols = sorted(tab[row_key].unique()), sorted(tab[col_key].unique())
    fig, ax = plt.subplots(figsize=(WIDTH_IN * 1.25, 2.4), constrained_layout=True)
    colors = {"delay": "#4c78a8", "delay+oracle": "#9ecae9", "burst": "#54a24b", "mix": "#b279a2", "mix+oracle": "#e45756", "-": "#dddddd"}
    for i, rv in enumerate(rows):
        for j, cv in enumerate(cols):
            r = tab[(tab[row_key] == rv) & (tab[col_key] == cv)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=colors[r["winner"]], alpha=0.85, lw=0))
            txt = ("infeasible" if r["winner"] == "-" else r["winner"].replace("+oracle", "+or.")) + f"\n{r['winner_usd'] / 1000:.1f}k" if r["winner"] != "-" else "infeasible"
            if r["winner"] != "-" and not np.isnan(r["mixor_burst_share"]):
                txt += f"\nmix+or b={r['mixor_burst_share'] * 100:.0f}%"
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=4.5)
    ax.set_xlim(0, len(cols)); ax.set_ylim(0, len(rows))
    ax.set_xticks(np.arange(len(cols)) + 0.5, [col_fmt(c) for c in cols])
    ax.set_yticks(np.arange(len(rows)) + 0.5, [row_fmt(r) for r in rows])
    ax.set_title(title)
    save(fig, name)


def ratio_table(df, keys):
    """Per setting: median over events of cost ratio (mix / delay with the same predictor; burst / delay last2), and events won."""
    rows = []
    for k, g in df.groupby(keys):
        g = g[g["feasible"]]
        piv = g.pivot_table(index="event_id", columns="who", values="usd_per_mwh", aggfunc="first")
        r = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        for a, b in (("mix", "delay"), ("mix+oracle", "delay+oracle"), ("burst", "delay")):
            if a in piv and b in piv:
                d = (piv[a] / piv[b].replace(0, np.nan)).dropna()
                r[f"{a}/{b}"] = d.median()
                r[f"{a}<{b}"] = int((piv[a] < piv[b]).sum())
        r["events"] = len(piv)
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    A = load(RESULTS / "e2" / "runs_A.csv")
    A = A[A["cap"] == 1000]
    ev_inf = A.groupby("event_id")["feasible"].first()
    print(f"== Sweep A: {A['event_id'].nunique()} events, infeasible (P* < floor): {int((~ev_inf).sum())} -> {sorted(ev_inf[~ev_inf].index)}")
    for col, tag in W.items():
        t = winner_table(A, ["boot", "price"], col)
        t.round(1).to_csv(OUT / f"sweepA_winners_{tag}.csv", index=False)
        print(f"\nSweep A winners ({tag}), median $/MWh over feasible events, compliant policies only:")
        print(t[["boot", "price", "winner", "winner_usd"] + POLICIES + ["mix_burst_share", "mixor_burst_share"]].round(2).to_string(index=False))
        winner_map(t, "boot", "price", f"f7_winner_price_boot_{tag}", f"winner, price x boot ({tag}), pool auto", lambda b: f"boot {int(b)} s", lambda p: f"${p:g}/h")
    rt = ratio_table(A, ["boot", "price"])
    rt.round(2).to_csv(OUT / "sweepA_ratios.csv", index=False)
    print("\nSweep A cost ratios (w_fail 0; median over events; '<' = events where the first is cheaper):")
    print(rt.round(2).to_string(index=False))

    B = load(RESULTS / "e2b" / "runs_*.csv")
    base = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(RESULTS / "e2b" / "baseline_*.csv")))])
    print("\n== Sweep B: baseline wait per pool"); print(base.sort_values("pool_gpus", ascending=False).to_string(index=False))
    inf = B.groupby(["pool", "event_id"])["feasible"].first().groupby("pool").apply(lambda s: int((~s).sum()))
    print("infeasible events per pool (of 10):", inf.to_dict())
    pm = B[(B["boot"] == 300) & (B["cap"] == 1000)]
    for col, tag in W.items():
        t = winner_table(pm, ["pool", "price"], col)
        t.round(1).to_csv(OUT / f"sweepB_pool_price_{tag}.csv", index=False)
        print(f"\nSweep B winner map, pool x price ({tag}), cap 1000, boot 300:")
        print(t[["pool", "price", "infeasible_events", "winner", "winner_usd"] + POLICIES].round(2).to_string(index=False))
        winner_map(t, "pool", "price", f"f7b_winner_pool_price_{tag}", f"winner, pool x price ({tag})", lambda p: f"{int(p)} GPUs", lambda p: f"${p:g}/h")
    sat = B[np.isclose(B["price"], 3.06) & (B["boot"] == 300)]
    for col, tag in W.items():
        t = winner_table(sat, ["pool", "cap"], col)
        t.round(1).to_csv(OUT / f"sweepB_pool_cap_{tag}.csv", index=False)
        print(f"\nSweep B saturation, pool x cloud cap ({tag}), price 3.06:")
        print(t[["pool", "cap", "infeasible_events", "winner", "winner_usd"] + POLICIES + [f"{w}_compl" for w in ("burst", "mix", "mix+oracle")]].round(2).to_string(index=False))
    rb = ratio_table(B[B["boot"] == 300], ["pool", "cap", "price"])
    rb.round(2).to_csv(OUT / "sweepB_ratios.csv", index=False)
    print("\nSweep B cost ratios:"); print(rb.round(2).to_string(index=False))

    # sanity: sweep A and sweep B (auto pool) must agree on the shared settings
    a = A[(A["boot"] == 300) & (A["price"].isin([0.43, 1.5, 3.06, 6.12]))].set_index(["event_id", "who", "price"])["usd_per_mwh"]
    b = B[(B["pool"] == 2300) & (B["boot"] == 300) & (B["cap"] == 1000)].set_index(["event_id", "who", "price"])["usd_per_mwh"]
    j = pd.concat([a.rename("A"), b.rename("B")], axis=1).dropna()
    print(f"\nsanity: sweep A vs sweep B (auto pool) on {len(j)} shared rows, max abs difference in $/MWh = {(j['A'] - j['B']).abs().max():.6f}")

    T = load(RESULTS / "t4" / "runs_*.csv")
    T["pert"] = T["setting"].str.replace("pert=", "", regex=False)
    d0 = A[np.isclose(A["price"], 3.06) & (A["boot"] == 300)].copy()
    d0["pert"] = "default"
    T = pd.concat([d0, T], ignore_index=True)
    print("\n== T4: infeasible events per perturbation:", T.groupby("pert").apply(lambda g: int((~g.groupby("event_id")["feasible"].first()).sum())).to_dict())
    lines = {}
    for col, tag in W.items():
        rows = []
        for pert, g in T.groupby("pert", sort=False):
            st = cell_stats(g, col)
            rows.append(dict(perturbation=pert, **{w: st[w]["med"] if w in st else np.nan for w in POLICIES},
                             **{f"{w}_ok": st[w]["ok"] if w in st else False for w in POLICIES}, winner=winner(st) or "-"))
        t = pd.DataFrame(rows)
        t.round(1).to_csv(OUT / f"t4_{tag}.csv", index=False)
        print(f"\nT4 median $/MWh ({tag}), feasible events; * = median compliance < 99%:")
        print(t[["perturbation"] + POLICIES + ["winner"]].round(0).to_string(index=False))
        lines[tag] = t
    t = lines["w0"]
    tex = [r"\begin{tabular}{lrrrrr}", r"\toprule", r"Perturbation & delay & delay+or. & burst & mix & mix+or. \\", r"\midrule"]
    for i in range(len(t)):
        name = t.loc[i, "perturbation"].replace("x", "$\times$")
        cells = " & ".join(f"{t.loc[i, w]:,.0f}" for w in POLICIES)
        tex.append(name + " & " + cells + " \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper" / "tables" / "t4.tex").write_text("\n".join(tex) + "\n")
    print("wrote paper/tables/t4.tex")


if __name__ == "__main__":
    main()
