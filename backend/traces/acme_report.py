"""Step 35 (S-A1) report: python -m traces.acme_report (from backend/). Reads results/acme_baseline, results/acme_e1
(merged), the stored Philly E1 results and paper/numbers.tex; writes results/acme_e1/report_*.csv and
docs/track-2/reports/SA1_acme.md. No simulation here."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
PRED = ["oracle", "last2", "gbdt", "lightgbm", "transformer"]
CELLS = [(d, c) for d in (1, 2, 4) for c in (0.1, 0.2, 0.3)]
COST = {"0": "cost_usd", "1": "cost_usd_wfail1"}
USD = {"0": "usd_per_mwh", "1": "usd_per_mwh_wfail1"}


def label(r):
    s, p = r["setting"], r["predictor"]
    if r["policy"] == "cap":
        return "cap"
    if s.startswith("price="):
        return f"{r['policy']}{'-p90' if 'cq=p90' in s else ''} @{s.split(';')[0][6:]}/{p}"
    if r["policy"] == "mix":
        return ("mix-nocloud" if "nocloud" in s else "mix-p90" if "cq=p90" in s else "mix") + "/" + p
    return f"{r['policy']}/{p}"


def load(path):
    d = pd.read_csv(path, dtype={"selector": str, "predictor": str, "setting": str, "job_id": str})
    return d.fillna({"selector": "", "predictor": "", "setting": ""})


def md(df, floatfmt=2):
    df = df.copy()
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("n/a" if pd.isna(v) else (f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.{floatfmt}f}"))
            else:
                cells.append(f"{v:,}" if isinstance(v, (int, np.integer)) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def ratio(a, b):
    """Per-event a/b on common events: median ratio, events with a < b, events, sum a / sum b."""
    x = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    r = x["a"] / x["b"].where(x["b"] > 0)
    return float(r.median()), int((x["a"] < x["b"] - 1e-9).sum()), len(x), float(x["a"].sum() / x["b"].sum())


def piv(df, col):
    return df.pivot_table(index="event_id", columns="who", values=col, aggfunc="first")


def in_cell(df, d, c):
    return df[(df["duration_h"] == d) & (df["cut"].sub(c).abs() < 1e-9)]


def macros():
    t = (ROOT / "paper" / "numbers.tex").read_text(encoding="utf-8")
    out = {}
    for line in t.splitlines():                      # one macro per line: \newcommand{\Name}{value}
        m = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", line)
        if m:
            out[m.group(1)] = m.group(2).replace("{,}", ",")
    return out


def main():
    out = RES / "acme_e1"
    runs = load(out / "runs.csv")
    runs["who"] = runs.apply(label, axis=1)
    ev = pd.read_csv(out / "events.csv")
    bursts = load(out / "bursts.csv")
    fb = pd.read_csv(out / "fallback.csv").groupby("predictor")["jobs_without_rows"].max()
    val = pd.read_csv(RES / "acme_baseline" / "validation.csv")
    cal = pd.read_csv(RES / "acme_baseline" / "calibration.csv").iloc[0]
    feas = runs[runs["feasible"].astype(str) != "False"]
    infeas = runs[runs["feasible"].astype(str) == "False"]["event_id"].unique()
    main_ = feas[~feas["setting"].str.startswith("price=")]
    L = []

    # ---- 1. baseline
    L += ["# SA1: Acme (Seren) simulator run", "",
          "Step 35. Same engine, policies and metrics as Philly through `--data acme` (`traces/acme.py`, `flex.yaml` section `acme:`); "
          "Philly results unchanged (default-cell event re-run before and after: identical). Seren only, window from `ml/config.py` "
          "(simulation from ACME_SIM_START, events in [ACME_TEST_START, ACME_TEST_END)), one seed.", "",
          "## 1. Baseline check (test window) and power calibration", "",
          md(val, 3), "",
          f"- Pool G = 2,288 A100 GPUs. Simulated and real busy GPUs agree (the pool never binds, so simulated waits are 0, as on Philly).",
          f"- Calibration: simulated busy share {cal.busy_share_sim:.3f}; one cluster-wide u = (169 - 63) / ((400 - 63) x {cal.busy_share_sim:.3f}) = "
          f"**{cal.u_calibrated:.3f}** (used: {cal.u_used:.3f}); simulated mean GPU power {cal.sim_mean_gpu_power_w:.1f} W against the Seren CDF mean "
          f"{cal.cdf_gpu_mean_w:.1f} W (August 2023 CDF; the test window is July). Facility mean {cal.facility_mean_kw:,.0f} kW; all-idle floor {cal.floor_kw:,.0f} kW.",
          f"- Host power from the GPU-node server-power CDF (8 GPUs per node): low end p1 = {cal.srv_p1_w:,.0f} W = idle host + 8 x 63 W, so idle host = "
          f"{cal.srv_p1_w - 8 * 63:,.0f} W and h_idle = {cal.h_idle_p1_w:.1f} W per GPU; high end p99 = {cal.srv_p99_w:,.0f} W = idle host + 8 x (400 W + h_busy), "
          f"so h_busy = ({cal.srv_p99_w:,.0f} - {cal.srv_p1_w - 8 * 63:,.0f} - 3,200) / 8 = {cal.h_busy_p99_w:.1f} W per GPU. Config: 115 / 163 W. "
          f"(With p10 / p90 instead: {cal.h_idle_p10_w:.0f} / {cal.h_busy_p90_w:.0f} W; not run.)",
          f"- Events: {runs['event_id'].nunique()} (10 start days x 9 cells), start 17:00-19:30 China time; infeasible (P* below the all-idle floor): "
          f"{len(infeas)} {sorted(infeas) if len(infeas) else ''}.",
          f"- Jobs without prediction rows (observed-value fallback), per predictor: {fb.to_dict()}.", ""]

    # ---- 2. per cell
    L += ["## 2. Per cell", "", "### 2a. Median $/MWh per cell (w_f = 0; n/a = shed below 10% of target)", ""]
    whos = ["cap"] + [f"{p}/{q}" for q in PRED for p in ("delay", "burst", "mix", "mix-p90", "mix-nocloud")]
    for k, title in (("0", None), ("1", "### 2b. Median $/MWh per cell, w_f = 1")):
        if title:
            L += [title, ""]
        t = pd.DataFrame({f"{d}h/{int(c * 100)}%": in_cell(main_, d, c).groupby("who")[USD[k]].median() for d, c in CELLS}).reindex(whos)
        t.insert(0, "policy/predictor", t.index)
        t.to_csv(out / f"report_usd_per_mwh_wf{k}.csv", index=False)
        L += [md(t, 0), ""]
    comp = pd.DataFrame({f"{d}h/{int(c * 100)}%": in_cell(main_, d, c).groupby("who")["compliance"].median() for d, c in CELLS}).reindex(
        ["cap"] + [f"{p}/transformer" for p in ("delay", "burst", "mix", "mix-p90", "mix-nocloud")] + ["burst/oracle", "mix/oracle"])
    comp.insert(0, "policy/predictor", comp.index)
    L += ["### 2c. Median compliance per cell (selected rows)", "",
          md(comp, 2), ""]
    allc = main_.groupby("who")["compliance"].apply(lambda s: float((s >= 0.99).mean()))
    L += [f"Share of events with compliance >= 0.99, all feasible events: " + ", ".join(f"{w} {allc.get(w, np.nan):.2f}" for w in
          ["cap", "delay/transformer", "burst/transformer", "burst/oracle", "mix-p90/transformer", "mix-nocloud/transformer"]), ""]

    # ratios
    rows = []
    for k, col in COST.items():
        pa = piv(main_, col)
        for p in PRED:
            for scope, pv in (("all", pa), ("2h/20%", piv(in_cell(main_, 2, 0.2), col))):
                if f"delay/{p}" not in pv:
                    continue
                r, w, n, tot = ratio(pv[f"mix-p90/{p}"], pv[f"delay/{p}"])
                r2, w2, n2, _ = ratio(pv[f"mix/{p}"], pv[f"delay/{p}"])
                r3, w3, n3, _ = ratio(pv[f"mix-p90/{p}"], pv[f"mix-nocloud/{p}"])
                rb = (pv[f"burst/{p}"] / pv[f"delay/{p}"].where(pv[f"delay/{p}"] > 0)).median()
                rows.append({"w_f": k, "predictor": p, "events": scope, "mix-p90/delay median": r, "mix-p90 cheaper": f"{w} of {n}",
                             "sum ratio": tot, "mix(p50)/delay median": r2, "mix(p50) cheaper": f"{w2} of {n2}",
                             "mix-p90/mix-nocloud median": r3, "mix-p90 cheaper than nocloud": f"{w3} of {n3}", "burst/delay median": rb})
    rt = pd.DataFrame(rows)
    rt.to_csv(out / "report_ratios.csv", index=False)
    L += ["### 2d. Per-event cost ratios (median over events; 'cheaper' = events where the first is cheaper)", "", md(rt, 3), ""]
    cellr = []
    for d, c in CELLS:
        if in_cell(main_, d, c).empty:
            continue
        pv = piv(in_cell(main_, d, c), "cost_usd")
        row = {"cell": f"{d}h/{int(c * 100)}%"}
        for p in PRED:
            r, w, n, _ = ratio(pv[f"mix-p90/{p}"], pv[f"delay/{p}"])
            row[p] = f"{r:.3f} ({w}/{n})"
        row["burst/delay (transformer)"] = (pv["burst/transformer"] / pv["delay/transformer"]).median()
        row["burst/delay (oracle)"] = (pv["burst/oracle"] / pv["delay/oracle"]).median()
        cellr.append(row)
    L += ["### 2e. mix-p90 / delay per cell, w_f = 0 (median ratio, cheaper in X of N)", "", md(pd.DataFrame(cellr), 1), ""]
    # cheapest policy per cell (transformer and oracle)
    best = []
    for d, c in CELLS:
        if in_cell(main_, d, c).empty:
            continue
        g = in_cell(main_, d, c).groupby("who")["usd_per_mwh"].median()
        for p in ("transformer", "oracle"):
            cand = g[[w for w in g.index if w == "cap" or w.endswith("/" + p)]].dropna()
            best.append({"cell": f"{d}h/{int(c * 100)}%", "predictor": p, "cheapest (median $/MWh, w_f 0)": cand.idxmin(), "$/MWh": cand.min()})
    L += ["### 2f. Cheapest policy per cell by median $/MWh (w_f = 0)", "", md(pd.DataFrame(best), 0), ""]

    # price runs
    pr = feas[(feas["duration_h"] == 2) & (feas["cut"].sub(0.2).abs() < 1e-9)]
    pp = []
    for k, col in COST.items():
        pv = piv(pr, col)
        for price in ("1.5", "4.1"):
            for w in (f"burst @{price}/transformer", f"burst @{price}/oracle", f"mix @{price}/oracle", f"mix-p90 @{price}/transformer"):
                base = "delay/" + w.split("/")[1]
                if w in pv:
                    r, win, n, _ = ratio(pv[w], pv[base])
                    pp.append({"w_f": k, "run": w, "vs": base, "median ratio": r, "cheaper": f"{win} of {n}",
                               "median $/MWh": pr[pr.who == w][USD[k]].median()})
        for w in ("burst/transformer", "burst/oracle", "mix/oracle", "mix-p90/transformer"):
            r, win, n, _ = ratio(pv[w], pv["delay/" + w.split("/")[1]])
            pp.append({"w_f": k, "run": w + " (at 2.745)", "vs": "delay/" + w.split("/")[1], "median ratio": r, "cheaper": f"{win} of {n}",
                       "median $/MWh": pr[pr.who == w][USD[k]].median()})
    L += ["### 2g. Cloud price, default cell (c_delay stays at 2.745 $/GPU-h, as in Philly E2)", "", md(pd.DataFrame(pp), 3), ""]

    # ---- 3. bursts
    b = bursts[~bursts["setting"].str.startswith("price=")].copy()
    b["who"] = b.apply(label, axis=1)
    bs = []
    for w, g in b.groupby("who"):
        cost = g["usd"].sum()
        tshare = g.groupby("type")["usd"].sum().sort_values(ascending=False) / cost if cost > 0 else pd.Series(dtype=float)
        bs.append({"policy/predictor": w, "events with bursts": g["event_id"].nunique(), "burst jobs": len(g),
                   "median remaining (min)": g["burst_rem_s"].median() / 60, "median GPUs": g["gpus"].median(),
                   "% Failed (waste) jobs": 100 * (g["sim_status"] == "Failed").mean(),
                   "jobs by type": ", ".join(f"{t} {n}" for t, n in g["type"].value_counts().items()),
                   "cloud $ by type": ", ".join(f"{t} {100 * v:.0f}%" for t, v in tshare.head(3).items())})
    bst = pd.DataFrame(bs)
    bst.to_csv(out / "report_bursts.csv", index=False)
    nb = main_.groupby("who")["n_burst"].sum()
    L += ["## 3. What gets burst (all events, main settings)", "", md(bst, 1), "",
          f"Burst-capable policies that never burst: {', '.join(w for w in nb.index if nb[w] == 0 and not w.startswith(('delay', 'cap', 'mix-nocloud'))) or 'none'}.", ""]
    e1 = ev.drop_duplicates("t_start")
    L += [f"Running GPUs at the 10 event starts: median {e1['running_gpus'].median():,.0f} (min {e1['running_gpus'].min():,}, max {e1['running_gpus'].max():,}); "
          f"GPUs held by jobs larger than the cloud limit (1,000 GPUs): median {e1['gpus_above_cloud_limit'].median():,.0f}, max {e1['gpus_above_cloud_limit'].max():,} "
          f"(in {int((e1['gpus_above_cloud_limit'] > 0).sum())} of 10 starts; jobs: {int(e1['jobs_above_cloud_limit'].sum())}). Such jobs can be delayed or capped but not burst.", ""]
    e1t = e1[["t_start", "running_jobs", "running_gpus", "gpus_above_cloud_limit", "base_mean_kw"]].copy()
    e1t["start (China time)"] = pd.to_datetime(e1t["t_start"] + pd.Timestamp("2023-06-21 16:00", tz="UTC").timestamp(), unit="s", utc=True).dt.tz_convert("Asia/Shanghai").dt.strftime("%m-%d %H:%M")
    L += [md(e1t[["start (China time)", "running_jobs", "running_gpus", "gpus_above_cloud_limit", "base_mean_kw"]], 0), ""]

    # ---- 4. Philly side by side
    M = macros()
    pa0, pa1 = piv(main_, "cost_usd"), piv(main_, "cost_usd_wfail1")
    dc = in_cell(main_, 2, 0.2)
    pd0, pd1 = piv(dc, "cost_usd"), piv(dc, "cost_usd_wfail1")
    med = dc.groupby("who")[["usd_per_mwh", "usd_per_mwh_wfail1", "compliance"]].median()
    lg = {p: ratio(pa0[f"mix-p90/{p}"], pa0[f"delay/{p}"]) for p in ("gbdt", "lightgbm", "transformer")}
    lg1 = {p: ratio(pa1[f"mix-p90/{p}"], pa1[f"delay/{p}"]) for p in ("gbdt", "lightgbm", "transformer")}
    nc0 = ratio(pa0["mix-p90/transformer"], pa0["mix-nocloud/transformer"])
    side = [
        ("Delay-only, default cell, median $/MWh (w_f 0 / 1), transformer", f"{M['DefDelayUsdZero']} / {M['DefDelayUsdOne']}",
         f"{med.loc['delay/transformer', 'usd_per_mwh']:,.0f} / {med.loc['delay/transformer', 'usd_per_mwh_wfail1']:,.0f}"),
        ("Cap-only, default cell, median compliance", M["DefCapCompliance"], f"{med.loc['cap', 'compliance']:.2f}"),
        ("Burst/delay per event, default cell, median (w_f 0 / 1), transformer", f"{M['BurstOverDelayDefaultZero']} / {M['BurstOverDelayDefaultOne']}",
         f"{(pd0['burst/transformer'] / pd0['delay/transformer']).median():.1f} / {(pd1['burst/transformer'] / pd1['delay/transformer']).median():.1f}"),
        ("Burst/delay per event, all feasible events (w_f 0 / 1), transformer", f"{M['BurstOverDelayAllZero']} / {M['BurstOverDelayAllOne']}",
         f"{(pa0['burst/transformer'] / pa0['delay/transformer']).median():.1f} / {(pa1['burst/transformer'] / pa1['delay/transformer']).median():.1f}"),
        ("mix-p90/delay, transformer, median (cheaper in X of N), w_f 0", f"{M['MixPNineOverDelayZero']} ({M['MixPNineOverDelayCheaperZero']} of {M['MixPNineOverDelayEventsZero']})",
         f"{lg['transformer'][0]:.3f} ({lg['transformer'][1]} of {lg['transformer'][2]})"),
        ("mix-p90/delay, gbdt, w_f 0", f"{M['LearnedGbdtMixOverDelayZero']} ({M['LearnedGbdtMixCheaperZero']} of 84)", f"{lg['gbdt'][0]:.3f} ({lg['gbdt'][1]} of {lg['gbdt'][2]})"),
        ("mix-p90/delay, lightgbm, w_f 0", f"{M['LearnedLightgbmMixOverDelayZero']} ({M['LearnedLightgbmMixCheaperZero']} of 84)", f"{lg['lightgbm'][0]:.3f} ({lg['lightgbm'][1]} of {lg['lightgbm'][2]})"),
        ("mix-p90/delay, transformer, w_f 1", f"{M['MixPNineOverDelayOne']} ({M['MixPNineOverDelayCheaperOne']} of {M['MixPNineOverDelayEventsOne']})",
         f"{lg1['transformer'][0]:.3f} ({lg1['transformer'][1]} of {lg1['transformer'][2]})"),
        ("mix-p90/mix-nocloud, transformer, median (cheaper), w_f 0", f"{M['MixOverNocloudTransPNineZeroZero']} ({M['MixOverNocloudTransPNineZeroCheaperZero']})",
         f"{nc0[0]:.3f} ({nc0[1]} of {nc0[2]})"),
        ("Infeasible events (P* below the idle floor)", M["SetupEventsInfeasible"], str(len(infeas))),
    ]
    L += ["## 4. Side by side with Philly", "", "Philly values are the paper macros (`paper/numbers.tex`); Acme values computed here the same way.", "",
          md(pd.DataFrame(side, columns=["quantity", "Philly", "Acme (Seren)"])), ""]
    (out / "report_tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
