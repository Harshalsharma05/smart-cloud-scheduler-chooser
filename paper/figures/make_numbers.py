"""S16: regenerate paper/numbers.tex from saved results only (no simulation, no hand-typed numbers).

Sources: results/motivation/stats.csv, results/e1, results/e1nc, results/e3 (contract and hold rule), results/e5,
ml/results/predictor_metrics.csv, backend/configs/flex.yaml, data/processed/philly_pool.json (pool and installed GPUs).
compute() returns an ordered dict {MacroName: (value, kind)} and is also used by check_claims.py.
Macro names contain letters only (LaTeX). Run from anywhere: python paper/figures/make_numbers.py
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import RESULTS, ROOT, load_runs  # noqa: E402

DEFAULT = (2, 0.2)
W = {"Zero": "cost_usd", "One": "cost_usd_wfail1"}                 # w_fail = 0 (primary) and 1
USD = {"Zero": "usd_per_mwh", "One": "usd_per_mwh_wfail1"}
PRED_NAMES = {"last2": "LastTwo", "gbdt": "Gbdt", "lightgbm": "Lightgbm", "transformer": "Transformer", "oracle": "Oracle"}


def camel(name):
    digits = dict(zip("0123456789", "Zero One Two Three Four Five Six Seven Eight Nine".split()))     # macro names: letters only
    return "".join((lambda q: q[:1].upper() + q[1:])("".join(digits.get(c, c) for c in p)) for p in name.split("_"))


def read_nc():
    d = pd.concat([pd.read_csv(p, dtype={"selector": str, "predictor": str, "setting": str}) for p in sorted(glob.glob(str(RESULTS / "e1nc" / "runs_d*.csv")))], ignore_index=True)
    d = d.fillna({"predictor": "", "setting": ""})
    return d[d["feasible"].astype(str) != "False"]


def in_cell(d, dur, cut):
    return d[(d["duration_h"] == dur) & (d["cut"].sub(cut).abs() < 1e-9)]


def ratio_stats(a, b):
    """Per-event ratio a/b over the common events: median ratio, number of events where a < b, number of events."""
    x = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    r = x["a"] / x["b"].where(x["b"] > 0)
    return float(r.median()), int((x["a"] < x["b"] - 1e-9).sum()), len(x), float(x["a"].sum() / x["b"].sum())


def compute():
    M = {}                                                              # name -> (value, kind); kind: int | f1 | f2 | f3 | pct0 ...

    def put(name, v, kind):
        M[name] = (float(v), kind)

    # ---- motivation (S10): stats.csv -> \Mot...
    S = pd.read_csv(RESULTS / "motivation" / "stats.csv", dtype={"value": str})
    for r in S.itertuples():
        try:
            v = float(r.value)
        except ValueError:
            M[camel("mot_" + r.name)] = (r.value, "str")
            continue
        n = r.name
        kind = ("f3" if n.endswith("_over_overall") else "f1" if (n.endswith("_pct") or n.endswith("_min")) and abs(v) < 100 else
                "int" if (n.endswith("_pct") or n.endswith("_min") or n.endswith("_kw")) else "int" if float(v).is_integer() else "int")
        if "_mean_kw" in n or n.endswith("_kw"):
            kind = "int"
        M[camel("mot_" + n)] = (v, kind)

    # ---- setup
    cfg = __import__("yaml").safe_load((ROOT / "backend" / "configs" / "flex.yaml").read_text(encoding="utf-8"))
    pool = json.loads((ROOT / "data" / "processed" / "philly_pool.json").read_text(encoding="utf-8"))
    put("SetupPoolGpus", pool.get("pool_gpus", pool.get("auto")), "int")
    put("SetupInstalledGpus", pool.get("installed_gpus", pool.get("installed")), "int")
    put("SetupWarmupDays", cfg["philly"]["warmup_days"], "int")
    put("SetupCloudCapGpus", cfg["cloud"]["max_cloud_gpus"], "int")
    raw = pd.read_csv(RESULTS / "e1" / "runs.csv", dtype={"selector": str, "predictor": str, "setting": str})
    n_ev, n_inf = raw["event_id"].nunique(), raw[raw["feasible"].astype(str) == "False"]["event_id"].nunique()
    put("SetupEventsTotal", n_ev, "int")
    put("SetupEventsInfeasible", n_inf, "int")
    put("SetupEventsFeasible", n_ev - n_inf, "int")
    inf_cuts = sorted(raw[raw["feasible"].astype(str) == "False"]["cut"].unique())
    assert len(inf_cuts) == 1, f"infeasible events at several cuts: {inf_cuts}"   # the text says "all at one cut"
    put("SetupInfeasibleCutPct", 100 * inf_cuts[0], "int")
    cov = raw[(raw["predictor"] == "transformer") & (raw["feasible"].astype(str) != "False")]["pred_coverage"].dropna()
    put("MethNoPredGpuShareMedianPct", 100 * (1 - cov.median()), "int")       # running GPUs at an event start without a prediction row

    # ---- Section 5 setup numbers (methodology_stats.py: trace loader, no simulation) and the baseline check (S9c validation.csv)
    for r in pd.read_csv(RESULTS / "methodology" / "stats.csv", dtype={"value": str}).itertuples():
        nm = camel("meth_" + r.name)
        try:
            M[nm] = (float(r.value), "int")
        except ValueError:
            M[nm] = (r.value, "str")
    val = pd.read_csv(RESULTS / "philly_baseline" / "validation.csv").set_index("metric")
    for metric, kind in (("busy_gpus_mean", "int"), ("utilisation_mean", "f2"), ("wait_mean_min", "int"), ("wait_p50_min", "f1"),
                         ("wait_p90_min", "int"), ("frac_wait_gt_60s", "f2")):
        put(camel("base_real_" + metric), val.loc[metric, "real"], kind)
        put(camel("base_sim_" + metric), val.loc[metric, "simulated"], kind)

    # ---- E1: default cell medians and ratios (transformer = the realistic predictor)
    e1 = load_runs("e1")
    e1 = e1[e1["setting"] != "attribution"]
    dc = in_cell(e1, *DEFAULT)
    med = dc.groupby("who")[["usd_per_mwh", "usd_per_mwh_wfail1", "shed_pct", "compliance"]].median()
    for who, nm in (("delay", "Delay"), ("cap", "Cap"), ("burst", "Burst"), ("mix", "Mix"), ("mix-p90", "MixPNine"), ("delay (last2)", "DelayLastTwo"),
                    ("mix (last2)", "MixLastTwo"), ("delay+oracle", "DelayOracle"), ("mix+oracle", "MixOracle"), ("burst+oracle", "BurstOracle")):
        for k, col in USD.items():
            put(f"Def{nm}Usd{k}", med.loc[who, col], "int")
    put("DefCapSharePct", med.loc["cap", "shed_pct"], "int")
    put("DefCapCompliance", med.loc["cap", "compliance"], "f2")
    capc = e1[e1.who == "cap"].groupby(["duration_h", "cut"])["compliance"].median()
    put("CapNoncompliantCells", int((capc < 0.99).sum()), "int")
    put("CellsTotal", len(capc), "int")
    for dur in (1, 2, 4):
        put(f"CapShareDur{camel({1: 'one', 2: 'two', 4: 'four'}[dur])}Pct", in_cell(e1[e1.who == "cap"], dur, 0.2)["shed_pct"].median(), "int")

    def piv(df, col):
        return df.pivot_table(index="event_id", columns="who", values=col, aggfunc="first")
    for k, col in W.items():
        p_all, p_def = piv(e1, col), piv(dc, col)
        put(f"BurstOverDelayDefault{k}", (p_def["burst"] / p_def["delay"]).median(), "int")           # per-event median, default cell
        put(f"BurstOverDelayAll{k}", (p_all["burst"] / p_all["delay"]).median(), "int")               # per-event median, all feasible events
        put(f"BurstCheaperThanDelay{k}", int((p_all["burst"] < p_all["delay"] - 1e-9).sum()), "int")  # W11: events where burst-only beats delay
        for who, nm in (("mix-p90", "MixPNine"), ("mix", "Mix"), ("mix (last2)", "MixLastTwo"), ("mix+oracle", "MixOracle")):
            base = {"mix (last2)": "delay (last2)", "mix+oracle": "delay+oracle"}.get(who, "delay")
            r, wins, n, tot = ratio_stats(p_all[who], p_all[base])
            put(f"{nm}OverDelay{k}", r, "f3" if r < 10 else "int")
            put(f"{nm}OverDelayCheaper{k}", wins, "int")
            put(f"{nm}OverDelayEvents{k}", n, "int")
            put(f"{nm}OverDelayTotal{k}", tot, "f2")

    # ---- E1nc: mix vs mix-nocloud, nocloud vs delay
    nc = read_nc()
    nc["who"] = nc.apply(lambda r: f"nc:{r['predictor']}:{'p90' if 'cq=p90' in r['setting'] else 'p50'}", axis=1)
    for k, col in W.items():
        pn = nc.pivot_table(index="event_id", columns="who", values=col, aggfunc="first")
        pm = piv(e1, col)
        pairs = {"TransPNineZero": ("mix-p90", "nc:transformer:p90"), "TransPFiveZero": ("mix", "nc:transformer:p50"),
                 "Oracle": ("mix+oracle", "nc:oracle:p50"), "LastTwo": ("mix (last2)", "nc:last2:p50")}
        for nm, (mw, nw) in pairs.items():
            r, wins, n, _ = ratio_stats(pm[mw], pn[nw])
            put(f"MixOverNocloud{nm}{k}", r, "f2")
            put(f"MixOverNocloud{nm}Cheaper{k}", wins, "int")
        r, wins, n, tot = ratio_stats(pn["nc:transformer:p90"], pm["delay"])
        put(f"NocloudOverDelay{k}", r, "f3")
        put(f"NocloudOverDelayCheaper{k}", wins, "int")
        put(f"NocloudOverDelayTotal{k}", tot, "f2")
        put(f"NocloudOverDelayReductionPct{k}", 100 * (1 - tot), "int")
        put(f"MixPNineOverDelayReductionPct{k}", 100 * (1 - M[f"MixPNineOverDelayTotal{k}"][0]), "int")
        # oracle in the deep cells
        for dur, cut, nm in ((4, 0.3, "FourThirty"), (4, 0.2, "FourTwenty")):
            a = in_cell(e1[e1.who == "mix+oracle"], dur, cut).set_index("event_id")[col]
            b = in_cell(nc[nc.who == "nc:oracle:p50"], dur, cut).set_index("event_id")[col]
            put(f"MixOverNocloudOracle{nm}{k}", ratio_stats(a, b)[0], "f2")
        pd_def = in_cell(nc[nc.who == "nc:transformer:p90"], *DEFAULT)
        put(f"NocloudDefaultUsd{k}", pd_def[USD[k]].median(), "int")
        put(f"NocloudOverDelayDefault{k}", ratio_stats(pd_def.set_index("event_id")[col], dc[dc.who == "delay"].set_index("event_id")[col])[0], "f2")

    # ---- E3: contract rule vs hold_beyond_grid, transformer mix, default cell (ratio of medians over the 10 events)
    def e3med(rule):
        f = RESULTS / "e3" / ("runs_default_hold.csv" if rule == "hold" else "runs_default.csv")
        d = pd.read_csv(f, dtype={"setting": str, "predictor": str, "selector": str}).fillna("")
        d = d[(d["policy"] == "mix") & (d["predictor"] == "transformer") & (d["selector"] == "predictor:transformer")]
        return d.groupby(d["setting"].str.contains("cq=p90"))[["cost_usd", "cost_usd_wfail1"]].median()
    c, h = e3med("contract"), e3med("hold")
    rat = []
    for q, nm in ((False, "PFifty"), (True, "PNine")):
        for k, col in W.items():
            r = c.loc[q, col] / h.loc[q, col]
            rat.append(r)
            put(f"ContractOverHold{nm}{k}", r, "int")
    put("ContractOverHoldMin", min(rat), "int")
    put("ContractOverHoldMax", max(rat), "int")

    # ---- W7 (Section 6): default-cell mix-p90 vs delay, clean-jobs (oos) run, burst at 30% cuts, contract-rule burst picks
    dly = dc[dc.who == "delay"].set_index("event_id")
    mp9 = dc[dc.who == "mix-p90"].set_index("event_id")
    o = pd.read_csv(RESULTS / "e3" / "runs_oos_hold.csv", dtype={"setting": str, "predictor": str, "selector": str}).fillna("")
    o = o[(o.policy == "mix") & (o.predictor == "transformer") & (o.setting.str.contains("cq=p90"))].set_index("event_id")
    for k, col in W.items():
        put(f"DefMixPNineOverDelay{k}", ratio_stats(mp9[col], dly[col])[0], "f2")
        put(f"OosMixPNineOverDelay{k}", ratio_stats(o[col], dly[col])[0], "f2")     # burst only jobs submitted in the test window
    b30 = e1[(e1.who == "burst") & (e1["cut"].sub(0.3).abs() < 1e-9)]
    put("BurstThirtyEvents", len(b30), "int")
    put("BurstThirtyCompliantEvents", int((b30["compliance"] >= 0.99).sum()), "int")
    cap3 = pd.read_csv(RESULTS / "s12b" / "burst_cap3000.csv")
    c3 = cap3[cap3["cap"] == 3000]
    put("DiagCapThreeKEvents", len(c3), "int")
    put("DiagCapThreeKCompliantEvents", int((c3["compliance"] >= 0.99).sum()), "int")
    put("DiagCapThreeKCloudKusdMin", c3["cloud_usd"].min() / 1000, "int")
    put("DiagCapThreeKCloudKusdMax", c3["cloud_usd"].max() / 1000, "int")
    sb = pd.read_csv(RESULTS / "s14b" / "tables.csv")
    bu, run = sb[sb["set"] == "burst"], sb[sb["set"] == "all running"]
    put("ContractBurstFloorMinPct", 100 * bu["floor60"].min(), "int")
    put("ContractBurstFloorMaxPct", 100 * bu["floor60"].max(), "int")
    put("ContractRunningFloorMinPct", 100 * run["floor60"].min(), "int")
    put("ContractRunningFloorMaxPct", 100 * run["floor60"].max(), "int")
    put("ContractBurstActualRemMinH", bu["actual_rem_med_h"].min(), "int")
    put("ContractBurstActualRemMaxH", bu["actual_rem_med_h"].max(), "int")
    rm = pd.read_csv(RESULTS / "robust_missing" / "summary.csv")
    for fb, fn in (("oracle", "Oracle"), ("uninformed", "Uninf")):
        for k, wn in (("Zero", "wfail0"), ("One", "wfail1")):
            for pol, pn in (("mix", "Mix"), ("nocloud", "Nocloud")):
                r = rm[(rm.fallback == fb) & (rm.wfail == wn) & (rm.policy == pol)].iloc[0]
                put(f"Robust{fn}{pn}OverDelay{k}", r.median_ratio, "f3")
                put(f"Robust{fn}{pn}Cheaper{k}", r.cheaper, "int")
                put(f"Robust{fn}{pn}Total{k}", r.total_ratio, "f2")
                put(f"Robust{fn}{pn}ReductionPct{k}", 100 * (1 - r.total_ratio), "int")

    # E3 (default cell, hold rule): median mix cost per event (w_fail 0) with each predictor's own p50 estimate
    e3h = pd.read_csv(RESULTS / "e3" / "runs_default_hold.csv", dtype={"setting": str, "predictor": str, "selector": str}).fillna("")
    for p_, nm_ in PRED_NAMES.items():
        q = e3h[(e3h.policy == "mix") & (e3h.predictor == p_) & (e3h.selector == f"predictor:{p_}") & ~e3h.setting.str.contains("cq=p90")]
        if len(q):
            put(f"Ethree{nm_}MixUsdEvent", q["cost_usd"].median(), "int")

    # W4 leakage check: jobs submitted before TRAIN_END get uninformed values from the learned predictors
    ri = pd.read_csv(RESULTS / "robust_insample" / "summary.csv")
    for k, wn in (("Zero", "wfail0"), ("One", "wfail1")):
        for pol, pn in (("mix", "Mix"), ("nocloud", "Nocloud")):
            r = ri[(ri.variant == "insample_uninformed") & (ri.wfail == wn) & (ri.policy == pol)].iloc[0]
            put(f"Insample{pn}OverDelay{k}", r.median_ratio, "f3")
            put(f"Insample{pn}Cheaper{k}", r.cheaper, "int")
            put(f"Insample{pn}ReductionPct{k}", 100 * (1 - r.total_ratio), "int")
        r = ri[(ri.variant == "insample_over_main") & (ri.wfail == wn)].iloc[0]
        put(f"InsampleDelayOverMain{k}", r.median_ratio, "f2")
    put("InsampleRunningJobsPct", ri[ri.policy == "running_jobs_pct_median"].median_ratio.iloc[0], "int")
    put("InsampleRunningGpusPct", ri[ri.policy == "running_gpus_pct_median"].median_ratio.iloc[0], "int")

    # W8: p90 mix vs delay-only with gbdt and lightgbm (each with its own delay-only; also vs the transformer's delay-only)
    ls_ = pd.read_csv(RESULTS / "e1_learned" / "summary.csv")
    nl = pd.concat([pd.read_csv(f, dtype={"selector": str, "predictor": str, "setting": str})
                    for f in sorted((RESULTS / "e1_learned").glob("runs_*.csv"))]).fillna({"setting": ""})
    nl = nl[nl["feasible"].astype(str) != "False"]
    dly_all = e1[e1.who == "delay"].set_index("event_id")
    for p_, nm_ in (("gbdt", "Gbdt"), ("lightgbm", "Lightgbm")):
        for k, wn in (("Zero", "wfail0"), ("One", "wfail1")):
            r = ls_[(ls_.predictor == p_) & (ls_.wfail == wn)].iloc[0]
            put(f"Learned{nm_}MixOverDelay{k}", r.median_ratio, "f3")
            put(f"Learned{nm_}MixCheaper{k}", r.cheaper, "int")
            mixp = nl[(nl.policy == "mix") & (nl.predictor == p_) & (nl.setting == "cq=p90")].set_index("event_id")[W[k]]
            put(f"Learned{nm_}MixOverTransDelay{k}", ratio_stats(mixp, dly_all[W[k]])[0], "f2")

    # W8b: three learned predictors x {main, masked}: mix-p90 / delay-only, mix-p90 / mix-nocloud-p90, events cheaper, total
    la = pd.read_csv(RESULTS / "e1_learned_nc" / "summary.csv")
    pn_ = {"transformer": "Trans", "gbdt": "Gbdt", "lightgbm": "Lightgbm"}
    for r in la.itertuples():
        base = f"Set{pn_[r.predictor]}{r.variant.capitalize()}"
        for k, wn in (("Zero", "wfail0"), ("One", "wfail1")):
            put(f"{base}MixOverDelay{k}", getattr(r, f"mix_delay_{wn}"), "f3")
            put(f"{base}MixOverNocloud{k}", getattr(r, f"mix_nocloud_{wn}"), "f2")
            put(f"{base}Cheaper{k}", getattr(r, f"cheaper_{wn}"), "int")
            put(f"{base}ReductionPct{k}", 100 * (1 - getattr(r, f"total_{wn}")), "f1")
    main_ = la[la.variant == "main"]
    ratios = main_.mix_delay_wfail0                     # W9b: the gain claim is scoped to w_f = 0
    put("LearnedGainMinPct", 100 * (1 - ratios.max()), "int")       # per-event median gain of the p90 mix over delay-only
    put("LearnedGainMaxPct", 100 * (1 - ratios.min()), "int")
    ncr = pd.concat([la.mix_nocloud_wfail0, la.mix_nocloud_wfail1])
    put("LearnedMixOverNocloudMin", ncr.min(), "f2")
    put("LearnedMixOverNocloudMax", ncr.max(), "f2")
    shift = max(abs(la[(la.predictor == p) & (la.variant == "masked")][c].iloc[0] - la[(la.predictor == p) & (la.variant == "main")][c].iloc[0])
                for p in pn_ for c in ("mix_delay_wfail0", "mix_delay_wfail1"))
    put("LearnedMaskShiftMax", shift, "f3")

    # ---- W9 (review fixes)
    sens = pd.read_csv(RESULTS / "t4" / "sens_summary.csv").set_index("perturbation")      # make_t4_compact.py
    for key, nm_ in (("default", "Default"), ("pessimistic cap curve", "Pessimistic"), ("host power x0.5", "HostHalf"),
                     ("host power x1.5", "HostOneHalf")):
        put(f"Sens{nm_}Ratio", sens.loc[key, "ratio"], "f2")
        put(f"Sens{nm_}Cheaper", sens.loc[key, "cheaper"], "int")
    put("SensEvents", sens.loc["default", "events"], "int")
    sa = pd.read_csv(RESULTS / "s13" / "sweepA_ratios.csv")
    sa = sa[sa["boot"] == cfg["cloud"]["boot_delay_s"]].set_index("price")                 # default boot delay; $/MWh ratios, default cell
    for price, nm_ in ((0.43, "Spot"), (1.5, "Low"), (3.06, "Base"), (6.12, "Double")):
        put(f"Price{nm_}Usd", price, "f2")
        put(f"Price{nm_}BurstOverDelay", sa.loc[price, "burst/delay"], "int")
    put("PriceSpotOracleMixOverDelay", sa.loc[0.43, "mix+oracle/delay+oracle"], "f2")
    put("PriceSpotOracleMixCheaper", sa.loc[0.43, "mix+oracle<delay+oracle"], "int")
    put("PriceSpotMixPNineOverDelay", sa.loc[0.43, "mix-p90/delay"], "f2")
    put("PriceBaseOracleMixOverDelay", sa.loc[3.06, "mix+oracle/delay+oracle"], "f2")
    bt = pd.read_csv(RESULTS / "bootstrap" / "ci.csv")                                      # make_bootstrap.py
    for r in bt.itertuples():
        b_ = f"Boot{ {'transformer': 'Trans', 'gbdt': 'Gbdt', 'lightgbm': 'Lightgbm'}[r.predictor]}"
        k_ = "Zero" if r.wfail == "wfail0" else "One"
        put(f"{b_}Lo{k_}", r.ci_lo, "f2")
        put(f"{b_}Hi{k_}", r.ci_hi, "f2")
    put("BootResamples", bt["resamples"].iloc[0], "int")
    put("BootDays", bt["days"].iloc[0], "int")
    mach = pd.read_parquet(ROOT / "data" / "processed" / "philly_machines.parquet")
    put("MethMachines", len(mach), "int")
    put("MethMachineGpus", mach["num_gpus"].sum(), "int")
    sh = pd.read_csv(RESULTS / "s14b" / "hold" / "tables.csv").set_index(["variant", "set"])  # last2 under the hold rule, default cell
    put("LastTwoHoldBurstFloorPct", 100 * sh.loc[("last2-p50", "burst"), "floor60"], "int")
    put("LastTwoHoldRunningFloorPct", 100 * sh.loc[("last2-p50", "all running"), "floor60"], "int")
    put("LastTwoHoldBurstActualRemH", sh.loc[("last2-p50", "burst"), "actual_rem_med_h"], "int")
    tf = json.loads((ROOT / "runs" / "tf_v1" / "summary.json").read_text(encoding="utf-8"))      # Track 1 run summary (read only)
    put("TfParams", tf["params"], "int")
    put("TfEpochs", tf["epochs_run"], "int")
    put("TfTrainMin", tf["epochs_run"] * tf["seconds_per_epoch"] / 60, "int")
    bl = pd.concat([pd.read_csv(f, dtype={"selector": str, "predictor": str, "setting": str})
                    for f in sorted((RESULTS / "e1_blind").glob("runs_*.csv"))]).fillna({"setting": ""})
    bl = bl[bl["feasible"].astype(str) != "False"].set_index("event_id")
    bdc = in_cell(bl.reset_index(), *DEFAULT)
    for k, col in W.items():
        put(f"BlindDelayUsd{k}", bdc[USD[k]].median(), "int")
        put(f"BlindOverTransDelay{k}", ratio_stats(bl[col], dly_all[col])[0], "f2")
        put(f"BlindOverOracleDelay{k}", ratio_stats(bl[col], e1[e1.who == "delay+oracle"].set_index("event_id")[col])[0], "f2")

    bs = pd.read_csv(RESULTS / "e1_blind" / "summary.csv")                                  # make_blind.py: policy / blind delay-only
    for pol, nm_ in (("transformer mix-p90", "TransMix"), ("gbdt mix-p90", "GbdtMix"), ("lightgbm mix-p90", "LightgbmMix"),
                     ("oracle mix", "OracleMix"), ("gbdt delay", "GbdtDelay"), ("lightgbm delay", "LightgbmDelay")):
        for k, wn in (("Zero", "wfail0"), ("One", "wfail1")):
            r = bs[(bs.policy == pol) & (bs.wfail == wn)].iloc[0]
            put(f"{nm_}OverBlind{k}", r.median_ratio, "f2")
            put(f"{nm_}OverBlindCheaper{k}", r.cheaper, "int")

    pwf = pd.read_csv(RESULTS / "robust_planwf" / "summary.csv")                           # make_planwf.py (w_f = 1, plan_wf = match)
    m_ = pwf[(pwf.plan_wf == "match") & (pwf.reference == "blind_delay")]
    for r in m_.itertuples():
        b_ = f"Planwf{ {'transformer': 'Trans', 'gbdt': 'Gbdt', 'lightgbm': 'Lightgbm'}[r.predictor]}"
        put(f"{b_}OverBlind", r.median, "f3")
        put(f"{b_}OverBlindLo", r.ci_lo, "f2")
        put(f"{b_}OverBlindHi", r.ci_hi, "f2")
        put(f"{b_}OverBlindCheaper", r.cheaper, "int")
    put("PlanwfGainMinPct", 100 * (1 - m_["median"].max()), "f1")
    put("PlanwfGainMaxPct", 100 * (1 - m_["median"].min()), "f1")
    put("PlanwfDelayOverBlindMax", pwf[pwf.reference == "delay_vs_blind"]["median"].max(), "f2")

    # ---- predictor accuracy (test split)
    pm_ = pd.read_csv(ROOT / "ml" / "results" / "predictor_metrics.csv", dtype={"elapsed_s": str})
    allr = pm_[pm_["elapsed_s"] == "all"].set_index("predictor")
    for p, nm in PRED_NAMES.items():
        if p not in allr.index:
            continue
        put(f"Pred{nm}Pinball", allr.loc[p, "pinball"], "f3")
        put(f"Pred{nm}MaeMin", allr.loc[p, "mae_p50_min"], "f1")
        put(f"Pred{nm}Coverage", allr.loc[p, "coverage_p10_p90"], "f2")
        put(f"Pred{nm}Logloss", allr.loc[p, "logloss"], "f3")
        put(f"Pred{nm}Auroc", allr.loc[p, "auroc_failed"], "f3")
    put("PredTestRows", allr.loc["last2", "n"], "int")

    # ---- E5 carbon
    ev = pd.read_csv(RESULTS / "e5" / "e5_events.csv")
    summ = pd.read_csv(RESULTS / "e5" / "e5_carbon.csv")
    cap = {"delay": "Delay", "mix": "Mix", "nocloud": "Nocloud", "oracle": "Oracle"}
    rg = {"us_west_2": "West", "us_east_1": "EastOne", "us_east_2": "EastTwo"}
    for (pol, reg), g in ev.groupby(["policy", "region"]):
        put(f"Ecarbon{cap[pol]}{rg[reg]}AddedPct", 100 * g["added_kg"].sum() / g["avoided_kg"].sum(), "f1")
        put(f"Ecarbon{cap[pol]}{rg[reg]}Net", g["net_kg_per_mwh"].median(), "f1")
        put(f"Ecarbon{cap[pol]}{rg[reg]}Share", 100 * (g["net_kg"] > 0).mean(), "int")
    put("EcarbonEvents", ev["event_id"].nunique(), "int")
    put("EcarbonLocal", cfg["carbon"]["local_camx"], "int")
    for reg in rg:
        put(f"EcarbonCloud{rg[reg]}", cfg["carbon"][f"cloud_{reg}"], "int")
    for pol in ("mix", "oracle"):
        g = ev[(ev.policy == pol) & (ev.region == "us_east_2")]
        put(f"Ecarbon{cap[pol]}BurstEventsPct", 100 * (g["cloud_kwh_window"] > 0).mean(), "int")
    acme(M, put, cfg)
    return M


def acme(M, put, cfg):
    """Step 35 (S-A1): Acme Seren macros (prefix Acme), from results/acme_baseline and results/acme_e1. Skipped if absent."""
    if not (RESULTS / "acme_e1" / "runs.csv").exists():
        return
    a = cfg["acme"]
    put("AcmePoolGpus", a["pool_gpus"], "int")
    put("AcmeUtil", a["util"], "f2")
    put("AcmeTargetGpuPowerW", a["target_gpu_power_w"], "int")
    for k, nm in (("p_idle_w", "PIdleW"), ("p_max_w", "PMaxW"), ("h_idle_w", "HIdleW"), ("h_busy_w", "HBusyW")):
        put(f"Acme{nm}", a["power"][k], "int")
    put("AcmePriceGpuH", a["cloud"]["price_per_gpu_h"], "f3")
    val = pd.read_csv(RESULTS / "acme_baseline" / "validation.csv").set_index("metric")
    for metric, kind in (("busy_gpus_mean", "int"), ("utilisation_mean", "f2"), ("wait_mean_min", "f1")):
        put(camel("acme_real_" + metric), val.loc[metric, "real"], kind)
        put(camel("acme_sim_" + metric), val.loc[metric, "simulated"], kind)
    r = pd.read_csv(RESULTS / "acme_e1" / "runs.csv", dtype={"selector": str, "predictor": str, "setting": str}).fillna({"predictor": "", "setting": ""})
    put("AcmeEvents", r["event_id"].nunique(), "int")
    put("AcmeEventsInfeasible", r.loc[r["feasible"].astype(str) == "False", "event_id"].nunique(), "int")
    r = r[r["feasible"].astype(str) != "False"]

    def who(x):
        s = x["setting"]
        if x["policy"] == "cap":
            return "cap"
        if s.startswith("price="):
            return f"{x['policy']}{'-p90' if 'cq=p90' in s else ''}@{s.split(';')[0][6:]}/{x['predictor']}"
        if x["policy"] == "mix":
            return ("nocloud" if "nocloud" in s else "mix-p90" if "cq=p90" in s else "mix") + "/" + x["predictor"]
        return f"{x['policy']}/{x['predictor']}"
    r["who"] = r.apply(who, axis=1)
    dc = in_cell(r, *DEFAULT)
    med = dc.groupby("who")[["usd_per_mwh", "usd_per_mwh_wfail1", "compliance"]].median()
    for k, col in USD.items():
        put(f"AcmeDefDelayUsd{k}", med.loc["delay/transformer", col], "int")
        put(f"AcmeDefBurstUsd{k}", med.loc["burst/transformer", col], "int")
        put(f"AcmeDefMixPNineUsd{k}", med.loc["mix-p90/transformer", col], "int")
    put("AcmeDefCapCompliance", med.loc["cap", "compliance"], "f2")
    for k, col in W.items():
        pa = r.pivot_table(index="event_id", columns="who", values=col, aggfunc="first")
        pdc = dc.pivot_table(index="event_id", columns="who", values=col, aggfunc="first")
        put(f"AcmeBurstOverDelayDefault{k}", (pdc["burst/transformer"] / pdc["delay/transformer"]).median(), "f1")
        put(f"AcmeBurstOverDelayAll{k}", (pa["burst/transformer"] / pa["delay/transformer"]).median(), "f1")
        for p, nm in PRED_NAMES.items():
            rr, wins, n, tot = ratio_stats(pa[f"mix-p90/{p}"], pa[f"delay/{p}"])
            put(f"AcmeMixPNineOverDelay{nm}{k}", rr, "f3")
            put(f"AcmeMixPNineOverDelayCheaper{nm}{k}", wins, "int")
            put(f"AcmeMixPNineOverDelayEvents{nm}{k}", n, "int")
            rr, wins, n, _ = ratio_stats(pa[f"mix-p90/{p}"], pa[f"nocloud/{p}"])
            put(f"AcmeMixOverNocloud{nm}{k}", rr, "f3")
            put(f"AcmeMixOverNocloudCheaper{nm}{k}", wins, "int")
        for w in ("burst@1.5/oracle", "burst@1.5/transformer", "mix@1.5/oracle"):
            if w in pdc:
                put(camel("acme_cheap_" + w.split("@")[0] + "_" + w.split("/")[1]) + f"OverDelay{k}",
                    (pdc[w] / pdc["delay/" + w.split("/")[1]]).median(), "f2")
        # step 36 (W11): ranges over the learned predictors, burst-cheaper counts, the $4.10 sensitivity
        learned = ("gbdt", "lightgbm", "transformer")
        mp = [ratio_stats(pa[f"mix-p90/{p}"], pa[f"delay/{p}"]) for p in learned]
        put(f"AcmeLearnedMixMin{k}", min(x[0] for x in mp), "f3")
        put(f"AcmeLearnedMixMax{k}", max(x[0] for x in mp), "f3")
        put(f"AcmeLearnedMixCheaperMin{k}", min(x[1] for x in mp), "int")
        put(f"AcmeLearnedMixCheaperMax{k}", max(x[1] for x in mp), "int")
        m5 = [ratio_stats(pa[f"mix/{p}"], pa[f"delay/{p}"])[0] for p in learned]
        put(f"AcmeLearnedMixPFiveMin{k}", min(m5), "f1")
        put(f"AcmeLearnedMixPFiveMax{k}", max(m5), "f1")
        nc = [ratio_stats(pa[f"mix-p90/{p}"], pa[f"nocloud/{p}"])[0] for p in learned]
        put(f"AcmeLearnedNocloudMin{k}", min(nc), "f3")
        put(f"AcmeLearnedNocloudMax{k}", max(nc), "f3")
        bc = [int((pa[f"burst/{p}"] < pa[f"delay/{p}"] - 1e-9).sum()) for p in learned]
        put(f"AcmeLearnedBurstCheaperMax{k}", max(bc), "int")
        put(f"AcmeBurstCheaperTransformer{k}", bc[learned.index("transformer")], "int")
        put(f"AcmeOracleBurstCheaper{k}", int((pa["burst/oracle"] < pa["delay/oracle"] - 1e-9).sum()), "int")
        put(f"AcmeOracleMixGainPct{k}", 100 * (1 - M[f"AcmeMixPNineOverDelayOracle{k}"][0]), "int")
        if "burst@4.1/transformer" in pdc:
            put(f"AcmeHighBurstTransformerOverDelay{k}", (pdc["burst@4.1/transformer"] / pdc["delay/transformer"]).median(), "f1")
    put("AcmeCapCompliantEvents", int(((r["who"] == "cap") & (r["compliance"] >= 0.99)).sum()), "int")
    # W12: largest gain of a learned p90 mix over delay-only on Acme, over both weightings (w_f = 1: 0.985 -> 1.5%)
    put("AcmeLearnedMixGainMaxPct", 100 * (1 - min(M["AcmeLearnedMixMinZero"][0], M["AcmeLearnedMixMinOne"][0])), "f1")
    put("AcmeHighPrice", 4.1, "f2")
    put("AcmeCheapPrice", 1.5, "f2")
    # what burst-only (transformer) moves: jobs with > 1 h left, share of jobs and of cloud dollars
    b = pd.read_csv(RESULTS / "acme_e1" / "bursts.csv", dtype={"setting": str, "predictor": str, "job_id": str}).fillna({"setting": "", "predictor": ""})
    g = b[(b["policy"] == "burst") & (b["predictor"] == "transformer") & (b["setting"] == "")]
    lng = g["burst_rem_s"] > 3600
    put("AcmeBurstLongJobsPct", 100 * lng.mean(), "int")
    put("AcmeBurstLongUsdPct", 100 * g.loc[lng, "usd"].sum() / g["usd"].sum(), "int")
    put("AcmeBurstMedianRemMin", g["burst_rem_s"].median() / 60, "int")
    # calibration and set-up
    cal = pd.read_csv(RESULTS / "acme_baseline" / "calibration.csv").iloc[0]
    put("AcmeSimGpuPowerW", cal["sim_mean_gpu_power_w"], "f1")
    put("AcmeCdfGpuPowerW", cal["cdf_gpu_mean_w"], "f1")
    put("AcmeFloorKw", cal["floor_kw"], "int")
    put("AcmeFacilityMeanKw", cal["facility_mean_kw"], "int")
    st = pd.read_csv(RESULTS / "acme_motivation" / "stats.csv").set_index("name")["value"]
    for name, kind in (("seren_evening_long_gpu_share_pct", "int"), ("philly_evening_long_gpu_share_pct", "int"),
                       ("kalos_evening_running_jobs_mean", "f1"), ("seren_wall_median_min", "f1"), ("philly_wall_median_min", "f1"),
                       ("seren_gpuh_share_ge256_pct", "int"), ("seren_max_gpus", "int"), ("sim_jobs", "int"), ("test_jobs", "int"),
                       ("warmup_jobs", "int"), ("carry_jobs", "int")):
        put(camel("acme_mot_" + name), st[name], kind)
    # Acme predictor accuracy (Track 1, A2)
    am = pd.read_csv(ROOT / "ml" / "results" / "acme_predictor_metrics.csv", dtype={"elapsed_s": str})
    am = am[am["elapsed_s"] == "all"].set_index("predictor")
    for p, nm in list(PRED_NAMES.items()) + [("lightgbm_notype", "LightgbmNotype")]:
        if p in am.index and p != "oracle":
            put(f"AcmePred{nm}Pinball", am.loc[p, "pinball"], "f3")
            put(f"AcmePred{nm}Auroc", am.loc[p, "auroc_failed"], "f3")
            put(f"AcmePred{nm}Logloss", am.loc[p, "logloss"], "f3")


def fmt(v, kind):
    if kind == "str":
        return v
    if kind == "int":
        return f"{int(round(v)):,}".replace(",", "{,}")
    return f"{v:.{int(kind[1])}f}"


def main():
    M = compute()
    B = chr(92)
    lines = ["% Written by paper/figures/make_numbers.py from saved results only. Do not edit by hand; never type a result here."]
    for name, (v, kind) in M.items():
        val = fmt(v, kind)
        assert re.fullmatch(r"[A-Za-z]+", name), f"macro name must be letters only (LaTeX): {name}"
        assert not re.search(r"(?<!\\)[#%&_$]", val), f"unescaped special character in value of {name}: {val}"
        lines.append(f"{B}newcommand{{{B}{name}}}{{{val}}}")
    (ROOT / "paper" / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote paper/numbers.tex:", len(M), "macros")
    return M


if __name__ == "__main__":
    main()
