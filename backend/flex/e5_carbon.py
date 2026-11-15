"""E5 carbon side check (SIM_PLAN S15, WORK_ORDER step 26). Event-window emissions only.

  python -m flex.e5_carbon run --duration 2 --cut 0.2   # re-run mix(transformer p90) and mix(oracle) on the E1 events,
                                                        # measure the cloud energy inside [t_start, t_end)
  python -m flex.e5_carbon table                        # merge the parts, join shed from results/e1 and results/e1nc, write the table

Nothing in the simulator is changed. The re-runs use the E1 seed, settings and prediction lookup rule, so the shed
energy must equal the stored E1 value (checked). Cloud energy mirrors power.cloud_energy_kwh (GPU power at the job's
utilisation, no cap, plus busy-host share, times the cloud PUE) but is clipped to the event window and covers
READY..RELEASED only.
"""
import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from .experiments import ROOT, load_predictors, load_setup, run_events, DEFAULT_EVENT
from .models import load_config
from .policies import run_policy, world_from_config
from .power import cloud_energy_kwh, gpu_power_w

OUT = ROOT / "results" / "e5"
REGIONS = ["us_west_2", "us_east_1", "us_east_2"]
POLICIES = [("delay", "delay (transformer)"), ("mix", "mix (transformer, p90)"),
            ("nocloud", "mix-nocloud (transformer, p90)"), ("oracle", "mix (oracle)")]


def window_cloud_kwh(engine, params, t0, t1):
    """Cloud energy (kWh) of all instances restricted to [t0, t1): same formula as power.cloud_energy_kwh."""
    total_w_s = 0.0
    for inst in engine.cloud.instances:
        if inst.ready is None:
            continue
        end = inst.released if inst.released is not None else engine.now
        dt = min(end, t1) - max(inst.ready, t0)
        if dt > 0:
            j = engine.jobs[inst.job_id]
            total_w_s += inst.gpus * (gpu_power_w(params, j.util) + params.h_busy_w) * dt
    return params.cloud_pue * total_w_s / 3.6e6


def run(args):
    cfg = load_config()
    cfg, jobs, pool, events = load_setup(cfg, "philly", False)
    events = [e for e in events if (args.duration is None or e.duration_h == args.duration)
              and (args.cut is None or abs(e.cut - args.cut) < 1e-9)]
    predictors = load_predictors(cfg, "philly")
    world = world_from_config(cfg, predictors["transformer"])
    cases = [("mix", "transformer", "p90"), ("oracle", "oracle", "p50")]
    rows_out = []

    def per_event(snap, ev, base):
        for label, pname, q in cases:
            w = dataclasses.replace(world, predictor=predictors[pname], cost_quantile=q)
            row, _, pol = run_policy(snap, ev, "mix", w, pname, "predictor", cfg["seed"], base, return_runs=True)
            eng = pol.engine
            rows_out.append(dict(event_id=ev.event_id if hasattr(ev, "event_id") else row["event_id"], policy=label,
                                 shed_mwh_rerun=row["shed_mwh"], n_burst=row["n_burst"],
                                 cloud_kwh_window=window_cloud_kwh(eng, w.params, ev.t_start, ev.t_end),
                                 cloud_kwh_all=cloud_energy_kwh(eng, w.params) if eng.cloud.instances else 0.0,
                                 cloud_gpu_h_row=row["cloud_gpu_h"]))
        return []
    run_events(cfg, jobs, pool, world, events, per_event, "e5")
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"d{args.duration:g}_c{args.cut:g}" if args.duration is not None else "all"
    pd.DataFrame(rows_out).to_csv(OUT / f"cloud_{tag}.csv", index=False)
    print("wrote", OUT / f"cloud_{tag}.csv", len(rows_out), "rows")


def table(cfg):
    ci = cfg["carbon"]
    local = ci["local_camx"]
    cl = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("cloud_*.csv"))], ignore_index=True)
    e1 = pd.read_csv(ROOT / "results" / "e1" / "runs.csv", dtype={"selector": str, "predictor": str, "setting": str}).fillna({"setting": ""})
    nc = pd.concat([pd.read_csv(f, dtype={"setting": str}) for f in sorted((ROOT / "results" / "e1nc").glob("runs_d*.csv"))])
    feas = e1[e1.policy == "cap"].set_index("event_id").feasible
    ok = set(feas[feas].index)
    tf = e1[(e1.predictor == "transformer")]
    base = {
        "delay": tf[tf.policy == "delay"].set_index("event_id").shed_mwh,
        "mix": tf[(tf.policy == "mix") & (tf.setting == "cq=p90")].set_index("event_id").shed_mwh,
        "nocloud": nc[(nc.predictor == "transformer") & (nc.setting == "nocloud;cq=p90")].set_index("event_id").shed_mwh,
        "oracle": e1[(e1.predictor == "oracle") & (e1.policy == "mix") & (e1.setting == "")].set_index("event_id").shed_mwh,
    }
    cl = cl.set_index(["policy", "event_id"])
    for p in ("mix", "oracle"):                         # the re-run must reproduce the stored E1 shed
        sub = cl.loc[p]
        diff = (sub.shed_mwh_rerun - base[p].reindex(sub.index)).abs().max()
        print(f"check: max |shed rerun - stored E1| for {p} = {diff:.2e} MWh")
        assert diff < 1e-6, "re-run differs from stored E1 results"
    recs = []
    for p, _ in POLICIES:
        for ev in sorted(ok):
            shed = base[p][ev]
            add_kwh = cl.loc[(p, ev), "cloud_kwh_window"] if p in ("mix", "oracle") else 0.0
            for r in REGIONS:
                avoided = shed * local                                    # MWh x g/kWh = kg
                added = add_kwh * ci[f"cloud_{r}"] / 1000.0               # kWh x g/kWh / 1000 = kg
                recs.append(dict(policy=p, event_id=ev, region=r, shed_mwh=shed, cloud_kwh_window=add_kwh,
                                 avoided_kg=avoided, added_kg=added, net_kg=avoided - added,
                                 net_kg_per_mwh=(avoided - added) / shed if shed > 0 else np.nan))
    d = pd.DataFrame(recs)
    d["default_cell"] = d.event_id.map(e1.drop_duplicates("event_id").set_index("event_id").pipe(
        lambda x: (x.duration_h == DEFAULT_EVENT[0]) & (abs(x.cut - DEFAULT_EVENT[1]) < 1e-9)))
    d.to_csv(OUT / "e5_events.csv", index=False)
    g = d.groupby(["policy", "region"])
    s = g.agg(events=("event_id", "nunique"), median_net_kg_per_mwh=("net_kg_per_mwh", "median"),
              share_net_positive=("net_kg", lambda x: float((x > 0).mean())),
              median_added_kg=("added_kg", "median"), median_avoided_kg=("avoided_kg", "median"),
              total_cloud_kwh_window=("cloud_kwh_window", lambda x: float(x.sum()) / 1 if False else float(x.sum()))).reset_index()
    dc = d[d.default_cell].groupby(["policy", "region"]).net_kg_per_mwh.median().rename("median_net_kg_per_mwh_default_cell")
    s = s.merge(dc.reset_index(), on=["policy", "region"])
    s.to_csv(OUT / "e5_carbon.csv", index=False)
    print(s.to_string())
    write_tex(d, ci)


def write_tex(d, ci):
    """T6: cloud emissions as % of avoided emissions, pooled over the feasible events (sum added / sum avoided)."""
    names = dict(POLICIES)
    pooled = {k: 100.0 * g["added_kg"].sum() / g["avoided_kg"].sum() for k, g in d.groupby(["policy", "region"])}
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Cloud emissions as a percentage of the emissions avoided on site, pooled over the " + str(d.event_id.nunique())
             + r" feasible events. Event window only; eGRID2023 annual average rates, average not marginal.}",
             r"\label{tab:carbon}", r"\begin{tabular}{lrrr}", r"\toprule",
             r"Policy & us-west-2 & us-east-1 & us-east-2 \\",
             rf" & ({ci['cloud_us_west_2']:.0f} g/kWh) & ({ci['cloud_us_east_1']:.0f} g/kWh) & ({ci['cloud_us_east_2']:.0f} g/kWh) \\", r"\midrule"]
    for p, _ in POLICIES:
        lines.append(f"{names[p]} & " + " & ".join(f"{pooled[(p, r)]:.1f}" + r"\%" for r in REGIONS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (ROOT / "paper" / "tables" / "t6_carbon.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote paper/tables/t6_carbon.tex")



def main():
    ap = argparse.ArgumentParser(prog="python -m flex.e5_carbon")
    ap.add_argument("what", choices=["run", "table"])
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--cut", type=float, default=None)
    args = ap.parse_args()
    if args.what == "run":
        run(args)
    else:
        table(load_config())


if __name__ == "__main__":
    main()
