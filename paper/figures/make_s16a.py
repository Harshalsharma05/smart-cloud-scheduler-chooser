"""S16a report numbers from results/e1/runs.csv (E1 under hold_beyond_grid): winner per cell, mix vs delay-only, attribution control.

Infeasible events (P* below the all-idle floor) are dropped by load_runs. A policy is compliant in a cell when its median
compliance is >= 0.99 (as in F6); winners are the lowest median $/MWh among compliant policies. Writes results/e1/s16a_*.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import RESULTS, load_runs  # noqa: E402

C_DELAY = 3.06          # flex.yaml cloud price x c_delay_factor (w_fail = 0: delay $ = C_DELAY x dgpuh_good)


def winners(df, who_set, col):
    rows = []
    for (d, c), cell in df.groupby(["duration_h", "cut"]):
        best = None
        for w in who_set:
            s = cell[cell["who"] == w]
            if s.empty or s["compliance"].median() < 0.99 or s[col].dropna().empty:
                continue
            v = s[col].median()
            if best is None or v < best[1]:
                best = (w, v)
        rows.append(dict(duration_h=d, cut=c, winner=best[0] if best else "none", usd_per_mwh=best[1] if best else np.nan))
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)
    df = load_runs("e1")
    raw = pd.read_csv(RESULTS / "e1" / "runs.csv")
    inf = raw[raw["feasible"].astype(str) == "False"].groupby(["duration_h", "cut"])["event_id"].nunique()
    print("infeasible events per cell:", {k: int(v) for k, v in inf.items()}, "| total", int(inf.sum()), "of", raw["event_id"].nunique())
    main_df = df[df["setting"] != "attribution"]
    sets = {"transformer": ["delay", "cap", "burst", "mix", "mix-p90"], "last2": ["delay (last2)", "cap", "burst (last2)", "mix (last2)"],
            "oracle": ["delay+oracle", "cap", "burst+oracle", "mix+oracle"]}
    for p, ws in sets.items():
        for col in ("usd_per_mwh", "usd_per_mwh_wfail1"):
            w = winners(main_df, ws, col)
            print(f"\nwinner per cell, predictor {p}, {col}:")
            print(w.pivot(index="duration_h", columns="cut", values="winner").to_string())
            print("share of cells won:", w["winner"].value_counts().to_dict())
    # mix vs delay-only, same predictor, per event
    print("\nmix vs delay-only (same predictor), per event over all feasible events")
    rows = []
    for p, (m, dl) in {"transformer": ("mix", "delay"), "transformer p90": ("mix-p90", "delay"), "last2": ("mix (last2)", "delay (last2)"), "oracle": ("mix+oracle", "delay+oracle")}.items():
        for col in ("cost_usd", "cost_usd_wfail1"):
            a = main_df[main_df["who"] == m].set_index("event_id")[col]
            b = main_df[main_df["who"] == dl].set_index("event_id")[col]
            x = pd.concat([a.rename("mix"), b.rename("delay")], axis=1).dropna()
            rows.append(dict(predictor=p, cost=col, n=len(x), mix_cheaper=int((x["mix"] < x["delay"] - 1e-9).sum()), equal=int((abs(x["mix"] - x["delay"]) <= 1e-9).sum()),
                             median_ratio=float((x["mix"] / x["delay"].where(x["delay"] > 0)).median()), median_mix=x["mix"].median(), median_delay=x["delay"].median()))
    t = pd.DataFrame(rows)
    print(t.round(3).to_string(index=False))
    t.to_csv(RESULTS / "e1" / "s16a_mix_vs_delay.csv", index=False)
    # attribution control on the default cell
    d = df[(df["duration_h"] == 2) & (df["cut"].sub(0.2).abs() < 1e-9)].copy()
    d["delay_usd"] = C_DELAY * d["dgpuh_good"]
    d["deadline_usd"] = d["cost_usd"] - d["cloud_usd"] - d["delay_usd"]
    keep = ["delay", "mix", "mix-p90", "attr:delay/gbdt", "attr:delay/lightgbm", "delay (last2)", "mix (last2)", "delay+oracle", "mix+oracle"]
    g = d[d["who"].isin(keep)].groupby("who").agg(n=("event_id", "size"), cost=("cost_usd", "median"), cloud=("cloud_usd", "median"), delay=("delay_usd", "median"),
                                                  deadline=("deadline_usd", "median"), n_susp=("n_suspended", "median"), n_burst=("n_burst", "median"),
                                                  cap_level=("cap_level", "median"), missed=("missed_deadlines", "median"), shed=("shed_mwh", "median"),
                                                  usd_mwh=("usd_per_mwh", "median")).loc[lambda x: [k for k in keep if k in x.index]]
    print("\nattribution control, default cell (median per event, w_fail = 0):")
    print(g.round(1).to_string())
    g.to_csv(RESULTS / "e1" / "s16a_attribution.csv")
    a = d[d["who"] == "mix"].set_index("event_id")
    b = d[d["who"] == "delay"].set_index("event_id")
    x = pd.DataFrame({"d_cost": a["cost_usd"] - b["cost_usd"], "d_cloud": a["cloud_usd"] - b["cloud_usd"], "d_delay": a["delay_usd"] - b["delay_usd"],
                      "d_deadline": a["deadline_usd"] - b["deadline_usd"], "d_susp": a["n_suspended"] - b["n_suspended"], "n_burst_mix": a["n_burst"]})
    print("\nmix/transformer minus delay/transformer per event (negative = mix cheaper):")
    print(x.round(1).to_string())
    print("median:", x.median().round(1).to_dict())
    print("\npred_coverage by predictor, all feasible events (median, min):", main_df[main_df["pred_coverage"].notna()].groupby("predictor")["pred_coverage"].agg(["median", "min"]).round(3).to_dict("index"))
    print("runtime_s per run: median", round(raw["runtime_s"].median(), 2), "sum (h)", round(raw["runtime_s"].sum() / 3600, 2))


if __name__ == "__main__":
    main()
