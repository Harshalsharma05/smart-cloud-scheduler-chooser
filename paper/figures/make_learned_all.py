"""W8b: the three learned predictors (transformer, gbdt, lightgbm), main runs and in-sample masked runs (jobs submitted before
2017-11-01 get uninformed values), all feasible E1 events, hold rule, p90 cloud bill.
Per predictor x variant: per-event median ratio mix-p90 / delay-only and mix-p90 / mix-nocloud-p90 (w_f 0 and 1), events where
the mix is cheaper than delay-only, total-cost ratio. Sources: results/e1 (+ e1nc), results/robust_insample (transformer masked),
results/e1_learned (gbdt/lightgbm delay + mix), results/e1_learned_nc (gbdt/lightgbm nocloud, and the masked runs).
Writes results/e1_learned_nc/summary.csv.   python paper/figures/make_learned_all.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_robust_missing import COST, read, stats  # noqa: E402
from style import RESULTS  # noqa: E402

OUT = RESULTS / "e1_learned_nc"


def by(d, policy, pred, setting):
    s = d[(d["policy"] == policy) & (d["predictor"] == pred) & (d["setting"] == setting)]
    assert len(s) and s["event_id"].is_unique, (policy, pred, setting, len(s))
    return s.set_index("event_id")


def main():
    e1 = read(RESULTS / "e1" / "runs.csv")
    nc = read(RESULTS / "e1nc" / "runs_d*.csv")
    ins = read(RESULTS / "robust_insample" / "runs_ins_*.csv")
    lr = read(RESULTS / "e1_learned" / "runs_*.csv")
    lnc = read(OUT / "runs_main_*.csv")
    lmask = read(OUT / "runs_mask_*.csv")
    m = "missing=oracle;uninformed<2017-11-01"
    u = "uninformed<2017-11-01"
    sets = {
        ("transformer", "main"): (by(e1, "delay", "transformer", ""), by(e1, "mix", "transformer", "cq=p90"), by(nc, "mix", "transformer", "nocloud;cq=p90")),
        ("transformer", "masked"): (by(ins, "delay", "transformer", m), by(ins, "mix", "transformer", m + ";cq=p90"), by(ins, "mix", "transformer", m + ";nocloud;cq=p90")),
    }
    for p in ("gbdt", "lightgbm"):
        sets[(p, "main")] = (by(lr, "delay", p, ""), by(lr, "mix", p, "cq=p90"), by(lnc, "mix", p, "nocloud;cq=p90"))
        sets[(p, "masked")] = (by(lmask, "delay", p, u), by(lmask, "mix", p, u + ";cq=p90"), by(lmask, "mix", p, u + ";nocloud;cq=p90"))
    rows = []
    for (p, v), (dl, mx, ncl) in sets.items():
        r = dict(predictor=p, variant=v)
        for w, col in COST.items():
            a = stats(mx, dl, col)
            r.update({f"mix_delay_{w}": a["median_ratio"], f"cheaper_{w}": a["cheaper"], f"total_{w}": a["total_ratio"], f"n_{w}": a["n"],
                      f"mix_nocloud_{w}": stats(mx, ncl, col)["median_ratio"], f"nocloud_delay_{w}": stats(ncl, dl, col)["median_ratio"]})
        rows.append(r)
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(s[["predictor", "variant", "mix_delay_wfail0", "mix_delay_wfail1", "mix_nocloud_wfail0", "mix_nocloud_wfail1",
             "cheaper_wfail0", "cheaper_wfail1", "total_wfail0", "total_wfail1", "n_wfail0"]].round(3).to_string(index=False))
    w = s.set_index(["predictor", "variant"])
    shift = {p: max(abs(w.loc[(p, "masked"), c] - w.loc[(p, "main"), c]) for c in ("mix_delay_wfail0", "mix_delay_wfail1"))
             for p in ("transformer", "gbdt", "lightgbm")}
    print("masking shift of mix/delay (max over w_f):", {k: round(v, 3) for k, v in shift.items()}, "-> use masked values:", any(v > 0.03 for v in shift.values()))
    mn = s[["mix_nocloud_wfail0", "mix_nocloud_wfail1"]]
    print("mix/nocloud within 0.99-1.01 for all:", bool(((mn >= 0.99) & (mn <= 1.01)).all().all()))


if __name__ == "__main__":
    main()
