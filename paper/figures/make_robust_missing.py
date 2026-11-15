"""W7 robustness: jobs without prediction rows use their true values (oracle fallback, all main results) or fixed
"uninformed" values (30-day remaining runtime, p_fail = p_cancel = 0). Transformer, hold_beyond_grid, all feasible E1 events.

Reads results/e1/runs.csv (delay, mix cq=p90), results/e1nc/runs_d*.csv (mix-nocloud cq=p90) for the oracle fallback and
results/robust_missing/runs_uninf_*.csv for the uninformed fallback. Checks that results/robust_missing/runs_check_oracle_*.csv
reproduces the stored E1 rows. Writes results/robust_missing/summary.csv.   python paper/figures/make_robust_missing.py
"""
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import RESULTS  # noqa: E402

OUT = RESULTS / "robust_missing"
COST = {"wfail0": "cost_usd", "wfail1": "cost_usd_wfail1"}


def read(pattern):
    d = pd.concat([pd.read_csv(p, dtype={"selector": str, "predictor": str, "setting": str}) for p in sorted(glob.glob(str(pattern)))],
                  ignore_index=True).fillna({"predictor": "", "setting": ""})
    return d[d["feasible"].astype(str) != "False"]


def pick(d, policy, setting_has, setting_not=()):
    s = d[(d["policy"] == policy) & (d["predictor"] == "transformer")]
    s = s[s["setting"].apply(lambda x: all(t in x for t in setting_has) and not any(t in x for t in setting_not))]
    assert s["event_id"].is_unique, (policy, setting_has)
    return s.set_index("event_id")


def stats(mix, delay, col):
    x = pd.concat([mix[col].rename("m"), delay[col].rename("d")], axis=1).dropna()
    r = x["m"] / x["d"].where(x["d"] > 0)
    return dict(n=len(x), median_ratio=float(r.median()), cheaper=int((x["m"] < x["d"] - 1e-9).sum()),
                cheaper_share=float((x["m"] < x["d"] - 1e-9).mean()), total_ratio=float(x["m"].sum() / x["d"].sum()))


def main():
    e1 = read(RESULTS / "e1" / "runs.csv")
    e1 = e1[e1["setting"] != "attribution"]
    nc = read(RESULTS / "e1nc" / "runs_d*.csv")
    un = read(OUT / "runs_uninf_*.csv")
    variants = {
        "oracle": dict(delay=pick(e1, "delay", []), mix=pick(e1, "mix", ["cq=p90"]), nocloud=pick(nc, "mix", ["nocloud", "cq=p90"])),
        "uninformed": dict(delay=pick(un, "delay", ["missing=uninformed"], ["cq="]), mix=pick(un, "mix", ["cq=p90"], ["nocloud"]),
                           nocloud=pick(un, "mix", ["nocloud", "cq=p90"])),
    }
    # the check run (oracle fallback through the new code path) must reproduce the stored E1 / E1nc rows
    chk = read(OUT / "runs_check_oracle_*.csv")
    for pol, has, nothave, ref in (("delay", [], ["cq="], variants["oracle"]["delay"]), ("mix", ["cq=p90"], ["nocloud"], variants["oracle"]["mix"]),
                                   ("mix", ["nocloud", "cq=p90"], [], variants["oracle"]["nocloud"])):
        c = pick(chk, pol, has, nothave)
        diff = (c["cost_usd"] - ref["cost_usd"].reindex(c.index)).abs().max()
        print(f"check {pol} {has}: {len(c)} events, max |cost - stored| = {diff:.3g}")
        assert diff < 1e-6, "the oracle-fallback check run does not reproduce the stored results"
    rows = []
    for v, d in variants.items():
        for w, col in COST.items():
            for who in ("mix", "nocloud"):
                rows.append(dict(fallback=v, wfail=w, policy=who, **stats(d[who], d["delay"], col)))
            rows.append(dict(fallback=v, wfail=w, policy="mix_over_nocloud", **stats(d["mix"], d["nocloud"], col)))
    for w, col in COST.items():                                       # how much the fallback changes delay-only itself
        rows.append(dict(fallback="uninformed_over_oracle", wfail=w, policy="delay",
                         **stats(variants["uninformed"]["delay"], variants["oracle"]["delay"], col)))
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.round(3).to_string(index=False))
    u = s[(s.fallback == "uninformed") & (s.policy == "mix")].set_index("wfail")["median_ratio"]
    print("STOP CONDITION (mix-p90 / delay per-event median >= 0.98 at either w_fail under uninformed):", bool((u >= 0.98).any()), u.round(4).to_dict())


if __name__ == "__main__":
    main()
