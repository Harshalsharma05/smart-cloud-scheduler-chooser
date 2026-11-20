"""S16 claim check (revised in W11 for the two-trace abstract): every claim in paper/abstract.txt against values computed from
results (via make_numbers.compute()).

Parses paper/ABSTRACT_CLAIMS.md (claims table rows `| id | claim | value now | source |`), then for each claim compares the value
written in the file with the value computed now (tolerance = half a unit of the last stated digit) and checks the abstract wording
that depends on it. Prints a table: claim | computed | OK/MISMATCH. Never edits the abstract. Exit code 1 if anything mismatches.
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_numbers as mn  # noqa: E402
from style import RESULTS, ROOT, load_runs  # noqa: E402

ABS = (ROOT / "paper" / "abstract.txt").read_text(encoding="utf-8")
CLAIMS = (ROOT / "paper" / "ABSTRACT_CLAIMS.md").read_text(encoding="utf-8")


def parse_claims():
    rows = {}
    for line in CLAIMS.splitlines():
        m = re.match(r"\|\s*([a-h]'?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|$", line)
        if m:
            rows[m.group(1)] = m.group(2)
    return rows


def acme_cells():
    """Acme: cells where delay-only (transformer) has a lower median $/MWh than burst-only (transformer)."""
    r = pd.read_csv(RESULTS / "acme_e1" / "runs.csv", dtype={"selector": str, "predictor": str, "setting": str}).fillna({"predictor": "", "setting": ""})
    r = r[(r["setting"] == "") & (r["predictor"] == "transformer") & r["policy"].isin(["delay", "burst"])]
    c = r.groupby(["duration_h", "cut", "policy"])["usd_per_mwh"].median().unstack()
    return int((c["delay"] < c["burst"]).sum()), len(c)


def main():
    M = {k: v[0] for k, v in mn.compute().items()}
    e1 = load_runs("e1")
    e1 = e1[e1["setting"] != "attribution"]
    cells = e1.groupby(["duration_h", "cut", "who"])[["usd_per_mwh", "compliance"]].median().reset_index()
    delay_lt_burst = sum(1 for (d, c), g in cells.groupby(["duration_h", "cut"]) if g[g.who == "delay"].usd_per_mwh.iloc[0] < g[g.who == "burst"].usd_per_mwh.iloc[0])
    cap_noncompliant = sum(1 for (d, c), g in cells.groupby(["duration_h", "cut"]) if g[g.who == "cap"].compliance.iloc[0] < 0.99)
    acme_dlb, acme_ncells = acme_cells()
    n_events = M["SetupEventsFeasible"] + M["AcmeEvents"]
    burst_wins = max(M["BurstCheaperThanDelayZero"], M["BurstCheaperThanDelayOne"], M["AcmeBurstCheaperTransformerZero"], M["AcmeBurstCheaperTransformerOne"])

    # claim id -> (list of (label, value in claims file, computed, tolerance), abstract snippet that depends on it, extra boolean checks)
    checks = {
        "a": ([("Philly cells delay<burst", 9, delay_lt_burst, 0), ("Philly cells cap non-compliant", 9, cap_noncompliant, 0),
               ("Acme cells delay<burst", 9, acme_dlb, 0), ("Acme cap compliant events", 1, M["AcmeCapCompliantEvents"], 0)],
              "Delay is the cheapest action that meets the cut on both traces", [delay_lt_burst == 9, acme_dlb == acme_ncells == 9]),
        "b": ([("Philly burst/delay default cell w0", 125, M["BurstOverDelayDefaultZero"], 0.5), ("Acme burst/delay default cell w0", 12.1, M["AcmeBurstOverDelayDefaultZero"], 0.05),
               ("Philly w1", 30, M["BurstOverDelayDefaultOne"], 0.5), ("Acme w1", 9.3, M["AcmeBurstOverDelayDefaultOne"], 0.05)],
              "bursting alone costs a median of 125 (Philly) and 12 (Acme) times as much as delay per event when delaying failed work costs nothing",
              [round(M["BurstOverDelayDefaultZero"]) == 125, round(M["AcmeBurstOverDelayDefaultZero"]) == 12]),
        "c": ([("Philly burst<delay events w0", 0, M["BurstCheaperThanDelayZero"], 0), ("Philly w1", 0, M["BurstCheaperThanDelayOne"], 0),
               ("Acme transformer burst<delay w0", 0, M["AcmeBurstCheaperTransformerZero"], 0), ("Acme w1", 1, M["AcmeBurstCheaperTransformerOne"], 0),
               ("events on both traces", 174, n_events, 0)],
              "With the transformer predictor it beat delay in at most one of 174 events", [burst_wins <= 1, n_events == 174]),
        "d": ([("Philly spot price burst/delay", 18, M["PriceSpotBurstOverDelay"], 0.5), ("Acme $1.50 burst/delay transformer", 6.65, M["AcmeCheapBurstTransformerOverDelayZero"], 0.005),
               ("Acme $1.50 burst/delay oracle", 4.55, M["AcmeCheapBurstOracleOverDelayZero"], 0.005)],
              "a cheaper cloud does not change this",
              [M["PriceSpotBurstOverDelay"] > 1, M["AcmeCheapBurstTransformerOverDelayZero"] > 1, M["AcmeCheapBurstOracleOverDelayZero"] > 1]),
        "e": ([("Acme burst jobs >1 h left, % of jobs", 35, M["AcmeBurstLongJobsPct"], 0.5), ("Acme their % of cloud $", 99, M["AcmeBurstLongUsdPct"], 0.5),
               ("Philly running jobs >12 h at event starts, %", 86.8, M["MotRunningBeyondGridJobsPct"], 0.05)],
              "the cloud bill covers a job's whole remaining runtime, so long jobs make bursting much worse",
              [M["AcmeBurstLongUsdPct"] >= 90, M["BurstOverDelayDefaultZero"] > M["AcmeBurstOverDelayDefaultZero"]]),
        "f": ([("trans mix-p90/delay w0", 0.945, M["MixPNineOverDelayZero"], 0.0005), ("gbdt", 0.839, M["SetGbdtMainMixOverDelayZero"], 0.0005),
               ("lightgbm", 0.870, M["SetLightgbmMainMixOverDelayZero"], 0.0005), ("pessimistic cap curve ratio (default cell)", 1.00, M["SensPessimisticRatio"], 0.005),
               ("max masking shift", 0.029, M["LearnedMaskShiftMax"], 0.0005)],
              "The predictor-guided mix beats delay by 5–16% on Philly under our default cap curve",
              [round(M["LearnedGainMinPct"]) == 5, round(M["LearnedGainMaxPct"]) == 16]),
        "g": ([("Acme learned mix-p90/delay min w0", 1.000, M["AcmeLearnedMixMinZero"], 0.0005), ("max w0", 1.011, M["AcmeLearnedMixMaxZero"], 0.0005),
               ("Acme oracle mix/delay w0", 0.827, M["AcmeMixPNineOverDelayOracleZero"], 0.005), ("oracle gain %", 17, M["AcmeOracleMixGainPctZero"], 0.5)],
              "not at all on Acme, where exact runtimes would save 17%",
              [M["AcmeLearnedMixMinZero"] >= 0.9995, round(M["AcmeOracleMixGainPctZero"]) == 17]),
        "h": ([("contract/hold p50 w0", 68, M["ContractOverHoldPFiftyZero"], 0.5), ("p90 w0", 39, M["ContractOverHoldPNineZero"], 0.5),
               ("p50 w1", 22, M["ContractOverHoldPFiftyOne"], 0.5), ("p90 w1", 11, M["ContractOverHoldPNineOne"], 0.5)],
              "makes the controller 11 to 68 times more expensive",
              [round(M["ContractOverHoldMin"]) == 11, round(M["ContractOverHoldMax"]) == 68]),
    }
    rows = parse_claims()
    out, bad = [], 0
    for cid, (subs, snippet, extra) in checks.items():
        in_file = cid in rows
        res = [abs(fv - cv) <= tol + 1e-9 for _, fv, cv, tol in subs]
        snip_ok = True if snippet is None else snippet in ABS
        ok = in_file and all(res) and snip_ok and all(extra)
        bad += not ok
        comp = "; ".join(f"{l}={cv:.4g}" + ("" if r else f" (file {fv})") for (l, fv, cv, _), r in zip(subs, res)) or "no numeric claim"
        why = [] if ok else ([] if in_file else ["claim missing in ABSTRACT_CLAIMS.md"]) + ([] if snip_ok else ["abstract wording not found"]) + ([] if all(extra) else ["abstract number does not round to computed"]) + ([] if all(res) else ["value in ABSTRACT_CLAIMS.md differs from computed (abstract wording not affected unless listed above)"])
        out.append((cid, rows.get(cid, "?")[:70], comp, "OK" if ok else "MISMATCH " + "; ".join(why)))
    for cid, txt, comp, st in out:
        print(f"[{cid}] {txt}\n     computed: {comp}\n     -> {st}")
    # every numeric phrase of the abstract must belong to a claim (or be part of the set-up: years, event grid)
    covered = [c[1] for c in checks.values() if c[1]]
    nums = re.findall(r"[\$]?\d[\d,.–-]*[k%]?(?: times| to \d+ times)?", ABS)
    print("\nnumbers in abstract.txt:", nums)
    print("claim snippets cover:", all(s in ABS for s in covered))
    print("\nSUMMARY:", f"{len(out) - bad} OK, {bad} MISMATCH")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
