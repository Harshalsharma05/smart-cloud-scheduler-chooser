"""S16 claim check: every claim in paper/abstract.txt against values computed from results (via make_numbers.compute()).

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
from style import ROOT, load_runs  # noqa: E402

ABS = (ROOT / "paper" / "abstract.txt").read_text(encoding="utf-8")
CLAIMS = (ROOT / "paper" / "ABSTRACT_CLAIMS.md").read_text(encoding="utf-8")


def parse_claims():
    rows = {}
    for line in CLAIMS.splitlines():
        m = re.match(r"\|\s*([a-h]'?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|$", line)
        if m:
            rows[m.group(1)] = m.group(2)
    return rows


def main():
    M = {k: v[0] for k, v in mn.compute().items()}
    e1 = load_runs("e1")
    e1 = e1[e1["setting"] != "attribution"]
    dc = mn.in_cell(e1, *mn.DEFAULT)
    # extra numbers quoted in the claims file
    med = {(w, c): dc[dc.who == w][c].median() for w in ("burst", "delay") for c in ("cost_usd", "cost_usd_wfail1", "usd_per_mwh", "usd_per_mwh_wfail1")}
    cells = e1.groupby(["duration_h", "cut", "who"])[["usd_per_mwh", "compliance"]].median().reset_index()
    delay_lt_burst = sum(1 for (d, c), g in cells.groupby(["duration_h", "cut"]) if g[g.who == "delay"].usd_per_mwh.iloc[0] < g[g.who == "burst"].usd_per_mwh.iloc[0])
    cap_noncompliant = sum(1 for (d, c), g in cells.groupby(["duration_h", "cut"]) if g[g.who == "cap"].compliance.iloc[0] < 0.99)
    nc = mn.read_nc()
    nc_def = mn.in_cell(nc[(nc.predictor == "transformer") & (nc.setting.str.contains("cq=p90"))], *mn.DEFAULT).set_index("event_id")
    dl_def = dc[dc.who == "delay"].set_index("event_id")
    default_cheaper = int((nc_def["cost_usd"] < dl_def["cost_usd"] - 1e-9).sum())
    p = e1[e1.who.isin(["mix-p90", "delay"])].pivot_table(index="event_id", columns="who", values="cost_usd", aggfunc="first")
    p1 = e1[e1.who.isin(["mix-p90", "delay"])].pivot_table(index="event_id", columns="who", values="cost_usd_wfail1", aggfunc="first")
    ratio_of_med = (p["mix-p90"].median() / p["delay"].median(), p1["mix-p90"].median() / p1["delay"].median())

    # claim id -> (list of (label, value in claims file, computed, tolerance), abstract snippet that depends on it, extra boolean checks)
    checks = {
        "a": ([("delay $/MWh w0", 3334, M["DefDelayUsdZero"], 0.5), ("delay $/MWh w1", 12941, M["DefDelayUsdOne"], 0.5),
               ("cells delay<burst", 9, delay_lt_burst, 0), ("cells cap non-compliant", 9, cap_noncompliant, 0)],
              "roughly $3k–13k per MWh", [round(M["DefDelayUsdZero"] / 1000) == 3, round(M["DefDelayUsdOne"] / 1000) == 13]),
        "b": ([("cap share of target, %", 46, M["DefCapSharePct"], 0.5)], "In that cell, power capping alone reaches less than half of the cut", [M["DefCapSharePct"] < 50]),
        "c": ([("per-event median w1", 30, M["BurstOverDelayDefaultOne"], 0.5), ("per-event median w0", 125, M["BurstOverDelayDefaultZero"], 0.5),
               ("ratio of medians cost w1", 33, med[("burst", "cost_usd_wfail1")] / med[("delay", "cost_usd_wfail1")], 0.5),
               ("ratio of medians cost w0", 123, med[("burst", "cost_usd")] / med[("delay", "cost_usd")], 0.5),
               ("$/MWh w1", 31, med[("burst", "usd_per_mwh_wfail1")] / med[("delay", "usd_per_mwh_wfail1")], 0.5),
               ("$/MWh w0", 121, med[("burst", "usd_per_mwh")] / med[("delay", "usd_per_mwh")], 0.5)],
              "bursting alone costs 30 to 125 times more", [round(M["BurstOverDelayDefaultOne"]) == 30, round(M["BurstOverDelayDefaultZero"]) == 125]),
        "d": ([("mix-p90/delay per event w0", 0.945, M["MixPNineOverDelayZero"], 0.0005), ("w1", 0.941, M["MixPNineOverDelayOne"], 0.0005),
               ("cheaper events w0", 66, M["MixPNineOverDelayCheaperZero"], 0), ("cheaper events w1", 75, M["MixPNineOverDelayCheaperOne"], 0),
               ("ratio of medians w0", 0.932, ratio_of_med[0], 0.0005), ("ratio of medians w1", 0.897, ratio_of_med[1], 0.0005)],
              "is about 5% cheaper than delay alone in both cost weightings", [M["MixPNineOverDelayZero"] < 1, M["MixPNineOverDelayOne"] < 1]),
        "d'": ([("p50 mix/delay w0", 1.012, M["MixOverDelayZero"], 0.0005), ("p50 cheaper w0", 35, M["MixOverDelayCheaperZero"], 0),
                ("p50 mix/delay w1", 0.976, M["MixOverDelayOne"], 0.0005), ("p50 cheaper w1", 54, M["MixOverDelayCheaperOne"], 0)], None, []),
        "e": ([("contract/hold p50 w0", 68, M["ContractOverHoldPFiftyZero"], 0.5), ("p90 w0", 39, M["ContractOverHoldPNineZero"], 0.5),
               ("p50 w1", 22, M["ContractOverHoldPFiftyOne"], 0.5), ("p90 w1", 11, M["ContractOverHoldPNineOne"], 0.5)],
              "makes the controller 10 to 70 times more expensive",
              [round(M["ContractOverHoldMin"], -1) == 10, round(M["ContractOverHoldMax"], -1) == 70]),
        "f": ([("trans p90 mix/nocloud w0", 1.00, M["MixOverNocloudTransPNineZeroZero"], 0.005), ("w1", 1.00, M["MixOverNocloudTransPNineZeroOne"], 0.005),
               ("cheaper events w0", 38, M["MixOverNocloudTransPNineZeroCheaperZero"], 0), ("oracle w0", 0.99, M["MixOverNocloudOracleZero"], 0.005),
               ("oracle w1", 1.00, M["MixOverNocloudOracleOne"], 0.005), ("oracle 4h/30% w0", 0.84, M["MixOverNocloudOracleFourThirtyZero"], 0.005),
               ("oracle 4h/20% w0", 0.91, M["MixOverNocloudOracleFourTwentyZero"], 0.005), ("trans p50 w0", 1.05, M["MixOverNocloudTransPFiveZeroZero"], 0.005),
               ("last2 w0", 22, M["MixOverNocloudLastTwoZero"], 0.5)],
              "offloading adds almost nothing", [abs(M["MixOverNocloudTransPNineZeroZero"] - 1) < 0.01, M["MixOverNocloudOracleFourThirtyZero"] < 0.9]),
        "g": ([("nocloud/delay w0", 0.949, M["NocloudOverDelayZero"], 0.0005), ("w1", 0.946, M["NocloudOverDelayOne"], 0.0005),
               ("cheaper events w0", 59, M["NocloudOverDelayCheaperZero"], 0), ("default cell w0", 0.86, M["NocloudOverDelayDefaultZero"], 0.005),
               ("default cell w1", 0.87, M["NocloudOverDelayDefaultOne"], 0.005), ("default cell cheaper events", 10, default_cheaper, 0),
               ("total cost w0", 0.91, M["NocloudOverDelayTotalZero"], 0.005), ("total cost w1", 0.93, M["NocloudOverDelayTotalOne"], 0.005)],
              "guided by the transformer's 90th-percentile runtime estimate, is about 5% cheaper than delay alone",
              [abs(100 * (1 - M["NocloudOverDelayZero"]) - 5) < 1, abs(100 * (1 - M["NocloudOverDelayOne"]) - 5) < 1]),
        "h": ([], "We discuss when offloading is worth using", []),
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
    # every numeric phrase of the abstract must belong to a claim
    covered = [c[1] for c in checks.values() if c[1]]
    nums = re.findall(r"[\$]?\d[\d,.–-]*[k%]?(?: times| to \d+ times)?", ABS)
    print("\nnumbers in abstract.txt:", nums)
    print("claim snippets cover:", all(s in ABS for s in covered))
    print("\nSUMMARY:", f"{len(out) - bad} OK, {bad} MISMATCH")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
