"""T2: default event (2 h, 20%), median over its 10 events, per policy; cost under w_fail = 0 (config) and w_fail = 1.
Writes paper/tables/t2.tex and prints the table. A policy with median compliance < 99% shows "shed x% of target"
instead of $/MWh. Run from the repo root."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import ROOT, load_runs  # noqa: E402

runs = load_runs("e1")
runs = runs[(runs["duration_h"] == 2) & (runs["cut"].sub(0.2).abs() < 1e-9)]
order = ["delay", "cap", "burst", "mix", "mix-p90", "delay (last2)", "burst (last2)", "mix (last2)", "delay+oracle", "burst+oracle", "mix+oracle"]
rows = []
for w in order:
    s = runs[runs["who"] == w]
    if s.empty:
        continue
    ok = s["compliance"].median() >= 0.99
    show = (lambda c: f"{s[c].median():,.0f}") if ok else (lambda c: f"shed {s['shed_pct'].median():.0f}\\% of target")
    rows.append(dict(policy=w, n=len(s), shed_mwh=s["shed_mwh"].median(), compliance=s["compliance"].median(),
                     usd_mwh_w0=show("usd_per_mwh"), usd_mwh_w1=show("usd_per_mwh_wfail1"), cloud_usd=s["cloud_usd"].median(),
                     dgpuh_good=s["dgpuh_good"].median(), missed=s["missed_deadlines"].median(), rebound_mwh=s["rebound_mwh"].median(),
                     compliant_events=int(s["compliance_ok"].sum())))
df = pd.DataFrame(rows)
print(df.round(3).to_string(index=False))
lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r"Policy & Shed (MWh) & Compl. & \$/MWh ($w{=}0$) & \$/MWh ($w{=}1$) & Cloud \$ & $\Delta$GPU-h good & Missed & Rebound (MWh) \\", r"\midrule"]
for r in df.itertuples():
    lines.append(f"{r.policy.replace('+', '+')} & {r.shed_mwh:.2f} & {r.compliance:.2f} & {r.usd_mwh_w0} & {r.usd_mwh_w1} & "
                 f"{r.cloud_usd:,.0f} & {r.dgpuh_good:,.0f} & {r.missed:.0f} & {r.rebound_mwh:.2f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
out = ROOT / "paper" / "tables" / "t2.tex"
out.write_text("\n".join(lines) + "\n")
print("wrote", out)
