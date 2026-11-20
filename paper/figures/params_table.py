"""Table T1: every parameter of backend/configs/flex.yaml that the experiments use, with value, unit and source.

Run from the repo root:  python paper/figures/params_table.py   ->  paper/tables/params.tex
Values come from flex.yaml (never typed here). The source column is "assumption" for every line whose yaml comment
starts with ASSUMPTION; otherwise a short name from SHORT below (no URLs in the table; footnotes come later).
"""
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "backend" / "configs" / "flex.yaml"
OUT = ROOT / "paper" / "tables" / "params.tex"

# (label, section, key, unit, formatter name). Row order = table order. PAPER_ROWS: the rows printed in the paper table (W6).
ROWS = [
    ("GPU idle power", "power", "p_idle_w", "W", "g"),
    ("GPU max power", "power", "p_max_w", "W", "g"),
    ("Host power/GPU, idle", "power", "h_idle_w", "W", "g"),
    ("Host power/GPU, if busy", "power", "h_busy_w", "W", "g"),
    ("On-prem PUE", "power", "pue", "--", "g"),
    ("Cloud PUE", "power", "cloud_pue", "--", "g"),
    ("Cap curve (r, speed)", "cap_curve", "optimistic", "--", "curve"),
    ("Cap curve (r, speed), pessimistic", "cap_curve", "pessimistic", "--", "curve"),
    ("Cap levels (x max)", None, "cap_levels", "--", "list"),
    ("Cloud price per GPU-h", "cloud", "price_per_gpu_h", "USD", "g"),
    ("Cloud boot delay", "cloud", "boot_delay_s", "s", "g"),
    ("Migration overhead", "cloud", "migrate_s", "s", "g"),
    ("Resume overhead", "cloud", "restore_s", "s", "g"),
    ("Minimum billed time", "cloud", "min_billed_s", "s", "g"),
    ("Cloud GPU limit", "cloud", "max_cloud_gpus", "GPUs", "g"),
    ("Delay cost / cloud price", "cost", "c_delay_factor", "--", "g"),
    ("Missed-deadline factor", "cost", "m", "--", "g"),
    ("Delay weight of Failed/Killed", "cost", "w_fail", "--", "g"),
    ("Jobs with a deadline", "deadlines", "fraction", "--", "g"),
    ("Deadline slack (min, max)", "deadlines", "slack_min", "x run time", "slack"),
    ("Event duration", "grid", "durations_h", "h", "list"),
    ("Event cut", "grid", "cuts", "fraction", "list"),
    ("Events per duration and cut", "grid", "events_per_cell", "--", "g"),
    ("Look-ahead after an event", "grid", "h_after_h", "h", "g"),
    ("Trace warm-up before the test window", "philly", "warmup_days", "days", "g"),
    ("Job GPU utilisation (until measured)", "philly", "util_default", "--", "g"),
]

PAPER_ROWS = ['GPU idle power', 'GPU max power', 'Host power/GPU, idle', 'Host power/GPU, if busy', 'On-prem PUE', 'Cap curve (r, speed)', 'Cap levels (x max)', 'Cloud price per GPU-h', 'Cloud boot delay', 'Migration overhead', 'Resume overhead', 'Cloud GPU limit', 'Missed-deadline factor']

# Short source names for rows that have a verified source (everything else must be an ASSUMPTION or a design choice).
SHORT = {
    ("power", "p_idle_w"): r"\cite{you23}",
    ("power", "p_max_w"): "P100 datasheet",
    ("power", "pue"): "Uptime Inst.\\ 2023",
    ("power", "cloud_pue"): "Google 2025",
    ("cloud", "price_per_gpu_h"): "AWS on-demand",
    ("cloud", "boot_delay_s"): r"\cite{mao12}",
}
RAW_SOURCES = {r"\cite{you23}", r"\cite{mao12}", "Uptime Inst.\\ 2023"}   # already LaTeX, not escaped

# W11 (step 36): Acme (Seren) column. Values come from the flex.yaml section acme: (power, cloud); other rows are shared.
ACME_SHORT = {
    ("power", "p_idle_w"): r"\cite{hu24}",
    ("power", "p_max_w"): "datasheet",
    ("power", "h_idle_w"): r"\cite{hu24}",
    ("power", "h_busy_w"): r"\cite{hu24}",
    ("cloud", "price_per_gpu_h"): "AWS on-demand",
}
PHILLY_SHORT_TWO = {("power", "p_max_w"): "datasheet"}


def read_comments(text):
    """{(section, key): comment text} for the top-level sections and their two-space-indented keys."""
    out, section = {}, None
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z_]\w*):\s*(#\s*(.*))?$", line)
        if m:
            section = m.group(1)
            out[(section, None)] = m.group(3) or ""
            continue
        m = re.match(r"^  ([A-Za-z_]\w*):[^#]*(#\s*(.*))?$", line)
        if m and section:
            out[(section, m.group(1))] = m.group(3) or ""
        m = re.match(r"^([A-Za-z_]\w*):[^#]*#\s*(.*)$", line)     # top-level scalar/list with a comment (cap_levels)
        if m:
            out[(None, m.group(1))] = m.group(2)
    return out


def tex(s):
    return str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def fmt(value, kind, cfg_section):
    if kind == "g":
        return f"{value:g}"
    if kind == "list":
        return ", ".join(f"{v:g}" for v in value)
    if kind == "curve":
        return " ".join(f"({r:g},{s:g})" for r, s in value)
    if kind == "slack":
        return f"{cfg_section['slack_min']:g}, {cfg_section['slack_max']:g}"
    raise ValueError(kind)


def source_of(comment, section, key):
    if comment.startswith("ASSUMPTION"):
        return "assumption"
    if "SIM_PLAN" in comment:
        return "design choice"
    if (section, key) in SHORT:
        return SHORT[(section, key)]
    raise SystemExit(f"no short source name for {section}.{key}: {comment[:60]!r}; add it to SHORT")


def main():
    text = CONFIG.read_text(encoding="utf-8")
    cfg = yaml.safe_load(text)
    comments = read_comments(text)
    lines, other = [], []
    acme = cfg.get("acme", {})
    for label, section, key, unit, kind in ROWS:
        node = cfg[key] if section is None else cfg[section][key]
        sec = None if section is None else cfg[section]
        value = fmt(node, kind, sec)
        comment = comments.get((section, key))
        if comment is None:
            raise SystemExit(f"no comment for {section}.{key} in flex.yaml")
        if label not in PAPER_ROWS:
            other.append(label)
            continue
        src = source_of(comment, section, key)
        src = r"\textit{assumption}" if src == "assumption" else (src if src in RAW_SOURCES else tex(src))
        a_val = acme.get(section, {}).get(key) if section in ("power", "cloud") else None
        if a_val is None or a_val == node:                     # shared by both traces: one value over both columns
            lines.append(f"{tex(label)} & \\multicolumn{{2}}{{>{{\\raggedright\\arraybackslash}}p{{1.75cm}}}}{{{tex(value)}}} & {tex(unit)} & {src} \\\\")
        else:
            a_src = ACME_SHORT[(section, key)]
            p_src = PHILLY_SHORT_TWO.get((section, key), src)
            both = p_src if p_src == a_src else f"{p_src} / {a_src}"
            lines.append(f"{tex(label)} & {tex(value)} & {tex(fmt(a_val, kind, None))} & {tex(unit)} & {both} \\\\")
    if acme:
        lines.append(f"GPU utilisation & per job & {acme['util']:g} & -- & trace / calibrated \\\\")
    assert len(lines) == len(PAPER_ROWS) + bool(acme), "a PAPER_ROWS label does not match ROWS"
    body = "\n".join(lines)
    out = (
        "% Generated by paper/figures/params_table.py from backend/configs/flex.yaml. Do not edit.\n"
        f"% Not printed (given in the text or in the artifact): {'; '.join(other)}.\n"
        "\\begin{table}[t]\n\\centering\n\\caption{Main simulator parameters (one value: both traces). Source \\emph{x / y}: "
        "Philly / Acme; \\textit{assumption}: no published source.}\n\\label{tab:params}\n"
        "\\footnotesize\n\\setlength{\\tabcolsep}{2.5pt}\n"
        "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{2.55cm}>{\\raggedright\\arraybackslash}p{0.8cm}"
        ">{\\raggedright\\arraybackslash}p{0.8cm}>{\\raggedright\\arraybackslash}p{1.15cm}>{\\raggedright\\arraybackslash}p{2.0cm}@{}}\n\\toprule\n"
        "Parameter & Philly & Acme & Unit & Source \\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(out, encoding="utf-8")
    n_assumed = sum("assumption" in l for l in lines)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(lines)} rows, {n_assumed} marked assumption")


if __name__ == "__main__":
    sys.exit(main())
