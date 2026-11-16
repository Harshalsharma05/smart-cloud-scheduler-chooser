"""S16: one command that rebuilds every figure PDF/PNG and table .tex used in the paper from SAVED results, then numbers.tex.
No re-simulation. fig_e2.py and fig_e3.py are early synthetic-era scripts (superseded by F7/F7b and F8) and are not run. The two scripts that need a simulator run (F1 teaser: one event; F2 motivation: baseline run) are skipped
unless --with-sim is given. Prints the list of files written (modified or created during this run).
Run from anywhere: python paper/figures/make_all.py [--with-sim]
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "paper" / "figures"
PY = sys.executable
STEPS = [                                   # (label, command, cwd)
    ("Section 5 setup stats (trace loader, no simulation)", [PY, str(FIG / "methodology_stats.py")], ROOT),
    ("T1 parameters", [PY, str(FIG / "params_table.py")], ROOT),
    ("T2 default cell + F6", [PY, str(FIG / "make_t2.py")], ROOT),
    ("F6 E1 lines (w_fail 0)", [PY, str(FIG / "fig_e1.py")], ROOT),
    ("F6 E1 lines (w_fail 1)", [PY, str(FIG / "fig_e1.py"), "wfail1"], ROOT),
    ("F9 E4 user impact", [PY, str(FIG / "fig_e4.py")], ROOT),
    ("S13c sweeps, F7/F7b, T4", [PY, str(FIG / "make_s13.py")], ROOT),
    ("S13d T5 + nocloud figure", [PY, str(FIG / "make_s13d.py")], ROOT),
    ("S14 E3 merge", [PY, str(FIG / "make_s14.py")], ROOT),
    ("S14c F8 + T3 (final)", [PY, str(FIG / "make_s14c.py")], ROOT),
    ("S16a E1 tables (csv)", [PY, str(FIG / "make_s16a.py")], ROOT),
    ("W7 missing-prediction robustness (csv)", [PY, str(FIG / "make_robust_missing.py")], ROOT),
    ("W4 in-sample leakage check (csv)", [PY, str(FIG / "make_robust_insample.py")], ROOT),
    ("W8 gbdt / lightgbm p90 mix (csv)", [PY, str(FIG / "make_learned.py")], ROOT),
    ("W8b three learned predictors x masking (csv)", [PY, str(FIG / "make_learned_all.py")], ROOT),
    ("W9 sensitivity table (t4_sens)", [PY, str(FIG / "make_t4_compact.py")], ROOT),
    ("W9 bootstrap CI over event days (csv)", [PY, str(FIG / "make_bootstrap.py")], ROOT),
    ("W9 no-predictor delay-only comparison (csv)", [PY, str(FIG / "make_blind.py")], ROOT),
    ("W9b plan_wf variant at w_f = 1 (csv)", [PY, str(FIG / "make_planwf.py")], ROOT),
    ("W7 main table (t_main)", [PY, str(FIG / "make_t_main.py")], ROOT),
    ("T6 carbon table", [PY, "-m", "flex.e5_carbon", "table"], ROOT / "backend"),
]
SIM_STEPS = [
    ("F1 teaser (one event re-run)", [PY, str(FIG / "fig_teaser.py")], ROOT),
    ("F2 motivation (baseline run)", [PY, str(FIG / "motivation.py")], ROOT),
]
WATCH = [ROOT / "paper" / "figures" / "out", ROOT / "paper" / "tables", ROOT / "results" / "e5", ROOT / "results" / "e1", ROOT / "results" / "e3"]


def scrub_figures():
    """Double-blind: blank Creator/Producer in every figure PDF (same length, so xref offsets stay valid) and drop PNG text chunks.
    Covers figures that were not re-rendered in this run (old or simulation-only ones)."""
    import re
    from PIL import Image
    n = 0
    for f in sorted((FIG / "out").glob("*")):
        if f.suffix == ".pdf":
            b = f.read_bytes()
            c = re.sub(rb"(/(?:Creator|Producer)\s*\()([^)]*)(\))", lambda m: m.group(1) + b" " * len(m.group(2)) + m.group(3), b)
            if c != b:
                f.write_bytes(c)
                n += 1
        elif f.suffix == ".png":
            im = Image.open(f)
            if im.info.keys() - {"dpi"}:
                dpi = im.info.get("dpi")
                im.load()
                im.save(f, dpi=dpi) if dpi else im.save(f)
                n += 1
    print(f"[ok] metadata scrub: {n} figure files changed")


def main():
    steps = (SIM_STEPS if "--with-sim" in sys.argv else []) + STEPS
    t0 = time.time()
    failed = []
    for label, cmd, cwd in steps:
        t = time.time()
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        print(f"[{'ok' if r.returncode == 0 else 'FAIL'}] {label} ({time.time() - t:.0f} s)")
        if r.returncode:
            failed.append(label)
            print("   ", (r.stderr or r.stdout).strip().splitlines()[-1:] )
    r = subprocess.run([PY, str(FIG / "make_numbers.py")], cwd=ROOT, capture_output=True, text=True)
    print(f"[{'ok' if r.returncode == 0 else 'FAIL'}] numbers.tex: {r.stdout.strip() or r.stderr.strip()[-300:]}")
    if r.returncode:
        failed.append("make_numbers")
    if "--with-sim" not in sys.argv:
        print("skipped (need a simulator run; use --with-sim): F1 teaser, F2 motivation")
    scrub_figures()
    out = []
    for d in WATCH + [ROOT / "paper"]:
        for p in ([d / "numbers.tex"] if d == ROOT / "paper" else sorted(d.glob("*"))):
            if p.is_file() and p.stat().st_mtime >= t0 - 1 and p.suffix in (".pdf", ".png", ".tex", ".csv"):
                out.append(p.relative_to(ROOT))
    print(f"\n{len(out)} outputs written:")
    for p in out:
        print("  ", p.as_posix())
    if failed:
        print("FAILED:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
