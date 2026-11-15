"""Local build of the paper with Tectonic (tools/tectonic/tectonic.exe, git-ignored), then every page rendered to PNG.

    python paper/build.py            -> paper/build/main.pdf, paper/build/page-01.png ... (110 dpi), and a warning summary
Tectonic downloads the packages it needs on the first run. The summary lists overfull hboxes above 5 pt and undefined
references/citations from the log. PyMuPDF renders the pages (already installed; not in requirements.txt).
"""
import re
import subprocess
import sys
from pathlib import Path

P = Path(__file__).resolve().parent
ROOT = P.parent
TECTONIC = ROOT / "tools" / "tectonic" / ("tectonic.exe" if sys.platform == "win32" else "tectonic")
OUT = P / "build"


def main():
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("page-*.png"):
        old.unlink()
    r = subprocess.run([str(TECTONIC), "main.tex", "--outdir", str(OUT), "--keep-logs", "--keep-intermediates"],
                       cwd=P, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
        sys.exit("tectonic failed")
    log = (OUT / "main.log").read_text(encoding="utf-8", errors="replace")
    over = [(float(m.group(1)), m.group(0)) for m in re.finditer(r"Overfull \\hbox \((\d+\.?\d*)pt too wide\)[^\n]*", log)]
    big = [s for v, s in over if v > 5]
    undef = sorted(set(re.findall(r"(?:Reference|Citation) `([^']+)' [^\n]*undefined", log)))
    import pymupdf
    doc = pymupdf.open(OUT / "main.pdf")
    for i, page in enumerate(doc, 1):
        page.get_pixmap(dpi=110).save(OUT / f"page-{i:02d}.png")
    print(f"pages: {doc.page_count}  ->  {OUT.relative_to(ROOT).as_posix()}/main.pdf and page-NN.png")
    print(f"overfull hboxes: {len(over)} in total, {len(big)} above 5pt")
    for s in big:
        print("  ", s)
    print("undefined references/citations:", undef or "none")
    other = sorted(set(re.findall(r"(?:LaTeX|LaTeX Font|Package \w+) Warning:[^\n]*", log)))
    print("other warnings:", len(other))
    for s in other[:20]:
        print("  ", s)


if __name__ == "__main__":
    main()
