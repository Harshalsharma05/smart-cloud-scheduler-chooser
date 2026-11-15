"""Build paper/overleaf_upload.zip with only what Overleaf needs to compile main.tex (main.tex at the zip root).

Contents: main.tex, local class/style files (*.cls, *.bst, *.sty in paper/, if any), refs.bib, numbers.tex, sections/*.tex,
tables/*.tex, figures/*.tex, and the figures/out/*.pdf files that some \\includegraphics actually uses. No scripts, results,
logs, PNGs or caches. Then it follows every \\input / \\include / \\includegraphics / \\bibliography reached from main.tex and
prints the paths that are missing from the zip (the list must be empty).   python paper/make_overleaf_zip.py
"""
import re
import sys
import zipfile
from pathlib import Path

P = Path(__file__).resolve().parent
OUT = P / "overleaf_upload.zip"
BS = chr(92)
E = re.escape(BS)


def code(path):
    """File text without comments (a % not preceded by a backslash starts a comment)."""
    return "\n".join(re.split(r"(?<!" + E + r")%", l)[0] for l in path.read_text(encoding="utf-8").splitlines())


def tex_path(name):
    return name if name.endswith(".tex") else name + ".tex"


def references(start="main.tex"):
    """Every file path reached from main.tex: inputs (recursively), graphics, bibliography."""
    need, graphics, todo, seen = set(), set(), [start], set()
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        need.add(f)
        if not (P / f).exists():
            continue
        s = code(P / f)
        for m in re.findall(E + r"(?:input|include)\{([^}]+)\}", s):
            todo.append(tex_path(m.strip()))
        for m in re.findall(E + r"includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", s):
            g = m.strip()
            graphics.add(g if Path(g).suffix else g + ".pdf")
        for m in re.findall(E + r"bibliography\{([^}]+)\}", s):
            need.update(b.strip() + ".bib" for b in m.split(","))
        for m in re.findall(E + r"documentclass(?:\[[^\]]*\])?\{([^}]+)\}", s):
            if (P / (m + ".cls")).exists():
                need.add(m + ".cls")
    return need, graphics


def main():
    need, graphics = references()
    files = ["main.tex", "refs.bib", "numbers.tex"]
    files += sorted(p.name for ext in ("*.cls", "*.bst", "*.sty") for p in P.glob(ext))
    for d in ("sections", "tables", "figures"):
        files += sorted(f"{d}/{p.name}" for p in (P / d).glob("*.tex"))
    files += sorted(g for g in graphics if (P / g).exists())
    files = list(dict.fromkeys(files))
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(P / f, f)
    inzip = set(files)
    missing = sorted((need | graphics) - inzip)
    print(f"wrote {OUT.relative_to(P.parent).as_posix()}: {len(files)} files, {OUT.stat().st_size / 1024:.1f} KiB")
    for f in files:
        print("  ", f)
    print("referenced from main.tex:", len(need), "tex/bib files,", len(graphics), "graphics")
    print("missing (referenced but not in the zip):", missing or "none")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
