"""S16 plain-text checks of paper/ (no LaTeX is installed here, so this is NOT a PDF build check).

1. macro hygiene: number macros used in tex files vs defined in numbers.tex (used-but-undefined, defined-but-unused)
2. [CITE?] and TODO markers with file and line
3. double-blind scan: names, university, repo links, e-mail, local paths in tex/bib/txt and in figure PDFs/PNGs (bytes)
4. structure: balanced braces, begin/end pairs, ref/label resolution, cite keys vs refs.bib (missing, uncited)
Prints a report; exit code 1 if anything is unresolved except unused macros.   python paper/figures/check_paper.py
"""
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "paper"
TEX = sorted(list((P / "sections").glob("*.tex")) + list((P / "tables").glob("*.tex")) + [P / "main.tex"])
TEXT = TEX + [P / "refs.bib", P / "abstract.txt", P / "numbers.tex"]
BS = chr(92)
bad = 0


def strip_comments(s):
    return "\n".join(re.sub(r"(?<!" + re.escape(BS) + r")%.*", "", l) for l in s.splitlines())


def lines(path):
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def cmd(name_pat):
    """regex for a LaTeX command with one braced argument: \\name{arg}"""
    return re.escape(BS) + name_pat + r"\{([^}]+)\}"


# 1. macros
defined = set(re.findall(cmd(r"newcommand").replace(r"\{([^}]+)\}", r"\{" + re.escape(BS) + r"([A-Za-z]+)\}"), (P / "numbers.tex").read_text(encoding="utf-8")))
prefixes = {re.match(r"[A-Z][a-z]+", n).group(0) for n in defined if re.match(r"[A-Z][a-z]+", n)}
used = Counter()
for f in TEX:
    for m in re.findall(re.escape(BS) + r"([A-Z][A-Za-z]+)", strip_comments(f.read_text(encoding="utf-8"))):
        mm = re.match(r"[A-Z][a-z]+", m)
        if mm and mm.group(0) in prefixes:
            used[m] += 1
undefined = sorted(set(used) - defined)
unused = sorted(defined - set(used))
print(f"== 1. macros: {len(defined)} defined, {len(used)} distinct used, {len(undefined)} used but undefined, {len(unused)} defined but unused")
print("   used but undefined:", undefined or "none")
bad += bool(undefined)
print("   unused by prefix:", dict(Counter(re.match(r"[A-Z][a-z]+", n).group(0) for n in unused)))
print("   unused (full list):", " ".join(unused))

# 2. markers
print("\n== 2. [CITE?] and TODO markers")
n_mark = 0
for f in TEXT:
    for i, l in enumerate(lines(f), 1):
        if "[CITE?" in l or re.search(r"\bTODO\b|\bFIXME\b|\bTBD\b", l):
            n_mark += 1
            print(f"   {f.relative_to(ROOT).as_posix()}:{i}: {l.strip()[:130]}")
print("   total:", n_mark)

# 3. double blind
print("\n== 3. double-blind scan")
PAT = re.compile(r"harshal|sharma|unaccomplished|pdeu|pandit|universit|college|github|gitlab|bitbucket|[\w.+-]+@[\w-]+\.[\w.]+|[A-Za-z]:" + re.escape(BS) + r"|/home/|/Users/|\bAdmin\b|smart-scheduler|minor project", re.I)
hits = 0
for f in TEXT:
    for i, l in enumerate(lines(f), 1):
        for m in PAT.finditer(l):
            hits += 1
            print(f"   {f.relative_to(ROOT).as_posix()}:{i}: [{m.group(0)}] {l.strip()[:110]}")
for f in sorted((P / "figures" / "out").glob("*")):
    b = f.read_bytes()
    m = re.search(rb"(?i)harshal|sharma|unaccomplished|pdeu|github|admin|C:" + re.escape(BS.encode()) + rb"|/Users/|matplotlib|@gmail", b)
    if m:
        hits += 1
        print(f"   {f.relative_to(ROOT).as_posix()}: bytes contain {m.group(0)!r}")
print("   findings:", hits, "(text findings are reported, not edited)")
bad += hits > 0

# 4. structure
print("\n== 4. structure of sections and tables")
labels, refs, cites = set(), Counter(), Counter()
for f in TEX:
    s = strip_comments(f.read_text(encoding="utf-8"))
    d = 0
    for ch in re.sub(re.escape(BS) + r"[{}]", "", s):
        d += (ch == "{") - (ch == "}")
        if d < 0:
            break
    begins = Counter(re.findall(re.escape(BS) + r"begin\{(\w+\*?)\}", s))
    ends = Counter(re.findall(re.escape(BS) + r"end\{(\w+\*?)\}", s))
    if d != 0 or begins != ends:
        bad += 1
        print(f"   UNBALANCED {f.relative_to(ROOT).as_posix()}: brace depth {d}, begin-end diff {dict((begins - ends) + (ends - begins))}")
    labels |= set(re.findall(cmd("label"), s))
    for g in re.findall(re.escape(BS) + r"(?:ref|eqref|autoref|cref|Cref)\{([^}]+)\}", s):
        refs.update(k.strip() for k in g.split(","))
    for g in re.findall(re.escape(BS) + r"cite[a-z]*\{([^}]+)\}", s):
        cites.update(k.strip() for k in g.split(","))
bib = set(re.findall(r"@\w+\{([^,\s]+),", (P / "refs.bib").read_text(encoding="utf-8")))
print("   braces and begin/end balanced in all", len(TEX), "files unless listed above")
print("   unresolved ref:", sorted(set(refs) - labels) or "none")
print("   cite keys missing in refs.bib:", sorted(set(cites) - bib) or "none")
print("   uncited bib entries:", sorted(bib - set(cites)) or "none")
print(f"   labels {len(labels)}, refs {len(refs)}, cite keys {len(cites)}, bib entries {len(bib)}")
bad += bool(set(refs) - labels) + bool(set(cites) - bib)
print("\nNOTE: no LaTeX here, so nothing above says the PDF builds.")
sys.exit(1 if bad else 0)
