"""Wording-only check for the W16 restyle.

    python paper/tools/invariants.py save      -> paper/build/invariants_before.json
    python paper/tools/invariants.py check [file ...]   (default: all)  compares with the baseline
Per file: cite keys (multiset + sentence index), ref/eqref/label targets, macro names, math segments,
literal numbers, word count. Everything except sentence index and word count must be identical;
the word count must not increase.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

P = Path(__file__).resolve().parents[1]
FILES = ["main.tex"] + [f"sections/{n}.tex" for n in (
    "01_introduction", "02_background", "03_problem", "04_design", "05_methodology", "06_evaluation",
    "07_related", "08_discussion", "09_references")]
BASE = P / "build" / "invariants_before.json"
MATH = re.compile(r"\$\$.*?\$\$|\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]"
                  r"|\\begin\{(equation\*?|align\*?|eqnarray\*?)\}.*?\\end\{\1\}", re.S)
CMD_WITH_ARG = re.compile(r"\\(?:cite\w*|ref|eqref|label|input|includegraphics|url|href|footnote\w*)(\[[^\]]*\])?\{[^}]*\}")


def strip_comments(t):
    return re.sub(r"(?<!\\)%[^\n]*", "", t)


def body(name, t):
    """main.tex: only the abstract."""
    if name == "main.tex":
        return re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", t, re.S).group(1)
    return t


def sentences(t):
    t = re.sub(r"\s+", " ", t)
    return re.split(r"(?<=[.!?])\s+(?=[A-Z\\$\[(])", t)


def extract(name):
    raw = (P / name).read_text(encoding="utf-8")
    t = body(name, strip_comments(raw))
    out = {}
    cites = []
    for i, s in enumerate(sentences(t)):
        for m in re.finditer(r"\\cite\w*\{([^}]*)\}", s):
            for k in m.group(1).split(","):
                cites.append((k.strip(), i))
    out["cites"] = sorted(cites)
    out["cite_keys"] = sorted(k for k, _ in cites)
    out["refs"] = sorted(re.findall(r"\\(?:ref|eqref|label|autoref|cref)\{([^}]*)\}", t))
    out["macros"] = sorted(Counter(re.findall(r"\\([A-Za-z]+)", t)).items())
    out["math"] = sorted(re.sub(r"\s+", "", m.group(0)) for m in MATH.finditer(t))
    text = CMD_WITH_ARG.sub(" ", MATH.sub(" ", t))
    out["numbers"] = sorted(re.findall(r"\d+(?:[.,]\d+)*", text))
    words = re.sub(r"\\[A-Za-z]+\*?(\[[^\]]*\])?", " ", text)   # a number macro counts as no word; fine for before/after
    words = re.sub(r"[{}~\\]", " ", words)
    out["words"] = len(words.split())
    return out


def norm(x):
    return json.loads(json.dumps(x))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    names = sys.argv[2:] or FILES
    names = [n if n.endswith(".tex") else f"sections/{n}.tex" for n in names]
    now = {n: extract(n) for n in names}
    if mode == "save":
        BASE.parent.mkdir(exist_ok=True)
        BASE.write_text(json.dumps(now, indent=1), encoding="utf-8")
        for n, d in now.items():
            print(f"{n:34s} words {d['words']:5d} cites {len(d['cite_keys']):3d} numbers {len(d['numbers']):3d} math {len(d['math']):3d}")
        return
    base = json.loads(BASE.read_text(encoding="utf-8"))
    ok_all = True
    for n, d in now.items():
        b = base[n]
        bad = []
        for k in ("cite_keys", "refs", "macros", "math", "numbers"):
            if norm(d[k]) != b[k]:
                bb, dd = Counter(map(json.dumps, b[k])), Counter(map(json.dumps, d[k]))
                bad.append(f"{k}: missing {list((bb - dd).elements())[:6]} extra {list((dd - bb).elements())[:6]}")
        moved = norm(d["cites"]) != b["cites"]
        ok = not bad and d["words"] <= b["words"]
        ok_all &= ok
        print(f"{n:30s} words {b['words']:5d} -> {d['words']:5d} ({d['words'] - b['words']:+d})  "
              f"identical: {'yes' if not bad else 'NO'}  cite-sentence-index changed: {'yes' if moved else 'no'}  {'OK' if ok else 'FAIL'}")
        for s in bad:
            print("   ", s)
    sys.exit(0 if ok_all else 1)


main()
