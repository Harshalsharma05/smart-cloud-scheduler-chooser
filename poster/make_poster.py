"""Build the college evaluation poster from the college template (step 40).

Edits a copy of poster/Poster Template.pptx (A1 portrait, kept at A1), fills it with the project, replaces the three
SmartArt diagrams by native shapes, exports a PDF with PowerPoint (COM, via PowerShell) or LibreOffice, scales it to
A3 with pypdf and renders poster/preview.png.

Every result number comes from paper/numbers.tex (or runs/*/summary.json, ml/model.py); nothing is typed by hand.
    python poster/make_poster.py
"""
import copy
import re
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymupdf
from lxml import etree
from PIL import Image, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Emu, Pt
from pypdf import PdfReader, PdfWriter, Transformation

POSTER = Path(__file__).resolve().parent
ROOT = POSTER.parent
TEMPLATE = next(p for p in (POSTER / "Poster_Template.pptx", POSTER / "Poster Template.pptx") if p.exists())
LOGO = next(p for p in (POSTER / "assets" / "school_logo.png", POSTER / "school_logo.png") if p.exists())
OUT = POSTER / "Minor_Project_Poster.pptx"
PDF_A1 = POSTER / "build" / "Minor_Project_Poster_A1.pdf"
PDF_A3 = POSTER / "Minor_Project_Poster_A3.pdf"
FIG = POSTER / "figures"
PAPER_PDF = ROOT / "paper" / "build" / "main.pdf"

BLANK = "________"
FONT = "Times New Roman"
M = 1.3                      # safe margin on the A1 canvas, cm (>= 1.2)
W_SLIDE, H_SLIDE = 59.41, 84.10
X0, X1 = M, W_SLIDE - M      # usable x range
SIZE = dict(title=48, names=30, body_max=32, body_min=28, cap=24, ref=24)

# template colours (read from ppt/diagrams/drawing*.xml + theme1.xml; tinted/transparent ones as PowerPoint renders them)
ORANGE, LGREEN, GOLD, BLUE, TEAL, GREEN = "ED7D31", "E2EFDA", "FFC000", "4472C4", "43BA8D", "70AD47"
TINT_BLUE, TINT_TEAL, TINT_ORANGE = "C0C9E6", "C5DDC8", "F8D6CD"
TEXT_ON = {ORANGE: "FFFFFF", BLUE: "FFFFFF", TEAL: "FFFFFF", GREEN: "FFFFFF", LGREEN: "000000", GOLD: "000000"}

EMU_CM = 360000
cm = lambda v: Emu(int(round(v * EMU_CM)))


# ---------------------------------------------------------------- numbers (never typed by hand)
def numbers():
    s = (ROOT / "paper" / "numbers.tex").read_text(encoding="utf-8")
    return {k: v.replace("{,}", ",") for k, v in re.findall(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$", s, re.M)}


def num(N, key):
    return float(N[key].replace(",", ""))


def transformer_facts():
    import json
    params = json.loads((ROOT / "runs" / "tf_v1" / "summary.json").read_text())["params"]
    src = (ROOT / "ml" / "model.py").read_text(encoding="utf-8")
    heads = re.search(r"heads=(\d+)", src).group(1)
    layers = re.search(r"layers=(\d+)", src).group(1)
    hist = re.search(r"^H = (\d+)", (ROOT / "ml" / "config.py").read_text(encoding="utf-8"), re.M).group(1)
    return layers, heads, f"~{round(params / 1000)}k", hist


def bib():
    s = (ROOT / "paper" / "refs.bib").read_text(encoding="utf-8")
    out = {}
    for key, body in re.findall(r"@\w+\{([^,]+),(.*?)\n\}", s, re.S):
        f = dict(re.findall(r"(\w+)\s*=\s*\{(.*)\},?\s*$", body, re.M))
        first = f["author"].split(" and ")[0].strip()
        surname = first.split(",")[0] if "," in first else first.split()[-1]
        out[key] = dict(surname=surname.strip("{}"), year=f["year"],
                        venue=f.get("booktitle", f.get("journal", "")).replace("{", "").replace("}", ""))
    return out


# ---------------------------------------------------------------- text fitting (Times New Roman metrics)
_F = {}


def font(bold=False, italic=False):
    name = {(0, 0): "times.ttf", (1, 0): "timesbd.ttf", (0, 1): "timesi.ttf", (1, 1): "timesbi.ttf"}[(bold, italic)]
    if name not in _F:
        _F[name] = ImageFont.truetype(f"C:/Windows/Fonts/{name}", 1000)
    return _F[name]


def text_w(s, size, bold=False, italic=False):
    return font(bold, italic).getlength(s) / 1000 * size / 72 * 2.54      # cm


def n_lines(text, size, width_cm, bold=False, italic=False):
    """wrapped line count; a vertical tab is a forced line break"""
    if "\v" in text:
        return sum(n_lines(t, size, width_cm, bold, italic) for t in text.split("\v"))
    lines, cur = 1, ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if text_w(t, size, bold, italic) <= width_cm * 0.96:              # 4% slack for PowerPoint's own wrapping
            cur = t
        else:
            if cur:
                lines += 1
            cur = w
    return lines


def text_h(paras, size, width_cm, bold=False, italic=False, gap=0.35):
    """height in cm of paragraphs at one size: line pitch 1.2 em, gap*em between paragraphs"""
    pt = sum(n_lines(p, size, width_cm, bold, italic) for p in paras) * 1.2 * size + gap * size * (len(paras) - 1)
    return pt / 72 * 2.54


def fit(paras, width_cm, height_cm, lo, hi, bold=False, italic=False, gap=0.35, what=""):
    for size in range(hi, lo - 1, -1):
        if text_h(paras, size, width_cm, bold, italic, gap) <= height_cm:
            return size
    raise SystemExit(f"text does not fit at {lo} pt in {width_cm:.1f} x {height_cm:.1f} cm: {what or paras[0][:40]}")


# ---------------------------------------------------------------- drawing helpers
def set_xfrm(el, x, y, w, h):
    xfrm = el.find(".//" + qn("a:xfrm"))
    xfrm.find(qn("a:off")).set("x", str(int(cm(x)))); xfrm.find(qn("a:off")).set("y", str(int(cm(y))))
    xfrm.find(qn("a:ext")).set("cx", str(int(cm(w)))); xfrm.find(qn("a:ext")).set("cy", str(int(cm(h))))


def style_runs(p, size, bold=False, italic=False, color="000000"):
    for r in p.runs:
        r.font.name, r.font.size, r.font.bold, r.font.italic = FONT, Pt(size), bold, italic
        r.font.color.rgb = RGBColor.from_string(color)
        rpr = r._r.get_or_add_rPr()
        for tag in ("a:cs",):
            e = rpr.find(qn(tag))
            if e is None:
                e = etree.SubElement(rpr, qn(tag))
            e.set("typeface", FONT)


def fill_frame(tf, paras, size, bold=False, italic=False, color="000000", align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
               bullets=False, ins=0.25, gap=0.35, labels=False):
    """paras: list of str; with labels=True a 'Label:' prefix is set bold"""
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = cm(ins)
    for i, text in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = 1.0
        if i:
            p.space_before = Pt(size * gap)
        if labels and ":" in text:
            head, tail = text.split(":", 1)
            r1 = p.add_run(); r1.text = head + ":"
            r2 = p.add_run(); r2.text = tail
            style_runs(p, size, bold, italic, color)
            r1.font.bold = True
        else:
            p.text = text                                   # paragraph.text turns "\v" into <a:br/>
            style_runs(p, size, bold, italic, color)
        if bullets:
            pPr = p._p.get_or_add_pPr()
            ind = int(cm(0.9))
            pPr.set("marL", str(ind)); pPr.set("indent", str(-ind))
            for tag in ("a:buFont", "a:buChar"):
                e = pPr.find(qn(tag))
                if e is not None:
                    pPr.remove(e)
            etree.SubElement(pPr, qn("a:buFont")).set("typeface", "Arial")
            etree.SubElement(pPr, qn("a:buChar")).set("char", "\u2022")


def textbox(slide, x, y, w, h, paras, size, **kw):
    tb = slide.shapes.add_textbox(cm(x), cm(y), cm(w), cm(h))
    fill_frame(tb.text_frame, paras, size, **kw)
    return tb


def shape(slide, kind, x, y, w, h, color, rot=0, radius=None):
    s = slide.shapes.add_shape(kind, cm(x), cm(y), cm(w), cm(h))
    s.fill.solid(); s.fill.fore_color.rgb = RGBColor.from_string(color)
    s.line.fill.background()
    s.shadow.inherit = False
    if rot:
        s.rotation = rot
    if radius is not None:
        s.adjustments[0] = radius
    return s


def box_size(w, h, text, lo, hi, ins=0.35, right_ins=None, bold=False):
    ri = ins if right_ins is None else right_ins
    return fit([text], w - ins - ri, h - 2 * ins, lo, hi, bold=bold, what=text)


def box(slide, x, y, w, h, color, text, size, ins=0.35, right_ins=None, bold=False):
    """rounded rectangle in a template colour with centred text at a given size (auto-fit off, word wrap on)"""
    s = shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, color, radius=0.1)
    ri = ins if right_ins is None else right_ins
    fill_frame(s.text_frame, [text], size, bold=bold, color=TEXT_ON[color], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE,
               ins=ins)
    s.text_frame.margin_right = cm(ri)
    return size


def picture(slide, path, x, y, w, h):
    """largest picture with the image's aspect ratio, centred in (x, y, w, h)"""
    iw, ih = Image.open(path).size
    s = min(w / iw, h / ih)
    pw, ph = iw * s, ih * s
    slide.shapes.add_picture(str(path), cm(x + (w - pw) / 2), cm(y + (h - ph) / 2), cm(pw), cm(ph))
    return x + (w - pw) / 2, y + (h - ph) / 2, pw, ph


# ---------------------------------------------------------------- images
def make_images(N):
    FIG.mkdir(exist_ok=True)
    doc = pymupdf.open(PAPER_PDF)
    # Fig. 2 (system overview) is vector art on page 4: crop it by the bounding box of its drawings and labels
    page = next(p for p in doc if any(b[4].startswith("Fig. 2.") for b in p.get_text("blocks")))
    cap_top = next(b[1] for b in page.get_text("blocks") if b[4].startswith("Fig. 2."))
    rects = [d["rect"] for d in page.get_drawings() if d["rect"].y1 < cap_top and d["rect"].x1 < page.rect.width / 2]
    rects += [pymupdf.Rect(b[:4]) for b in page.get_text("blocks") if b[3] < cap_top and b[2] < page.rect.width / 2]
    clip = rects[0]
    for r in rects[1:]:
        clip |= r
    clip = clip + (-3, -3, 3, 3)
    page.get_pixmap(dpi=600, clip=clip).save(FIG / "system_overview.png")
    doc[0].get_pixmap(dpi=200).save(FIG / "paper_page1.png")

    # grouped bars: median $/MWh in the default cell (2 h, 20 % cut, w_f = 0), transformer predictor
    vals = {"Philly (2017)": [num(N, "DefDelayUsdZero"), num(N, "DefMixPNineUsdZero"), num(N, "DefBurstUsdZero")],
            "Acme Seren (2023)": [num(N, "AcmeDefDelayUsdZero"), num(N, "AcmeDefMixPNineUsdZero"), num(N, "AcmeDefBurstUsdZero")]}
    names = ["Delay only", "Mix (90th-pct. runtimes)", "Burst only"]
    colours = ["#" + GREEN, "#" + BLUE, "#" + ORANGE]
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 30})
    fig, ax = plt.subplots(figsize=(34.0 / 2.54, 17.0 / 2.54))
    bw = 0.26
    for i, (nm, c) in enumerate(zip(names, colours)):
        xs = [g + (i - 1) * bw for g in range(len(vals))]
        ys = [v[i] for v in vals.values()]
        bars = ax.bar(xs, ys, bw * 0.92, color=c, label=nm)
        for b, y in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, y * 1.12, f"${y:,.0f}", ha="center", va="bottom", fontsize=28)
    ax.set_yscale("log")
    ax.set_ylim(1000, 3e6)
    ax.set_xticks(range(len(vals)), list(vals))
    ax.set_ylabel("Median $ per MWh shed (log)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", which="major", color="#DDDDDD", linewidth=1)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=28)
    fig.tight_layout()
    fig.savefig(FIG / "cost_per_mwh.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- the poster
def build(N):
    shutil.copyfile(TEMPLATE, OUT)
    prs = Presentation(OUT)
    slide = prs.slides[0]
    tree = slide.shapes._spTree
    grp = next(s for s in slide.shapes if s.shape_type == 6)
    g = grp._element
    # identity group transform, so child coordinates are slide coordinates
    gx = g.grpSpPr.find(qn("a:xfrm"))
    gx.find(qn("a:off")).attrib.update(gx.find(qn("a:chOff")).attrib)
    kids = {s.name: s for s in grp.shapes}

    # remove the SmartArt frames, the lavender placeholder boxes and the diagram parts
    for nm in ("Diagram 10", "Diagram 13", "Diagram 14", "TextBox 11", "TextBox 111", "TextBox 114", "TextBox 117",
               "TextBox 25", "TextBox 23"):
        kids[nm]._element.getparent().remove(kids[nm]._element)
    for rId, rel in list(slide.part.rels.items()):
        if "diagram" in rel.reltype.lower():
            slide.part.drop_rel(rId)

    # ---- title bar (template rectangle, logos left and right)
    layers, heads, params, hist = transformer_facts()
    title = f"Cloud Bursting Rarely Pays for Demand Response in\vOn-Premise GPU Clusters (Project No. {BLANK})"
    names = (f"Harshal Sharma (23BCP298), Anuj Sharma (23BCP289) | Guide: Dr. Riddhi Thakkar, Assistant Professor, "
             "Department of Computer Science and Engineering, School of Technology, PDEU")
    logo_w = 5.9
    tw = X1 - X0 - 2 * (logo_w + 0.5) - 0.4
    th = text_h([title], SIZE["title"], tw, bold=True) + text_h([names], SIZE["names"], tw, bold=True, italic=True) + 0.3
    bar_h = max(5.44, th + 0.6)
    bar = kids["Rectangle 1"]
    set_xfrm(bar._element, X0, M, X1 - X0, bar_h)
    tf = bar.text_frame
    bp = tf._txBody.find(qn("a:bodyPr")); bp.set("wrap", "square")
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = cm(logo_w + 0.7)
    p1, p2 = tf.paragraphs[0], tf.paragraphs[1]
    for p, text, size, it in ((p1, title, SIZE["title"], False), (p2, names, SIZE["names"], True)):
        p.alignment = PP_ALIGN.CENTER
        for r in p.runs[1:]:
            r._r.getparent().remove(r._r)
        r0 = p.runs[0]._r
        r0.find(qn("a:t")).text = text.split("\v")[0]
        for part in text.split("\v")[1:]:                     # forced line break, same run formatting
            br = OxmlElement("a:br")
            br.append(copy.deepcopy(r0.find(qn("a:rPr"))))
            r_new = copy.deepcopy(r0)
            r_new.find(qn("a:t")).text = part
            r0.addnext(br)
            br.addnext(r_new)
            r0 = r_new
        for r in p.runs:
            r.font.size = Pt(size)
            r.font.italic = it
    end = p1._p.find(qn("a:endParaRPr"))
    if end is not None:
        end.set("sz", str(SIZE["title"] * 100))
    # university logo (template picture, kept) and school logo box
    uni = next(s for s in slide.shapes if s.name == "Picture 7")
    uh = (logo_w - 0.2) * uni.height / uni.width                # template picture, aspect ratio kept
    uni.left, uni.top, uni.width, uni.height = cm(X0 + 0.1), cm(M + (bar_h - uh) / 2), cm(logo_w - 0.2), cm(uh)
    sbox = next(s for s in slide.shapes if s.name == "Rectangle 5")
    sbox.left, sbox.top, sbox.width, sbox.height = cm(X1 - logo_w), cm(M), cm(logo_w), cm(bar_h)
    for p in list(sbox.text_frame._txBody.findall(qn("a:p")))[1:]:
        p.getparent().remove(p)
    for r in sbox.text_frame.paragraphs[0].runs:
        r._r.getparent().remove(r._r)
    picture(slide, LOGO, X1 - logo_w + 0.35, M + 0.35, logo_w - 0.7, bar_h - 0.7)

    # ---- band geometry (cm)
    HB = 1.57                                   # header bar height (template)
    y = M + bar_h + 0.4
    band = {}
    heights = {"A": 9.9, "M": 22.3, "R": 18.8}
    for key in ("A", "M", "R"):
        band[key] = (y, y + HB, y + HB + heights[key])   # header top, content top, content bottom
        y = band[key][2] + 0.2
    band["O"] = (y, y + HB, H_SLIDE - M)
    print("title bar", round(bar_h, 2), "bands", {k: tuple(round(v, 2) for v in b) for k, b in band.items()})
    mid = X0 + (X1 - X0) * 29.79 / 59.41                 # template split between Motivation and Objective
    split_o = X0 + (X1 - X0) * 33.07 / 59.41             # template split between Outcomes and Budget
    hdr = {"TextBox 22": (X0, band["A"][0], mid - X0), "TextBox 21": (mid, band["A"][0], X1 - mid),
           "TextBox 101": (X0, band["M"][0], X1 - X0), "TextBox 92": (X0, band["R"][0], X1 - X0),
           "TextBox 18": (X0, band["O"][0], split_o - X0), "TextBox 108": (split_o, band["O"][0], X1 - split_o)}
    for nm, (hx, hy, hw) in hdr.items():
        set_xfrm(kids[nm]._element, hx, hy, hw, HB)
    line = next(s for s in slide.shapes if s.name == "Straight Connector 20")
    line.left, line.top, line.width, line.height = cm(mid), cm(band["A"][1] + 0.3), 0, cm(heights["A"] - 0.6)
    # group bounds = bounds of what is left in it
    xs = [(int(k.find(".//" + qn("a:off")).get("x")), int(k.find(".//" + qn("a:off")).get("y")),
           int(k.find(".//" + qn("a:ext")).get("cx")), int(k.find(".//" + qn("a:ext")).get("cy")))
          for k in g if k.tag in (qn("p:sp"),)]
    bx0, by0 = min(a for a, *_ in xs), min(b for _, b, *_ in xs)
    bx1, by1 = max(a + c for a, _, c, _ in xs), max(b + d for _, b, _, d in xs)
    for tag, vals in (("a:off", (bx0, by0)), ("a:chOff", (bx0, by0))):
        gx.find(qn(tag)).attrib.update({"x": str(vals[0]), "y": str(vals[1])})
    for tag in ("a:ext", "a:chExt"):
        gx.find(qn(tag)).attrib.update({"cx": str(bx1 - bx0), "cy": str(by1 - by0)})

    used = {}
    # ---- A: Brief Motivation | Objective
    mot = ["Grids ask large loads to cut power for a few hours.",
           "GPU clusters are now megawatt-scale loads.",
           "A cluster can delay jobs, cap GPU power, or burst jobs to the cloud.",
           "Nobody had priced cloud bursting as a grid response against delay and capping."]
    obj = ["Price delay, capping and bursting on one scale ($ per MWh shed).",
           "Build a trace-driven GPU-cluster simulator with grid events.",
           "Choose jobs with predicted remaining runtime and failure risk (transformer + baselines).",
           "Evaluate on two real traces (Philly 2017, AcmeTrace 2023)."]
    a_top, a_bot = band["A"][1] + 0.3, band["A"][2] - 0.2
    a_size = min(fit(paras, x1 - x0 - 0.6 - 0.5 - 0.9, a_bot - a_top - 0.5, SIZE["body_min"], SIZE["body_max"], gap=0.4)
                 for paras, x0, x1 in ((mot, X0, mid), (obj, mid, X1)))
    for paras, x0, x1 in ((mot, X0, mid), (obj, mid, X1)):
        textbox(slide, x0 + 0.3, a_top, x1 - x0 - 0.6, a_bot - a_top, paras, a_size, bullets=True, gap=0.4)
    used["motivation/objective"] = a_size

    # ---- M: Methodology, left = 5-box snake (template process5), right = system overview figure + notes
    m_top, m_bot = band["M"][1], band["M"][2]
    steps = ["Real job traces: Philly (2017), Acme Seren (2023)",
             "Clean data, build leak-free user-history examples",
             "Train predictors: last-two, GBDT, LightGBM, Transformer",
             "Simulate grid\vevents: 1\u20134 h,\v10\u201330% power cut",
             "Price delay / cap /\vburst in $ per\vMWh shed"]
    bw_, bh_, gx_, gy_ = 8.7, 5.5, 2.5, 2.1
    sx = X0 + 0.5
    sy = m_top + (m_bot - m_top - (3 * bh_ + 2 * gy_)) / 2
    pos = [(sx, sy), (sx + bw_ + gx_, sy), (sx + bw_ + gx_, sy + bh_ + gy_), (sx, sy + bh_ + gy_), (sx, sy + 2 * (bh_ + gy_))]
    cols = [ORANGE, LGREEN, GOLD, BLUE, GREEN]
    m_size = min(box_size(bw_, bh_, t, SIZE["body_min"], SIZE["body_max"]) for t in steps)
    for (px, py), c, t in zip(pos, cols, steps):
        box(slide, px, py, bw_, bh_, c, t, m_size)
    aw, ah = 1.7, 1.9                                        # arrow length x thickness (template ratio 1.46 x 1.71)
    shape(slide, MSO_SHAPE.RIGHT_ARROW, sx + bw_ + (gx_ - aw) / 2, sy + (bh_ - ah) / 2, aw, ah, ORANGE)
    for (px, py), c in (((pos[1][0] + bw_ / 2, pos[1][1] + bh_ + gy_ / 2), LGREEN), ((pos[3][0] + bw_ / 2, pos[3][1] + bh_ + gy_ / 2), BLUE)):
        shape(slide, MSO_SHAPE.RIGHT_ARROW, px - aw / 2, py - ah / 2, aw, ah, c, rot=90)
    shape(slide, MSO_SHAPE.RIGHT_ARROW, sx + bw_ + (gx_ - aw) / 2, pos[2][1] + (bh_ - ah) / 2, aw, ah, GOLD, rot=180)
    used["methodology boxes"] = m_size

    rx0 = sx + 2 * bw_ + gx_ + 1.2
    rw = X1 - rx0
    notes = ["Paired runs: same cluster with and without the event",
             "Controller: greedy plan over cap levels and per-job suspend / burst",
             f"Transformer: {layers} layers, {heads} heads, {params} parameters, history of the user's last {hist} jobs"]
    n_size = fit(notes, rw - 0.5 - 0.9, 5.0, SIZE["body_min"], SIZE["body_max"], gap=0.3)
    n_h = text_h(notes, n_size, rw - 0.5 - 0.9, gap=0.3) + 0.6
    cap = "System overview (Fig. 2 of the paper): the controller reads the grid target and the job predictions, then caps, suspends or bursts jobs."
    c_size = fit([cap], rw - 0.5, 2.6, SIZE["cap"], SIZE["cap"] + 2, italic=True)
    c_h = text_h([cap], c_size, rw - 0.5, italic=True) + 0.5
    fig_h = m_bot - m_top - 0.4 - c_h - n_h
    fx, fy, fw, fh = picture(slide, FIG / "system_overview.png", rx0, m_top + 0.3, rw, fig_h)
    textbox(slide, rx0, fy + fh + 0.1, rw, c_h, [cap], c_size, italic=True, align=PP_ALIGN.CENTER)
    textbox(slide, rx0, fy + fh + 0.1 + c_h, rw, n_h, notes, n_size, bullets=True, gap=0.3, labels=True)
    used["methodology notes"], used["captions"] = n_size, c_size

    # ---- R: Expected Results, left = 3 stepped boxes (template vProcess5), right = chart
    r_top, r_bot = band["R"][1], band["R"][2]
    res = ["Delay is the cheapest action on both traces",
           f"Bursting alone costs {N['BurstOverDelayDefaultZero']}\u00d7 (Philly) and {N['AcmeBurstOverDelayDefaultZero']}\u00d7 (Acme) "
           "as much as delay",
           f"Better prediction accuracy did not lower the cost; reading predictions naively cost "
           f"{N['ContractOverHoldMin']}\u2013{N['ContractOverHoldMax']}\u00d7 more"]
    vw, vh, vstep_x, vstep_y = 20.5, 5.6, 1.55, 6.3
    vy = r_top + (r_bot - r_top - (2 * vstep_y + vh)) / 2
    vx = X0 + 0.3
    arw = 3.0                                               # down-arrow size (template 3.31 for 17.6 x 5.1 boxes)
    r_ins = [arw + 0.5, arw + vstep_x + 0.6, arw + vstep_x + 0.6]      # keep text clear of the own and the incoming arrow
    r_size = min(box_size(vw, vh, t, SIZE["body_min"], SIZE["body_max"], right_ins=ri) for t, ri in zip(res, r_ins))
    for i, (t, c) in enumerate(zip(res, [BLUE, TEAL, GREEN])):
        box(slide, vx + i * vstep_x, vy + i * vstep_y, vw, vh, c, t, r_size, right_ins=r_ins[i])
    for i, c in enumerate([TINT_BLUE, TINT_TEAL]):
        ax_ = vx + i * vstep_x + vw - arw - 0.25
        shape(slide, MSO_SHAPE.DOWN_ARROW, ax_, vy + i * vstep_y + vh - 1.25, arw, arw, c)
    used["result boxes"] = r_size
    cx0 = vx + 2 * vstep_x + vw + 1.0
    cw = X1 - cx0
    ccap = ("Median cost per MWh shed for a 2-hour, 20% cut (transformer predictor, failed work valued at zero). "
            "Log scale: burst-only is far above the other two.")
    cc_size = fit([ccap], cw - 0.5, 2.6, SIZE["cap"], SIZE["cap"] + 2, italic=True)
    cc_h = text_h([ccap], cc_size, cw - 0.5, italic=True) + 0.5
    px_, py_, pw_, ph_ = picture(slide, FIG / "cost_per_mwh.png", cx0, r_top + 0.3, cw, r_bot - r_top - 0.5 - cc_h)
    textbox(slide, cx0, py_ + ph_ + 0.1, cw, cc_h, [ccap], cc_size, italic=True, align=PP_ALIGN.CENTER)
    used["captions"] = min(used["captions"], cc_size)

    # ---- O: Outcomes (2 x 2 like the template matrix2) + first page of the paper | Budget + references
    o_top, o_bot = band["O"][1], band["O"][2]
    outs = ["Research paper submitted to IEEE CCGrid 2027 (Track\u00a07)",
            "Trace-driven GPU-cluster simulator with power model",
            "Transformer and baseline predictors on two traces",
            "Guideline: when cloud offloading can help"]
    qh = o_bot - o_top - 0.4
    qw = 23.0
    qx, qy = X0 + 0.3, o_top + 0.2
    shape(slide, MSO_SHAPE.QUAD_ARROW, qx, qy, qw, qh, TINT_ORANGE)
    ob_w, ob_h, ogap = 10.6, (qh - 2.0 - 0.8) / 2, 0.8
    ox0 = qx + (qw - 2 * ob_w - ogap) / 2
    oy0 = qy + 1.0
    o_size = min(box_size(ob_w, ob_h, t, SIZE["body_min"], SIZE["body_max"], ins=0.3) for t in outs)
    for i, (t, c) in enumerate(zip(outs, [ORANGE, LGREEN, GOLD, BLUE])):
        box(slide, ox0 + (i % 2) * (ob_w + ogap), oy0 + (i // 2) * (ob_h + ogap), ob_w, ob_h, c, t, o_size, ins=0.3)
    used["outcome boxes"] = o_size
    ix0 = qx + qw + 0.6
    iw = split_o - 0.4 - ix0
    pc = "First page of the paper"
    pc_h = text_h([pc], SIZE["cap"], iw, italic=True) + 0.4
    ix, iy, iww, ihh = picture(slide, FIG / "paper_page1.png", ix0, o_top + 0.3, iw, o_bot - o_top - 0.3 - pc_h)
    slide.shapes[-1].line.color.rgb = RGBColor(0x80, 0x80, 0x80)
    slide.shapes[-1].line.width = Pt(1)
    textbox(slide, ix0, iy + ihh + 0.05, iw, pc_h, [pc], SIZE["cap"], italic=True, align=PP_ALIGN.CENTER)

    # budget table
    bx0 = split_o + 0.4
    bw = X1 - bx0
    rows = [("Datasets", "public (free)"), ("Compute", "own laptop (GTX 1650 GPU)"),
            ("Software", "open-source (Python, PyTorch, LightGBM)"), ("Cloud", "no spending (cloud costs are simulated)"),
            ("Total", "Rs. 0")]
    c1 = max(text_w(a, SIZE["body_min"], bold=True) for a, _ in rows) + 0.9
    c2 = bw - c1
    t_size = min(fit([b], c2 - 0.5, 1.2 * SIZE["body_max"] / 72 * 2.54 + 0.01, SIZE["body_min"], SIZE["body_max"]) for _, b in rows)
    row_h = 1.2 * t_size / 72 * 2.54 + 0.25
    gf = slide.shapes.add_table(len(rows), 2, cm(bx0), cm(o_top + 0.4), cm(bw), cm(row_h * len(rows)))
    tbl = gf.table
    tbl.first_row = tbl.horz_banding = False
    tbl.columns[0].width, tbl.columns[1].width = cm(c1), cm(c2)
    for i, (a, b) in enumerate(rows):
        tbl.rows[i].height = cm(row_h)
        for j, text in enumerate((a, b)):
            cell = tbl.cell(i, j)
            cell.margin_left = cell.margin_right = cm(0.25)
            cell.margin_top = cell.margin_bottom = cm(0.08)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            r = p.add_run(); r.text = text
            style_runs(p, t_size, bold=(j == 0 or a == "Total"))
            tcPr = cell._tc.get_or_add_tcPr()
            for e in list(tcPr):
                tcPr.remove(e)
            for side in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
                ln = etree.SubElement(tcPr, qn(side), w="12700")
                etree.SubElement(etree.SubElement(ln, qn("a:solidFill")), qn("a:srgbClr"), val="7F7F7F")
            etree.SubElement(etree.SubElement(tcPr, qn("a:solidFill")), qn("a:srgbClr"), val="FFFFFF")
    used["budget table"] = t_size

    # references (author surname and year from refs.bib; short venue names)
    B = bib()
    refs = [("jeon19", "USENIX ATC", "USENIX Annual Technical Conference", " (Philly trace)"),
            ("hu24", "USENIX NSDI", "NSDI", " (AcmeTrace)"),
            ("tsafrir07", "IEEE TPDS", "Transactions on Parallel and Distributed Systems", ""),
            ("colangelo26", "Nature Energy", "Nature Energy", ""),
            ("acun26dr", "ACM e-Energy", "Future and Sustainable Energy Systems", "")]
    ref_lines = []
    for i, (k, short, must, note) in enumerate(refs, 1):
        e = B[k]
        assert must in e["venue"], (k, e["venue"])
        ref_lines.append(f"[{i}] {e['surname']} et al., {short} {e['year']}{note}")
    ry = o_top + 0.4 + row_h * len(rows) + 0.3
    hdr_h = 1.2 * SIZE["body_min"] / 72 * 2.54 + 0.4
    textbox(slide, bx0, ry, bw, hdr_h, ["References"], SIZE["body_min"], bold=True)
    r_size = fit(ref_lines, bw - 0.5, o_bot - ry - hdr_h - 0.2, SIZE["ref"], SIZE["ref"] + 2, gap=0.15)
    textbox(slide, bx0, ry + hdr_h, bw, o_bot - ry - hdr_h, ref_lines, r_size, gap=0.15)
    used["references"] = r_size

    prs.save(OUT)
    return used


# ---------------------------------------------------------------- export, A3 scaling, preview
PS_EXPORT = r"""
$app = New-Object -ComObject PowerPoint.Application
try { $p = $app.Presentations.Open('{src}', $true, $false, $false); $p.SaveAs('{dst}', 32); $p.Close() } finally { $app.Quit() }
"""


def export_pdf():
    PDF_A1.parent.mkdir(exist_ok=True)
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice:
        subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(PDF_A1.parent), str(OUT)], check=True)
        (PDF_A1.parent / (OUT.stem + ".pdf")).replace(PDF_A1)
    else:
        cmd = PS_EXPORT.replace("{src}", str(OUT)).replace("{dst}", str(PDF_A1))
        subprocess.run(["powershell", "-NoProfile", "-Command", cmd], check=True)


def scale_a3():
    A3 = (841.89, 1190.55)
    out = PdfWriter(clone_from=PDF_A1)
    page = out.pages[0]
    w, h = float(page.mediabox.width), float(page.mediabox.height)
    s = min(A3[0] / w, A3[1] / h)
    page.add_transformation(Transformation().scale(s, s).translate((A3[0] - w * s) / 2, (A3[1] - h * s) / 2))
    for box_ in ("mediabox", "cropbox", "trimbox", "bleedbox", "artbox"):
        setattr(page, box_, pymupdf_box(A3))
    out.write(PDF_A3)
    return s


def pymupdf_box(a3):
    from pypdf.generic import RectangleObject
    return RectangleObject([0, 0, a3[0], a3[1]])


def preview():
    doc = pymupdf.open(PDF_A3)
    p = doc[0]
    p.get_pixmap(dpi=150).save(POSTER / "preview.png")
    # margin check: bounding box of non-white pixels, in mm from each edge
    pix = p.get_pixmap(dpi=72)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("L").point(lambda v: 255 if v < 245 else 0)
    l, t, r, b = img.getbbox()
    mm = 25.4 / 72
    return dict(page_mm=(round(p.rect.width * mm, 1), round(p.rect.height * mm, 1)),
                margins_mm=dict(left=round(l * mm, 1), top=round(t * mm, 1), right=round((pix.width - r) * mm, 1),
                                bottom=round((pix.height - b) * mm, 1)))


if __name__ == "__main__":
    N = numbers()
    make_images(N)
    used = build(N)
    export_pdf()
    s = scale_a3()
    info = preview()
    print("font sizes on A1 (pt):", dict(title=SIZE["title"], names=SIZE["names"], headers=30.7, **used))
    print("A3 scale factor:", round(s, 5), info)
