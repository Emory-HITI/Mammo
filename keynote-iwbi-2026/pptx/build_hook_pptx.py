#!/usr/bin/env python3
"""Hook section — editable PPTX. 6 slides reproducing slides/hook.html.
One text box per blurb; native shapes for cards/steps; mammogram canvas rendered
as a dark viewport placeholder (HTML uses a <canvas>, no embedded image data)."""
import sys, os
sys.path.insert(0, "/home/user/Mammo/keynote-iwbi-2026/pptx")
from pptxlib import *
from bs4 import BeautifulSoup
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

SLIDES = "/home/user/Mammo/keynote-iwbi-2026/slides"
H = open(SLIDES + "/hook.html").read()
soup = BeautifulSoup(H, 'html.parser')
secs = soup.select('section.slide')
notes = [s.get('data-note', '') for s in secs]

prs = new_prs()


def viewport(s, l, t, w, h, head_left="R MLO · screening", head_right="AI —"):
    """Dark mammogram 'viewport' panel: header bar + black body placeholder."""
    panel = card(s, l, t, w, h, fill=PANEL, edge=AMBERDP, edge_w=0.75)
    _, tf = box(s, l + 0.12, t + 0.06, w - 0.24, 0.28, anchor=MSO_ANCHOR.MIDDLE)
    run(para(tf, True), head_left, 9, INK2, font=MONO)
    _, tf = box(s, l + 0.12, t + 0.06, w - 0.24, 0.28, anchor=MSO_ANCHOR.MIDDLE)
    run(para(tf, True, align=PP_ALIGN.RIGHT), head_right, 9, CYAN, font=MONO)
    g = rect(s, l + 0.08, t + 0.42, w - 0.16, 0.01, LINE); g.height = Pt(1)
    rect(s, l + 0.12, t + 0.52, w - 0.24, h - 0.66, BLACK)
    _, tf = box(s, l + 0.18, t + h - 0.42, w - 0.36, 0.3)
    run(para(tf, True), "[ de-identified Emory screening case ]", 8, INK3, font=MONO)
    return panel


# ============ SLIDE 1 — title card ============
s = slide(prs); note(s, notes[0])
_, tf = box(s, 0.85, 0.95, 11.6, 0.4)
run(para(tf, True), "International Workshop on Breast Imaging · 2026", 12, CYAN, bold=True, font=MONO)
_, tf = box(s, 0.85, 2.35, 11.6, 2.2)
p = para(tf, True, line=0.98)
run(p, "Thirty Years,", 54, INK, bold=True)
p2 = para(tf, line=0.98)
run(p2, "Three Eras.", 54, AMBER, bold=True)
_, tf = box(s, 0.85, 4.95, 9.5, 0.9)
run(para(tf, True, line=1.3), "AI in Breast Imaging — from CAD to Clinical Intelligence", 22, INK2)
_, tf = box(s, 0.85, 6.1, 11.6, 0.5)
p = para(tf, True)
run(p, "Hari Trivedi, MD", 14, INK, bold=True, font=MONO)
run(p, "  ·  Emory University", 14, INK3, font=MONO)

# ============ SLIDE 2 — open: normal mammogram ============
s = slide(prs); note(s, notes[1])
viewport(s, 0.85, 1.15, 4.1, 5.5, head_right="AI —")
_, tf = box(s, 5.55, 2.6, 7.0, 2.4)
run(para(tf, True, line=1.1), "A normal screening mammogram — read as normal.", 34, INK, bold=True)

# ============ SLIDE 3 — anchor: what are we not seeing ============
s = slide(prs); note(s, notes[2])
_, tf = box(s, 0.85, 2.4, 11.0, 2.6, anchor=MSO_ANCHOR.MIDDLE)
p = para(tf, True, line=1.05)
run(p, "What are we ", 56, INK, bold=True)
run(p, "not seeing?", 56, AMBER, bold=True)

# ============ SLIDE 4 — report: AI reads same mammogram ============
s = slide(prs); note(s, notes[3])
viewport(s, 0.85, 1.0, 3.55, 5.6, head_right="AI ✓✓✓")
RX = 4.95; RW = 7.55
_, tf = box(s, RX, 0.95, RW, 1.15)
p = para(tf, True, line=1.12)
run(p, "AI reads this ", 19, INK, bold=True)
run(p, "same mammogram", 19, AMBER, bold=True)
run(p, " — and her eventual biopsy slide — for information not visible to us.", 19, INK, bold=True)

findings = [
    ("01", "Image-based risk", " — high probability of cancer within ~5 years",
     "which she developed"),
    ("02", "Cardiovascular disease", " — breast arterial calcification",
     "an independent risk marker, with a dose-response relationship"),
    ("03", "Recurrence prediction", " — a genomic recurrence score from the biopsy slide",
     None),
]
fy = 2.25
for ix, vlabel, rest, sub in findings:
    fh = 1.0 if sub else 0.72
    c = card(s, RX, fy, RW, fh, fill=CARD, edge=AMBER, edge_w=1.0)
    rect(s, RX, fy, 0.045, fh, AMBER)
    tf = c.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.22); tf.margin_right = Inches(0.18)
    tf.margin_top = Inches(0.08); tf.margin_bottom = Inches(0.08)
    p = para(tf, True, line=1.12)
    run(p, ix + "   ", 11, AMBER, bold=True, font=MONO)
    run(p, vlabel, 14, CYAN, bold=True)
    run(p, rest, 14, INK, bold=True)
    if sub:
        p2 = para(tf, before=3, line=1.15)
        run(p2, sub, 11, INK2)
    fy += fh + 0.16
_, tf = box(s, RX, fy + 0.02, RW, 0.9)
p = para(tf, True, line=1.3)
run(p, "None of these is hypothetical. Each is published and externally validated — and ", 12, INK2)
run(p, "one is already in this year's screening guidelines.", 12, INK, bold=True)

# ============ SLIDE 5 — pivot ============
s = slide(prs); note(s, notes[4])
_, tf = box(s, 0.85, 2.15, 1.0, 0.5)
run(para(tf, True), "~30 YRS", 11, INK3, font=MONO)
_, tf = box(s, 2.05, 1.9, 10.4, 1.6)
p = para(tf, True, line=1.18)
run(p, "We built tools to ", 30, INK3, bold=True)
run(p, "find the lesion", 30, INK3, bold=True)
run(p, " — to mark the spot. That was CAD.", 30, INK3, bold=True)
g = rect(s, 0.85, 3.95, 11.6, 0.01, LINE); g.height = Pt(1)
_, tf = box(s, 0.85, 4.45, 1.0, 0.5)
run(para(tf, True), "NOW", 11, INK3, font=MONO)
_, tf = box(s, 2.05, 4.25, 10.4, 1.6)
p = para(tf, True, line=1.18)
run(p, "What's changing is the ", 30, INK, bold=True)
run(p, "scope of the question", 30, AMBER, bold=True)
run(p, " we ask.", 30, INK, bold=True)

# ============ SLIDE 6 — spine: from lesions to patients ============
s = slide(prs); note(s, notes[5])
_, tf = box(s, 0.85, 1.5, 11.6, 1.2)
p = para(tf, True, line=1.04)
run(p, "From lesions to ", 44, INK, bold=True)
run(p, "patients.", 44, AMBER, bold=True)

steps = [
    ("01", "Past", "Lesion", "find & characterize"),
    ("02", "Present", "Image", "risk & opportunistic signal"),
    ("03", "Future", "Patient", "pathology + integration"),
]
fills = [CARD, RGBColor(0x3A, 0x2E, 0x16), AMBER]
edges = [LINE, AMBERDP, AMBERDP]
sx_l = 0.85
sw_w = 3.75
gap = 0.275
sy = 3.3
sh = 2.5
for i, (ix, era, word, desc) in enumerate(steps):
    l = sx_l + i * (sw_w + gap)
    solid = (i == 2)
    c = card(s, l, sy, sw_w, sh, fill=fills[i], edge=edges[i], edge_w=1.25)
    rect(s, l + 0.25, sy + 0.22, 0.45, 0.035, BLACK if solid else AMBER)
    tf = c.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = Inches(0.28); tf.margin_right = Inches(0.24)
    tf.margin_top = Inches(0.45); tf.margin_bottom = Inches(0.2)
    ink_word = RGBColor(0x1A, 0x12, 0x06) if solid else INK
    ink_era = RGBColor(0x1A, 0x12, 0x06) if solid else AMBER
    ink_desc = RGBColor(0x2A, 0x1E, 0x0A) if solid else INK2
    ink_ix = RGBColor(0x40, 0x30, 0x14) if solid else INK3
    run(para(tf, True), ix, 11, ink_ix, font=MONO)
    p = para(tf, before=4)
    run(p, era.upper(), 10, ink_era, bold=True, font=MONO)
    p = para(tf, before=6)
    run(p, word, 26, ink_word, bold=True)
    p = para(tf, before=6, line=1.25)
    run(p, desc, 12.5, ink_desc)

# ---- 7 step into a time machine ----
import re as _re
_H = open("/home/user/Mammo/keynote-iwbi-2026/slides/hook.html").read()
_svgs = _re.findall(r'<svg[\s\S]*?</svg>', _H)
s = slide(prs)
note(s, "To understand where this is going, we have to start with where it has been — because this field has overpromised before. So let's step into a time machine and go back to the beginning, to 1998.")
_, tf = box(s, 1.0, 1.25, 11.3, 1.1)
p = para(tf, True, align=PP_ALIGN.CENTER)
run(p, "Let's step into a ", 40, INK, bold=True); run(p, "time machine.", 40, AMBER, bold=True)
if _svgs: embed_svg(s, _svgs[-1], 4.15, 2.5, 5.0, 3.8, center=True)
_, tf = box(s, 1.0, 6.55, 11.3, 0.5)
run(para(tf, True, align=PP_ALIGN.CENTER), "This field has overpromised before. Let's start there — in 1998.", 14, INK2)

out = "/home/user/Mammo/keynote-iwbi-2026/pptx/IWBI2026_Trivedi_00_hook.pptx"
save(prs, out)
