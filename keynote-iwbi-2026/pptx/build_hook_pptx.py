#!/usr/bin/env python3
"""Build the IWBI 2026 hook deck as an editable PPTX — one text box per blurb."""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ---- theme ----
BG      = RGBColor(0x0F,0x14,0x1A)
PANEL   = RGBColor(0x06,0x09,0x0C)
INK     = RGBColor(0xEC,0xEF,0xF3)
INK2    = RGBColor(0x9D,0xA9,0xB5)
INK3    = RGBColor(0x5C,0x69,0x75)
AMBER   = RGBColor(0xE7,0xAC,0x51)
AMBERDP = RGBColor(0xC9,0x8A,0x2E)
CYAN    = RGBColor(0x5F,0xB7,0xC9)
DARK    = RGBColor(0x1A,0x12,0x06)
LINE    = RGBColor(0x2A,0x33,0x3D)
SANS, MONO = "Arial", "Consolas"

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

def slide():
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    # thin amber accent hairline top-left
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(6), Pt(3))
    bar.fill.solid(); bar.fill.fore_color.rgb = AMBER; bar.line.fill.background()
    bar.shadow.inherit = False
    return s

def box(s, l, t, w, h, anchor=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tb, tf

def run(p, text, size, color, bold=False, font=SANS, italic=False):
    r = p.add_run(); r.text = text
    f = r.font; f.size = Pt(size); f.bold = bold; f.italic = italic
    f.name = font; f.color.rgb = color
    return r

def para(tf, first=False, align=PP_ALIGN.LEFT, space_after=0, space_before=0, line=1.0):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    if space_after:  p.space_after  = Pt(space_after)
    if space_before: p.space_before = Pt(space_before)
    try: p.line_spacing = line
    except Exception: pass
    return p

# ============ SLIDE 1 — title ============
s = slide()
_, tf = box(s, 0.85, 0.7, 11.6, 0.4)
run(para(tf, True), "INTERNATIONAL WORKSHOP ON BREAST IMAGING  ·  2026", 12, INK3, font=MONO)

_, tf = box(s, 0.85, 2.5, 11.6, 2.2)
p = para(tf, True, line=1.0); run(p, "Thirty Years,", 54, INK, bold=True)
p = para(tf, line=1.0);       run(p, "Three Eras.", 54, AMBER, bold=True)

_, tf = box(s, 0.85, 4.95, 11.6, 0.6)
run(para(tf, True), "AI in Breast Imaging — from CAD to Clinical Intelligence", 20, INK2)

_, tf = box(s, 0.85, 5.7, 11.6, 0.5)
p = para(tf, True); run(p, "Hari Trivedi, MD", 14, INK, bold=True); run(p, "    ·    Emory University", 14, INK3)

# ============ SLIDE 2 — the mammogram ============
s = slide()
ph = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.85), Inches(1.2), Inches(3.7), Inches(5.1))
ph.fill.solid(); ph.fill.fore_color.rgb = PANEL
ph.line.color.rgb = AMBERDP; ph.line.width = Pt(0.75); ph.shadow.inherit = False
tf = ph.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.BOTTOM
run(para(tf, True, align=PP_ALIGN.CENTER), "[ real de-identified Emory screening case ]", 10, INK3, font=MONO)

_, tf = box(s, 5.1, 2.6, 7.4, 2.2, anchor=MSO_ANCHOR.MIDDLE)
run(para(tf, True, line=1.08), "A normal screening mammogram — read as normal.", 30, INK, bold=True)

# ============ SLIDE 3 — what are we not seeing ============
s = slide()
_, tf = box(s, 1.0, 2.6, 11.3, 2.2, anchor=MSO_ANCHOR.MIDDLE)
p = para(tf, True, align=PP_ALIGN.LEFT)
run(p, "What are we ", 48, INK, bold=True); run(p, "not seeing?", 48, AMBER, bold=True)

# ============ SLIDE 4 — three models ============
s = slide()
ph = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.85), Inches(1.5), Inches(3.3), Inches(4.6))
ph.fill.solid(); ph.fill.fore_color.rgb = PANEL
ph.line.color.rgb = AMBERDP; ph.line.width = Pt(0.75); ph.shadow.inherit = False

_, tf = box(s, 4.5, 0.95, 8.0, 1.2)
p = para(tf, True, line=1.1)
run(p, "AI reads this ", 19, INK, bold=True)
run(p, "same mammogram", 19, AMBER, bold=True)
run(p, " — and her eventual biopsy slide — for information not visible to us.", 19, INK, bold=True)

findings = [
    ("01", "Image-based risk", " — high probability of cancer within ~5 years", "which she developed"),
    ("02", "Cardiovascular disease", " — breast arterial calcification", "an independent risk marker, with a dose-response relationship"),
    ("03", "Recurrence prediction", " — a genomic recurrence score from the biopsy slide", None),
]
ty = 2.35
for ix, cat, rest, sub in findings:
    h = 0.95
    bx = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(4.5), Inches(ty), Inches(8.0), Inches(h))
    bx.fill.solid(); bx.fill.fore_color.rgb = RGBColor(0x1B,0x23,0x2D)
    bx.line.color.rgb = AMBER; bx.line.width = Pt(1.0); bx.shadow.inherit = False
    tf = bx.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.18); tf.margin_right = Inches(0.14)
    p = para(tf, True, line=1.05)
    run(p, ix + "   ", 12, AMBER, bold=True, font=MONO)
    run(p, cat, 15, CYAN, bold=True)
    run(p, rest, 15, INK)
    if sub:
        p2 = para(tf, space_before=2); run(p2, sub, 11.5, INK2, italic=True)
    ty += h + 0.12

_, tf = box(s, 4.5, 5.55, 8.0, 0.8)
p = para(tf, True, line=1.25)
run(p, "None of these is hypothetical. Each is published and externally validated — and ", 12, INK2)
run(p, "one is already in this year’s screening guidelines.", 12, INK, bold=True)

# ============ SLIDE 5 — the shift ============
s = slide()
_, tf = box(s, 0.85, 2.35, 1.7, 0.5)
run(para(tf, True), "~30 YRS", 12, INK3, font=MONO)
_, tf = box(s, 2.7, 2.05, 9.7, 1.2, anchor=MSO_ANCHOR.MIDDLE)
run(para(tf, True, line=1.1), "We built tools to find the lesion — to mark the spot. That was CAD.", 26, INK3, bold=True)

ln = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(2.7), Inches(3.55), Inches(9.7), Pt(1))
ln.fill.solid(); ln.fill.fore_color.rgb = LINE; ln.line.fill.background(); ln.shadow.inherit = False

_, tf = box(s, 0.85, 4.05, 1.7, 0.5)
run(para(tf, True), "NOW", 12, AMBER, font=MONO)
_, tf = box(s, 2.7, 3.75, 9.7, 1.2, anchor=MSO_ANCHOR.MIDDLE)
p = para(tf, True, line=1.1)
run(p, "What’s changing is the ", 26, INK, bold=True)
run(p, "scope of the question", 26, AMBER, bold=True)
run(p, " we ask.", 26, INK, bold=True)

# ============ SLIDE 6 — spine ============
s = slide()
_, tf = box(s, 0.85, 0.95, 11.6, 1.1)
p = para(tf, True)
run(p, "From lesions to ", 40, INK, bold=True); run(p, "populations.", 40, AMBER, bold=True)

steps = [
    ("01", "Lesion",     "find & characterize",        RGBColor(0x2B,0x2E,0x30), INK,  INK2, INK3),
    ("02", "Image",      "risk & opportunistic signal",RGBColor(0x48,0x41,0x35), INK,  INK2, INK3),
    ("03", "Patient",    "pathology + integration",    RGBColor(0x77,0x61,0x3D), INK,  INK2, INK3),
    ("04", "Population",  "access, equity, governance", AMBER,                  DARK, DARK, RGBColor(0x3a,0x2a,0x10)),
]
x = 0.85; w = 2.71; gap = 0.30; top = 2.7; h = 3.0
for ix, word, sub, fill, cword, csub, cix in steps:
    bx = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(top), Inches(w), Inches(h))
    bx.fill.solid(); bx.fill.fore_color.rgb = fill
    bx.line.color.rgb = AMBER; bx.line.width = Pt(1.0); bx.shadow.inherit = False
    tf = bx.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = Inches(0.2); tf.margin_right = Inches(0.16); tf.margin_top = Inches(0.22)
    run(para(tf, True), ix, 11, cix, font=MONO)
    p = para(tf, space_before=8); run(p, word, 26, cword, bold=True)
    p = para(tf, space_before=6); run(p, sub, 12, csub)
    x += w + gap

import os
out = "/home/user/Mammo/keynote-iwbi-2026/pptx/IWBI2026_Trivedi_hook.pptx"
os.makedirs(os.path.dirname(out), exist_ok=True)
prs.save(out)
print("saved", out, "slides:", len(prs.slides._sldIdLst))
