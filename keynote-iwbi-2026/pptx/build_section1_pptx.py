#!/usr/bin/env python3
"""Era I (CAD) — editable PPTX, hand-tuned to match slides/section1.html
(current 7 visible slides; the evidence/Table-2 slide is hidden in final and
dropped here). One text box per blurb; native cards; the adoption curve and
Mahoney bar chart are embedded from the HTML's own SVGs; CAD mammogram embedded."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pptxlib import embed_img, embed_svg
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from bs4 import BeautifulSoup
import re

BG    = RGBColor(0x0F,0x14,0x1A); PANEL = RGBColor(0x06,0x09,0x0C)
INK   = RGBColor(0xEC,0xEF,0xF3); INK2 = RGBColor(0x9D,0xA9,0xB5); INK3 = RGBColor(0x5C,0x69,0x75)
AMBER = RGBColor(0xE7,0xAC,0x51); AMBERDP = RGBColor(0xC9,0x8A,0x2E); CYAN = RGBColor(0x5F,0xB7,0xC9)
WARN  = RGBColor(0xD9,0x78,0x5B); CARD = RGBColor(0x1B,0x23,0x2D); LINE = RGBColor(0x2A,0x33,0x3D)
SANS, MONO = "Arial", "Consolas"

SLIDES = "/home/user/Mammo/keynote-iwbi-2026/slides"
soup = BeautifulSoup(open(SLIDES + "/section1.html").read(), 'html.parser')
SECS = soup.select('section.slide')
NOTES = [s.get('data-note', '') for s in SECS]
def svg_of(i):
    sv = SECS[i].find('svg'); return str(sv) if sv is not None else None
def img_of(i):
    for im in SECS[i].find_all('img'):
        if im.get('src', '').startswith('data:'): return im['src']
    return None

prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

def slide(note_idx=None):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid(); s.background.fill.fore_color.rgb = BG
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(6), Pt(3))
    bar.fill.solid(); bar.fill.fore_color.rgb = AMBER; bar.line.fill.background(); bar.shadow.inherit = False
    if note_idx is not None: s.notes_slide.notes_text_frame.text = NOTES[note_idx] or ""
    return s

def box(s, l, t, w, h, anchor=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tb, tf

def para(tf, first=False, align=PP_ALIGN.LEFT, before=0, line=1.0):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    if before: p.space_before = Pt(before)
    try: p.line_spacing = line
    except Exception: pass
    return p

def run(p, text, size, color, bold=False, font=SANS, italic=False):
    r = p.add_run(); r.text = text; f = r.font
    f.size = Pt(size); f.bold = bold; f.italic = italic; f.name = font; f.color.rgb = color
    return r

def eyebrow(s):
    _, tf = box(s, 0.85, 0.55, 6, 0.35)
    run(para(tf, True), "ERA I", 11, AMBER, bold=True, font=MONO)

def card(s, l, t, w, h, fill=CARD, edge=AMBER, edge_w=1.0):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(l), Inches(t), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = edge; sh.line.width = Pt(edge_w); sh.shadow.inherit = False
    return sh

def rect(s, l, t, w, h, color):
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(l), Inches(t), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = color; sh.line.fill.background(); sh.shadow.inherit = False
    return sh

# ============ 1 — divider: The year is 1998 ============
s = slide(0)
_, tf = box(s, 0.85, 0.72, 11.6, 0.4)
p = para(tf, True)
run(p, "ERA I · 1998 · LESION", 11, AMBER, bold=True, font=MONO)
run(p, "    →    IMAGE    →    PATIENT", 11, INK3, font=MONO)
_, tf = box(s, 0.85, 2.5, 7.4, 1.7)
p = para(tf, True); run(p, "The year is ", 50, INK, bold=True); run(p, "1998.", 50, AMBER, bold=True)
_, tf = box(s, 0.85, 4.35, 7.2, 2.2)
run(para(tf, True, line=1.3), "Computer-aided detection has just arrived — a mark on every mammogram flagging a possible cancer. A second set of eyes, automatically. It feels like the future.", 18, INK2)
di = img_of(0)
if di: embed_img(s, di, 8.55, 1.35, 4.0, 4.9, card_bg='black')

# ============ 2 — adoption timeline (embed HTML curve SVG) ============
s = slide(1); eyebrow(s)
_, tf = box(s, 0.85, 1.2, 11.6, 1.1)
p = para(tf, True, line=1.05)
run(p, "Within a decade, CAD was used on ", 30, INK, bold=True); run(p, "most US mammograms.", 30, AMBER, bold=True)
sv = svg_of(1)
if sv: embed_svg(s, sv, 1.3, 2.45, 10.7, 3.5, center=True)
_, tf = box(s, 0.85, 6.2, 11.6, 0.8)
p = para(tf, True, line=1.3)
run(p, "A 2002 reimbursement code drove adoption — not evidence of benefit. ", 13, INK2)
run(p, "The definitive study landed in 2015", 13, INK, bold=True)
run(p, " — after CAD already ran on ~9 in 10 mammograms.", 13, INK2)

# ============ 3 — false positives + Mahoney bar chart (embed SVG) ============
s = slide(2); eyebrow(s)
_, tf = box(s, 0.85, 1.2, 5.4, 1.6)
p = para(tf, True, line=1.12)
run(p, "Conventional CAD put a ", 24, INK, bold=True)
run(p, "false-positive mark", 24, WARN, bold=True)
run(p, " on most normal exams.", 24, INK, bold=True)
_, tf = box(s, 0.85, 3.15, 5.4, 1.0)
run(para(tf, True), "70–85%", 38, AMBER, bold=True)
_, tf = box(s, 0.85, 3.98, 5.2, 0.7)
p = para(tf, True, line=1.25); run(p, "of normal mammograms had ", 12, INK); run(p, "≥1 false positive", 12, WARN, bold=True)
_, tf = box(s, 0.85, 4.75, 5.4, 1.0)
run(para(tf, True), "0.5–1.5", 38, AMBER, bold=True)
_, tf = box(s, 0.85, 5.58, 5.2, 0.8)
p = para(tf, True, line=1.25); run(p, "false positives ", 12, WARN, bold=True); run(p, "per image — about 2–4 per 4-view case", 12, INK)
_, tf = box(s, 0.85, 6.5, 6.0, 0.4)
run(para(tf, True), "Kim 2009 · Mahoney 2011 · Watanabe 2019 · Leon 2009", 10, INK3, font=MONO)
sv = svg_of(2)
if sv: embed_svg(s, sv, 6.7, 1.4, 6.0, 4.4, center=True)
_, tf = box(s, 6.7, 5.95, 6.0, 0.8)
run(para(tf, True, align=PP_ALIGN.CENTER, line=1.3), "Mean false-positive marks per case · ImageChecker v7.2 — Mahoney & Meganathan, J Digit Imaging 2011 (Table 4) · PMC3180536.", 9, INK3, font=MONO)

# ============ 4 — reckoning ($400M / −47%) ============
s = slide(3); eyebrow(s)
_, tf = box(s, 1.4, 2.2, 5.0, 1.4)
p = para(tf, True); run(p, "$400M", 66, AMBER, bold=True); run(p, " /yr", 26, INK2)
_, tf = box(s, 1.4, 3.75, 4.6, 0.9)
run(para(tf, True, line=1.3), "U.S. spend ≈ $1 of every $10,000 in health care", 12, INK, font=MONO)
_, tf = box(s, 7.1, 2.2, 5.2, 1.4)
p = para(tf, True); run(p, "−47%", 66, AMBER, bold=True); run(p, " odds", 22, INK3)
_, tf = box(s, 7.1, 3.75, 5.2, 1.1)
run(para(tf, True, line=1.3), "odds of detecting a malignant lesion — CAD on vs off, same readers (OR 0.53)", 12, INK, font=MONO)
_, tf = box(s, 1.4, 5.1, 10.6, 0.8)
run(para(tf, True, line=1.3), "When the definitive study finally ran, CAD added cost and, for the same readers, lowered the odds of catching a cancer.", 18, INK2)
_, tf = box(s, 1.4, 6.1, 10.6, 0.4)
run(para(tf, True), "Lehman et al., JAMA Internal Medicine 2015 · 323,973 women · 271 radiologists", 10, INK3, font=MONO)

# ============ 5 — verdict quote ============
s = slide(4); eyebrow(s)
_, tf = box(s, 0.85, 1.5, 7.6, 3.4)
run(para(tf, True, line=1.25), "“Computer-aided detection does not improve diagnostic accuracy of mammography. These results suggest that insurers pay more for CAD with no established benefit to women.”", 25, INK, bold=True)
_, tf = box(s, 0.85, 5.25, 7.6, 0.5)
run(para(tf, True), "— Lehman et al., JAMA Internal Medicine 2015", 13, AMBER, font=MONO)
side = [("85.3% vs 87.3%", "sensitivity — with vs without CAD"),
        ("4.1 = 4.1 /1,000", "cancer detection — identical"),
        ("0.871 vs 0.919", "AUC — with vs without (Fenton, NEJM 2007)")]
sy = 1.8
for big, lab in side:
    _, tf = box(s, 9.0, sy, 3.5, 0.55); run(para(tf, True), big, 18, INK, bold=True)
    _, tf = box(s, 9.0, sy + 0.5, 3.5, 0.6); run(para(tf, True, line=1.15), lab, 10.5, INK3, font=MONO)
    sy += 1.35

# ============ 6 — what went wrong ============
s = slide(6); eyebrow(s)
_, tf = box(s, 0.85, 1.05, 11.6, 0.8); run(para(tf, True), "What went wrong.", 30, INK, bold=True)
reasons = [
    ("01", "It flagged only what we can already describe.",
     [("Hand-tuned features", True), (" are limited to what we can design — so it added marks, not new information.", False)]),
    ("02", "Readers learned to ignore it.",
     [("70–85% of normal exams", True), (" carried at least one false-positive mark — eroding trust.", False)]),
    ("03", "It was deployed at scale before it was ever validated.",
     [("Reimbursed in 2002 and run on ", False), ("~92% of US screening mammograms", True),
      (" by 2016; the definitive accuracy study didn't arrive until 2015 — and found no benefit.", False)]),
]
ry = 2.05
for ix, head, subruns in reasons:
    c = card(s, 0.85, ry, 11.6, 1.2, fill=CARD, edge=AMBER, edge_w=1.0)
    tf = c.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.25); tf.margin_right = Inches(0.25)
    p = para(tf, True); run(p, ix + "   ", 13, AMBER, bold=True, font=MONO); run(p, head, 17, INK, bold=True)
    p2 = para(tf, before=3, line=1.22)
    for t, b in subruns: run(p2, t, 12.5, INK if b else INK2, bold=b)
    ry += 1.36
_, tf = box(s, 0.85, ry + 0.02, 11.6, 0.4)
run(para(tf, True), "Lehman CD, et al. JAMA Intern Med 2015;175(11):1828–1837 · Keen JD, et al. J Am Coll Radiol 2018 (adoption ~92%)", 9.5, INK3, font=MONO)

# ============ 7 — a paradigm shift (turn-grid) ============
s = slide(7); eyebrow(s)
_, tf = box(s, 0.85, 1.5, 11.6, 0.9)
p = para(tf, True); run(p, "A paradigm ", 30, INK, bold=True); run(p, "shift.", 30, AMBER, bold=True)
cw, gap, cy, ch = 5.4, 0.8, 2.7, 3.6
# left (old)
c = card(s, 0.85, cy, cw, ch, fill=CARD, edge=LINE, edge_w=1.0)
tf = c.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.3); tf.margin_right = Inches(0.26); tf.margin_top = Inches(0.28)
run(para(tf, True), "TRADITIONAL CAD", 11, INK3, bold=True, font=MONO)
p = para(tf, before=8); run(p, "Programmed", 24, INK2, bold=True)
p = para(tf, before=8, line=1.3)
run(p, "We wrote the rules for what cancer looks like. It ", 13, INK2)
run(p, "pointed", 13, INK2, italic=True); run(p, " at spots for a human to check.", 13, INK2)
# right (new)
c = card(s, 0.85 + cw + gap, cy, cw, ch, fill=CARD, edge=AMBERDP, edge_w=1.25)
tf = c.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.3); tf.margin_right = Inches(0.26); tf.margin_top = Inches(0.28)
run(para(tf, True), "MODERN DEEP LEARNING", 11, AMBER, bold=True, font=MONO)
p = para(tf, before=8); run(p, "Trained", 24, INK, bold=True)
p = para(tf, before=8, line=1.3)
run(p, "The model learns the patterns from the images themselves. It can ", 13, INK2)
run(p, "read", 13, INK, italic=True); run(p, " the image — as a second reader, or on its own.", 13, INK2)
# arrow
_, tf = box(s, 0.85 + cw, cy + ch / 2 - 0.35, gap, 0.7, anchor=MSO_ANCHOR.MIDDLE)
run(para(tf, True, align=PP_ALIGN.CENTER), "→", 28, AMBER)

out = "/home/user/Mammo/keynote-iwbi-2026/pptx/sections/IWBI2026_Trivedi_01_CAD.pptx"
os.makedirs(os.path.dirname(out), exist_ok=True)
prs.save(out)
print("saved", out, "slides:", len(prs.slides._sldIdLst))
