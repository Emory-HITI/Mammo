#!/usr/bin/env python3
"""Build Era I (CAD) deck as editable PPTX — one text box per blurb. Slide 6 hidden."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pptxlib import embed_img

BG    = RGBColor(0x0F,0x14,0x1A); PANEL = RGBColor(0x06,0x09,0x0C)
INK   = RGBColor(0xEC,0xEF,0xF3); INK2 = RGBColor(0x9D,0xA9,0xB5); INK3 = RGBColor(0x5C,0x69,0x75)
AMBER = RGBColor(0xE7,0xAC,0x51); AMBERDP = RGBColor(0xC9,0x8A,0x2E); CYAN = RGBColor(0x5F,0xB7,0xC9)
WARN  = RGBColor(0xD9,0x78,0x5B); CARD = RGBColor(0x1B,0x23,0x2D); LINE = RGBColor(0x2A,0x33,0x3D)
DARK  = RGBColor(0x1A,0x12,0x06)
SANS, MONO = "Arial", "Consolas"

prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

def slide(hidden=False):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid(); s.background.fill.fore_color.rgb = BG
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(6), Pt(3))
    bar.fill.solid(); bar.fill.fore_color.rgb = AMBER; bar.line.fill.background(); bar.shadow.inherit = False
    if hidden:
        try: s._element.set('show', '0')
        except Exception: pass
    return s

def box(s, l, t, w, h, anchor=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tb, tf

def para(tf, first=False, align=PP_ALIGN.LEFT, before=0, after=0, line=1.0):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    if before: p.space_before = Pt(before)
    if after:  p.space_after = Pt(after)
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

# ============ SLIDE 1 — divider: The year is 1998 ============
s = slide()
_, tf = box(s, 0.85, 0.72, 11.6, 0.4)
p = para(tf, True)
run(p, "ERA I · 1998 · LESION", 11, AMBER, bold=True, font=MONO)
run(p, "    →    IMAGE    →    PATIENT", 11, INK3, font=MONO)
_, tf = box(s, 0.85, 2.5, 7.4, 1.7)
p = para(tf, True); run(p, "The year is ", 50, INK, bold=True); run(p, "1998.", 50, AMBER, bold=True)
_, tf = box(s, 0.85, 4.35, 7.2, 2.2)
run(para(tf, True, line=1.3), "Computer-aided detection has just arrived — software that puts a mark on every mammogram to flag a possible cancer. A second set of eyes, automatically. It feels like the future.", 18, INK2)
# real CAD-marked mammogram from the HTML deck (base64), if present
import re as _re, base64 as _b64
_h = open("/home/user/Mammo/keynote-iwbi-2026/slides/section1.html").read()
_m = _re.search(r'src="(data:image/[^"]+)"', _h)
if _m:
    embed_img(s, _m.group(1), 8.55, 1.35, 4.0, 4.9, card_bg='black')
else:
    ph = card(s, 8.7, 1.5, 3.8, 4.6, fill=PANEL, edge=AMBERDP, edge_w=0.75)
    tf = ph.text_frame; tf.vertical_anchor = MSO_ANCHOR.BOTTOM; tf.word_wrap = True
    run(para(tf, True, align=PP_ALIGN.CENTER), "[ CAD-marked screening mammogram ]", 10, INK3, font=MONO)

# ============ SLIDE 2 — adoption timeline ============
s = slide(); eyebrow(s)
_, tf = box(s, 0.85, 1.25, 11.6, 1.1)
p = para(tf, True, line=1.05)
run(p, "Within a decade, CAD was used on ", 30, INK, bold=True); run(p, "most US mammograms.", 30, AMBER, bold=True)
# timeline line
ln = rect(s, 1.0, 4.05, 11.3, 0.02*72/72, LINE)  # thin line
ln.height = Pt(2)
miles = [
    (1.0,  "1998", "FDA clears first CAD\n(R2 ImageChecker)", AMBER),
    (3.7,  "2002", "CMS reimburses CAD\n— drives adoption", AMBER),
    (6.4,  "2008", "~74% of US screening\nmammograms use CAD", INK),
    (9.0,  "2016", "~92% — near-universal", INK),
    (10.9, "2015", "The verdict lands:\nno benefit", WARN),
]
for lx, yr, lab, col in miles:
    dot = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(lx-0.06), Inches(4.0), Inches(0.16), Inches(0.16))
    dot.fill.solid(); dot.fill.fore_color.rgb = col; dot.line.fill.background(); dot.shadow.inherit = False
    _, tf = box(s, lx-0.2, 3.2, 2.3, 0.7)
    run(para(tf, True), yr, 17, col, bold=True, font=MONO)
    _, tf = box(s, lx-0.2, 4.35, 2.3, 1.2)
    for i, lnz in enumerate(lab.split("\n")):
        run(para(tf, i==0, line=1.15), lnz, 11, INK2)
_, tf = box(s, 0.85, 6.25, 11.6, 0.7)
run(para(tf, True, line=1.3), "A 2002 reimbursement code drove adoption — not evidence of benefit. The definitive study landed in 2015, after CAD already ran on ~9 in 10 mammograms.", 13, INK2)

# ============ SLIDE 3 — false positives + Mahoney bar chart ============
s = slide(); eyebrow(s)
_, tf = box(s, 0.85, 1.2, 5.2, 1.6)
p = para(tf, True, line=1.12)
run(p, "Conventional CAD put a ", 24, INK, bold=True)
run(p, "false-positive mark", 24, WARN, bold=True)
run(p, " on most normal exams.", 24, INK, bold=True)
# stat 1
_, tf = box(s, 0.85, 3.15, 5.2, 1.2)
run(para(tf, True), "70–85%", 38, AMBER, bold=True)
_, tf = box(s, 0.85, 3.95, 5.0, 0.7)
p = para(tf, True, line=1.25); run(p, "of normal mammograms had ", 12, INK); run(p, "≥1 false positive", 12, WARN, bold=True)
# stat 2
_, tf = box(s, 0.85, 4.75, 5.2, 1.2)
run(para(tf, True), "0.5–1.5", 38, AMBER, bold=True)
_, tf = box(s, 0.85, 5.55, 5.0, 0.8)
p = para(tf, True, line=1.25); run(p, "false positives ", 12, WARN, bold=True); run(p, "per image — about 2–4 per 4-view case", 12, INK)
_, tf = box(s, 0.85, 6.5, 6.0, 0.4)
run(para(tf, True), "Kim 2009 · Mahoney 2011 · Watanabe 2019 · Leon 2009", 10, INK3, font=MONO)

# Mahoney grouped bar chart (drawn shapes), right side
cx0, ybase, per = 7.5, 5.55, 1.05   # x of first group baseline, y baseline, inches per mark
# y gridlines + labels (0..3)
for v in range(0,4):
    gy = ybase - v*per
    g = rect(s, 7.0, gy, 5.6, 0.01, LINE); g.height = Pt(1)
    _, tf = box(s, 6.5, gy-0.13, 0.45, 0.3); run(para(tf, True, align=PP_ALIGN.RIGHT), str(v), 9, INK3, font=MONO)
groups = [("All marks",2.6,1.8),("Masses",2.0,1.5),("Calcifications",0.6,0.3)]
gx = 7.3; gw = 1.75; bw = 0.55
for name, nd, dn in groups:
    for j,(val,col) in enumerate([(nd,AMBER),(dn,CYAN)]):
        bl = gx + j*(bw+0.05)
        h = val*per
        rect(s, bl, ybase-h, bw, h, col)
        _, tf = box(s, bl-0.1, ybase-h-0.32, bw+0.2, 0.3)
        run(para(tf, True, align=PP_ALIGN.CENTER), f"{val:.1f}", 11, INK, bold=True)
    _, tf = box(s, gx-0.25, ybase+0.08, gw+0.3, 0.4)
    run(para(tf, True, align=PP_ALIGN.CENTER), name, 9.5, INK2)
    gx += gw + 0.35
# legend
lg = rect(s, 9.7, 1.55, 0.18, 0.18, AMBER); _, tf = box(s, 9.95, 1.5, 1.2, 0.3); run(para(tf, True), "Non-dense", 9.5, INK2)
lg = rect(s, 11.1, 1.55, 0.18, 0.18, CYAN); _, tf = box(s, 11.35, 1.5, 1.0, 0.3); run(para(tf, True), "Dense", 9.5, INK2)
_, tf = box(s, 7.0, 6.05, 5.6, 0.5)
run(para(tf, True, align=PP_ALIGN.CENTER), "Mean false-positive marks per case · ImageChecker v7.2 — Mahoney & Meganathan, J Digit Imaging 2011 (Table 4)", 9, INK3, font=MONO)

# ============ SLIDE 4 — reckoning ($400M / -47%) ============
s = slide(); eyebrow(s)
# stat 1
_, tf = box(s, 1.4, 2.2, 5.0, 1.4)
p = para(tf, True); run(p, "$400M", 66, AMBER, bold=True); run(p, " /yr", 26, INK2)
_, tf = box(s, 1.4, 3.75, 4.6, 0.9)
run(para(tf, True, line=1.3), "U.S. spend ≈ $1 of every $10,000 in health care", 12, INK, font=MONO)
# stat 2
_, tf = box(s, 7.1, 2.2, 5.2, 1.4)
p = para(tf, True); run(p, "−47%", 66, AMBER, bold=True); run(p, " odds", 22, INK3)
_, tf = box(s, 7.1, 3.75, 5.2, 1.1)
run(para(tf, True, line=1.3), "odds of detecting a malignant lesion — CAD on vs off, same readers (OR 0.53)", 12, INK, font=MONO)
_, tf = box(s, 1.4, 5.1, 10.6, 0.8)
run(para(tf, True, line=1.3), "When the definitive study finally ran, CAD added cost and, for the same readers, lowered the odds of catching a cancer.", 18, INK2)
_, tf = box(s, 1.4, 6.1, 10.6, 0.4)
run(para(tf, True), "Lehman et al., JAMA Internal Medicine 2015 · 323,973 women · 271 radiologists", 10, INK3, font=MONO)

# ============ SLIDE 5 — verdict quote ============
s = slide(); eyebrow(s)
_, tf = box(s, 0.85, 1.5, 7.6, 3.4, anchor=MSO_ANCHOR.TOP)
run(para(tf, True, line=1.25), "“Computer-aided detection does not improve diagnostic accuracy of mammography. These results suggest that insurers pay more for CAD with no established benefit to women.”", 25, INK, bold=True)
_, tf = box(s, 0.85, 5.25, 7.6, 0.5)
run(para(tf, True), "— Lehman et al., JAMA Internal Medicine 2015", 13, AMBER, font=MONO)
side = [("85.3% vs 87.3%","sensitivity — with vs without CAD"),
        ("4.1 = 4.1 /1,000","cancer detection — identical"),
        ("0.871 vs 0.919","AUC — with vs without (Fenton, NEJM 2007)")]
sy = 1.8
for big, lab in side:
    _, tf = box(s, 9.0, sy, 3.5, 0.55); run(para(tf, True), big, 18, INK, bold=True)
    _, tf = box(s, 9.0, sy+0.5, 3.5, 0.6); run(para(tf, True, line=1.15), lab, 10.5, INK3, font=MONO)
    sy += 1.35

# ============ SLIDE 6 — evidence (HIDDEN) ============
s = slide(hidden=True); eyebrow(s)
_, tf = box(s, 7.0, 0.5, 5.3, 0.4)
run(para(tf, True, align=PP_ALIGN.RIGHT), "⚑ HIDDEN IN FINAL", 10, WARN, bold=True, font=MONO)
_, tf = box(s, 0.85, 1.15, 11.6, 0.9)
p = para(tf, True, line=1.08); run(p, "Same readers, same detection rate — ", 22, INK, bold=True); run(p, "with CAD or without.", 22, AMBER, bold=True)
# native table: Lehman Table 2 (condensed)
rows = [("Measure","CAD","No CAD","aOR","P"),
        ("Cancers detected /1000","4.1","4.1","0.99",".86"),
        ("Sensitivity, %","85.3","87.3","0.81",".18"),
        ("Specificity, %","91.6","91.4","1.02",".58"),
        ("Recall rate /100","8.7","9.1","0.96",".35")]
gf = s.shapes.add_table(len(rows), 5, Inches(0.85), Inches(2.4), Inches(8.2), Inches(2.6))
tbl = gf.table
tbl.columns[0].width = Inches(3.4)
for c in range(1,5): tbl.columns[c].width = Inches(1.2)
for ri,row in enumerate(rows):
    for ci,val in enumerate(row):
        cell = tbl.cell(ri,ci); cell.fill.solid(); cell.fill.fore_color.rgb = BG
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf = cell.text_frame; p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if ci==0 else PP_ALIGN.RIGHT
        r = p.add_run(); r.text = val; f = r.font; f.size = Pt(12); f.name = SANS
        f.bold = (ri==0); f.color.rgb = INK2 if ri==0 else (AMBER if ci==1 else INK)
_, tf = box(s, 0.85, 5.2, 11.6, 0.5)
run(para(tf, True), "Pooled ROC across 271 radiologists: pAUC 0.88 without CAD vs 0.84 with. No adjusted OR favors CAD.  Lehman 2015, Table 2.", 11, INK2)

# ============ SLIDE 7 — what went wrong ============
s = slide(); eyebrow(s)
_, tf = box(s, 0.85, 1.2, 11.6, 0.9); run(para(tf, True), "What went wrong.", 30, INK, bold=True)
reasons = [
    ("01","It targeted the cancers we already find well.","Hand-tuned rules aimed at cancers radiologists already catch — so it added marks, not new information."),
    ("02","Readers learned to ignore it.","70–85% of normal mammograms carried a false mark. After enough false alarms, you stop trusting the marks."),
    ("03","It could lower a good reader's sensitivity.","As an imperfect second reader it pulled attention toward its marks — the over-dependence Lehman described (OR 0.53)."),
]
ry = 2.35
for ix, head, sub in reasons:
    c = card(s, 0.85, ry, 11.6, 1.25, fill=CARD, edge=AMBER, edge_w=1.0)
    tf = c.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.25); tf.margin_right = Inches(0.25)
    p = para(tf, True); run(p, ix+"   ", 13, AMBER, bold=True, font=MONO); run(p, head, 17, INK, bold=True)
    p2 = para(tf, before=3); run(p2, sub, 12.5, INK2)
    ry += 1.45

# ============ SLIDE 8 — lesson + turn ============
s = slide(); eyebrow(s)
_, tf = box(s, 0.85, 1.2, 11.6, 0.8)
p = para(tf, True); run(p, "The lesson, and ", 28, INK, bold=True); run(p, "the turn.", 28, AMBER, bold=True)
# two cards
c = card(s, 0.85, 2.3, 5.3, 2.0, fill=CARD, edge=LINE, edge_w=1.0)
tf = c.text_frame; tf.word_wrap=True; tf.margin_left=Inches(0.22); tf.margin_right=Inches(0.22); tf.margin_top=Inches(0.18)
run(para(tf, True), "TRADITIONAL CAD", 11, INK3, font=MONO)
p=para(tf, before=6); run(p, "Programmed", 22, INK2, bold=True)
p=para(tf, before=6, line=1.25); run(p, "We wrote the rules for what cancer looks like. It pointed at spots for a human to check.", 13, INK2)
c = card(s, 7.15, 2.3, 5.3, 2.0, fill=CARD, edge=AMBERDP, edge_w=1.25)
tf = c.text_frame; tf.word_wrap=True; tf.margin_left=Inches(0.22); tf.margin_right=Inches(0.22); tf.margin_top=Inches(0.18)
run(para(tf, True), "MODERN DEEP LEARNING", 11, AMBER, font=MONO)
p=para(tf, before=6); run(p, "Trained", 22, INK, bold=True)
p=para(tf, before=6, line=1.25); run(p, "The model learns the patterns from the images. It can read the image — as a second reader, or on its own.", 13, INK2)
_, tf = box(s, 0.85, 4.75, 11.6, 1.8)
run(para(tf, True, line=1.3), "The lesson from Era I: CAD was reimbursed before it was proven, and never shown to help. For two decades we taught machines to point at mammograms; the change was teaching them to read the image — and validating that before billing for it. The next section picks up sixteen years later.", 16, INK)

import os
out = "/home/user/Mammo/keynote-iwbi-2026/pptx/IWBI2026_Trivedi_01_CAD.pptx"
os.makedirs(os.path.dirname(out), exist_ok=True)
prs.save(out)
print("saved", out, "slides:", len(prs.slides._sldIdLst))
