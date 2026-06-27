#!/usr/bin/env python3
"""Build Era III risk deck (From Detection to Risk) as editable PPTX — one text box per blurb.
AsymMirai/Mirai architecture redrawn as labeled rounded rects + arrows; placeholder figure slot
reproduced as dashed-bordered rectangle. 9 slides, none hidden. No embedded photos in source."""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.oxml.ns import qn

BG    = RGBColor(0x0F,0x14,0x1A); PANEL = RGBColor(0x06,0x09,0x0C)
INK   = RGBColor(0xEC,0xEF,0xF3); INK2 = RGBColor(0x9D,0xA9,0xB5); INK3 = RGBColor(0x5C,0x69,0x75)
AMBER = RGBColor(0xE7,0xAC,0x51); AMBERDP = RGBColor(0xC9,0x8A,0x2E); CYAN = RGBColor(0x5F,0xB7,0xC9)
WARN  = RGBColor(0xD9,0x78,0x5B); CARD = RGBColor(0x1B,0x23,0x2D); LINE = RGBColor(0x2A,0x33,0x3D)
PURPLE= RGBColor(0x8E,0x7F,0xC0)
SANS, MONO = "Arial", "Consolas"

HERE = os.path.dirname(os.path.abspath(__file__))
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

def eyebrow(s, txt="Era III"):
    _, tf = box(s, 0.85, 0.55, 9, 0.35)
    run(para(tf, True), txt.upper(), 11, AMBER, bold=True, font=MONO)

def notes(s, txt):
    s.notes_slide.notes_text_frame.text = txt

def rect(s, l, t, w, h, color):
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(l), Inches(t), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = color; sh.line.fill.background(); sh.shadow.inherit = False
    return sh

def chip(s, l, t, w, h, parts):
    c = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(l), Inches(t), Inches(w), Inches(h))
    c.fill.solid(); c.fill.fore_color.rgb = CARD; c.line.color.rgb = CYAN; c.line.width = Pt(1.0)
    c.shadow.inherit = False
    tf = c.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.12); tf.margin_right = Inches(0.1); tf.margin_top = Inches(0.04); tf.margin_bottom = Inches(0.04)
    p = para(tf, True, line=1.05)
    for txt, col, bold in parts:
        run(p, txt, 10.5, col, bold=bold, font=MONO)
    return c

def labeled_box(s, l, t, w, h, txt, fill, edge, tsize=9, tcol=INK):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(l), Inches(t), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = fill; sh.line.color.rgb = edge; sh.line.width = Pt(1.0)
    sh.shadow.inherit = False
    tf = sh.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.04); tf.margin_right = Inches(0.04); tf.margin_top = Inches(0.02); tf.margin_bottom = Inches(0.02)
    run(para(tf, True, align=PP_ALIGN.CENTER), txt, tsize, tcol)
    return sh

def conn(s, x1, y1, x2, y2, color=INK3, w=1.2, arrow=True):
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    ln.line.color.rgb = color; ln.line.width = Pt(w); ln.shadow.inherit = False
    if arrow:
        lnEl = ln.line._get_or_add_ln()
        tail = lnEl.makeelement(qn('a:tailEnd'), {'type': 'triangle', 'w': 'med', 'len': 'med'})
        lnEl.append(tail)
    return ln

# blend two RGBColor toward dark (approx the translucent fills in the SVG)
def blend(c, frac=0.18):
    r = int(c[0]*frac + 0x10*(1-frac)); g = int(c[1]*frac + 0x16*(1-frac)); b = int(c[2]*frac + 0x1c*(1-frac))
    return RGBColor(r, g, b)

# ============ SLIDE 1 — divider: hidden signals ============
s = slide()
_, tf = box(s, 0.85, 0.72, 11, 0.4)
p = para(tf, True)
run(p, "LESION", 11, INK3, font=MONO); run(p, "  →  ", 11, INK3, font=MONO)
run(p, "IMAGE", 11, AMBER, bold=True, font=MONO); run(p, "  →  PATIENT  →  POPULATION", 11, INK3, font=MONO)
_, tf = box(s, 0.85, 2.1, 11.6, 1.4)
p = para(tf, True, line=1.02)
run(p, "Hidden signals ", 50, INK, bold=True); run(p, "in the image.", 50, AMBER, bold=True)
_, tf = box(s, 0.85, 3.85, 9.6, 1.6)
p = para(tf, True, line=1.4)
run(p, "A normal mammogram. Two readers, both right — no cancer today. But a model reads the same pixels and forecasts her five-year risk. The question changes: from ", 17, INK2)
run(p, "“is there cancer now?”", 17, INK, bold=True)
run(p, " to ", 17, INK2)
run(p, "“what is her risk, and how hard should we look?”", 17, INK, bold=True)
_, tf = box(s, 0.85, 5.65, 9.6, 1.0)
run(para(tf, True, line=1.4), "The mammogram stops being only a detector. It becomes a biosensor — and this one is already cleared by the FDA.", 14, INK3)
notes(s, "Come back to today. Same reading room, a normal mammogram — two readers, both right, no cancer. But one model just flagged her: about a five-year risk. It was not looking for a lesion; there wasn't one. It was reading the tissue. There were hidden signals in the image. That is the turn this section is about — the mammogram stops being only a detector and becomes a biosensor. And this one is already FDA-cleared.")

# ============ SLIDE 2 — the old world (quote) ============
s = slide(); eyebrow(s, "Era III · the old world")
bar = rect(s, 0.85, 2.0, 0.05, 2.0, AMBERDP)
_, tf = box(s, 1.1, 2.0, 7.0, 2.5)
p = para(tf, True, line=1.28)
run(p, "“For forty years, the only risk signal we could read off the image was ", 26, INK, bold=True)
run(p, "density", 26, AMBER, bold=True)
run(p, " — coarse, subjective, and a weak predictor.”", 26, INK, bold=True)
_, tf = box(s, 1.1, 4.7, 7.0, 0.5)
run(para(tf, True), "Tyrer-Cuzick, Gail, BCSC — questionnaire-and-density models", 11, AMBER, font=MONO)
# side stats
sx = 8.7
stats = [("~0.62", "Tyrer-Cuzick v8 · 5-yr AUC"),
         ("~0.59–0.61", "density-augmented Gail / TC")]
sy = 1.7
for n, l in stats:
    _, tf = box(s, sx, sy, 3.8, 0.6); run(para(tf, True), n, 22, INK, bold=True)
    _, tf = box(s, sx, sy + 0.55, 3.8, 0.5); run(para(tf, True), l, 10, INK3, font=MONO)
    sy += 1.35
_, tf = box(s, sx, sy, 3.8, 1.0)
p = para(tf, True, line=1.3)
run(p, "Strong at the population level.", 13, INK2)
p = para(tf, line=1.3); run(p, "Weak for the woman in front of you.", 13, INK2)
notes(s, "For forty years our only imaging risk marker was density — a coarse, subjective category. The questionnaire models we layered on top were weak: Tyrer-Cuzick predicts five-year risk at an AUC around 0.62, barely better than a coin flip at the individual level. We were sorting women into risk tiers with tools that could not really tell them apart.")

# ============ SLIDE 3 — the pivot: DL reads risk ============
s = slide(); eyebrow(s, "Era III · the pivot")
_, tf = box(s, 0.85, 1.15, 11.6, 1.0)
p = para(tf, True, line=1.1)
run(p, "Deep-learning models predict risk from the ", 28, INK, bold=True)
run(p, "mammogram itself.", 28, AMBER, bold=True)
_, tf = box(s, 0.85, 2.35, 11.6, 1.0)
p = para(tf, True, line=1.4)
run(p, "Yala et al. (Radiology 2019) showed a network reading the image itself out-predicted the classical models. ", 15, INK2)
run(p, "Mirai", 15, INK, bold=True)
run(p, " (Yala et al., Science Translational Medicine 2021) was built to predict across time points, tolerate missing risk-factor data, and stay consistent across scanners.", 15, INK2)
chip(s, 0.85, 3.6, 5.5, 0.62, [("C-index 0.76 / 0.81 / 0.79", CYAN, True), (" — validated US · Sweden · Taiwan", INK, False)])
chip(s, 6.6, 3.6, 5.9, 0.62, [("41.5%", CYAN, True), (" of future cancers flagged high-risk ", INK, False), ("(vs 22.9% Tyrer-Cuzick)", INK3, False)])
_, tf = box(s, 0.85, 4.5, 11.6, 0.9)
p = para(tf, True, line=1.35)
run(p, "But a number like 0.81 is a ", 14, INK3); run(p, "black box", 14, INK2, bold=True)
run(p, " answer. For a screening program, “it works” is not enough — we have to know ", 14, INK3)
run(p, "why", 14, INK2, bold=True); run(p, ".", 14, INK3)
_, tf = box(s, 0.85, 5.6, 11.6, 0.4)
run(para(tf, True), "Yala et al., Radiology 2019 · Mirai — Yala et al., Sci Transl Med 2021 (DOI 10.1126/scitranslmed.aba4373)", 10, INK3, font=MONO)
notes(s, "Then the pixels turned out to carry risk. Yala and colleagues at MIT and MGH showed a deep network reading the mammogram itself beat the questionnaire models. Mirai, the next version, was built to predict across time, tolerate missing risk-factor data, and stay consistent across machines. Externally validated in three countries, C-index 0.76 to 0.81. Among women who went on to develop cancer within five years, Mirai flagged 41 percent as high-risk — versus 23 percent for Tyrer-Cuzick.")

# ============ SLIDE 4 — AsymMirai vs Mirai architecture (redrawn) ============
s = slide(); eyebrow(s, "Era III · explainability done right")
_, tf = box(s, 0.85, 1.05, 11.6, 0.7)
p = para(tf, True, line=1.1)
run(p, "AsymMirai: much of the signal is ", 22, INK, bold=True)
run(p, "local bilateral asymmetry.", 22, AMBER, bold=True)
_, tf = box(s, 0.85, 1.85, 11.6, 0.75)
p = para(tf, True, line=1.3)
run(p, "Mirai is accurate but opaque. ", 12.5, INK2); run(p, "AsymMirai", 12.5, CYAN, bold=True)
run(p, " (Donnelly et al., Radiology 2024 — Duke + Emory) reverse-engineered what it keys on: the left-vs-right tissue difference. A transparent model on that signal alone nearly matched the black box.", 12.5, INK2)

# ----- LEFT pipeline: AsymMirai (interpretable) -----
lx = 0.85
labeled_box(s, lx, 2.7, 3.6, 0.32, "4 image tiles (L/R, CC + MLO)", PANEL, LINE, 9, INK2)
for i in range(4):
    labeled_box(s, lx + i*0.92, 3.2, 0.8, 0.32, "CNN", blend(CYAN), CYAN, 9, CYAN)
conn(s, lx+1.8, 3.05, lx+1.8, 3.18, INK3, 1.0)
labeled_box(s, lx, 3.72, 1.75, 0.34, "CC Asymmetry", blend(AMBER), AMBER, 9)
labeled_box(s, lx+1.85, 3.72, 1.75, 0.34, "MLO Asymmetry", blend(AMBER), AMBER, 9)
conn(s, lx+0.9, 3.54, lx+0.9, 3.7, INK3, 1.0)
conn(s, lx+2.7, 3.54, lx+2.7, 3.7, INK3, 1.0)
labeled_box(s, lx, 4.26, 1.75, 0.34, "Prediction Window", blend(AMBERDP), AMBERDP, 8.5)
labeled_box(s, lx+1.85, 4.26, 1.75, 0.34, "Prediction Window", blend(AMBERDP), AMBERDP, 8.5)
conn(s, lx+0.9, 4.08, lx+0.9, 4.24, INK3, 1.0)
conn(s, lx+2.7, 4.08, lx+2.7, 4.24, INK3, 1.0)
labeled_box(s, lx, 4.8, 3.6, 0.34, "Average", blend(INK2), INK2, 9)
conn(s, lx+1.8, 4.62, lx+1.8, 4.78, INK3, 1.0)
labeled_box(s, lx+1.0, 5.34, 1.6, 0.36, "Risk Score", BG, INK, 10)
conn(s, lx+1.8, 5.16, lx+1.8, 5.32, INK3, 1.0)
_, tf = box(s, lx, 5.85, 3.6, 0.3)
run(para(tf, True, align=PP_ALIGN.CENTER), "AsymMirai · interpretable", 10, AMBER, bold=True, font=MONO)

# divider
dl = rect(s, 4.75, 2.7, 0.012, 3.0, LINE); dl.width = Pt(1)

# ----- RIGHT pipeline: Mirai (black box) -----
rx = 5.0
labeled_box(s, rx, 2.7, 3.6, 0.32, "4 image tiles", PANEL, LINE, 9, INK2)
for i in range(4):
    labeled_box(s, rx + i*0.92, 3.18, 0.8, 0.3, "CNN", blend(CYAN), CYAN, 9, CYAN)
for i in range(4):
    labeled_box(s, rx + i*0.92, 3.62, 0.8, 0.3, "Maxpool", blend(PURPLE), PURPLE, 8, PURPLE)
labeled_box(s, rx, 4.06, 3.6, 0.32, "Transformer", blend(AMBER, 0.14), AMBERDP, 9.5)
labeled_box(s, rx, 4.5, 2.55, 0.32, "Risk Factor Predictor", blend(WARN), WARN, 8.5)
for i in range(5):
    labeled_box(s, rx + i*0.6, 4.94, 0.52, 0.3, "AHL", blend(WARN, 0.13), WARN, 8.5)
for i in range(5):
    labeled_box(s, rx + i*0.6, 5.38, 0.52, 0.38, "Yr %d" % (i+1), BG, INK, 8.5)
    conn(s, rx + i*0.6 + 0.26, 5.26, rx + i*0.6 + 0.26, 5.36, INK3, 0.8)
_, tf = box(s, rx, 5.9, 3.6, 0.3)
run(para(tf, True, align=PP_ALIGN.CENTER), "Mirai · black box", 10, INK2, font=MONO)

# attention inset (simplified)
ix = 9.0
inset = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(ix-0.05), Inches(2.7), Inches(3.45), Inches(3.05))
inset.fill.solid(); inset.fill.fore_color.rgb = blend(AMBER, 0.04); inset.line.color.rgb = AMBERDP; inset.line.width = Pt(1.0); inset.shadow.inherit = False
inset.line._get_or_add_ln().append(inset.line._get_or_add_ln().makeelement(qn('a:prstDash'), {'val':'dash'}))
_, tf = box(s, ix, 2.78, 3.3, 0.3); run(para(tf, True, align=PP_ALIGN.CENTER), "ASYMMETRY MODULE — DETAIL", 9, AMBER, font=MONO)
labeled_box(s, ix+1.0, 3.18, 1.3, 0.32, "Split", blend(PURPLE), PURPLE, 9, PURPLE)
labeled_box(s, ix+0.05, 3.74, 1.0, 0.3, "Linear", blend(AMBER,0.13), AMBER, 8.5)
labeled_box(s, ix+1.2, 3.74, 1.0, 0.3, "Linear", blend(AMBER,0.13), AMBER, 8.5)
labeled_box(s, ix+2.35, 3.74, 1.0, 0.3, "Linear", blend(AMBER,0.13), AMBER, 8.5)
labeled_box(s, ix+0.05, 4.3, 3.3, 0.32, "Attention", blend(AMBER), AMBER, 9.5)
labeled_box(s, ix+1.0, 4.86, 1.3, 0.32, "Linear", blend(WARN), WARN, 9)
conn(s, ix+1.65, 4.62, ix+1.65, 4.84, INK3, 1.0)
conn(s, ix+1.65, 3.5, ix+1.65, 3.72, INK3, 1.0)
_, tf = box(s, 0.85, 6.25, 11.6, 0.9)
p = para(tf, True, line=1.3)
run(p, "Redrawn in-theme from Donnelly et al., “AsymMirai,” Radiology 2024;310(3):e232780. ", 9.5, INK3, font=MONO)
run(p, "Interpretable AsymMirai 1-yr AUC 0.79 vs Mirai 0.84; 3-yr AUC 0.92 in stable tissue — a signal you can point to.", 9.5, CYAN, font=MONO)
notes(s, "Here is why explainability matters and what it looks like done right. On the right is Mirai: four images into CNNs, pooled, through a transformer and a risk-factor predictor, out come five years of risk. A black box. On the left is AsymMirai — our group's work with Duke. We asked what Mirai was actually keying on, and the answer was surprisingly simple: local bilateral dissimilarity, the difference between left and right breast tissue in the same region. A transparent model built only on that asymmetry nearly matched the black box: one-year AUC 0.79 versus 0.84. A signal you can point to on the image.")

# ============ SLIDE 5 — longitudinal figure slot (placeholder) ============
s = slide(); eyebrow(s, "Era III · the signal builds early")
_, tf = box(s, 0.85, 1.05, 11.6, 0.7)
p = para(tf, True, line=1.1)
run(p, "The asymmetry was present ", 22, INK, bold=True)
run(p, "years before", 22, AMBER, bold=True)
run(p, " the diagnosis.", 22, INK, bold=True)
_, tf = box(s, 0.85, 1.85, 11.6, 0.75)
run(para(tf, True, line=1.3), "One patient, left MLO, same region across three screening rounds. The model's attention localizes to a growing asymmetry the readers reasonably called normal — and that is where the cancer arose.", 12.5, INK2)
# dashed placeholder figure box (left)
ph = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.85), Inches(2.75), Inches(7.3), Inches(3.4))
ph.fill.solid(); ph.fill.fore_color.rgb = RGBColor(0x0B,0x11,0x16)
ph.line.color.rgb = AMBER; ph.line.width = Pt(1.0); ph.shadow.inherit = False
ph.line._get_or_add_ln().append(ph.line._get_or_add_ln().makeelement(qn('a:prstDash'), {'val':'dash'}))
tf = ph.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
tf.margin_left = Inches(0.3); tf.margin_right = Inches(0.3)
run(para(tf, True, align=PP_ALIGN.CENTER), "▢", 30, AMBER, font=MONO)
p = para(tf, before=6, align=PP_ALIGN.CENTER); run(p, "LONGITUDINAL CASE — 2014 / 2015 / 2017", 12, INK, font=MONO)
p = para(tf, before=6, align=PP_ALIGN.CENTER, line=1.4)
run(p, "drop the triptych here (L MLO, age 69 → 17 mo → 21 mo; red box on the localized asymmetry that became cancer)", 10, INK3, font=MONO)
_, tf = box(s, 0.85, 6.2, 7.3, 0.6)
run(para(tf, True, line=1.3), "Source: Donnelly et al., “AsymMirai,” Radiology 2024;310(3):e232780 · pubs.rsna.org/doi/10.1148/radiol.232780", 9, INK3, font=MONO)
# fnote (right)
nl = rect(s, 8.55, 2.85, 0.012, 3.0, LINE); nl.width = Pt(1)
_, tf = box(s, 8.8, 3.1, 3.7, 1.3)
p = para(tf, True, line=1.3); run(p, "Risk is a trajectory, not a snapshot.", 15, INK, bold=True)
p = para(tf, before=4, line=1.35); run(p, "The score was rising ", 12, INK2); run(p, "for years", 12, CYAN, bold=True); run(p, " before the diagnosis.", 12, INK2)
_, tf = box(s, 8.8, 4.7, 3.7, 1.4)
p = para(tf, True, line=1.3); run(p, "Same phenomenon as detection.", 15, INK, bold=True)
p = para(tf, before=4, line=1.35); run(p, "Hickman (§2): detection scores also climb across a decade ", 12, INK2); run(p, "before", 12, CYAN, bold=True); run(p, " we call it cancer.", 12, INK2)
notes(s, "And the signal is visible early. This is one patient, left MLO, the same region tracked over three screening rounds. In 2014, at age 69, the model's attention sits on a subtle asymmetry the readers called normal. Seventeen months later it intensifies. By 2017 it is unmistakable — and that is where the cancer arose. The risk score was rising for years before the diagnosis. This is the same phenomenon Hickman described in the detection models: the signal builds on the image long before we call it cancer.")

# ============ SLIDE 6 — the trap: population != individual ============
s = slide(); eyebrow(s, "Era III · the trap in a good AUC")
_, tf = box(s, 0.85, 1.15, 11.6, 1.0)
p = para(tf, True, line=1.1)
run(p, "A population AUC is not an ", 28, INK, bold=True)
run(p, "individual", 28, AMBER, bold=True)
run(p, " prediction.", 28, INK, bold=True)
_, tf = box(s, 0.85, 2.35, 11.6, 1.2)
p = para(tf, True, line=1.4)
run(p, "Discrimination ≠ calibration. ", 15, INK, bold=True)
run(p, "An AUC measures how well a model ranks a million women. Whether a predicted “8% risk” actually means 8% — calibration — is a separate property, less often reported, and it is the one that governs ", 15, INK2)
run(p, "her", 15, INK2, italic=True); run(p, " decision.", 15, INK2)
chip(s, 0.85, 3.7, 3.7, 0.62, [("~28%", CYAN, True), (" of future cancers in the AI top decile", INK, False)])
chip(s, 4.7, 3.7, 3.7, 0.62, [("~21%", CYAN, True), (" in the clinical-model top decile", INK, False)])
chip(s, 8.55, 3.7, 3.95, 0.62, [("the two high-risk groups ", INK, False), ("overlap only partially", CYAN, True)])
_, tf = box(s, 0.85, 4.65, 11.6, 0.8)
p = para(tf, True, line=1.35)
run(p, "The image model and the clinical model flag ", 15, INK2); run(p, "different women", 15, INK, bold=True)
run(p, ". That disagreement — not the average AUC — is the decision in front of you.", 15, INK2)
_, tf = box(s, 0.85, 5.65, 11.6, 0.4)
run(para(tf, True), "Arasu et al., Radiology 2023;307(5):e222733", 10, INK3, font=MONO)
notes(s, "Now the caveat, and it is the whole ballgame for using these clinically. Every number I just showed — 0.76, 0.81, 0.84 — is a population number. An AUC tells you how a model sorts a million women; it tells you little about the one in your clinic. Discrimination is not calibration: whether a predicted eight-percent risk really means eight percent is a separate, less-reported property. And the image model and the clinical model flag different women. Arasu showed the AI's top-risk decile and the clinical model's overlapped only partially. For a large share of women, the old number and the new number disagree — and that disagreement, not the average AUC, is the decision in front of you.")

# ============ SLIDE 7 — four-quadrant 2x2 ============
s = slide(); eyebrow(s, "Era III · the clinically relevant view")
_, tf = box(s, 0.85, 1.1, 11.6, 0.9)
p = para(tf, True, line=1.1)
run(p, "Image-based and clinical risk models often flag ", 24, INK, bold=True)
run(p, "different women.", 24, AMBER, bold=True)
# axis labels
_, tf = box(s, 3.4, 2.15, 4.0, 0.35); run(para(tf, True, align=PP_ALIGN.CENTER), "Image model · LOW", 10, INK3, font=MONO)
_, tf = box(s, 8.0, 2.15, 4.0, 0.35); run(para(tf, True, align=PP_ALIGN.CENTER), "Image model · HIGH", 10, INK3, font=MONO)
_, tf = box(s, 0.85, 3.45, 2.4, 0.35); run(para(tf, True), "Classical · HIGH", 10, INK3, font=MONO)
_, tf = box(s, 0.85, 5.55, 2.4, 0.35); run(para(tf, True), "Classical · LOW", 10, INK3, font=MONO)

def qcell(s, l, t, w, h, tag, tagcol, title, desc_parts, edge):
    c = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(l), Inches(t), Inches(w), Inches(h))
    c.fill.solid(); c.fill.fore_color.rgb = CARD; c.line.color.rgb = edge; c.line.width = Pt(1.5)
    c.shadow.inherit = False
    tf = c.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = Inches(0.16); tf.margin_right = Inches(0.16); tf.margin_top = Inches(0.12)
    run(para(tf, True), tag, 9, tagcol, font=MONO)
    p = para(tf, before=4, line=1.1); run(p, title, 13, INK, bold=True)
    p = para(tf, before=4, line=1.25)
    for txt, col, bold, it in desc_parts:
        run(p, txt, 10.5, col, bold=bold, italic=it)
    return c
# row 1 (Classical HIGH)
qcell(s, 3.4, 2.6, 4.2, 1.95, "⚠ discordant", WARN, "Germline risk, quiet image",
      [("BRCA / strong family history, low image score. The image was not trained on inherited risk — do ", INK2, False, False),
       ("not", INK, True, False), (" de-escalate. Highest-stakes error.", INK2, False, False)], WARN)
qcell(s, 7.75, 2.6, 4.2, 1.95, "✓ concordant high", CYAN, "Escalate",
      [("Supplemental MRI, short interval. Both axes agree.", INK2, False, False)], CYAN)
# row 2 (Classical LOW)
qcell(s, 3.4, 4.65, 4.2, 1.95, "✓ concordant low", CYAN, "Standardize / de-escalate",
      [("Routine screening, consider longer interval.", INK2, False, False)], CYAN)
qcell(s, 7.75, 4.65, 4.2, 1.95, "★ the new signal", AMBER, "“Average” by questionnaire, high by image",
      [("Tissue-state signal the form missed. Strongest near-term — ", INK2, False, False),
       ("look harder now", INK, True, False), (": short interval, supplemental imaging, a second look at ", INK2, False, False),
       ("this", INK2, False, True), (" exam.", INK2, False, False)], AMBER)
notes(s, "So picture the real decision as a two-by-two. The classical model on one axis, the image model on the other. When they agree, management is easy: both low, standardize or extend the interval; both high, escalate to MRI. The interesting cells are the disagreements. Classical-low but image-high: the tissue is doing something the questionnaire never captured — look harder now, short-interval follow-up, a second read of this mammogram. And the dangerous cell, classical-high but image-low: a BRCA carrier with a quiet-looking image. The reassuring image must not override known germline risk. De-escalating there would be the most consequential error this technology invites. The models measure different things — inherited risk versus current tissue state — and each carries its own action.")

# ============ SLIDE 8 — FDA-cleared + trials ============
s = slide(); eyebrow(s, "Era III · already cleared, already in trials")
_, tf = box(s, 0.85, 1.15, 11.6, 1.0)
p = para(tf, True, line=1.1)
run(p, "From research model to ", 28, INK, bold=True)
run(p, "FDA clearance", 28, AMBER, bold=True)
run(p, " and screening guidelines.", 28, INK, bold=True)
_, tf = box(s, 0.85, 2.35, 11.6, 0.9)
p = para(tf, True, line=1.4)
run(p, "Clairity Breast", 15, INK, bold=True)
run(p, " (FDA De Novo, June 2025) — the first tool to predict 5-year risk from a routine screening mammogram alone; added to the ", 15, INK2)
run(p, "2026 NCCN", 15, INK, bold=True); run(p, " screening guidance.", 15, INK2)
chip(s, 0.85, 3.5, 11.6, 0.7, [("WISDOM", CYAN, True), (" (JAMA 2026): risk-based non-inferior for stage ≥IIB · ~3,800 fewer mammograms /100k person-yr · ", INK, False), ("89%", CYAN, True), (" chose risk-based", INK, False)])
chip(s, 0.85, 4.35, 11.6, 0.7, [("MyPeBS", CYAN, True), (": ~85,000 women · 6 countries · screen-by-risk vs by-age · ", INK, False), ("ongoing", CYAN, True)])
_, tf = box(s, 0.85, 5.3, 11.6, 0.7)
run(para(tf, True, line=1.35), "Today the trials triage on genetics and classical models. The image-derived score is the second axis they have not yet switched on.", 13, INK3)
_, tf = box(s, 0.85, 6.15, 11.6, 0.6)
run(para(tf, True, line=1.3), "Clairity — FDA De Novo 2025 (no primary validation paper yet) · WISDOM — Esserman et al., JAMA 2026;335(9):763–774 · MyPeBS — BMC Cancer 2022", 9.5, INK3, font=MONO)
notes(s, "And this is no longer hypothetical. Clairity Breast — Constance Lehman's company — received FDA De Novo authorization in June 2025: the first tool to predict five-year risk from a routine screening mammogram alone, and it was added to the 2026 NCCN screening guidance. The path from research model to cleared product to guideline took under five years. Meanwhile the trials are testing whether to screen by risk instead of by age. WISDOM, reported in JAMA this year, found risk-based screening non-inferior for advanced cancers with about 3,800 fewer mammograms per hundred-thousand person-years — and 89 percent of women offered the choice picked it. MyPeBS, 85,000 women across six countries, will give the definitive answer.")

# ============ SLIDE 9 — bridge to §4 ============
s = slide(); eyebrow(s, "Era III · image → patient")
_, tf = box(s, 0.85, 1.3, 11.6, 1.0)
p = para(tf, True, line=1.1)
run(p, "The risk score guides ", 30, INK, bold=True)
run(p, "when and how closely", 30, AMBER, bold=True)
run(p, " to screen.", 30, INK, bold=True)
bar = rect(s, 0.85, 2.7, 0.05, 1.0, AMBER)
_, tf = box(s, 1.1, 2.7, 11.0, 1.0)
p = para(tf, True, line=1.4)
run(p, "Once we find something, the question changes again: from ", 18, INK)
run(p, "“is there cancer, and what is her risk?”", 18, AMBER, bold=True)
run(p, " to ", 18, INK)
run(p, "“what is this cancer, and what will it do?”", 18, AMBER, bold=True)
_, tf = box(s, 0.85, 4.0, 11.6, 0.4)
p = para(tf, True)
run(p, "LESION  →  ", 11, INK3, font=MONO); run(p, "IMAGE ✓", 11, AMBER, font=MONO)
run(p, "  →  ", 11, INK3, font=MONO); run(p, "PATIENT →", 11, AMBER, font=MONO); run(p, "  →  POPULATION", 11, INK3, font=MONO)
_, tf = box(s, 0.85, 4.7, 10.5, 1.2)
p = para(tf, True, line=1.45)
run(p, "That answer is not in radiology. It is across the hospital, on a glass slide. We move to the ", 18, INK2)
run(p, "pathology lab", 18, INK, bold=True)
run(p, " — where AI does something our images cannot.", 18, INK2)
notes(s, "So the image can now tell us when to look and how hard. But once we find something, the question changes again — from is there cancer and what is her risk, to what is this cancer and what will it do? That answer is not in radiology. It is across the hospital, on a glass slide. To understand the patient, we move to the pathology lab — where AI does something our images cannot.")

out = os.path.join(HERE, "IWBI2026_Trivedi_section3_risk.pptx")
prs.save(out)
print("saved", out, "slides:", len(prs.slides._sldIdLst))
