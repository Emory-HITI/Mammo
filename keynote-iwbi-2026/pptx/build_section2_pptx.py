#!/usr/bin/env python3
"""Build Era II (Modern AI 2020->2026) deck as editable PPTX — one text box per blurb.
Photos extracted from section2.html base64; bar/ROC charts redrawn as shapes; native tables.
10 slides, none hidden."""
import os, re, base64
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image

# ---- theme ----
BG    = RGBColor(0x0F,0x14,0x1A); PANEL = RGBColor(0x06,0x09,0x0C)
INK   = RGBColor(0xEC,0xEF,0xF3); INK2 = RGBColor(0x9D,0xA9,0xB5); INK3 = RGBColor(0x5C,0x69,0x75)
AMBER = RGBColor(0xE7,0xAC,0x51); AMBERDP = RGBColor(0xC9,0x8A,0x2E); CYAN = RGBColor(0x5F,0xB7,0xC9)
WARN  = RGBColor(0xD9,0x78,0x5B); CARD = RGBColor(0x1B,0x23,0x2D); LINE = RGBColor(0x2A,0x33,0x3D)
DARK  = RGBColor(0x1A,0x12,0x06)
SANS, MONO = "Arial", "Consolas"

HERE = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(HERE, "..", "slides", "section2.html")
TMP  = "/tmp/claude-0/-home-user-Mammo/6e3f6d6c-3e45-5480-8f5e-a0e6f5f1309e/scratchpad"
os.makedirs(TMP, exist_ok=True)

# ---- extract embedded images in document order ----
def extract_images():
    t = open(HTML).read()
    pat = re.compile(r'data:image/(png|jpeg|jpg);base64,([A-Za-z0-9+/=]+)')
    paths = []
    for i, m in enumerate(pat.finditer(t)):
        ext = m.group(1); ext = 'jpg' if ext == 'jpeg' else ext
        fn = os.path.join(TMP, 's2_img%d.%s' % (i, ext))
        open(fn, 'wb').write(base64.b64decode(m.group(2)))
        paths.append(fn)
    return paths
IMGS = extract_images()

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

def eyebrow(s, txt="Era II"):
    _, tf = box(s, 0.85, 0.55, 8, 0.35)
    run(para(tf, True), txt.upper(), 11, AMBER, bold=True, font=MONO)

def notes(s, txt):
    s.notes_slide.notes_text_frame.text = txt

def card(s, l, t, w, h, fill=CARD, edge=AMBER, edge_w=1.0):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(l), Inches(t), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = edge; sh.line.width = Pt(edge_w); sh.shadow.inherit = False
    return sh

def rect(s, l, t, w, h, color):
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(l), Inches(t), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = color; sh.line.fill.background(); sh.shadow.inherit = False
    return sh

def place_image(s, path, l, t, max_w, max_h, frame=True, foot=None, head=None):
    """Fit image into a box preserving aspect ratio; optional viewport frame + caption."""
    im = Image.open(path); iw, ih = im.size; ar = iw / ih
    box_ar = max_w / max_h
    if ar > box_ar:
        w = max_w; h = max_w / ar
    else:
        h = max_h; w = max_h * ar
    cx = l + (max_w - w) / 2; cy = t + (max_h - h) / 2
    if frame:
        fr = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(cx-0.06), Inches(cy-0.06), Inches(w+0.12), Inches(h+0.12))
        fr.fill.solid(); fr.fill.fore_color.rgb = PANEL; fr.line.color.rgb = AMBERDP; fr.line.width = Pt(0.75)
        fr.shadow.inherit = False
    s.shapes.add_picture(path, Inches(cx), Inches(cy), Inches(w), Inches(h))
    return cx, cy, w, h

def chip(s, l, t, w, h, parts):
    """parts: list of (text,color,bold). cyan left accent."""
    c = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(l), Inches(t), Inches(w), Inches(h))
    c.fill.solid(); c.fill.fore_color.rgb = CARD; c.line.color.rgb = CYAN; c.line.width = Pt(1.0)
    c.shadow.inherit = False
    tf = c.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.12); tf.margin_right = Inches(0.1); tf.margin_top = Inches(0.04); tf.margin_bottom = Inches(0.04)
    p = para(tf, True, line=1.05)
    for txt, col, bold in parts:
        run(p, txt, 10.5, col, bold=bold, font=MONO)
    return c

# native comparison table styled dark
def cmp_table(s, l, t, w, rows, col_w, ai_col=1):
    nrow = len(rows); ncol = len(rows[0])
    gf = s.shapes.add_table(nrow, ncol, Inches(l), Inches(t), Inches(w), Inches(0.42*nrow))
    tbl = gf.table
    for ci, cw in enumerate(col_w):
        tbl.columns[ci].width = Inches(cw)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci); cell.fill.solid(); cell.fill.fore_color.rgb = BG
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = Inches(0.08); cell.margin_right = Inches(0.08)
            cell.margin_top = Inches(0.02); cell.margin_bottom = Inches(0.02)
            tf = cell.text_frame; p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.RIGHT
            r = p.add_run(); r.text = val; f = r.font
            f.size = Pt(12.5); f.name = SANS
            if ri == 0:
                f.bold = True; f.color.rgb = INK2; f.name = MONO; f.size = Pt(10.5)
            elif ci == 0:
                f.color.rgb = INK2
            elif ci == ai_col:
                f.color.rgb = AMBER; f.bold = True
            else:
                f.color.rgb = INK
    return tbl

# ============ SLIDE 1 — divider: The year is 2020 ============
s = slide();
_, tf = box(s, 0.85, 0.72, 6, 0.4)
run(para(tf, True), "ERA II", 11, AMBER, bold=True, font=MONO)
_, tf = box(s, 0.85, 1.7, 6.4, 1.4)
p = para(tf, True); run(p, "The year is ", 50, INK, bold=True); run(p, "2020.", 50, AMBER, bold=True)
_, tf = box(s, 0.85, 3.35, 6.6, 1.6)
p = para(tf, True, line=1.4)
run(p, "You've spent four years hearing about AI — impressed, or worried about your job. The reading room is ", 15, INK2)
run(p, "still feeling the burn from CAD", 15, INK, bold=True)
run(p, "; some run CAD and AI in tandem, which only adds to the confusion.", 15, INK2)
_, tf = box(s, 0.85, 5.2, 6.6, 1.0)
run(para(tf, True, line=1.4), "So: what does mammography AI actually look like — and what is the evidence?", 15, INK3)
# image right (heatmap)
place_image(s, IMGS[0], 8.0, 1.5, 4.5, 4.5)
_, tf = box(s, 8.0, 6.15, 4.5, 0.7)
run(para(tf, True, align=PP_ALIGN.CENTER, line=1.3),
    "Example AI detection output (vendor removed): a suspicion score plus a localizing heatmap.", 9, INK3, font=MONO)
notes(s, "The year is 2020. You're a radiologist, and you've spent the last four years hearing about AI. Maybe you're impressed; maybe you're worried about being replaced. And the reading room is still feeling the burn from CAD — some colleagues are running CAD and AI in tandem, which only adds to the confusion. So the question is simple: what does mammography AI actually look like, and what is the evidence?")

# ============ SLIDE 2 — DREAM challenge ============
s = slide(); eyebrow(s)
_, tf = box(s, 0.85, 1.15, 6.4, 1.3)
p = para(tf, True, line=1.1)
run(p, "2016: the first large-scale push — and a ", 25, INK, bold=True)
run(p, "surge of optimism.", 25, AMBER, bold=True)
_, tf = box(s, 0.85, 2.5, 6.4, 1.0)
p = para(tf, True, line=1.35)
run(p, "The ", 14, INK2); run(p, "Digital Mammography DREAM Challenge", 14, INK, bold=True)
run(p, " was the first large-scale effort to build breast-cancer AI — and we took part. The hope was that a crowdsourced competition would simply solve it.", 14, INK2)
chip(s, 0.85, 3.55, 3.0, 0.6, [("1,000+", CYAN, True), (" participants · 126 teams · 44 countries", INK, False)])
chip(s, 4.0, 3.55, 2.6, 0.6, [("~640,000", CYAN, True), (" mammograms", INK, False)])
_, tf = box(s, 0.85, 4.35, 6.4, 1.0)
p = para(tf, True, line=1.35)
run(p, "Best model ", 14, INK2); run(p, "AUC 0.858", 14, INK, bold=True)
run(p, " → ensemble ", 14, INK2); run(p, "0.895", 14, INK, bold=True)
run(p, " → ensemble + a radiologist ", 14, INK2); run(p, "0.942", 14, INK, bold=True)
run(p, ". The winner, Therapixel, spun out ", 14, INK2); run(p, "MammoScreen", 14, INK, bold=True)
run(p, " — still in clinical use today.", 14, INK2)
_, tf = box(s, 0.85, 5.45, 6.4, 0.4)
run(para(tf, True), "Schaffter et al., JAMA Network Open 2020;3(3):e200265.", 10, INK3, font=MONO)
# DREAM ROC image right
place_image(s, IMGS[1], 8.0, 1.3, 4.5, 4.4)
_, tf = box(s, 7.7, 5.85, 5.1, 0.9)
run(para(tf, True, align=PP_ALIGN.CENTER, line=1.3),
    "Source: Schaffter et al., JAMA Netw Open 2020. Orange = best model (0.858); blue = ensemble (0.895); dark = ensemble + radiologist (0.942).", 9, INK3, font=MONO)
notes(s, "The optimism wasn't new in 2020. Back in 2016 came the first large-scale effort to build breast-cancer AI: the Digital Mammography DREAM Challenge. Over a thousand participants, 126 teams from 44 countries, on roughly 640,000 mammograms; we took part. There was an incredible surge of optimism that this would simply solve breast AI. On the held-out evaluation set, the best single model reached an AUC of 0.858; the ensemble of the eight best models reached 0.895; and the ensemble plus a single radiologist reached 0.942. The winner, Therapixel, spun out a company whose product, MammoScreen, is still in clinical use today.")

# ============ SLIDE 3 — McKinney ============
s = slide(); eyebrow(s)
_, tf = box(s, 0.85, 1.15, 7.6, 1.2)
p = para(tf, True, line=1.1)
run(p, "2020: Google publishes its ", 25, INK, bold=True)
run(p, "breast-cancer AI model.", 25, AMBER, bold=True)
_, tf = box(s, 0.85, 2.45, 7.6, 0.8)
p = para(tf, True, line=1.35)
run(p, "McKinney et al., Nature 2020", 14, INK, bold=True)
run(p, " (Google Health, UK + US) — a deep-learning system read screening mammograms stand-alone.", 14, INK2)
mk_chips = [
    [("false positives ", INK, False), ("−5.7% US / −1.2% UK", CYAN, True)],
    [("false negatives ", INK, False), ("−9.4% US / −2.7% UK", CYAN, True)],
    [("AUC ", INK, False), ("+11.5%", CYAN, True), (" vs avg radiologist", INK, False)],
    [("2nd-reader workload ", INK, False), ("−88%", CYAN, True)],
]
cy = 3.35
for i, parts in enumerate(mk_chips):
    chip(s, 0.85 + (i % 2) * 3.95, cy + (i // 2) * 0.72, 3.75, 0.62, parts)
_, tf = box(s, 0.85, 4.95, 7.6, 0.7)
run(para(tf, True, line=1.3), "Results are promising — but again, only on retrospective, clean data.", 15, INK, bold=True)
# McKinney Fig4 tall image right
place_image(s, IMGS[2], 9.0, 1.2, 3.4, 5.0)
_, tf = box(s, 8.7, 6.25, 4.0, 0.7)
run(para(tf, True, align=PP_ALIGN.CENTER, line=1.3),
    "Source: McKinney et al., Nature 2020 (Fig. 4). Orange boxes = AI-localized findings.", 9, INK3, font=MONO)
notes(s, "So what was the early evidence? In 2020, McKinney and colleagues at Google Health, in Nature, reported an AI that on US data cut false positives by 5.7% and false negatives by 9.4%, exceeded the average radiologist's AUC by over 11 points, and could cut a second reader's workload by 88%. The figure shows example cases the model localized. On paper, superhuman — but retrospective.")

# ============ SLIDE 4 — Salim independent validation ============
s = slide(); eyebrow(s)
_, tf = box(s, 0.85, 1.15, 6.4, 1.3)
p = para(tf, True, line=1.1)
run(p, "Three commercial models, independently validated on ", 23, INK, bold=True)
run(p, "8,800 women.", 23, AMBER, bold=True)
_, tf = box(s, 0.85, 2.65, 6.4, 0.7)
p = para(tf, True, line=1.35)
run(p, "Salim et al., JAMA Oncology 2020", 14, INK, bold=True)
run(p, " tested three commercial algorithms on an external Stockholm cohort.", 14, INK2)
sal_chips = [
    [("best (Algorithm 1) ", INK, False), ("AUC 0.956", CYAN, True)],
    [("AI + first reader ", INK, False), ("88.6% sens @ 93% spec", CYAN, True)],
    [("exceeded ", INK, False), ("two human readers", CYAN, True)],
]
cy = 3.5
for i, parts in enumerate(sal_chips):
    chip(s, 0.85, cy + i * 0.72, 5.8, 0.62, parts)
_, tf = box(s, 0.85, 5.8, 6.4, 0.6)
run(para(tf, True, line=1.3), "Three vendors, external data — the evidence is looking good.", 15, INK, bold=True)
# Salim ROC image right
place_image(s, IMGS[3], 7.6, 1.5, 4.9, 4.3)
_, tf = box(s, 7.6, 5.95, 4.9, 0.7)
run(para(tf, True, align=PP_ALIGN.CENTER, line=1.3),
    "Source: Salim et al., JAMA Oncology 2020 — ROC of Algorithms 1–3 on 8,805 women.", 9, INK3, font=MONO)
notes(s, "And it held up independently. Salim and colleagues, in JAMA Oncology, took three commercial AI algorithms and tested them on an external Stockholm cohort of about 8,800 women. The best — Algorithm 1 here — reached an AUC of 0.956, and combined with a first reader it reached 88.6% sensitivity at 93% specificity, exceeding two human readers. Three different vendors, external data — the evidence is looking good.")

# ============ SLIDE 5 — retrospective isn't enough + adoption stats ============
s = slide(); eyebrow(s)
_, tf = box(s, 0.85, 1.15, 11.6, 1.0)
p = para(tf, True, line=1.1)
run(p, "Retrospective data isn't enough to ", 30, INK, bold=True)
run(p, "drive adoption.", 30, AMBER, bold=True)
# prompt (amber left border)
bar = rect(s, 0.85, 2.45, 0.05, 1.0, AMBER)
_, tf = box(s, 1.05, 2.45, 11.3, 1.0)
run(para(tf, True, line=1.35),
    "Enriched cases, lab conditions, no real workflow — CAD had good retrospective numbers too. The bar is prospective deployment, and real-world uptake has been slow.", 16, INK)
adoption = [
    ("48%", "European radiologists use AI (2024) — up from 20% in 2018", INK),
    ("13.7%", "of them, for breast imaging — about 1 in 8", INK),
    ("~2%", "of US practices, by one estimate", INK),
]
ax = 0.85
for big, lab, col in adoption:
    _, tf = box(s, ax, 3.85, 3.7, 1.2)
    run(para(tf, True), big, 56, AMBER, bold=True)
    _, tf = box(s, ax, 5.15, 3.6, 1.0)
    run(para(tf, True, line=1.35), lab, 11, INK, font=MONO)
    ax += 4.0
_, tf = box(s, 0.85, 6.55, 11.6, 0.4)
run(para(tf, True), "ESR EuroAIM/EuSoMII survey, Insights Imaging 2024 (n=572) · US estimate: industry report", 10, INK3, font=MONO)
notes(s, "But here is the discipline this talk runs on. Retrospective reader studies have well-known limits: enriched case sets, lab conditions, no real workflow. CAD had good retrospective numbers too. That kind of data is not enough to drive adoption. And it shows: a 2024 European survey found about 48% of radiologists use AI, but of those only 13.7% use it for breast imaging, about one in eight. In the US, by one estimate, only around 2% of practices use it at all.")

# ============ SLIDE 6 — MASAI 2023 safety ============
s = slide(); eyebrow(s)
_, tf = box(s, 0.85, 1.15, 11.6, 0.9)
p = para(tf, True); run(p, "2023: MASAI publishes its ", 28, INK, bold=True); run(p, "safety analysis.", 28, AMBER, bold=True)
_, tf = box(s, 0.85, 2.1, 11.6, 0.6)
p = para(tf, True, line=1.3)
run(p, "A randomized trial, ", 15, INK2); run(p, "~80,000 women", 15, INK, bold=True)
run(p, " (≈40,000 per arm) — AI-supported reading vs standard double reading.", 15, INK2)
# bigline
_, tf = box(s, 0.85, 2.85, 6.0, 1.0, anchor=MSO_ANCHOR.MIDDLE)
p = para(tf, True); run(p, "6.1 ", 50, AMBER, bold=True); run(p, "vs", 30, INK3); run(p, " 5.1", 50, AMBER, bold=True)
_, tf = box(s, 0.85, 3.95, 6.4, 0.5)
run(para(tf, True), "CDR per 1,000 · ratio 1.2 (95% CI 1.0–1.5)", 12, INK, font=MONO)
# native table
cmp_table(s, 0.85, 4.65, 6.2,
          [("Measure", "AI", "Control"),
           ("Recall rate", "2.2%", "2.0%"),
           ("False-positive rate", "1.5%", "1.5%"),
           ("PPV of recall", "28.3%", "24.8%"),
           ("Invasive cancers", "75%", "81%")],
          [3.4, 1.4, 1.4])
_, tf = box(s, 7.4, 3.0, 5.1, 2.0)
run(para(tf, True, line=1.4), "Above the lowest acceptable detection limit for safety — safe, and detecting more.", 16, INK2)
_, tf = box(s, 0.85, 6.95, 11.6, 0.4)
run(para(tf, True), "Lång et al., Lancet Oncology 2023", 10, INK3, font=MONO)
notes(s, "In 2023, MASAI published its clinical safety analysis — a randomized trial of about eighty thousand women, roughly forty thousand per arm. Cancer detection was 6.1 per thousand with AI versus 5.1 in controls, a ratio of 1.2, above the lowest acceptable safety limit. Recall was essentially identical, 2.2 versus 2.0 percent; the false-positive rate was 1.5 percent in both; and the positive predictive value of recall was higher with AI, 28 versus 25 percent. Three-quarters of detected cancers were invasive. So: safe, and detecting more — the green light to continue.")

# ============ SLIDE 7 — PRAIM ============
s = slide(); eyebrow(s)
_, tf = box(s, 0.85, 1.15, 11.6, 0.9)
p = para(tf, True); run(p, "2025: PRAIM — ", 28, INK, bold=True); run(p, "real-world data.", 28, AMBER, bold=True)
_, tf = box(s, 0.85, 2.1, 11.6, 0.6)
p = para(tf, True, line=1.3)
run(p, "463,000 women", 15, INK, bold=True)
run(p, " · 12 sites · 119 radiologists (Germany, 2021–23) — ~261,000 read with AI support (Vara).", 15, INK2)
_, tf = box(s, 0.85, 2.85, 6.4, 1.0, anchor=MSO_ANCHOR.MIDDLE)
p = para(tf, True); run(p, "6.7 ", 50, AMBER, bold=True); run(p, "vs", 30, INK3); run(p, " 5.7", 50, AMBER, bold=True)
_, tf = box(s, 0.85, 3.95, 6.6, 0.5)
run(para(tf, True), "CDR per 1,000 · +17.6% (95% CI +5.7% to +30.8%), superior", 12, INK, font=MONO)
cmp_table(s, 0.85, 4.65, 6.5,
          [("Measure", "AI", "Control"),
           ("Recall rate /1,000", "37.4", "38.3"),
           ("PPV of recall", "17.9%", "14.9%"),
           ("PPV of biopsy", "64.5%", "59.2%")],
          [3.7, 1.4, 1.4])
_, tf = box(s, 7.6, 3.0, 4.9, 2.0)
run(para(tf, True, line=1.4), "The MASAI result holds outside a controlled trial — superior detection, non-inferior recall.", 16, INK2)
_, tf = box(s, 0.85, 6.6, 11.6, 0.4)
run(para(tf, True), "Eisemann et al., Nature Medicine 2025 — observational, real-world", 10, INK3, font=MONO)
notes(s, "Does it hold in routine practice? PRAIM is the largest real-world dataset: 463,000 women across 12 German sites, 119 radiologists, with about 261,000 read with AI support from the Vara system. Detection was 6.7 per thousand with AI, a 17.6 percent relative increase, statistically superior to 5.7 in controls; recall was non-inferior, and the predictive value of both recall and biopsy improved. The MASAI result holds outside a controlled trial.")

# ============ SLIDE 8 — MASAI 2026 interval cancer ============
s = slide(); eyebrow(s)
_, tf = box(s, 0.85, 1.15, 11.6, 0.9)
p = para(tf, True); run(p, "Today: MASAI ", 28, INK, bold=True); run(p, "interval-cancer", 28, AMBER, bold=True); run(p, " result.", 28, INK, bold=True)
_, tf = box(s, 0.85, 2.1, 11.6, 0.7)
p = para(tf, True, line=1.3)
run(p, "~105,000 women. A ", 15, INK2); run(p, "triage design", 15, INK, bold=True)
run(p, ": high-risk exams → standard double reading; low-risk → a single radiologist.", 15, INK2)
# two stat blocks
_, tf = box(s, 0.85, 3.1, 5.0, 1.4)
run(para(tf, True), "−12%", 64, AMBER, bold=True)
_, tf = box(s, 0.85, 4.55, 4.8, 0.9)
run(para(tf, True, line=1.35), "interval-cancer rate — non-inferior (ratio 0.88)", 12, INK, font=MONO)
_, tf = box(s, 6.5, 3.1, 6.0, 1.4)
p = para(tf, True); run(p, "80.5%", 64, AMBER, bold=True); run(p, " vs 73.8%", 22, INK3)
_, tf = box(s, 6.5, 4.55, 5.5, 0.9)
run(para(tf, True, line=1.35), "sensitivity — at the same specificity (98.5%)", 12, INK, font=MONO)
_, tf = box(s, 0.85, 5.8, 11.6, 0.4)
run(para(tf, True), "Gommers et al. (…Lång), Lancet 2026;407(10527):505–514", 10, INK3, font=MONO)
notes(s, "And this year, the result everyone was waiting for: MASAI's interval-cancer study, about 105,000 women. The design is a triage — high-risk exams went to standard human double reading, low-risk exams to a single radiologist. The interval-cancer rate — the cancers that surface between screens — was 12 percent lower, non-inferior. Sensitivity was 80.5 percent with AI versus 73.8 in controls, a near seven-point gain, at the same specificity of 98.5 percent, and the results held across age and breast-density subgroups. This is the level-1 evidence CAD never had.")

# ============ SLIDE 9 — subgroup failure + bar chart ============
s = slide(); eyebrow(s)
_, tf = box(s, 0.85, 1.1, 6.4, 1.4)
p = para(tf, True, line=1.1)
run(p, "2026: the averages look strong — so where do these models ", 22, INK, bold=True)
run(p, "still fail?", 22, AMBER, bold=True)
_, tf = box(s, 0.85, 2.55, 6.4, 0.7)
p = para(tf, True, line=1.3)
run(p, "Our DBT audit", 13, INK, bold=True)
run(p, " (EMBED, 163,449 exams): overall AUC ", 13, INK2); run(p, "0.91", 13, INK, bold=True)
run(p, " — but detection drops by histologic subtype.", 13, INK2)
_, tf = box(s, 0.85, 3.35, 6.4, 0.8)
p = para(tf, True, line=1.3)
run(p, "Common ", 13, INK2); run(p, "ductal", 13, INK, bold=True); run(p, " cancers are flagged ", 13, INK2)
run(p, "~86%", 13, AMBER, bold=True); run(p, " of the time; rarer special subtypes — tubular, papillary, colloid — far less, ", 13, INK2)
run(p, "57–62%", 13, WARN, bold=True); run(p, " (small n).", 13, INK2)
_, tf = box(s, 0.85, 4.3, 6.4, 0.9)
p = para(tf, True, line=1.3)
run(p, "Sensitivity also falls on ", 13, INK2); run(p, "dense breasts, in-situ, and calcifications", 13, INK, bold=True)
run(p, " (0.55–0.66) even as AUC stays ~0.80–0.88. And it ", 13, INK2); run(p, "doesn't transfer", 13, INK, bold=True)
run(p, " — specificity 31–54% on women with implants.", 13, INK2)
_, tf = box(s, 0.85, 5.55, 6.4, 0.7)
run(para(tf, True, line=1.3), "Trivedi group — DBT subgroup audit (Nat Commun 2026) · Du H et al. \"Beyond Screening\" 2026. Chart redrawn from the subtype figure.", 9.5, INK3, font=MONO)
# horizontal bar chart redrawn — right side
# data: (label, n, pct, color)
subt = [("IDC (NOS) · n=544", 86, AMBER), ("Mammary · n=67", 84, AMBER), ("Lobular · n=100", 82, AMBER),
        ("Papillary · n=8", 62, WARN), ("Tubular · n=7", 57, WARN), ("Colloid · n=7", 57, WARN),
        ("Medullary · n=1", 100, INK3)]
chart_l = 9.0; chart_r = 12.7; chart_w = chart_r - chart_l  # px-to-in scale for 0..100%
ctop = 1.5; bh = 0.34; gap = 0.12
_, tf = box(s, 7.4, 1.1, 5.4, 0.4)
run(para(tf, True, align=PP_ALIGN.CENTER), "% of cancers detected, by histologic subtype", 10.5, INK2, font=MONO)
by = ctop
for lab, pct, col in subt:
    _, tf = box(s, 7.4, by, 1.55, bh, anchor=MSO_ANCHOR.MIDDLE)
    run(para(tf, True, align=PP_ALIGN.RIGHT), lab, 9, INK2)
    bw = chart_w * pct / 100.0
    rect(s, chart_l, by + 0.02, bw, bh - 0.04, col)
    _, tf = box(s, chart_l + bw + 0.05, by, 0.7, bh, anchor=MSO_ANCHOR.MIDDLE)
    run(para(tf, True), "%d%%" % pct, 10, INK, bold=True)
    by += bh + gap
# axis labels
_, tf = box(s, chart_l - 0.15, by, 0.4, 0.3); run(para(tf, True, align=PP_ALIGN.CENTER), "0", 9, INK3, font=MONO)
_, tf = box(s, chart_l + chart_w/2 - 0.2, by, 0.5, 0.3); run(para(tf, True, align=PP_ALIGN.CENTER), "50%", 9, INK3, font=MONO)
_, tf = box(s, chart_r - 0.25, by, 0.5, 0.3); run(para(tf, True, align=PP_ALIGN.CENTER), "100%", 9, INK3, font=MONO)
_, tf = box(s, 7.4, by + 0.35, 5.4, 0.4)
run(para(tf, True, align=PP_ALIGN.CENTER), "Rare special subtypes are detected far less often (small n)", 9.5, INK3, font=MONO)
notes(s, "So it is 2026 — where do we actually stand? The averages look strong, but a single AUC describes average performance and says nothing about where a model fails. In our audit of a commercial DBT model across 163,000 exams, the overall AUC was 0.91 — but detection drops sharply by histologic subtype. The common invasive ductal cancers are flagged about 86 percent of the time; the rarer special subtypes — tubular, papillary, colloid — far less, 57 to 62 percent, though on small numbers. Sensitivity also falls on dense breasts, in-situ disease, and calcifications even as the AUC stays high. And in our Beyond Screening work, the models did not transfer to diagnostic exams or to women with implants. The two gaps: subgroup performance, and explainability.")

# ============ SLIDE 10 — bridge to Era III: Hickman + ROC line ============
s = slide(); eyebrow(s, "Era II → Era III")
_, tf = box(s, 0.85, 1.1, 6.4, 1.3)
p = para(tf, True, line=1.1)
run(p, "The detection score rises for ", 23, INK, bold=True)
run(p, "years before", 23, AMBER, bold=True)
run(p, " the cancer is visible.", 23, INK, bold=True)
_, tf = box(s, 0.85, 2.55, 6.4, 1.1)
p = para(tf, True, line=1.3)
run(p, "Hickman et al., Radiology 2026", 13, INK, bold=True)
run(p, " — three detection systems across sequential screens (31,394 individuals). For future cancers the AI score is elevated up to ", 13, INK2)
run(p, "10 years", 13, INK, bold=True); run(p, " before diagnosis; cancer-free individuals stay flat.", 13, INK2)
_, tf = box(s, 0.85, 3.75, 6.4, 0.5)
p = para(tf, True, line=1.3)
run(p, "Combined predictive AUC ", 13, INK2); run(p, "0.63–0.67", 13, INK, bold=True); run(p, " vs 0.57 for breast density.", 13, INK2)
_, tf = box(s, 0.85, 4.4, 6.4, 1.0)
p = para(tf, True, line=1.35)
run(p, "Detection is blurring into risk. We stop asking the image about the lesion — and start asking it about the patient. ", 14, INK, bold=True)
run(p, "That is Era III.", 14, AMBER, bold=True)
_, tf = box(s, 0.85, 5.75, 6.4, 0.6)
run(para(tf, True, line=1.3), "Chart redrawn from Hickman et al., Radiology 2026;319(3):e251309 (Table 2 / Fig 2A).", 9.5, INK3, font=MONO)
# line chart redrawn: future-cancer rises, cancer-free flat
# plot area
gx0, gy0, gw, gh = 8.3, 2.0, 3.9, 3.2  # inches; y0 top
# axes
ax = rect(s, gx0, gy0, 0.012, gh, LINE); ax.width = Pt(1.2)
ax2 = rect(s, gx0, gy0 + gh, gw, 0.012, LINE); ax2.height = Pt(1.2)
_, tf = box(s, 7.4, gy0 + gh - 0.13, 0.85, 0.3); run(para(tf, True, align=PP_ALIGN.RIGHT), "0", 9, INK3, font=MONO)
_, tf = box(s, 7.4, gy0 + gh/2 - 0.13, 0.85, 0.3); run(para(tf, True, align=PP_ALIGN.RIGHT), "25%", 9, INK3, font=MONO)
_, tf = box(s, 7.4, gy0 - 0.13, 0.85, 0.3); run(para(tf, True, align=PP_ALIGN.RIGHT), "50%", 9, INK3, font=MONO)
# future-cancer polyline (years: 10,6,4,2,1) pct approx from SVG (50% scale): map y to fraction
# SVG points (x:70..300, y:146,144,129,114,65,38) over y 20..200 mapped to 50%..0
fut_pts = [(0.0, 0.30), (0.25, 0.31), (0.48, 0.39), (0.70, 0.48), (0.90, 0.75), (1.0, 0.90)]
def conn(s, x1, y1, x2, y2, color, w=2.5, dash=False):
    ln = s.shapes.add_connector(2, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    ln.line.color.rgb = color; ln.line.width = Pt(w); ln.shadow.inherit = False
    if dash:
        from pptx.oxml.ns import qn
        d = ln.line._get_or_add_ln(); pd = d.makeelement(qn('a:prstDash'), {'val': 'dash'}); d.append(pd)
    return ln
prev = None
for fx, fr in fut_pts:
    px = gx0 + 0.15 + fx * (gw - 0.3)
    py = gy0 + gh - fr * gh
    if prev: conn(s, prev[0], prev[1], px, py, AMBER, 2.6)
    prev = (px, py)
# cancer-free flat dashed line
fy = gy0 + gh - 0.27 * gh
conn(s, gx0 + 0.15, fy, gx0 + gw - 0.15, fy, CYAN, 2.2, dash=True)
# x labels
xl = [("10 yr", 0.0), ("6 yr", 0.40), ("4 yr", 0.6), ("2 yr", 0.8), ("1 yr", 1.0)]
for lab, fx in xl:
    px = gx0 + 0.15 + fx * (gw - 0.3)
    _, tf = box(s, px - 0.35, gy0 + gh + 0.05, 0.7, 0.3)
    run(para(tf, True, align=PP_ALIGN.CENTER), lab, 9, INK2, font=MONO)
# legend
rect(s, 9.6, 1.55, 0.25, 0.05, AMBER)
_, tf = box(s, 9.9, 1.45, 1.5, 0.3); run(para(tf, True), "future cancer", 9.5, INK, font=MONO)
rect(s, 11.1, 1.55, 0.25, 0.05, CYAN)
_, tf = box(s, 11.4, 1.45, 1.2, 0.3); run(para(tf, True), "cancer-free", 9.5, INK2, font=MONO)
_, tf = box(s, 7.6, gy0 + gh + 0.4, 4.7, 0.5)
run(para(tf, True, align=PP_ALIGN.CENTER), "% flagged at 90th-centile score · years before diagnosis", 9.5, INK3, font=MONO)
notes(s, "And one recent result points straight at the next era. Hickman and colleagues, in Radiology this year, ran three commercial detection systems across sequential screening mammograms. For people later diagnosed with cancer, the AI detection score is already elevated up to ten years before diagnosis and rises steadily toward it; in cancer-free people it stays flat. The score we trained to find today's cancer already carries a signal about tomorrow's. Detection is blurring into risk — and that is exactly where Era III begins. We stop asking the image about the lesion, and start asking it about the whole image, and the patient.")

out = os.path.join(HERE, "IWBI2026_Trivedi_section2_EraII.pptx")
prs.save(out)
print("saved", out, "slides:", len(prs.slides._sldIdLst))
