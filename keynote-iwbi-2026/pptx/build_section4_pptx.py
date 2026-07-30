#!/usr/bin/env python3
"""Build Era III pathology deck (The Other Half of the Slide) as editable PPTX — one box per blurb.
Native foundation-model table styled dark. 8 slides, none hidden, no embedded photos."""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

BG    = RGBColor(0x0F,0x14,0x1A); PANEL = RGBColor(0x06,0x09,0x0C)
INK   = RGBColor(0xEC,0xEF,0xF3); INK2 = RGBColor(0x9D,0xA9,0xB5); INK3 = RGBColor(0x5C,0x69,0x75)
AMBER = RGBColor(0xE7,0xAC,0x51); AMBERDP = RGBColor(0xC9,0x8A,0x2E); CYAN = RGBColor(0x5F,0xB7,0xC9)
WARN  = RGBColor(0xD9,0x78,0x5B); CARD = RGBColor(0x1B,0x23,0x2D); LINE = RGBColor(0x2A,0x33,0x3D)
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

def twostat(s, items, t=2.9):
    """items: list of ((big, vs_suffix), label). Returns nothing."""
    sx = 0.85
    for (big, vs), lab in items:
        _, tf = box(s, sx, t, 5.6, 1.3)
        p = para(tf, True)
        run(p, big, 52, AMBER, bold=True)
        if vs: run(p, vs, 20, INK3)
        _, tf = box(s, sx, t + 1.35, 5.4, 0.9)
        run(para(tf, True, line=1.35), lab, 11.5, INK, font=MONO)
        sx += 6.2

# ============ SLIDE 1 — divider: across the hospital ============
s = slide()
_, tf = box(s, 0.85, 0.72, 11, 0.4)
p = para(tf, True)
run(p, "LESION  →  IMAGE  →  ", 11, INK3, font=MONO)
run(p, "PATIENT", 11, AMBER, bold=True, font=MONO)
run(p, "  →  POPULATION", 11, INK3, font=MONO)
_, tf = box(s, 0.85, 2.0, 11.6, 1.4)
p = para(tf, True, line=1.02)
run(p, "Across the hospital, in the ", 46, INK, bold=True); run(p, "pathology lab.", 46, AMBER, bold=True)
_, tf = box(s, 0.85, 3.75, 10.0, 1.6)
run(para(tf, True, line=1.4), "So far we have stayed in the reading room. The same arc — hand-built tools, then deep learning, then foundation models — has played out in pathology, and it has one capability our images do not: inferring molecular and genomic features directly from the stained slide.", 17, INK2)
bar = rect(s, 0.85, 5.65, 0.05, 1.0, AMBERDP)
_, tf = box(s, 1.1, 5.65, 10.5, 1.0)
p = para(tf, True, line=1.4)
run(p, "This is also where the ", 14, INK3); run(p, "third model", 14, AMBER, bold=True)
run(p, " from the opening lives — the recurrence score read from a glass slide.", 14, INK3)
notes(s, "For the whole talk we have stayed in the reading room. Now walk across the hospital to the pathology lab — a part of the building a radiology audience rarely sees. The same arc played out here: hand-built tools, then deep learning, then foundation models. But tissue AI can do one thing our images cannot, and it is the reason this matters to you. And remember the third model I opened with — the four-thousand-dollar recurrence score being read off a glass slide across town. We are about to meet it.")

# ============ SLIDE 2 — foundation models on tissue ============
s = slide(); eyebrow(s, "Era III · an unexpected fact")
_, tf = box(s, 0.85, 1.15, 11.6, 1.1)
p = para(tf, True, line=1.1)
run(p, "Some of the largest medical foundation models were trained on ", 26, INK, bold=True)
run(p, "pathology slides.", 26, AMBER, bold=True)
_, tf = box(s, 0.85, 2.55, 11.6, 1.3)
p = para(tf, True, line=1.4)
run(p, "Virchow", 15, INK, bold=True)
run(p, " (Paige + Microsoft, Nature Medicine 2024) was trained on ~1.5M whole-slide images from ~120,000 patients — ", 15, INK2)
run(p, "self-supervised, no hand labels", 15, INK, bold=True)
run(p, ". It learned the structure of human tissue on its own, and detects cancer across a dozen-plus tissue types at clinical-grade accuracy, including rare cancers.", 15, INK2)
chip(s, 0.85, 4.1, 3.0, 0.62, [("1.49M", CYAN, True), (" whole-slide images", INK, False)])
chip(s, 4.0, 4.1, 3.0, 0.62, [("119,629", CYAN, True), (" patients", INK, False)])
chip(s, 7.15, 4.1, 5.3, 0.62, [("0.95", CYAN, True), (" specimen-level AUC · pan-cancer incl. rare", INK, False)])
_, tf = box(s, 0.85, 5.1, 11.6, 0.4)
run(para(tf, True), "Vorontsov et al., Nature Medicine 2024 · DOI 10.1038/s41591-024-03141-0", 10, INK3, font=MONO)
notes(s, "Here is a fact that surprises most imaging people. Some of the largest foundation models in medicine were not built on radiology images — they were built on pathology slides. In 2024, Virchow, from Paige and Microsoft, was trained on about 1.5 million whole-slide images from roughly 120,000 patients, self-supervised, with no hand labels. It learned the structure of human tissue on its own, and detects cancer across more than a dozen tissue types at clinical-grade accuracy, including rare cancers where data is scarce.")

# ============ SLIDE 3 — morphology is molecular: CDH1/ILC ============
s = slide(); eyebrow(s, "Era III · morphology is molecular")
_, tf = box(s, 0.85, 1.15, 11.6, 1.1)
p = para(tf, True, line=1.1)
run(p, "Inferring molecular features from the H&E, including ", 25, INK, bold=True)
run(p, "invasive lobular carcinoma.", 25, AMBER, bold=True)
_, tf = box(s, 0.85, 2.5, 11.6, 1.5)
p = para(tf, True, line=1.4)
run(p, "Invasive lobular carcinoma", 15, INK, bold=True)
run(p, " hides in dense tissue and grows single-file — the hardest cancer to catch on mammography. It is defined molecularly by loss of E-cadherin (the ", 15, INK2)
run(p, "CDH1", 15, INK, bold=True)
run(p, " gene). A model predicts CDH1 status ", 15, INK2)
run(p, "from the H&E alone", 15, INK, bold=True)
run(p, ". The morphology was always a readout of the biology underneath; it just wasn't visible by eye.", 15, INK2)
chip(s, 0.85, 4.25, 3.6, 0.62, [("CDH1 from H&E · ", INK, False), ("AUC ~0.94", CYAN, True)])
chip(s, 4.6, 4.25, 7.4, 0.62, [("74%", CYAN, True), (" of AI-flagged cases had hidden CDH1 inactivation the assay missed", INK, False)])
_, tf = box(s, 0.85, 5.25, 11.6, 0.4)
run(para(tf, True), "Cancer Research 2024 (Mount Sinai) · PMID 39106449 — a dedicated model, not Virchow", 10, INK3, font=MONO)
notes(s, "And this is directly relevant to breast imaging. These models read features off the H&E that the eye cannot see. Take the cancer hardest to detect on a mammogram — invasive lobular carcinoma. It hides in dense tissue and grows single-file. Lobular cancer is defined molecularly by the loss of one protein, E-cadherin, the CDH1 gene. A model predicts CDH1 status from the H&E alone at an AUC around 0.94. The morphology was always a readout of the molecular biology underneath; it just was not visible by eye. And among cases the model flagged as CDH1-lost where the standard test found nothing, about three-quarters had another mechanism silencing the gene — biology the assay never tested for.")

# ============ SLIDE 4 — CAMELYON16 reader study ============
s = slide(); eyebrow(s, "Era III · pathology's reader study")
_, tf = box(s, 0.85, 1.15, 11.6, 1.1)
p = para(tf, True, line=1.1)
run(p, "CAMELYON16: ", 26, INK, bold=True)
run(p, "AI compared with pathologists", 26, AMBER, bold=True)
run(p, " on lymph-node metastasis.", 26, INK, bold=True)
twostat(s, [(("0.994", ""), "best algorithm · AUC"),
            (("0.810", ""), "pathologist panel · AUC (under a 2-hour limit)")], t=2.7)
_, tf = box(s, 0.85, 5.0, 11.6, 1.2)
p = para(tf, True, line=1.4)
run(p, "Lymph-node metastasis on H&E, 11 pathologists under realistic time pressure. ", 16, INK2)
run(p, "Under the clock, the best AI exceeded the panel", 16, INK, bold=True)
run(p, "; top algorithms matched experts given unlimited time. The point is fatigue and time, not replacement.", 16, INK2)
_, tf = box(s, 0.85, 6.3, 11.6, 0.4)
run(para(tf, True), "Ehteshami Bejnordi et al., JAMA 2017 · DOI 10.1001/jama.2017.14585", 10, INK3, font=MONO)
notes(s, "Pathology ran the same AI-versus-experts study radiology did. CAMELYON16: detecting lymph-node metastasis on H&E, eleven pathologists reading a test set under a realistic two-hour time limit. The best algorithm scored 0.994; the pathologist panel, under the clock, 0.810. The top algorithms matched experts given unlimited time. The framing is not replacement — it is fatigue and time pressure, which machines do not feel.")

# ============ SLIDE 5 — HER2-low ============
s = slide(); eyebrow(s, "Era III · a call that now picks a drug")
_, tf = box(s, 0.85, 1.15, 11.6, 1.1)
p = para(tf, True, line=1.1)
run(p, "HER2-low: AI improves agreement on the ", 26, INK, bold=True)
run(p, "0-versus-1+", 26, AMBER, bold=True)
run(p, " call.", 26, INK, bold=True)
_, tf = box(s, 0.85, 2.55, 11.6, 1.0)
p = para(tf, True, line=1.4)
run(p, "Since trastuzumab deruxtecan (DESTINY-Breast04), ", 15, INK2)
run(p, "HER2-low is treatable", 15, INK, bold=True)
run(p, ". That makes the HER2 0-vs-1+ distinction decisive — a call the assay wasn't designed for, on which pathologists disagree.", 15, INK2)
chip(s, 0.85, 3.85, 3.5, 0.62, [("pathologist agreement ", INK, False), ("72.4%", CYAN, True)])
chip(s, 4.5, 3.85, 3.0, 0.62, [("automated AI ", INK, False), ("92.1%", CYAN, True)])
chip(s, 7.65, 3.85, 4.85, 0.62, [("0-vs-1+ reader agreement ", INK, False), ("70% → 87%", CYAN, True)])
_, tf = box(s, 0.85, 4.85, 11.6, 0.4)
run(para(tf, True), "Krishnamurthy et al., JCO Precision Oncology 2024 · DOI 10.1200/PO.24.00353", 10, INK3, font=MONO)
notes(s, "Here is AI on a problem that decides treatment today. Since trastuzumab deruxtecan and DESTINY-Breast04, HER2-low disease is treatable. That makes the HER2 zero-versus-one-plus call decisive — and it is a call the assay was never designed to make, on which pathologists disagree. Baseline agreement is about 72 percent. Automated AI reaches 92. And on the hardest call, zero versus one-plus, AI lifts reader agreement from about 70 to about 87 percent. A distinction that now determines who is eligible for a drug that improves survival.")

# ============ SLIDE 6 — Oncotype from H&E (Orpheus) ============
s = slide(); eyebrow(s, "Era III · morphology → molecular (the third model)")
_, tf = box(s, 0.85, 1.15, 11.6, 1.1)
p = para(tf, True, line=1.1)
run(p, "Estimating the Oncotype DX recurrence score from the ", 24, INK, bold=True)
run(p, "H&E slide.", 24, AMBER, bold=True)
twostat(s, [(("0.89", " vs 0.73"), "flags high-risk (RS>25) vs a leading nomogram"),
            (("0.75", " vs 0.49"), "predicts recurrence in low-RS patients — beating the score itself")], t=2.6)
_, tf = box(s, 0.85, 4.95, 11.6, 1.2)
p = para(tf, True, line=1.4)
run(p, "Oncotype DX costs ", 16, INK2); run(p, "~$4,000", 16, INK, bold=True)
run(p, " and takes 1–2 weeks. ", 16, INK2)
run(p, "Orpheus", 16, INK, bold=True)
run(p, " (6,172 cases, 3 institutions) infers the 21-gene Recurrence Score ", 16, INK2)
run(p, "directly from the H&E", 16, INK, bold=True)
run(p, ". This is what imaging cannot do: morphology → molecular identity.", 16, INK2)
_, tf = box(s, 0.85, 6.35, 11.6, 0.4)
run(para(tf, True), "Boehm et al. (\"Orpheus\"), Nature Communications 2025 · concept: Kather et al., Nature Cancer 2020", 10, INK3, font=MONO)
notes(s, "And this is the capability imaging simply does not have, and the third model from the opening. Oncotype DX — the 21-gene recurrence score — costs about four thousand dollars and takes one to two weeks. The slide is already on the desk. Orpheus, in Nature Communications this year, infers that recurrence score directly from the H&E across six thousand cases. It flags the high-risk patients at an AUC of 0.89 versus 0.73 for a leading nomogram. And in the low-score patients — the ones the test calls low-risk — it predicts who actually recurs better than the recurrence score itself: 0.75 versus 0.49. Morphology to molecular identity, from a slide we already have.")

# ============ SLIDE 7 — foundation-model wave (native table) ============
s = slide(); eyebrow(s, "Era III · the substrate")
_, tf = box(s, 0.85, 1.15, 11.6, 0.9)
p = para(tf, True); run(p, "Pathology ", 28, INK, bold=True); run(p, "foundation models", 28, AMBER, bold=True); run(p, " (2024).", 28, INK, bold=True)
# native table
rows = [("Model (2024)", "Group", "Scale", "Note"),
        ("UNI", "Mahmood Lab, Harvard/BWH", ">100k slides · ~20 tissues", "open-source tissue encoder"),
        ("CONCH", "Mahmood Lab", "1.17M image–text pairs", "vision-language: query tissue with text"),
        ("Virchow", "Paige + Microsoft", "1.49M slides · 119,629 patients", "pan-cancer incl. rare (0.95 AUC)"),
        ("Prov-GigaPath", "Microsoft + Providence + UW", "1.3B tiles · 171k slides · 28 centers", "first whole-slide FM; SOTA 25/26 tasks")]
gf = s.shapes.add_table(len(rows), 4, Inches(0.85), Inches(2.2), Inches(11.6), Inches(2.5))
tbl = gf.table
for ci, cw in enumerate([2.0, 3.4, 3.6, 2.6]):
    tbl.columns[ci].width = Inches(cw)
for ri, row in enumerate(rows):
    for ci, val in enumerate(row):
        cell = tbl.cell(ri, ci); cell.fill.solid(); cell.fill.fore_color.rgb = BG
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_left = Inches(0.1); cell.margin_right = Inches(0.08)
        cell.margin_top = Inches(0.03); cell.margin_bottom = Inches(0.03)
        tf = cell.text_frame; tf.word_wrap = True; p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
        r = p.add_run(); r.text = val; f = r.font; f.name = SANS
        if ri == 0:
            f.size = Pt(10); f.bold = True; f.color.rgb = INK2; f.name = MONO
        elif ci == 0:
            f.size = Pt(12); f.bold = True; f.color.rgb = AMBER; f.name = MONO
        else:
            f.size = Pt(11); f.color.rgb = INK2
_, tf = box(s, 0.85, 5.1, 11.6, 1.5)
p = para(tf, True, line=1.4)
run(p, "They generalize across institutions and scanners — the reproducibility problem that had limited clinical pathology AI. ", 13, INK2)
run(p, "Honest caveat:", 13, INK, bold=True)
run(p, " the bottleneck now is scanning, digital workflow, and reimbursement — not the algorithm. Most US labs still read glass; Paige Prostate was the first FDA-authorized pathology AI (2021).", 13, INK2)
notes(s, "The reason all of this arrived at once is the substrate underneath it. Instead of training a new model per task, labs trained one self-supervised model on millions of unlabeled slides — a GPT for tissue — that learns the visual language of pathology and adapts with minimal fine-tuning. UNI, CONCH, Virchow, and Prov-GigaPath were published within months of each other in Nature and Nature Medicine. They cut the data a breast task needs and generalize across institutions and scanners, which is what had limited clinical pathology AI. The honest caveat: the bottleneck now is scanning, digital workflow, and reimbursement — not the algorithm. Most US labs still read glass. Paige Prostate was the first FDA-authorized pathology AI, in 2021; molecular-from-H&E is not yet replacing the assays.")

# ============ SLIDE 8 — bridge to §5 ============
s = slide(); eyebrow(s, "Era III · combining the two specialties")
_, tf = box(s, 0.85, 1.3, 11.6, 1.0)
p = para(tf, True, line=1.1)
run(p, "Analyzing radiology and pathology ", 30, INK, bold=True)
run(p, "together.", 30, AMBER, bold=True)
bar = rect(s, 0.85, 2.7, 0.05, 1.0, AMBER)
_, tf = box(s, 1.1, 2.7, 11.0, 1.0)
run(para(tf, True, line=1.4), "Radiology and pathology have been handled as separate departments, with separate images and separate reports. AI now makes it practical to analyze them together for the same patient.", 18, INK)
_, tf = box(s, 0.85, 4.0, 11.6, 0.4)
p = para(tf, True)
run(p, "LESION  →  ", 11, INK3, font=MONO); run(p, "IMAGE ✓", 11, AMBER, font=MONO)
run(p, "  →  ", 11, INK3, font=MONO); run(p, "PATIENT →", 11, AMBER, font=MONO); run(p, "  →  POPULATION", 11, INK3, font=MONO)
_, tf = box(s, 0.85, 4.7, 11.0, 1.3)
p = para(tf, True, line=1.45)
run(p, "Radiology AI localizes the lesion; pathology AI infers molecular features from the same disease; multimodal models combine the two. This is what the next section means by ", 18, INK2)
run(p, "clinical intelligence.", 18, INK, bold=True)
notes(s, "For decades, radiology and pathology have been handled as separate departments, with separate images and separate reports. AI now makes it practical to analyze them together for the same patient. Radiology AI localizes the lesion; pathology AI infers molecular features from the same disease; and multimodal models combine the two. That is where the next section goes, and it is what the talk means by clinical intelligence.")

out = os.path.join(HERE, "IWBI2026_Trivedi_section4_pathology.pptx")
prs.save(out)
print("saved", out, "slides:", len(prs.slides._sldIdLst))
