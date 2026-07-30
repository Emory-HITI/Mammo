#!/usr/bin/env python3
"""Era II · 2026 → brink of Era III — editable PPTX (20 slides; ICH+PE merged), hand-tuned to
slides/section2b.html. Inherits the gradient/motif background and transparent
SVG charts from pptxlib. Paper figures embedded from the HTML's base64; SVG
charts (PE, Mirai, PAD forest, time machines) embedded transparently."""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pptxlib import *
from bs4 import BeautifulSoup
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

SLIDES = "/home/user/Mammo/keynote-iwbi-2026/slides"
HIDE = re.compile(r'hide in final|⚑')
soup = BeautifulSoup(open(SLIDES + "/section2b.html").read(), 'html.parser')
VIS = [s for s in soup.select('section.slide') if not (HIDE.search(str(s)) or s.get('data-hide'))]
NOTES = [s.get('data-note', '') for s in VIS]
def imgs(i): return [im['src'] for im in VIS[i].find_all('img') if im.get('src', '').startswith('data:')]
def svgs(i): return [str(s) for s in VIS[i].find_all('svg')]

prs = new_prs()

def title2(s, y, w, a, b, sz=26, x=0.85, line=1.12):
    _, tf = box(s, x, y, w, 1.4); p = para(tf, True, line=line)
    if a: run(p, a, sz, INK, bold=True)
    if b: run(p, b, sz, AMBER, bold=True)
    return tf

def body(s, y, w, runs, sz=14, x=0.85, line=1.4, h=1.4):
    _, tf = box(s, x, y, w, h); p = para(tf, True, line=line)
    for t, col, bold in runs: run(p, t, sz, col, bold=bold)
    return tf

def cite(s, y, txt, x=0.85, w=11.6):
    _, tf = box(s, x, y, w, 0.5); run(para(tf, True, line=1.3), txt, 9.5, INK3, font=MONO)

def stat(s, x, y, big, lab, w=2.8, bigsz=30, bigcol=AMBER, labw=None):
    _, tf = box(s, x, y, w, 0.55); run(para(tf, True), big, bigsz, bigcol, bold=True)
    _, tf = box(s, x, y + bigsz / 100.0 + 0.10, labw or w, 0.6); run(para(tf, True, line=1.2), lab, 11, INK2, font=MONO)

from pptx.oxml.ns import qn
def _dash(shape):
    ln = shape.line._get_or_add_ln()
    for old in ln.findall(qn('a:prstDash')): ln.remove(old)
    d = ln.makeelement(qn('a:prstDash'), {'val': 'dash'}); ln.append(d)

def placeholder(s, x, y, w, h, label, sub):
    c = card(s, x, y, w, h, fill=BG, edge=AMBERDP, edge_w=1.0); _dash(c)
    try: c.fill.background()
    except Exception: pass
    tf = c.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    run(para(tf, True, align=PP_ALIGN.CENTER), label, 11, AMBER, font=MONO)
    run(para(tf, align=PP_ALIGN.CENTER, before=3), sub, 9, INK3, font=MONO)

def leftbar(s, x, y, w, runs, sz=14, h=0.9, col=AMBER):
    b = rect(s, x, y, 0.03, h, col)
    _, tf = box(s, x + 0.2, y, w - 0.2, h); p = para(tf, True, line=1.35)
    for t, c, bold in runs: run(p, t, sz, c, bold=bold)

def statline(s, x, y, big, lab, bigcol=AMBER, bigsz=24, w=6.4):
    _, tf = box(s, x, y, w, 0.5, anchor=MSO_ANCHOR.MIDDLE)
    p = para(tf, True)
    run(p, big, bigsz, bigcol, bold=True); run(p, "   " + lab, 13, INK2)

def chiprun(s, x, y, runs, fs=10.5):
    chars = sum(len(t) for t, _ in runs); w = chars * fs * 0.0082 + 0.42
    c = card(s, x, y, w, 0.42, fill=CARD, edge=LINE, edge_w=0.75); rect(s, x, y, 0.045, 0.42, CYAN)
    tf = c.text_frame; tf.word_wrap = False; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.15); tf.margin_right = Inches(0.10); tf.margin_top = Pt(0); tf.margin_bottom = Pt(0)
    p = para(tf, True)
    for t, bold in runs: run(p, t, fs, (CYAN if bold else INK), bold=bold, font=MONO)
    return w

def divider(i, yr, sub_a, sub_b, eb="Era II · 2026"):
    s = slide(prs); note(s, NOTES[i])
    # eyebrow + title + subtitle as one tight, vertically-centred block (matches HTML flexbox)
    _, tf = box(s, 0.85, 2.45, 9, 0.34); run(para(tf, True), eb.upper(), 11, AMBER, bold=True, font=MONO)
    _, tf = box(s, 0.85, 2.85, 6.9, 0.95); p = para(tf, True, line=1.0); run(p, "The year is ", 44, INK, bold=True); run(p, yr + ".", 44, AMBER, bold=True)
    _, tf = box(s, 0.85, 3.95, 6.3, 1.3); p = para(tf, True, line=1.4); run(p, sub_a, 17, INK2); run(p, sub_b, 17, INK, bold=True)
    sv = svgs(i)
    if sv: embed_svg(s, sv[0], 7.8, 1.05, 4.9, 4.7)
    return s

# ===== 0 divider 2026 =====
divider(0, "2026", "Ten years on, the early promise has been put to the test — ", "prospectively, in real screening programs.")

# ===== 1 two prospective studies =====
s = slide(prs); note(s, NOTES[1]); eyebrow(s, "Era II · 2026")
title2(s, 1.05, 11.6, "2026: two prospective studies show ", "AI screening works.", sz=27)
def studycard(x, head, headline, sub, cit):
    c = card(s, x, 2.35, 5.7, 2.55, fill=CARD, edge=LINE, edge_w=0.75); rect(s, x, 2.35, 0.05, 2.55, CYAN)
    tf = c.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.3); tf.margin_right = Inches(0.24); tf.margin_top = Inches(0.2)
    run(para(tf, True), head, 11, CYAN, font=MONO)
    p = para(tf, before=8, line=1.18)
    for t, col, b in headline: run(p, t, 16, col, bold=b)
    p = para(tf, before=7, line=1.3)
    for t, col, b in sub: run(p, t, 13, col, bold=b)
    run(para(tf, before=9), cit, 9, INK3, font=MONO)
studycard(0.85, "MASAI · SWEDEN · RCT (~105,000)",
          [("Interval cancers non-inferior", INK, True), (" (ratio 0.88)", INK3, False)],
          [("Sensitivity ", INK2, False), ("80.5%", INK, True), (" vs 73.8% at matched specificity · ", INK2, False), ("~44%", INK, True), (" less reading workload.", INK2, False)],
          "Gommers/Lång et al., Lancet 2026;407:505–514")
studycard(6.75, "PRAIM · GERMANY · REAL-WORLD (463,000)",
          [("+17.6%", AMBER, True), (" cancers detected", INK, True)],
          [("Recall ", INK2, False), ("non-inferior", INK, True), (" · 6.7 vs 5.7 CDR / 1,000 · 12 sites, 119 radiologists.", INK2, False)],
          "Eisemann et al., Nature Medicine 2025")
leftbar(s, 0.85, 5.35, 11.6, [("A third is underway: ", INK2, False), ("PRISM", INK, True),
     (" — the first large randomized trial of screening AI in the United States (Transpara; $16M PCORI; 7 sites; announced Sept 2025, recruiting). It tests the ", INK2, False), ("US single-reader workflow", INK, True), (" directly.", INK2, False)], sz=13.5, h=1.3)

# ===== 2 what counts as truth (EMBED cascade fig) =====
s = slide(prs); note(s, NOTES[2]); eyebrow(s, "Era II · 2026")
title2(s, 1.15, 6.3, "The averages look strong — but ", "what counts as the truth?", sz=24)
body(s, 3.1, 6.4, [("The label is not in the image — it comes from follow-up and pathology. Assigning it is a cascade of judgment calls (~163,000 EMBED exams, ", INK2, False), ("AUC 0.91", INK, True), ("), and it sets the denominator for ", INK2, False), ("every downstream number.", INK, True)], sz=14.5, h=2.6)
im = imgs(2)
if im: embed_img(s, im[0], 7.6, 1.2, 5.2, 5.6, card_bg='white')

# ===== 3 outcomes and cancer types (Fig 2 + Fig 3) =====
s = slide(prs); note(s, NOTES[3]); eyebrow(s, "Era II · 2026")
title2(s, 1.1, 11.6, "Performance across ", "outcomes and cancer types.", sz=26)
im = imgs(3)
if len(im) >= 2:
    embed_img(s, im[0], 0.85, 2.35, 5.7, 3.9, card_bg='white')
    embed_img(s, im[1], 6.75, 2.35, 5.7, 3.9, card_bg='white')
    _, tf = box(s, 0.85, 6.35, 5.7, 0.3); run(para(tf, True, align=PP_ALIGN.CENTER), "By clinical outcome (Fig. 2)", 10, INK2, font=MONO)
    _, tf = box(s, 6.75, 6.35, 5.7, 0.3); run(para(tf, True, align=PP_ALIGN.CENTER), "By pathology (Fig. 3)", 10, INK2, font=MONO)
cite(s, 6.78, "[your group], Nat Commun 2026 (DOI 10.1038/s41467-026-70637-3), Figs. 2–3.")

# ===== 4 imaging findings (Fig 6) =====
s = slide(prs); note(s, NOTES[4]); eyebrow(s, "Era II · 2026")
title2(s, 1.1, 11.6, "Performance across ", "imaging findings.", sz=26)
body(s, 2.35, 11.6, [("Exam-level model-score distributions by imaging feature, across clinical outcome.", INK2, False)], sz=14, h=0.6)
im = imgs(4)
if im: embed_img(s, im[0], 2.4, 3.0, 8.5, 3.3, card_bg='white')
cite(s, 6.5, "[your group], Nat Commun 2026 (DOI 10.1038/s41467-026-70637-3), Fig. 6.", x=0.85)

# ===== 5 not only breast — combined ICH + PE bar charts =====
s = slide(prs); note(s, NOTES[5]); eyebrow(s, "Era II · 2026 · not only breast")
title2(s, 1.05, 11.6, "The same fragility — ", "beyond breast.", sz=26)
body(s, 2.3, 11.6, [("Two FDA-relevant tools, two organs: overall sensitivity looks strong — then it ", INK2, False), ("collapses in the hard subgroups.", INK, True)], sz=14, h=0.7)
_sv = svgs(5)
_, tf = box(s, 0.85, 3.2, 5.8, 0.3); run(para(tf, True), "HEAD CT · INTRACRANIAL HEMORRHAGE", 10, AMBER, font=MONO)
_, tf = box(s, 6.85, 3.2, 5.8, 0.3); run(para(tf, True), "CHEST CT · PULMONARY EMBOLISM", 10, AMBER, font=MONO)
if len(_sv) >= 1: embed_svg(s, _sv[0], 0.85, 3.65, 5.8, 2.6)
if len(_sv) >= 2: embed_svg(s, _sv[1], 6.85, 3.65, 5.8, 2.6)
cite(s, 6.55, "ICH: [your group], npj Digital Medicine 2025 · PE: [your group], unpublished (Aidoc PE/iPE, 30,678 CTPA).")

# ===== 6 EMBED open data =====
s = slide(prs); note(s, NOTES[6]); eyebrow(s, "Era II · 2026 · open data")
title2(s, 1.1, 6.4, "Emory Breast Imaging Dataset ", "(EMBED).", sz=26)
body(s, 2.3, 6.2, [("Free to researchers through the ", INK2, False), ("AWS Open Data Program.", INK, True)], sz=15, h=0.7)
_, tf = box(s, 0.85, 3.05, 6.4, 0.35); run(para(tf, True), "EMBED V2 — IN PROGRESS", 11.5, AMBER, bold=True, font=MONO)
embed_chips = [[("~300,000 ", False), ("patients", True)],
               [("~1M ", False), ("exams", True), (" (2D · DBT · US · MRI)", False)],
               [("Free-text ", False), ("reports", True)],
               [("Patient ", False), ("risk data", True)],
               [("Outcomes ", False), ("registry-harmonized", True)]]
cx = 0.85; cy = 3.55
for runs in embed_chips:
    w = sum(len(t) for t, _ in runs) * 10.5 * 0.0082 + 0.42
    if cx + w > 7.25: cx = 0.85; cy += 0.55
    chiprun(s, cx, cy, runs); cx += w + 0.16
cite(s, 6.45, "v1: [your group], Radiology: Artificial Intelligence 2023 (Fig. 1) · aws.amazon.com/opendata", w=6.4)
placeholder(s, 7.2, 1.3, 5.6, 4.9, "MAP — EMBED USERS BY COUNTRY", "[ drop in PNG ]")
_, tf = box(s, 7.2, 6.3, 5.6, 0.3); run(para(tf, True, align=PP_ALIGN.CENTER), "opt: EMBED v1 schematic — Radiology: AI, Fig. 1", 9, INK3, font=MONO)

# ===== 7 brink of Era III (turn) =====
s = slide(prs); note(s, NOTES[7]); eyebrow(s, "Era II · 2026 → Era III")
title2(s, 1.6, 11.6, "We are at the ", "brink of Era III.", sz=32)
body(s, 3.2, 11.4, [("Detection is proven. But the same 2026 image — and the tissue that follows it — already tells us more than ", INK2, False), ("“is there cancer today?”", INK, True), (" From one screening mammogram and its biopsy: a ", INK2, False), ("future-risk", AMBER, True), (" estimate, a ", INK2, False), ("cardiovascular", AMBER, True), (" signal, and a ", INK2, False), ("molecular recurrence", AMBER, True), (" score — the shift from finding the lesion to ", INK2, False), ("characterizing the patient.", INK, True)], sz=17, line=1.45, h=3.0)

# ===== 8 density / Gail / TC (quote-stat) =====
s = slide(prs); note(s, NOTES[8]); eyebrow(s, "Era III · 2026")
body(s, 2.0, 6.7, [("For 40 years, the only risk signal we could read off the image was ", INK2, False), ("density", AMBER, False), (". The clinical models layered on top — ", INK2, False), ("Gail", AMBER, False), (" and ", INK2, False), ("Tyrer-Cuzick", AMBER, False), (" — were hardly better than a coin flip.", INK2, False)], sz=24, line=1.32, h=3.2)
cite(s, 5.6, "Gail · Tyrer-Cuzick · BCSC — questionnaire + density models", w=6.7)
ln = rect(s, 8.0, 2.2, 0.012, 2.4, LINE); ln.width = Pt(1)
stat(s, 8.4, 2.35, "~0.62", "Tyrer-Cuzick v8 · 5-yr AUC", w=4.6, bigsz=22, bigcol=INK)
stat(s, 8.4, 3.25, "~0.55–0.58", "Gail model · 5-yr AUC", w=4.6, bigsz=22, bigcol=INK)
body(s, 4.25, 4.6, [("Coin flip is 0.50. ", INK, False), ("For the woman in front of you, barely above it.", INK2, False)], sz=13, x=8.4, line=1.35, h=0.9)

# ===== 9 image-based risk (Mirai bar svg) =====
s = slide(prs); note(s, NOTES[9]); eyebrow(s, "Era III · 2026")
_, tf = box(s, 0.85, 1.1, 6.4, 1.4); p = para(tf, True, line=1.12)
run(p, "Image-based risk", 26, AMBER, bold=True); run(p, " — Mirai and a de novo FDA clearance.", 26, INK, bold=True)
stat(s, 0.85, 3.0, "0.76–0.81", "Mirai C-index · 3 countries", w=6.0, bigsz=40)
stat(s, 0.85, 4.5, "< 5 yr", "research → cleared → guideline", w=6.0, bigsz=40)
sv = svgs(9)
if sv: embed_svg(s, sv[0], 7.0, 1.6, 5.8, 4.6)

# ===== 10 AsymMirai (fig) =====
s = slide(prs); note(s, NOTES[10]); eyebrow(s, "Era III · 2026")
_, tf = box(s, 0.85, 1.1, 6.3, 1.4); p = para(tf, True, line=1.12)
run(p, "AsymMirai", 26, AMBER, bold=True); run(p, " — an explainable risk model.", 26, INK, bold=True)
body(s, 2.9, 6.2, [("1-yr AUC ", INK2, False), ("0.79", INK, True), (" vs Mirai 0.84  ·  3-yr AUC ", INK2, False), ("0.92", INK, True), (" (stable tissue)", INK2, False)], sz=15, h=1.0)
im = imgs(10)
if im: embed_img(s, im[0], 7.0, 1.4, 5.8, 5.0, card_bg='white')

# ===== 11 BAC continuous (segmentation fig) =====
s = slide(prs); note(s, NOTES[11]); eyebrow(s, "Era III · 2026")
title2(s, 1.1, 6.4, "Breast Arterial Calcification (BAC) as a ", "continuous measurement (mm²).", sz=23)
body(s, 3.0, 6.4, [("A transformer segments breast arterial calcification on the routine mammogram and quantifies its area — turning an incidental finding into a ", INK2, False), ("graded, quantitative risk marker.", INK, True)], sz=14.5, h=2.2)
cite(s, 6.5, "Segmentation: SCU-Net (your group). Dapamede et al., Eur Heart J 2026;47(18):2206–2220.", w=6.4)
im = imgs(11)
if im: embed_img(s, im[0], 7.0, 1.3, 5.8, 5.4, card_bg='black')

# ===== 12 BAC predicts MACE (KM fig) =====
s = slide(prs); note(s, NOTES[12]); eyebrow(s, "Era III · 2026")
title2(s, 1.1, 6.4, "BAC predicts MACE — ", "more calcium, more risk.", sz=24)
body(s, 2.7, 6.4, [("~120,000 women", INK, True), (", two sites, racially diverse. Dose-response: ", INK2, False), ("mild HR 1.18–1.22 · moderate 1.38–1.47 · severe 2.03–2.22.", INK, True)], sz=14.5, h=2.0)
cite(s, 6.5, "Event-free survival by BAC severity — MACE, AMI, stroke, HF, death. Dapamede et al., Eur Heart J Open 2025.", w=6.4)
im = imgs(12)
if im: embed_img(s, im[0], 7.0, 1.4, 5.8, 5.0, card_bg='white')

# ===== 13 BAC and PAD (forest svg + angiogram) =====
s = slide(prs); note(s, NOTES[13]); eyebrow(s, "Era III · 2026")
_, tf = box(s, 0.85, 1.1, 6.0, 1.2); p = para(tf, True); run(p, "BAC and ", 26, INK, bold=True); run(p, "PAD.", 26, AMBER, bold=True)
body(s, 2.3, 6.0, [("BAC is ", INK2, False), ("medial", INK, True), (" calcification (Mönckeberg's) — the pattern that dominates PAD. In ", INK2, False), ("59,854 women", INK, True), (" (~8-yr follow-up), it graded the risk of ", INK2, False), ("vascular intervention", INK, True), (" overall and within every high-risk subgroup.", INK2, False)], sz=13.5, h=2.4)
cite(s, 6.5, "[your group], abstract — adj. age, DM, HTN, HLD, CKD, smoking. Angiogram: Radiopaedia.org (CC).", w=6.0)
sv = svgs(13); im = imgs(13)
if sv: embed_svg(s, sv[0], 6.85, 1.3, 4.2, 5.2)
if im: embed_img(s, im[0], 11.05, 2.2, 1.9, 3.0, card_bg='black')

# ===== 14 BAC and valvular disease =====
s = slide(prs); note(s, NOTES[14]); eyebrow(s, "Era III · 2026")
_, tf = box(s, 0.85, 1.1, 6.2, 1.2); p = para(tf, True); run(p, "BAC and ", 26, INK, bold=True); run(p, "valvular disease.", 26, AMBER, bold=True)
body(s, 2.3, 6.2, [("122,011 women.", INK, True), (" Severe BAC vs none — ", INK2, False), ("AS 11.4 · AR 3.1 · MR 2.6 · TR 2.3", INK, True), ("; mitral stenosis null (negative control). 5-yr any-VHD ", INK2, False), ("19.3% vs 4.6%", AMBER, True), (" (NNS 7).", INK2, False)], sz=14, h=2.6)
cite(s, 6.5, "Any VHD by BAC severity. Dapamede et al., Fig. 1. Valve MRI: ECR 2024 EPOS #166904 (used w/ permission).", w=6.2)
im = imgs(14)
if len(im) >= 1: embed_img(s, im[0], 7.1, 1.4, 5.7, 3.4, card_bg='white')
if len(im) >= 2: embed_img(s, im[1], 8.7, 4.95, 2.5, 1.8, card_bg='black')

# ===== 15 BAC and CAC =====
s = slide(prs); note(s, NOTES[15]); eyebrow(s, "Era III · 2026")
_, tf = box(s, 0.85, 1.1, 6.2, 1.2); p = para(tf, True); run(p, "BAC and ", 26, INK, bold=True); run(p, "CAC.", 26, AMBER, bold=True)
body(s, 2.3, 6.2, [("In ", INK2, False), ("2,079 women", INK, True), (" who later had a cardiac CT, detectable ", INK2, False), ("coronary calcium rose 38% → 67%", INK, True), (" across BAC tiers; CAC>100 up ", INK2, False), ("2.5×", AMBER, True), (". BAC independently predicted CAC (OR 1.32 / SD). ", INK2, False), ("The mammogram previews the coronary CT.", INK, True)], sz=14, h=2.8)
cite(s, 6.5, "[your group], RSNA 2026 abstract (BAC–CAC). CAC detection by BAC tier (n=2,079).", w=6.2)
im = imgs(15)
if im: embed_img(s, im[0], 7.1, 1.4, 5.7, 4.8, card_bg='white')

# ===== 16 BAC and costochondral calcification (table) =====
s = slide(prs); note(s, NOTES[16]); eyebrow(s, "Era III · 2026")
_, tf = box(s, 0.85, 1.1, 6.4, 1.2); p = para(tf, True); run(p, "BAC and ", 26, INK, bold=True); run(p, "costochondral calcification.", 26, AMBER, bold=True)
body(s, 2.35, 6.2, [("A ", INK2, False), ("non-vascular", INK, True), (" marker — calcification in the costal cartilage on chest CT (", INK2, False), ("n=1,311", INK, True), ("). Correlated with BAC (ρ=0.089, p=0.001).", INK2, False)], sz=14, h=1.4)
# 2-col table
tx, tw, c1, rh, ty = 0.85, 5.6, 3.9, 0.5, 3.95
rect(s, tx, ty, tw, rh, CARD)  # header bg
_, tf = box(s, tx + 0.12, ty, c1, rh, anchor=MSO_ANCHOR.MIDDLE); run(para(tf, True), "HIGH BAC VS ZERO (ΔCCC)", 10, INK2, font=MONO)
_, tf = box(s, tx + c1, ty, tw - c1 - 0.12, rh, anchor=MSO_ANCHOR.MIDDLE); run(para(tf, True, align=PP_ALIGN.RIGHT), "ADJ. P", 10, INK2, font=MONO)
_, tf = box(s, tx + 0.12, ty + rh, c1, rh, anchor=MSO_ANCHOR.MIDDLE); run(para(tf, True), "Median (q0.50)", 13, INK2)
_, tf = box(s, tx + c1, ty + rh, tw - c1 - 0.12, rh, anchor=MSO_ANCHOR.MIDDLE); run(para(tf, True, align=PP_ALIGN.RIGHT), "NS · 0.47", 13, INK2)
rect(s, tx, ty + 2 * rh, tw, rh, RGBColor(0x22, 0x2C, 0x37))  # highlight row
_, tf = box(s, tx + 0.12, ty + 2 * rh, c1, rh, anchor=MSO_ANCHOR.MIDDLE); p = para(tf, True); run(p, "75th pct (q0.75)  ", 13, INK, bold=True); run(p, "+3,408 mm³", 13, AMBER, bold=True)
_, tf = box(s, tx + c1, ty + 2 * rh, tw - c1 - 0.12, rh, anchor=MSO_ANCHOR.MIDDLE); run(para(tf, True, align=PP_ALIGN.RIGHT), "<0.001", 13, INK, bold=True)
leftbar(s, 0.85, 5.7, 6.2, [("Vascular + valvular + skeletal — one free mammographic finding.", INK, False)], sz=13.5, h=0.7)
cite(s, 6.55, "[your group], RSNA 2026 abstract (BAC–CCC) · quantile regression, adj. age + DM.", w=6.4)
placeholder(s, 7.5, 1.3, 5.3, 5.0, "COSTOCHONDRAL CALCIFICATION", "[ drop in — confirm reuse rights ]")
_, tf = box(s, 7.5, 6.4, 5.3, 0.3); run(para(tf, True, align=PP_ALIGN.CENTER), "Illustrative. Source: Sarıyıldız et al., via ResearchGate.", 9, INK3, font=MONO)

# ===== 17 I-BAC pragmatic trial (fig) =====
s = slide(prs); note(s, NOTES[17]); eyebrow(s, "Era III · 2026")
title2(s, 1.1, 6.4, "Closing the loop: the ", "I-BAC pragmatic trial.", sz=24)
body(s, 2.7, 6.4, [("The gap: ", INK2, False), ("<7% (Emory) / 15% (Mayo)", INK, True), (" of women had the chart data to even compute a PREVENT/ASCVD score. ", INK2, False), ("I-BAC", AMBER, True), (" (R01HL167811) is a Hybrid Type 1 pragmatic RCT across Emory + Mayo AZ. Primary endpoint: 6-mo composite of ", INK2, False), ("lipid panel · preventive-med change · CAC testing.", INK, True)], sz=13.5, h=3.0)
cite(s, 6.5, "Deployment + pragmatic-trial structure (Aim 2). Precedent: NOTIFY-PICTURE — BAC has no guideline yet.", w=6.4)
im = imgs(17)
if im: embed_img(s, im[0], 7.0, 1.7, 5.9, 4.0, card_bg='white')

# ===== 18 verge of multimodal (turn) =====
s = slide(prs); note(s, NOTES[18]); eyebrow(s, "Era III · 2026 → 2035")
title2(s, 1.6, 11.6, "And we are on the verge of ", "multimodal, multi-omics.", sz=30)
body(s, 3.2, 11.4, [("Everything so far is still ", INK2, False), ("one model, one modality", INK, True), (" — the image, or the calcium, or the tissue, read separately. The frontier is to combine them — ", INK2, False), ("imaging + pathology + genomics + the clinical record", AMBER, True), (" — into a single model of the patient. To see where that goes, we have to leave 2026.", INK2, False)], sz=17, line=1.45, h=2.8)

# ===== 19 divider 2035 =====
divider(19, "2035", "From here, today's models are a decade old. The real question: what happens when imaging, pathology, and omics finally converge — ", "on one patient.", eb="Era III · 2035")

out = "/home/user/Mammo/keynote-iwbi-2026/pptx/sections/IWBI2026_Trivedi_02b_EraII_2026.pptx"
os.makedirs(os.path.dirname(out), exist_ok=True)
save(prs, out)
