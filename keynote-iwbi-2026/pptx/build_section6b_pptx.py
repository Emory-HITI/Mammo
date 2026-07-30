#!/usr/bin/env python3
"""Frontier models (§6b) — editable PPTX (7 slides), hand-tuned to
slides/section6b.html. Inherits gradient/motif background + transparent SVGs.
Bar chart (slide 2) transparent; hallucination taxonomy figure embedded."""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pptxlib import *
from bs4 import BeautifulSoup
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

SLIDES = "/home/user/Mammo/keynote-iwbi-2026/slides"
HIDE = re.compile(r'hide in final|⚑')
soup = BeautifulSoup(open(SLIDES + "/section6b.html").read(), 'html.parser')
VIS = [s for s in soup.select('section.slide') if not (HIDE.search(str(s)) or s.get('data-hide'))]
NOTES = [s.get('data-note', '') for s in VIS]
def imgs(i): return [im['src'] for im in VIS[i].find_all('img') if im.get('src', '').startswith('data:')]
def svgs(i): return [str(s) for s in VIS[i].find_all('svg')]
EB = "Frontier models"
prs = new_prs()

def title2(s, y, w, a, b, sz=26, x=0.85, line=1.12):
    _, tf = box(s, x, y, w, 1.4); p = para(tf, True, line=line)
    if a: run(p, a, sz, INK, bold=True)
    if b: run(p, b, sz, AMBER, bold=True)

def body(s, y, w, runs, sz=14, x=0.85, line=1.42, h=2.0):
    _, tf = box(s, x, y, w, h); p = para(tf, True, line=line)
    for t, col, bold in runs: run(p, t, sz, col, bold=bold)

def cite(s, y, txt, x=0.85, w=11.6):
    _, tf = box(s, x, y, w, 0.5); run(para(tf, True, line=1.3), txt, 9.5, INK3, font=MONO)

def leftbar(s, x, y, w, runs, sz=15, h=1.0, col=AMBER):
    rect(s, x, y, 0.03, h, col)
    _, tf = box(s, x + 0.22, y, w - 0.22, h); p = para(tf, True, line=1.4)
    for t, c, bold in runs: run(p, t, sz, c, bold=bold)

def chiprun(s, x, y, runs, fs=10.5):
    chars = sum(len(t) for t, _ in runs); w = chars * fs * 0.0082 + 0.42
    c = card(s, x, y, w, 0.42, fill=CARD, edge=LINE, edge_w=0.75); rect(s, x, y, 0.045, 0.42, CYAN)
    tf = c.text_frame; tf.word_wrap = False; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.15); tf.margin_right = Inches(0.10); tf.margin_top = Pt(0); tf.margin_bottom = Pt(0)
    p = para(tf, True)
    for t, bold in runs: run(p, t, fs, (CYAN if bold else INK), bold=bold, font=MONO)
    return w

def chiprow(s, x, y, chips, gap=0.16, maxx=12.5):
    cx = x; cy = y
    for runs in chips:
        w = sum(len(t) for t, _ in runs) * 10.5 * 0.0082 + 0.42
        if cx + w > maxx: cx = x; cy += 0.56
        chiprun(s, cx, cy, runs); cx += w + gap
    return (cy - y) + 0.42

# ===== 0 divider: will the generalists absorb the field =====
s = slide(prs); note(s, NOTES[0])
_, tf = box(s, 0.85, 2.3, 9, 0.34); p = para(tf, True)
run(p, "SPECIALIST ", 11, INK3, font=MONO); run(p, "vs ", 11, INK3, font=MONO); run(p, "GENERALIST", 11, AMBER, bold=True, font=MONO); run(p, " · AN OPEN QUESTION", 11, INK3, font=MONO)
_, tf = box(s, 0.85, 2.75, 10.5, 1.7); p = para(tf, True, line=1.05)
run(p, "Will the generalists ", 44, INK, bold=True); run(p, "absorb the field?", 44, AMBER, bold=True)
body(s, 4.65, 10.8, [("Every model in this talk was ", INK2, False), ("purpose-built.", INK, True), (" The frontier models from OpenAI, Google, and Anthropic were not built for medicine — yet they improve every few months. ", INK2, False), ("Do we still build our own?", INK, True)], sz=17, line=1.45, h=1.6)

# ===== 1 where generalists perform well =====
s = slide(prs); note(s, NOTES[1]); eyebrow(s, EB)
title2(s, 1.35, 11.0, "Where the generalists ", "already perform well.", sz=27)
body(s, 2.7, 11.4, [("Medical knowledge:", INK, True), (" frontier models score ", INK2, False), ("~90%+ on USMLE-style exams", INK, True), (" (Med-Gemini 91.1%), across the breadth of medicine. ", INK2, False), ("Rapid adaptation:", INK, True), (" fine-tuned on small labeled sets they often exceed bespoke models, and generalize better across sites, scanners, and populations.", INK2, False)], sz=15, line=1.45, h=1.6)
chiprow(s, 0.85, 4.5, [
    [("USMLE-style: ", False), ("~91%", True), (" (Med-Gemini)", False)],
    [("2D CXR reports ", False), ('"equivalent or better"', True), (" in ", False), ("43–96%", True), (" of cases", False)],
    [("exceeded prior SOTA on ", False), ("17/20", True), (" CXR tasks", False)]])
cite(s, 5.6, "Med-Gemini — Yang et al., arXiv:2405.03162, 2024 (preprint)")

# ===== 2 where they fall short — bar chart =====
s = slide(prs); note(s, NOTES[2]); eyebrow(s, EB)
title2(s, 1.5, 6.4, "Where they still fall short — ", "the image itself.", sz=24)
body(s, 3.1, 6.4, [("Their strength is ", INK2, False), ("text, not image content.", INK, True), (" On radiology interpretation, general VLM diagnostic accuracy ranges only ", INK2, False), ("~8–29%", INK, True), (", and roughly ", INK2, False), ("22%", INK, True), (" of reports contain factual errors. Purpose-built ", INK2, False), ("RadFM outperforms GPT-4V", INK, True), ("; Med-Gemini-3D CT reports were only ", INK2, False), ("53% clinically acceptable.", INK, True)], sz=14, line=1.45, h=3.0)
cite(s, 6.6, "Visual LLMs in Radiology, Life (MDPI) 2026;16(1):66 · RadFM — Wu et al., Nat Commun 2025 · Med-Gemini, 2024", w=6.4)
sv = svgs(2)
if sv: embed_svg(s, sv[0], 7.0, 1.6, 5.8, 4.4)

# ===== 3 specializing does not make it safer — taxonomy fig =====
s = slide(prs); note(s, NOTES[3]); eyebrow(s, EB)
title2(s, 1.25, 6.3, "Specializing a model ", "does not make it safer.", sz=24)
body(s, 2.85, 6.3, [("Medical hallucination is not one failure but many — ", INK2, False), ("factual errors, outdated references, spurious correlations, fabricated sources, and broken chains of reasoning", INK, True), (" — across diagnosis, procedure, and research.", INK2, False)], sz=13.5, line=1.4, h=1.6)
body(s, 4.45, 6.3, [("On a hallucination benchmark (", INK2, False), ("Med-HALT, 2025", INK, True), ("), general models were hallucination-free ", INK2, False), ("76.6%", AMBER, True), (" of the time vs only ", INK2, False), ("51.3%", WARN, True), (' for "specialized" models. The instinct to ', INK2, False), ("buy the medical one may be wrong.", INK, True)], sz=13.5, line=1.4, h=1.8)
cite(s, 6.6, "Taxonomy: Kim, Jeong et al., \"Medical Hallucination in Foundation Models,\" arXiv:2503.05777 · Med-HALT, arXiv:2307.15343", w=6.3)
im = imgs(3)
if im: embed_img(s, im[0], 7.0, 1.4, 5.8, 5.0, card_bg='white')

# ===== 4 the bitter lesson (turn) =====
s = slide(prs); note(s, NOTES[4]); eyebrow(s, EB)
title2(s, 1.7, 11.6, "The thesis: ", 'the "bitter lesson."', sz=30)
leftbar(s, 0.85, 3.0, 11.0, [("Over decades, ", INK2, False), ("general methods that scale with data and compute", INK, True), (" have repeatedly outperformed systems built on handcrafted domain knowledge — in chess, in vision, in language.", INK2, False)], sz=16, h=1.1)
body(s, 4.5, 11.4, [("Radiology AI has spent fifteen years handcrafting. The trajectory of other fields suggests the generalists eventually take over perception too. ", INK2, False), ("The open question is when — and what we build in the meantime.", INK, True)], sz=15, line=1.45, h=1.6)
cite(s, 6.2, "Rich Sutton, \"The Bitter Lesson,\" 2019 essay (an argument, not a study)")

# ===== 5 orchestration (primer) =====
s = slide(prs); note(s, NOTES[5]); eyebrow(s, EB)
title2(s, 1.35, 11.0, "A likely answer: ", "orchestration.", sz=27)
body(s, 2.7, 11.4, [("The probable future is a ", INK2, False), ("stack", INK, True), (", not a replacement. A ", INK2, False), ("frontier model as the reasoning and orchestration layer", INK, True), (" — it reads the EHR, talks to the patient, writes the report, decides the next step — calling ", INK2, False), ("specialist models as instruments:", INK, True), (" the detector, the risk model, the BAC quantifier, the tissue model.", INK2, False)], sz=15, line=1.45, h=1.8)
chiprow(s, 0.85, 4.75, [
    [("generalist = reasoning & orchestration", False)],
    [("specialists = the instruments it calls", False)],
    [("increasingly ", False), ("domain-adapted", True), (" (Med-Gemini)", False)]])
body(s, 5.65, 11.4, [('Measurement still rewards purpose-built precision and calibration. "Agentic radiology": the generalist conducts; the specialists play.', INK2, False)], sz=13, line=1.4, h=0.8)

# ===== 6 what it means for this room (turn) =====
s = slide(prs); note(s, NOTES[6]); eyebrow(s, EB)
title2(s, 1.4, 11.6, "What it means ", "for this room.", sz=30)
chiprow(s, 0.85, 3.0, [
    [("Value moves", True), (" to data, validation, integration — not architecture", False)],
    [("Concentration", True), (" is a governance problem: cost, sovereignty, access", False)]])
chiprow(s, 0.85, 3.7, [[("The radiologist", True), (" shifts toward validator & orchestrator", False)]])
body(s, 4.65, 11.4, [("Whether the result is a specialist model or a frontier generalist, built in Palo Alto or in this room, it does not change what makes it safe. ", INK2, False), ("The bar is the same one CAD failed and MASAI cleared.", INK, True)], sz=16, line=1.45, h=1.6)

out = "/home/user/Mammo/keynote-iwbi-2026/pptx/sections/IWBI2026_Trivedi_06b_frontier.pptx"
os.makedirs(os.path.dirname(out), exist_ok=True)
save(prs, out)
