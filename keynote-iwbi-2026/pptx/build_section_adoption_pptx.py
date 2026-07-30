#!/usr/bin/env python3
"""Adoption (§6c) — editable PPTX (2 slides), hand-tuned to slides/section_adoption.html.
Numbered reframe cards. Inherits gradient/motif background from pptxlib."""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pptxlib import *
from bs4 import BeautifulSoup
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

SLIDES = "/home/user/Mammo/keynote-iwbi-2026/slides"
HIDE = re.compile(r'hide in final|⚑')
soup = BeautifulSoup(open(SLIDES + "/section_adoption.html").read(), 'html.parser')
VIS = [s for s in soup.select('section.slide') if not (HIDE.search(str(s)) or s.get('data-hide'))]
NOTES = [s.get('data-note', '') for s in VIS]
prs = new_prs()

def title2(s, y, a, b, sz=27, w=11.4, x=0.85):
    _, tf = box(s, x, y, w, 1.0); p = para(tf, True, line=1.1)
    run(p, a, sz, INK, bold=True); run(p, b, sz, AMBER, bold=True)

def numrow(s, x, y, w, num, head, sub_runs, h=1.0):
    _, tf = box(s, x, y, 0.5, h, anchor=MSO_ANCHOR.TOP); run(para(tf, True), num, 15, AMBER, bold=True, font=MONO)
    _, tf = box(s, x + 0.55, y, w - 0.55, h); p = para(tf, True, line=1.34)
    run(p, head + "  ", 15, INK, bold=True)
    for t, c, b in sub_runs: run(p, t, 13.5, c, bold=b)

# ===== 0 one model for everything is the wrong goal =====
s = slide(prs); note(s, NOTES[0]); eyebrow(s, "Adoption")
title2(s, 1.05, "One model for everything is ", "the wrong goal.")
_, tf = box(s, 0.85, 2.2, 11.4, 0.6); p = para(tf, True, line=1.35)
run(p, "We chased better detection for three decades. But look at what we ask a ", 15, INK2)
run(p, "single", 15, INK, bold=True); run(p, " model to do at once:", 15, INK2)
ry = 3.05
numrow(s, 0.85, ry, 11.4, "01", "Hold up across", [("every scanner and vendor", INK, True), (" in the field", INK2, False)], h=0.6); ry += 0.7
numrow(s, 0.85, ry, 11.4, "02", "Perform in women of", [("all ages, all densities, all risk types", INK, True)], h=0.6); ry += 0.7
numrow(s, 0.85, ry, 11.4, "03", "Serve both the", [("expert breast radiologist", INK, True), (" and the ", INK2, False), ("generalist who reads mammo one day a week", INK, True)], h=0.8); ry += 0.95
_, tf = box(s, 0.85, ry + 0.1, 11.4, 0.7); p = para(tf, True, line=1.35)
run(p, "We should seriously reframe how we consider ", 16, INK2); run(p, "mammography-model deployment.", 16, AMBER, bold=True)

# ===== 1 how we should deploy instead =====
s = slide(prs); note(s, NOTES[1]); eyebrow(s, "Adoption")
title2(s, 0.95, "How we should deploy ", "instead.")
_, tf = box(s, 0.85, 2.0, 11.4, 0.4); run(para(tf, True), "Not one model imposed everywhere — three reframes:", 14, INK2)
ry = 2.65
numrow(s, 0.85, ry, 11.6, "1", "Operating points, not one threshold.",
       [("Sensitivity vs. false positives is a ", INK2, False), ("value judgment", INK, True), (", not a setting. Let a model run at a range of operating points, ", INK2, False), ("tailored to the site and the radiologist", INK, True), (" — the expert and the once-a-week reader don't want the same one.", INK2, False)], h=1.2); ry += 1.35
numrow(s, 0.85, ry, 11.6, "2", "Multimodal — and soon.",
       [("A model on imaging alone, or ", INK2, False), ("imaging plus basic clinical data", INK, True), (", will end in user frustration. The model a radiologist trusts pulls in ", INK2, False), ("image, clinical context, risk, and history", INK, True), (" together.", INK2, False)], h=1.1); ry += 1.25
numrow(s, 0.85, ry, 11.6, "3", "The integration tax.",
       [("A perfect model dies if it can't reach PACS. One-off integrations don't scale; onboarding happens through a platform or vendor you already run.", INK2, False)], h=0.9); ry += 1.0
_, tf = box(s, 0.85, 6.75, 11.6, 0.4)
run(para(tf, True), "Sharma, BMC Cancer 2023 (275,900 mammograms) · Eisemann, Nat Med 2025 (PRAIM) · Tejani, Radiology 2024 · Allen, Mayo Clin Proc Digit Health 2024", 9, INK3, font=MONO)

out = "/home/user/Mammo/keynote-iwbi-2026/pptx/sections/IWBI2026_Trivedi_06c_adoption.pptx"
os.makedirs(os.path.dirname(out), exist_ok=True)
save(prs, out)
