#!/usr/bin/env python3
"""Close (§7) — editable PPTX (5 slides), hand-tuned to slides/section7.html.
Divider · Lesion/Image/Patient recap · discipline-in-three-lines · the turn ·
thank-you. Inherits gradient/motif background from pptxlib."""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pptxlib import *
from bs4 import BeautifulSoup
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

SLIDES = "/home/user/Mammo/keynote-iwbi-2026/slides"
HIDE = re.compile(r'hide in final|⚑')
soup = BeautifulSoup(open(SLIDES + "/section7.html").read(), 'html.parser')
VIS = [s for s in soup.select('section.slide') if not (HIDE.search(str(s)) or s.get('data-hide'))]
NOTES = [s.get('data-note', '') for s in VIS]
prs = new_prs()

def title2(s, y, a, b, sz=27, w=11.4, x=0.85, line=1.1):
    _, tf = box(s, x, y, w, 1.2); p = para(tf, True, line=line)
    run(p, a, sz, INK, bold=True); run(p, b, sz, AMBER, bold=True)

def body(s, y, runs, sz=15, w=11.4, x=0.85, line=1.45, h=1.6):
    _, tf = box(s, x, y, w, h); p = para(tf, True, line=line)
    for t, col, bold in runs: run(p, t, sz, col, bold=bold)

def labelrow(s, x, y, w, label, head, sub_runs, h=0.95):
    _, tf = box(s, x, y + 0.02, 1.55, h); run(para(tf, True), label.upper(), 11, AMBER, bold=True, font=MONO)
    _, tf = box(s, x + 1.7, y, w - 1.7, h); p = para(tf, True, line=1.35)
    run(p, head + "  ", 15, INK, bold=True)
    for t, c, b in sub_runs: run(p, t, 14, c, bold=b)

def numrow(s, x, y, w, num, head, sub_runs, h=0.95):
    _, tf = box(s, x, y, 0.55, h); run(para(tf, True), num, 14, AMBER, bold=True, font=MONO)
    _, tf = box(s, x + 0.6, y, w - 0.6, h); p = para(tf, True, line=1.32)
    run(p, head + "  ", 15, INK, bold=True)
    for t, c, b in sub_runs: run(p, t, 13.5, c, bold=b)

# ===== 0 divider: what thirty years taught us =====
s = slide(prs); note(s, NOTES[0])
_, tf = box(s, 0.85, 2.3, 9, 0.34); p = para(tf, True)
run(p, "LESION ", 11, INK3, font=MONO); run(p, "→ ", 11, AMBER, font=MONO); run(p, "IMAGE ", 11, INK3, font=MONO); run(p, "→ ", 11, AMBER, font=MONO); run(p, "PATIENT", 11, INK3, font=MONO)
_, tf = box(s, 0.85, 2.75, 11.0, 1.5); p = para(tf, True, line=1.05)
run(p, "What thirty years ", 44, INK, bold=True); run(p, "taught us.", 44, AMBER, bold=True)
body(s, 4.55, [("Technology is fallible.", INK, True), (" We keep getting better at looking ahead — but the consequences of what we deploy are not always ones we foresee.", INK2, False)], sz=18, line=1.45, h=1.5)

# ===== 1 lesion, image, patient =====
s = slide(prs); note(s, NOTES[1]); eyebrow(s, "Close")
title2(s, 1.15, "Lesion, image, ", "patient.", sz=28)
ry = 2.65
labelrow(s, 0.85, ry, 11.6, "Lesion", "Find the spot.",
         [("CAD failed at it; whole-image detection later earned the randomized evidence CAD never had.", INK2, False)]); ry += 1.05
labelrow(s, 0.85, ry, 11.6, "Image", "Beyond the lesion.",
         [("The whole image reads a woman's future ", INK2, False), ("risk", AMBER, True), (" and a ", INK2, False), ("cardiovascular", AMBER, True), (" signal — from the same mammogram.", INK2, False)]); ry += 1.05
labelrow(s, 0.85, ry, 11.6, "Patient", "The whole person.",
         [("Image, tissue, gene, and history together — risk known ", INK2, False), ("before she enters the scanner.", INK, True)]); ry += 1.05

# ===== 2 the discipline, in three lines =====
s = slide(prs); note(s, NOTES[2]); eyebrow(s, "Close")
title2(s, 1.15, "The discipline, ", "in three lines.", sz=28)
ry = 2.7
numrow(s, 0.85, ry, 11.6, "01", "Prove it before you scale it.",
       [("Prospective, randomized evidence — MASAI is the bar, not a retrospective AUC.", INK2, False)]); ry += 1.1
numrow(s, 0.85, ry, 11.6, "02", "Audit the blind spots.",
       [("Explainability and subgroup performance are release criteria — a headline AUC is a sales number.", INK2, False)]); ry += 1.1
numrow(s, 0.85, ry, 11.6, "03", "Preserve clinical judgment; guard against de-skilling.",
       [("Design against automation bias — the clinician should stay a clinician, not a rubber stamp.", INK2, False)]); ry += 1.1

# ===== 3 the image hasn't changed (turn) =====
s = slide(prs); note(s, NOTES[3]); eyebrow(s, "Close")
title2(s, 1.9, "The image hasn't changed — ", "what we read from it will.", sz=30, line=1.12)
body(s, 3.5, [("Twenty-five years ago we could barely mark a suspicious spot. From that same image today: a woman's ", INK2, False), ("cancer risk", INK, True), (" years ahead, her ", INK2, False), ("cardiovascular risk", INK, True), (", and a link to her ", INK2, False), ("tissue and genome.", AMBER, True)], sz=18, line=1.5, h=2.0)

# ===== 4 thank you =====
s = slide(prs); note(s, NOTES[4])
_, tf = box(s, 0.85, 2.4, 11.6, 1.0); run(para(tf, True), "Thank you.", 44, INK, bold=True)
_, tf = box(s, 0.85, 3.7, 11.6, 0.6); run(para(tf, True), "EMORY", 20, AMBER, bold=True, font=MONO)
_, tf = box(s, 0.85, 4.25, 11.6, 0.4); run(para(tf, True), "Radiology & Imaging Sciences", 14, INK2)
_, tf = box(s, 0.85, 4.65, 11.6, 0.4); run(para(tf, True), "HITI Lab", 13, AMBER, font=MONO)
_, tf = box(s, 0.85, 6.5, 11.6, 0.4); run(para(tf, True), "Hari Trivedi, MD  ·  Emory University  ·  IWBI 2026", 12, INK3, font=MONO)

out = "/home/user/Mammo/keynote-iwbi-2026/pptx/sections/IWBI2026_Trivedi_07_close.pptx"
os.makedirs(os.path.dirname(out), exist_ok=True)
save(prs, out)
