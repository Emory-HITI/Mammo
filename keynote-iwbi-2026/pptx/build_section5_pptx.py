#!/usr/bin/env python3
"""Era III · 2035 · the whole patient (Prevention) — editable PPTX (5 slides),
hand-tuned to slides/section5alternate.html. Inherits gradient/motif background
and transparent SVG charts from pptxlib. Each slide carries one generated SVG
(Venn, reading queue, fusion engine, whole-body streams, two-tier), embedded
transparently so the slide background shows through."""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pptxlib import *
from bs4 import BeautifulSoup
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

SLIDES = "/home/user/Mammo/keynote-iwbi-2026/slides"
HIDE = re.compile(r'hide in final|⚑')
soup = BeautifulSoup(open(SLIDES + "/section5alternate.html").read(), 'html.parser')
VIS = [s for s in soup.select('section.slide') if not (HIDE.search(str(s)) or s.get('data-hide'))]
NOTES = [s.get('data-note', '') for s in VIS]
def svg0(i):
    sv = VIS[i].find('svg'); return str(sv) if sv is not None else None

EB = "Era III · 2035 · the whole patient"
prs = new_prs()

def head(s, i):
    note(s, NOTES[i]); eyebrow(s, EB)

def title2(s, y, a, b, sz=26, w=6.5, x=0.85):
    _, tf = box(s, x, y, w, 1.5); p = para(tf, True, line=1.12)
    run(p, a, sz, INK, bold=True); run(p, b, sz, AMBER, bold=True)

def body(s, y, runs, sz=14, w=6.5, x=0.85, line=1.4, h=2.0):
    _, tf = box(s, x, y, w, h); p = para(tf, True, line=line)
    for t, col, bold in runs: run(p, t, sz, col, bold=bold)

def cite(s, y, txt, w=6.5, x=0.85):
    _, tf = box(s, x, y, w, 0.6); run(para(tf, True, line=1.3), txt, 9.5, INK3, font=MONO)

def stat(s, x, y, big, lab, bigcol=AMBER, w=6.3, bigsz=30):
    _, tf = box(s, x, y, w, 0.5); run(para(tf, True), big, bigsz, bigcol, bold=True)
    _, tf = box(s, x, y + 0.46, w, 0.5); run(para(tf, True, line=1.2), lab, 11, INK2, font=MONO)

# ===== 0 She knows her risk (Venn) =====
s = slide(prs); head(s, 0)
title2(s, 2.0, "She knows her risk ", "before she ever lies down.", sz=27)
body(s, 3.5, [("Risk in 2035 is built from ", INK2, False), ("four independent inputs", INK, True),
     (" — polygenic, germline, clinical, and ", INK2, False), ("imaging", INK, True),
     (" (prior mammograms) — fused into one engine and updated at every scan.", INK2, False)], sz=15, line=1.45, h=2.0)
cite(s, 5.4, "Shieh, JAMA Oncol 2026 · Esserman (WISDOM), JAMA 2025 · Yala (Mirai), Sci Transl Med 2021 · Clairity FDA 2025")
if svg0(0): embed_svg(s, svg0(0), 7.6, 1.3, 5.2, 5.2)

# ===== 1 second reader is software (queue) =====
s = slide(prs); head(s, 1)
title2(s, 1.15, "The second reader is now ", "software.", sz=27)
body(s, 2.5, [("Double reading has been replaced — not by one study, but by ", INK2, False),
     ("convergent RCT evidence across four continents", INK, True),
     (". AI clears roughly two-thirds of normal screens; the radiologist takes the rest.", INK2, False)], sz=14, line=1.4, h=1.7)
stat(s, 0.85, 4.15, "−44% / +29%", "workload / cancers detected — MASAI (Lancet 2026)", bigsz=26)
stat(s, 0.85, 4.95, "−63.6%", "radiologist workload — Elías-Cabot (Nat Med 2026)", bigsz=26)
stat(s, 0.85, 5.75, "~99.9%", "NPV of AI-normal triage at ~6/1,000 prevalence", bigcol=CYAN, bigsz=26)
cite(s, 6.7, "MASAI, Lancet 2026 · PRAIM, Nat Med 2025 · Elías-Cabot, Nat Med 2026 · AI-STREAM · GEMINI · ScreenTrustCAD")
if svg0(1): embed_svg(s, svg0(1), 7.6, 1.5, 5.2, 4.8)

# ===== 2 multimodal engine (full-width fusion SVG) =====
s = slide(prs); head(s, 2)
title2(s, 1.0, "From one score to a multimodal ", "detection, treatment, and recurrence engine.", sz=22, w=11.8)
body(s, 1.95, [("Each layer's 2025 evidence, and where 2035 takes it — ", INK2, False),
     ("no single image makes the call, and no modality is acquired unless the model expects it to change the answer.", INK, True)], sz=13, w=11.8, line=1.35, h=0.8)
if svg0(2): embed_svg(s, svg0(2), 0.85, 2.75, 11.6, 3.6)
cite(s, 6.55, "MASAI (Lancet 2026) · PRAIM (Nat Med 2025) · ScreenTrustMRI (Nat Med 2024) · BMU-Net · UNI · BINDS · ctDNA MRD.", w=11.8)

# ===== 3 whole-body health visit (streams) =====
s = slide(prs); head(s, 3)
title2(s, 1.05, "The mammogram is now a ", "whole-body health visit.", sz=26)
body(s, 2.35, [("We built a cancer screen and found a ", INK2, False), ("window into the entire body", INK, True),
     (" — and we are not the only ones.", INK2, False)], sz=14.5, line=1.4, h=1.0)
stat(s, 0.85, 3.35, "2.8×", "5-yr mortality, severe BAC — 123,762 (Eur Heart J 2026)", bigsz=26)
stat(s, 0.85, 4.15, "49,196", "women: CVD from the image alone (Heart 2026)", bigsz=26)
stat(s, 0.85, 4.95, "biological age", "from the mammogram, ±4–6 yr — Mammo-AGE", bigcol=CYAN, bigsz=26)
body(s, 5.85, [("~40M US mammograms/yr", INK, True), (" — often her only reliable preventive touchpoint. By 2035 the modality stops mattering: one model, any routine image, one systemic-risk profile.", INK2, False)], sz=12.5, line=1.35, h=1.0)
cite(s, 6.95, "Dapamede, Eur Heart J 2026 · Barraclough, Heart 2026 · Pan (Mammo-AGE), Nat Commun 2025 · RETFound, Nature 2023")
if svg0(3): embed_svg(s, svg0(3), 7.7, 1.4, 5.1, 5.0)

# ===== 4 frontier plateaued (two-tier) =====
s = slide(prs); head(s, 4)
title2(s, 1.05, "The frontier plateaued. ", "The specialists compounded.", sz=25)
_, tf = box(s, 0.85, 2.45, 6.5, 0.8)
p = para(tf, True, line=1.25)
run(p, "Scale was no silver bullet — the failure was ", 16, INK, bold=True); run(p, "architectural.", 16, AMBER, bold=True)
body(s, 3.5, [("Frontier models aced the exam, then degraded on real clinical data — a model trained on the whole internet ", INK2, False),
     ("reasons from text priors, not the pixels in front of it.", INK, True),
     (" By 2035 that hardened into market structure: generalists won the ", INK2, False), ("data tier", AMBER, True),
     (" (extract, harmonize, label — human signs off); domain models own the ", INK2, False), ("disease tier.", AMBER, True)], sz=13.5, line=1.4, h=2.6)
cite(s, 6.4, "Wu, JMIR 2025 (e84120) · Raji/Daneshjou/Alsentzer, NEJM AI 2025 · RadFlag · Abdullah & Kim, JMIR 2025 · ScaleMAI · Mammo-CLIP.")
if svg0(4): embed_svg(s, svg0(4), 7.7, 1.3, 5.1, 5.2)

out = "/home/user/Mammo/keynote-iwbi-2026/pptx/sections/IWBI2026_Trivedi_05_EraIII_prevention.pptx"
os.makedirs(os.path.dirname(out), exist_ok=True)
save(prs, out)
