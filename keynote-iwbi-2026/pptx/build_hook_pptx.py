#!/usr/bin/env python3
"""Hook section — editable PPTX, hand-tuned to match slides/hook.html (current
6-slide structure): title · disclosures · normal-mammogram · what-are-we-not-
seeing · spine · time-machine. One text box per blurb; native cards/chips;
mammogram viewport as a dark placeholder (HTML uses a <canvas>)."""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pptxlib import *
from bs4 import BeautifulSoup
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

SLIDES = "/home/user/Mammo/keynote-iwbi-2026/slides"
H = open(SLIDES + "/hook.html").read()
soup = BeautifulSoup(H, 'html.parser')
secs = soup.select('section.slide')
notes = [s.get('data-note', '') for s in secs]
SVGS = re.findall(r'<svg[\s\S]*?</svg>', H)

prs = new_prs()


def viewport(s, l, t, w, h, head_left="R MLO · screening", head_right="AI —"):
    panel = card(s, l, t, w, h, fill=PANEL, edge=AMBERDP, edge_w=0.75)
    _, tf = box(s, l + 0.12, t + 0.06, w - 0.24, 0.28, anchor=MSO_ANCHOR.MIDDLE)
    run(para(tf, True), head_left, 9, INK2, font=MONO)
    _, tf = box(s, l + 0.12, t + 0.06, w - 0.24, 0.28, anchor=MSO_ANCHOR.MIDDLE)
    run(para(tf, True, align=PP_ALIGN.RIGHT), head_right, 9, CYAN, font=MONO)
    g = rect(s, l + 0.08, t + 0.42, w - 0.16, 0.01, LINE); g.height = Pt(1)
    rect(s, l + 0.12, t + 0.52, w - 0.24, h - 0.66, BLACK)
    _, tf = box(s, l + 0.18, t + h - 0.42, w - 0.36, 0.3)
    run(para(tf, True), "[ de-identified Emory screening case ]", 8, INK3, font=MONO)
    return panel


def chip_card(s, l, t, text, fs=11):
    """cyan-left-accent rounded chip sized to its text; returns width used."""
    w = len(text) * 0.55 * fs / 72 + 0.30          # mono char width ~0.55em
    c = card(s, l, t, w, 0.40, fill=CARD, edge=LINE, edge_w=0.75)
    ac = rect(s, l, t, 0.045, 0.40, CYAN)
    tf = c.text_frame; tf.word_wrap = False; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.14); tf.margin_right = Inches(0.10)
    tf.margin_top = Pt(0); tf.margin_bottom = Pt(0)
    run(para(tf, True), text, fs, INK, font=MONO)
    return w


def chip_row(s, x, y, maxw, items, fs=11, gap=0.12, rowh=0.40, vgap=0.10):
    """flow chips left-to-right, wrapping; returns total height used."""
    cx, cy = x, y
    for it in items:
        w = len(it) * 0.55 * fs / 72 + 0.30
        if cx + w > x + maxw and cx > x:
            cx = x; cy += rowh + vgap
        chip_card(s, cx, cy, it, fs=fs)
        cx += w + gap
    return (cy - y) + rowh


# ============ SLIDE 1 — title card ============
s = slide(prs); note(s, notes[0])
_, tf = box(s, 0.85, 0.95, 11.6, 0.4)
run(para(tf, True), "International Workshop on Breast Imaging · 2026", 12, CYAN, bold=True, font=MONO)
_, tf = box(s, 0.85, 2.35, 11.6, 2.2)
run(para(tf, True, line=0.98), "Thirty Years,", 54, INK, bold=True)
run(para(tf, line=0.98), "Three Eras.", 54, AMBER, bold=True)
_, tf = box(s, 0.85, 4.95, 9.5, 0.9)
run(para(tf, True, line=1.3), "AI in Breast Imaging — from CAD to Clinical Intelligence", 22, INK2)
_, tf = box(s, 0.85, 6.1, 11.6, 0.5)
p = para(tf, True)
run(p, "Hari Trivedi, MD", 14, INK, bold=True, font=MONO)
run(p, "  ·  Emory University", 14, INK3, font=MONO)

# ============ SLIDE 2 — disclosures ============
s = slide(prs); note(s, notes[1])
eyebrow(s, "Disclosures")
_, tf = box(s, 0.85, 1.18, 11.6, 0.8)
run(para(tf, True), "Disclosures", 38, INK, bold=True)
disc = soup.select_one('.s-discl')
roles = []
for grp in disc.select('.discl-roles > div'):
    k = grp.select_one('.drole-k')
    chips = [c.get_text(' ', strip=True) for c in grp.select('.dchip')]
    if k and chips: roles.append((k.get_text(' ', strip=True), chips))
supp_k = disc.select_one('.dsupp-k')
supp = [c.get_text(' ', strip=True) for c in disc.select('.discl-support .dchip')]
cy = 2.55
for k, chips in roles:
    _, tf = box(s, 0.85, cy, 11.6, 0.26)
    run(para(tf, True), k.upper(), 12, AMBER, bold=True, font=MONO); cy += 0.34
    cy += chip_row(s, 0.85, cy, 11.6, chips) + 0.20
if supp_k:
    _, tf = box(s, 0.85, cy, 11.6, 0.26)
    run(para(tf, True), supp_k.get_text(' ', strip=True).upper(), 12, AMBER, bold=True, font=MONO); cy += 0.34
    chip_row(s, 0.85, cy, 11.9, supp)

# ============ SLIDE 3 — normal mammogram (s-open) ============
s = slide(prs); note(s, notes[2])
viewport(s, 0.85, 1.15, 4.1, 5.5, head_right="AI —")
_, tf = box(s, 5.55, 2.5, 7.0, 2.4)
run(para(tf, True, line=1.12), "A normal screening mammogram — read as normal.", 34, INK, bold=True)

# ============ SLIDE 4 — what are we not seeing (s-report) ============
s = slide(prs); note(s, notes[3])
viewport(s, 0.85, 1.0, 3.55, 5.6, head_right="AI ✓✓✓")
RX = 4.95; RW = 7.55
_, tf = box(s, RX, 2.0, RW, 1.4)
p = para(tf, True, line=1.05)
run(p, "What are we ", 40, INK, bold=True)
run(p, "not seeing?", 40, AMBER, bold=True)
_, tf = box(s, RX, 3.55, RW, 1.6)
p = para(tf, True, line=1.32)
run(p, "AI reads this ", 18, INK)
run(p, "same mammogram", 18, AMBER, bold=True)
run(p, " and her biopsy slide — not just the ", 18, INK)
run(p, "lesion", 18, INK, bold=True)
run(p, ", but the ", 18, INK)
run(p, "patient.", 18, AMBER, bold=True)

# ============ SLIDE 5 — spine (from lesions to patients) ============
s = slide(prs); note(s, notes[4])
_, tf = box(s, 0.85, 1.5, 11.6, 1.2)
p = para(tf, True, line=1.04)
run(p, "From lesions to ", 44, INK, bold=True)
run(p, "patients.", 44, AMBER, bold=True)
steps = [("01", "Past", "Lesion", "find & characterize"),
         ("02", "Present", "Image", "risk & opportunistic signal"),
         ("03", "Future", "Patient", "pathology + integration")]
fills = [CARD, RGBColor(0x3A, 0x2E, 0x16), AMBER]
edges = [LINE, AMBERDP, AMBERDP]
sx_l, sw_w, gap, sy, sh = 0.85, 3.75, 0.275, 3.3, 2.5
for i, (ix, era, word, desc) in enumerate(steps):
    l = sx_l + i * (sw_w + gap); solid = (i == 2)
    c = card(s, l, sy, sw_w, sh, fill=fills[i], edge=edges[i], edge_w=1.25)
    rect(s, l + 0.25, sy + 0.22, 0.45, 0.035, BLACK if solid else AMBER)
    tf = c.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = Inches(0.28); tf.margin_right = Inches(0.24)
    tf.margin_top = Inches(0.45); tf.margin_bottom = Inches(0.2)
    ink_word = RGBColor(0x1A, 0x12, 0x06) if solid else INK
    ink_era = RGBColor(0x1A, 0x12, 0x06) if solid else AMBER
    ink_desc = RGBColor(0x2A, 0x1E, 0x0A) if solid else INK2
    ink_ix = RGBColor(0x40, 0x30, 0x14) if solid else INK3
    run(para(tf, True), ix, 11, ink_ix, font=MONO)
    run(para(tf, before=4), era.upper(), 10, ink_era, bold=True, font=MONO)
    run(para(tf, before=6), word, 26, ink_word, bold=True)
    run(para(tf, before=6, line=1.25), desc, 12.5, ink_desc)

# ============ SLIDE 6 — step into a time machine ============
s = slide(prs); note(s, notes[5])
_, tf = box(s, 1.0, 1.25, 11.3, 1.1)
p = para(tf, True, align=PP_ALIGN.CENTER)
run(p, "Let's step into a ", 40, INK, bold=True); run(p, "time machine.", 40, AMBER, bold=True)
if SVGS: embed_svg(s, SVGS[-1], 4.15, 2.5, 5.0, 3.8, center=True)
_, tf = box(s, 1.0, 6.55, 11.3, 0.5)
run(para(tf, True, align=PP_ALIGN.CENTER), "This field has overpromised before. Let's start there — in 1998.", 14, INK2)

out = "/home/user/Mammo/keynote-iwbi-2026/pptx/sections/IWBI2026_Trivedi_00_hook.pptx"
save(prs, out)
