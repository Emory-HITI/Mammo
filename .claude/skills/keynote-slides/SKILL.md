---
name: keynote-slides
description: >
  Build, edit, and export the breast-imaging keynote slides in this repo —
  self-contained dark-themed HTML decks in keynote-iwbi-2026/slides/ and
  editable PowerPoint exports in keynote-iwbi-2026/pptx/. Use whenever the
  task involves creating/editing keynote slides, the deck's HTML sections,
  the all_slides review deck, or generating/matching the .pptx files.
---

# Keynote slide system (IWBI breast-imaging deck)

Speaker: **Hari Trivedi, MD (Emory)**. Tone: plain, factual, scientific — **no
hyperbole, drama, or cheesy lines**. Real figures embedded base64 with tiny-font
attribution. The HTML decks are the **source of truth**; the PPTX is generated to
match them.

## Repo map
- `keynote-iwbi-2026/slides/*.html` — one self-contained HTML deck per section
  (`hook`, `section1`, `section2`, `section2b`, `section5alternate`,
  `section6b`, `section_adoption`, `section7`). Each = inline CSS/JS, base64
  images, `.slide{position:absolute;inset:0}` single-view nav.
- `keynote-iwbi-2026/slides/all_slides.html` — combined review deck (iframes);
  rebuilt by `slides/build_combined.py`.
- `keynote-iwbi-2026/pptx/` — editable PowerPoint pipeline (below).
- `*.md` files alongside = editable markdown mirrors of each section.

## Golden workflow
1. **Edit the HTML section file** (source of truth) — never edit `all_slides.html` directly.
2. **Rebuild the combined HTML deck:** `python3 keynote-iwbi-2026/slides/build_combined.py`.
3. **QA the HTML:** `python3 keynote-iwbi-2026/pptx/render_html.py <section> <out_dir>`
   → 2-col contact sheet `HTML_<section>.png` + per-slide `h_<section>_NN.png`.
4. **Rebuild the PPTX** for that section (below) and **QA** with `render_pptx.py`.
5. **Commit + push** to the working branch (fetch+rebase first). Deliver `.pptx`
   files with SendUserFile; publish HTML sections with the Artifact tool when asked.

## Design system (theme tokens)
`--floor #080B0F · --bg #0F141A · --bg-2 #151C24 · --bg-3 #1B232D · --ink #ECEFF3
· --ink-2 #9DA9B5 · --ink-3 #5C6975 · --amber #E7AC51 · --amber-deep #C98A2E
· --cyan #5FB7C9 · --warn #D9785B`. Violet `#8C7BD8` = genomics. Emory navy
`rgb(1,33,105)` for the "EMORY" wordmark. Fonts: Helvetica Neue/Arial (sans),
SFMono/Consolas (mono). 16:9, frame = 1180px CSS wide in HTML.

Common HTML slide classes: `s-divider` (era/year dividers), `s-primer`,
`s-caveat` (text-left + figure-right), `s-fig`, `s-turn` (turn-line w/ amber
left-border), `s-stat`, `s-act`, `s-quote`. Chips: `.chip`/`.dchip` (cyan
left-accent; `<b>` inside → cyan). Eras: I·1998 (CAD), II·2016 & 2026, III·2026
& 2035. `data-note` on each `<section>` = speaker note. Mark a slide
hidden-in-final with `data-hide="1"` + an HTML comment containing `hide in final`
(also matches `⚑`); `build_combined.py` and the PPTX builders drop these.

## PPTX pipeline (`keynote-iwbi-2026/pptx/`)
13.333×7.5in (16:9). Every slide gets the gradient/contour-motif background
(`assets/bg_motif.png`) behind an amber accent bar. Two build approaches:

### A) Hand-tuned per-section scripts (default for figure-heavy sections)
`build_hook_pptx.py`, `build_section1_pptx.py`, `build_section2_pptx.py`,
`build_section2b_pptx.py`, `build_section5_pptx.py`, `build_section6b_pptx.py`.
They use **`pptxlib.py`** helpers and place each element deliberately. Run e.g.
`python3 pptx/build_section2_pptx.py` → writes to `pptx/sections/`.

`pptxlib.py` exports: color constants; `new_prs()`; `slide(prs)` (adds bg motif
+ accent bar, returns slide); `box/para/run`; `eyebrow`; `card`; `rect`; `chip`;
`native_table`; `note`; `embed_img(s,datauri,l,t,w,h,card_bg='white'|'black')`;
`embed_svg(s,svg,l,t,w,h,bg=None)` (rasterizes via cairosvg — **bg=None =
transparent** so the gradient shows through; this is the default). `save(prs,path)`.

Sizes that match the HTML (pt): eyebrow 11 mono · divider/year hero 44 · s-turn
30 · section title 26 · body 14 · turn-line/emph 17–19 · stat 30 · cite 9–10 mono
· chip 10.5–11 mono. Left margin ≈ 0.85in for hand-tuned (0.69 in the HTML —
see exact-layout below).

### B) Exact-layout extractor (best pixel-match; used for text-heavy sections)
`extract_layout.py <section>` renders the HTML in Playwright and writes
`pptx/layout/<section>.json` with every text box's exact position, font px,
color, alignment, `text-transform`, left-border, and padding.
`build_from_layout.py <section>` builds the PPTX from that JSON — exact
positions/sizes/colors, cyan/amber runs, left-border bars (turn-lines/callouts),
transparent SVGs, embedded base64 figures, bg motif. Currently the chosen builder
for `section_adoption` + `section7`. Run:
`python3 extract_layout.py section7 && python3 build_from_layout.py section7`.

Scale math: frame px → inches by frame width; **font pt = fontPx × 960 / frameW**.
The extractor renders at viewport 1920 wide so vw-clamped fonts hit their design
max (matches presentation width). Single-line/title elements are widened so a
slightly different font can't force a wrap.

## Verification loop — MANDATORY (do NOT ask the user for screenshots)
After building or editing any section's PPTX, **self-check it against the HTML
and iterate until they match** — this is the step that was missing before and
caused the user to paste screenshots repeatedly. Do not deliver until done.

```
python3 pptx/compare.py <section>        # e.g. section_adoption
```
This renders the **HTML (reference, left)** and the **generated PPTX (right)**
side by side, one row per slide, into `compare_<section>.png`. **View that image
yourself**, scan every slide for mismatches in font size, spacing, color,
position, wrapping, and missing elements; fix the builder; rebuild the section;
re-run `compare.py`; repeat until the two columns are visually identical. Then
deliver. (LibreOffice does NOT work in this env, hence the PIL proxy.)

**Proxy caveat:** the right column uses DejaVu (wider than PowerPoint's Arial),
so a long one-line title can *look* wrapped/overlapping in the proxy yet be fine
in real PowerPoint — the exact-layout builder widens single-line/title boxes to
prevent real wraps. Treat long-title wrap-overlaps as proxy artifacts; trust
positions, colors, content. `render_pptx.py <file> <out>` (whole-deck contact
sheet) and `render_html.py <section> <dir>` (HTML-only) are the lower-level tools
`compare.py` is built from.

## Figures
- SVG charts authored in the HTML → embedded **transparent** (gradient shows
  through). `html.parser` lowercases SVG attrs; the builders restore camelCase
  (`viewBox`, `radialGradient`, markers…) before cairosvg.
- Base64 **photos** (mammograms, ROC/paper figures) have white/black backgrounds
  baked in → keep a `card_bg` ('white' usually, 'black' for dark images).
- Placeholder assets the user hasn't supplied → dashed amber box (`placeholder()`
  pattern) labeled with the asset name; matches the HTML's `[ to be added ]`.

## Gotchas / lessons
- Run git from `/home/user/Mammo` with full paths (cwd can drift after scratchpad cmds).
- On push reject (lock at X expected Y): `git fetch` + `git rebase origin/<branch>` then push.
- `cairosvg`, `playwright`+pre-installed Chromium, `PIL`, `python-pptx`, `bs4`
  are available; **numpy is NOT** (PIL-only pixel work). Chromium exe:
  `/opt/pw-browsers/chromium-1194/chrome-linux/chrome` (`--no-sandbox`).
- Era III spans two files: Era III·2026 = back half of `section2b`; Era III·2035 =
  `section5alternate`.
- When merging/splitting HTML slides, the hand-tuned scripts index `VIS[i]` by
  position — renumber indices after the merge (or prefer the exact-layout builder,
  which re-derives automatically).
- Deliver per-section `.pptx` from `pptx/sections/`; the user concatenates them.
