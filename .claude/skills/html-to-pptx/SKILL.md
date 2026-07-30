---
name: html-to-pptx
description: >
  Convert a keynote HTML slide/section into a pixel-faithful editable PowerPoint
  for the IWBI breast-imaging deck (keynote-iwbi-2026/). Invoke whenever the task
  is "make/build the PPTX", "turn this slide into PowerPoint", "export to pptx",
  or "match the pptx to the html". Runs the exact-layout pipeline
  (extract_layout → build_from_layout → compare) and iterates until the PPTX
  matches the HTML — no user screenshots needed.
---

# HTML → PPTX conversion (exact-layout pipeline)

The HTML deck in `keynote-iwbi-2026/slides/` is the **source of truth**. This
skill produces an editable `.pptx` that faithfully reproduces one HTML file by
reading the browser's *computed* layout (positions, font sizes, colors) — not by
re-guessing sizes. It supersedes the older hand-tuned per-section scripts for any
new conversion.

## Where the scripts live & how they find the deck
The pipeline scripts live in **two places**, kept identical:
- **In the repo:** `keynote-iwbi-2026/pptx/` (the fallback; auto-available to any
  session on the Mammo repo).
- **Bundled with this skill:** `scripts/` (so the skill is self-contained when
  uploaded to the Anthropic Skills library).

`deckpaths.py` locates the deck (`keynote-iwbi-2026/`) so the scripts work from
either place, in this order: **`$KEYNOTE_DECK`** (if it contains `slides/`) →
walk up from the current dir for `keynote-iwbi-2026/slides/` → the repo layout
relative to the script. So: run from anywhere inside a Mammo checkout and it just
works; or `export KEYNOTE_DECK=/path/to/keynote-iwbi-2026` to point it explicitly.
Run the commands from `keynote-iwbi-2026/pptx/` (repo) **or** the skill's
`scripts/` dir — both resolve the same deck. The scripts still need the deck's
`slides/*.html` present to read and write output into `<deck>/pptx/sections/`.

## The three scripts (do not re-invent)
- `extract_layout.py <name>` — opens `slides/<name>.html` in Playwright at a
  **1920-wide** viewport (so `clamp()`/`vw` fonts hit their design max), walks the
  active `.slide`, and writes `layout/<name>.json`: per unit → rect (as fractions
  of the frame), `fontPx`, `linePx`, `align`, `family`, `text-transform`,
  left-border, padding, `::before` marker, and colored/bold/italic runs. Unit
  types: `text`, `box` (bordered/filled container, kept only if bg alpha ≥ 0.15),
  `svg`, `img`. Hidden slides (`data-hide`, comment `hide in final`, or `⚑`) are
  flagged.
- `build_from_layout.py <name>` — builds `sections/<outfile>.pptx` from the JSON.
  Draws the bg motif + amber accent bar, then each unit: text boxes at exact
  pt/color, rounded/plain boxes (dashed if the CSS border is dashed), left-border
  accent bars, `::before` list markers, **transparent** SVGs, embedded base64
  images.
- `compare.py <name>` — renders **HTML (left)** vs **generated PPTX (right)** side
  by side, one row per visible slide, into `compare_<name>.png`.

**Scale math (already implemented):** inches = `frac × 13.333` (or `7.5` for y);
**font pt = `fontPx × 960 / frameW`**. Don't hand-tune sizes — trust the extractor.

## Procedure — follow in order

1. **Register the name → output filename** in the `MAP` dict of **both**
   `build_from_layout.py` **and** `compare.py` (they must match). This is the
   step most easily forgotten. Use the convention
   `IWBI2026_Trivedi_<slug>.pptx`. Example:
   ```python
   "slide_sbi_survey": "IWBI2026_Trivedi_SBI_survey.pptx"
   ```

2. **Extract → build → compare**, from `keynote-iwbi-2026/pptx/`:
   ```
   python3 extract_layout.py <name> && \
   python3 build_from_layout.py <name> && \
   python3 compare.py <name>
   ```

3. **View `compare_<name>.png` yourself** (Read the image). Scan every slide for
   mismatches in position, font size, color, wrapping, alignment, and missing
   elements. **Do not ask the user for screenshots** — drive the loop from this
   image.

4. **Fix → rebuild → re-compare** until the two columns are visually identical.
   Most fixes belong in `extract_layout.py` (capturing a layout fact the JSON is
   missing) or `build_from_layout.py` (rendering a captured fact correctly) —
   fix the *pipeline*, so every future section benefits, rather than special-
   casing one slide. See the symptom→fix table below. Re-run step 2's extract
   whenever you touch `extract_layout.py`.

5. **Deliver & commit.** Optionally `SendUserFile` the `.pptx`. Then from
   `/home/user/Mammo`: `git fetch` + `git rebase origin/<branch>`, `git add` the
   `.pptx` (in `sections/`) and any changed scripts, commit, push `-u`.
   `compare_*.png` and `layout/` are gitignored (QA artifacts) — never commit them.

## Symptom → fix (lessons already baked in; extend, don't relearn)
| In the PPTX column you see… | Cause | Fix location |
|---|---|---|
| Two spans collapse together (e.g. `Workflow efficiency73%`) | a `flex; justify-content:space-between` row was merged into one text unit | extractor already emits each child of a flex row (≥2 inline children) at its **own** measured rect — keep it |
| A colored/highlighted word lost its color or bold | run color/weight not captured | `runsOf()` captures per-text-node color/bold/italic; builder maps to RGB & bold |
| A container fill/border missing or a faint overlay drawn as solid white | box alpha threshold | boxes kept only if bg alpha ≥ 0.15 (`hasBg`); tune in extractor |
| List dash / bullet accent missing | `::before` marker | extractor captures `::before` bg/w/h; builder draws it at the element's outer-left |
| Callout/turn-line amber left stripe missing | `border-left` accent | extractor captures `bordL/bordC`; builder draws a bar and insets the text |
| Placeholder box solid instead of dashed outline | dashed border | builder applies `a:prstDash` when CSS `border-*-style: dashed` |
| SVG has a black box / wrong colors | camelCase attrs lowercased by parser | builder restores `viewBox`/`radialGradient`/markers and rasterizes with `background_color=None` (transparent) |
| An `UPPERCASE` label came out mixed-case | `text-transform` | extractor captures it; builder uppercases/lowercases the runs |
| Long one-line **title wraps/overlaps** the line below | **proxy artifact** — DejaVu is wider than PowerPoint's Arial | usually **not** a real bug: builder widens single-line/title boxes so real PowerPoint (Arial, narrower) fits it on one line. Trust positions/colors; only act if the HTML itself wraps there |

## Environment notes
- Chromium: `/opt/pw-browsers/chromium-1194/chrome-linux/chrome` (`--no-sandbox`).
- Available: `cairosvg`, `playwright`+Chromium, `PIL`, `python-pptx`, `bs4`.
  **numpy is NOT** — PIL-only for any pixel work.
- **LibreOffice does not work here** — that's why `compare.py` uses a PIL proxy
  renderer (`render_pptx.render_slides`). The proxy's fonts differ slightly from
  real PowerPoint; see the title-wrap caveat above.
- Output slides are 13.333×7.5 in (16:9). Every slide gets `assets/bg_motif.png`
  behind an amber accent bar.

## Packaging / dependencies
Bundled under `scripts/`: `deckpaths.py`, `extract_layout.py`,
`build_from_layout.py`, `compare.py`, `render_pptx.py`, and `assets/bg_motif.png`.
Python deps (pip): `playwright` (+ the Chromium at
`/opt/pw-browsers/chromium-1194/chrome-linux/chrome`), `cairosvg`, `python-pptx`,
`pillow`, `beautifulsoup4`. In a freshly recycled container these may need
reinstalling: `pip install playwright cairosvg python-pptx pillow beautifulsoup4`.
(Note: a piped `python3 -c import… | tail` masks a failed import behind `tail`'s
exit 0 — check imports directly.)

## Relationship to the `keynote-slides` skill
`keynote-slides` covers the whole deck (authoring HTML, the design-system tokens,
the review deck, and both PPTX approaches). Use **this** skill when the job is
specifically HTML→PPTX conversion. For the theme palette, slide classes, and HTML
editing conventions, consult `keynote-slides`.
