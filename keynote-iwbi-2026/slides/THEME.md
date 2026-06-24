# Keynote Deck — Master Theme ("Clinical Workstation")
### Locked design system for all IWBI 2026 slides. Reference: `slides/hook.html`.

The reference implementation is `hook.html` (the 5-slide opening). Every new section's slides reuse these exact tokens and component patterns so the deck stays consistent. When we export to PowerPoint, these tokens map 1:1 to the master slide.

---

## Palette (dark clinical workstation)

| Token | Hex | Use |
|---|---|---|
| `--floor` | `#080B0F` | stage behind the slide |
| `--bg` | `#0F141A` | slide ground (deep blue-charcoal) |
| `--bg-2` | `#151C24` | panels |
| `--bg-3` | `#1B232D` | raised cards (top of card gradient) |
| `--ink` | `#ECEFF3` | primary text |
| `--ink-2` | `#9DA9B5` | secondary text |
| `--ink-3` | `#5C6975` | faint / mono labels |
| `--line` | `rgba(255,255,255,.09)` | borders/hairlines |
| **`--amber`** | **`#E7AC51`** | **primary accent (warm gold)** — eyebrows, index numbers, emphasis, "Population" step, top rule |
| `--amber-deep` | `#C98A2E` | amber gradient end |
| **`--cyan`** | **`#5FB7C9`** | **data accent** — result values ("elevated", "present"), viewport tags, step ticks |
| `--viewport` | `#06090C` | dark PACS image panel |

**Accent discipline:** amber = structure/emphasis (the one bold voice); cyan = *data values only*; everything else is ink-scale. Don't introduce new accents.

## Type
- **Display/body:** `"Helvetica Neue", Arial, system-ui, sans-serif` — headlines heavy (700), tight tracking (-0.015 to -0.02em), `text-wrap: balance`.
- **Labels/data:** mono `"SFMono-Regular", Menlo, Consolas, monospace` — uppercase, letter-spaced (.12–.22em), for eyebrows, the "SAY"/counter, the `01/02` indices, axis/unit labels.
- Numbers always `font-variant-numeric: tabular-nums`.
- (No webfonts — CSP blocks CDNs; system stacks avoid silent fallback.)

## Signature motif
Faint **density-contour iso-lines** (`#motif` canvas) radiating from the upper-right, amber at very low alpha (~0.01–0.09). Echoes mammographic density / risk topography. It's the identity thread — keep it subtle, behind content, on every slide.

## Components (reuse these)
- **Frame:** 16:9, radius 12px, radial charcoal gradient, 1px border, top amber hairline (`::after`), deep shadow.
- **Eyebrow:** mono uppercase amber, with a short amber rule before it. Top-left.
- **PACS viewport:** dark panel, header strip (`R MLO · screening` left, cyan `AI` tag right), canvas body, optional mono corner note. Holds the (placeholder → real) mammogram.
- **Finding card:** `bg-3→bg-2` gradient, 2px amber left border, mono amber index, bold ink label, cyan result value, muted sub-line. Use for any "results/list" slide.
- **Step / process row:** equal cards, cyan top-tick; final/destination card filled amber with dark text.
- **Pivot line:** mono width-66px key on the left, bold statement on the right; "was" in `ink-3`, "now" in `ink` with amber bold.
- **Speaker bar:** `SAY` mono amber + italic-free muted note (data-note attribute drives it). This is presenter scaffolding — strip or move to real speaker notes in the final export.

## Slide grammar
- One idea per slide. Eyebrow names the section/beat.
- Big claim in ink; the single most important number/word in amber; live data values in cyan.
- Generous padding (`clamp(28px,4.1vw,64px)`); never let the body scroll sideways.
- Motion: quick opacity fade between slides only. Respect `prefers-reduced-motion`.

## Build/export notes
- Slides are authored in this HTML/Artifact format for fast preview. The plan: build each section as slides in this system → assemble the full deck → produce an editable **PowerPoint** mapping these tokens to a master slide for the podium.
- Placeholder mammogram is Canvas-rendered. Swap in the **real de-identified case** (and its risk heatmap for the §3 callback) before the talk.
