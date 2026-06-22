# Section 2 — The Deep-Learning Turn & the Prospective Evidence
### ~8 minutes · spine stage: **LESION** (modern) · the present-day state of screening AI

**Purpose:** Show that modern AI is not CAD 2.0. The defining feature of this era is not a higher AUC — it's that, for the first time, screening AI earned **level-1 randomized evidence** before scaling. This is the direct answer to Section 1's cautionary tale, and to the skeptic who says "AI is just CAD again."

---

## OPENING BEAT — pick up the gauntlet (picks up from §1's Kohli & Jha)

*Section 1 closed on the field's own post-mortem: skeptics cite CAD as proof AI won't deliver. Open Section 2 by answering them head-on.*

> *"So here's the fair challenge. When someone in this room — and someone always does — says **'we've seen this movie before; AI is just CAD with better marketing'** — they have a point. CAD earns that skepticism. So the burden is on us. Not 'is the AUC higher?' — CAD had decent numbers too. The real question is the one CAD never answered before it scaled to 92% of American mammograms: **does it hold up, prospectively, in a real screening program?** For the first time, we have the answer."*

> **The frame for the whole section:** CAD was *paid first, proven never.* Modern screening AI is *proven first.* That inversion is the story.

---

## Acts 1–3 — the road to credibility (run as a fast ~2.5-min montage, one slide each)

**Act 1 — "Superhuman on paper" (2020).** McKinney et al., *Nature* 2020 (Google Health/DeepMind, UK + US): false positives down **5.7% (US)/1.2% (UK)**, false negatives down **9.4% (US)/2.7% (UK)**; AI AUC beat the average radiologist by **11.5%** absolute; cut a simulated second reader's workload **88%**. *(Nature 577:89–94)* — *The "wow" moment.*

**Act 2 — the backlash that made the field grow up (2020).** Haibe-Kains et al., *Nature* 2020: McKinney withheld code and model details — **not independently reproducible.** *(Nature 586:E14–E16)* — *The hype-check, and a direct echo of CAD's "trust without verification." The field, this time, called itself out.*

**Act 3 — rigorous, independent validation (2020).**
- Salim et al., *JAMA Oncology* 2020 (Stockholm, 8,805 women): best algorithm **AUC 0.956**; AI + first reader hit **88.6% sensitivity at 93.0% specificity** — beating two human readers.
- Schaffter et al., *JAMA Network Open* 2020 — **the DREAM Challenge**: 126 teams / 44 countries; **no single AI beat radiologists**, but an **AI + radiologist ensemble reached AUC 0.942.** **[YOUR PAPER — first-person: "I was part of this one."]**

> **Montage takeaway (one line):** *"By 2020 we knew AI could match or complement readers — on old data, looking backward. CAD could clear that bar too. The question that matters is forward."*

---

## Act 4 — the prospective + randomized evidence (dwell here, ~5 min — this is what's new)

> **COLD-NUMBER LEAD (two slides, CAD-style):**
> **"+29% cancer detection — with no increase in false positives."**
> *(next slide)* **"44% less reading workload."**
> *"Same women. Same images. One randomized trial. This is what CAD never had."*

**MASAI — the RCT that anchors the talk (Sweden, Transpara).**
- *Safety analysis* — **Lång et al., Lancet Oncology 2023** (80,033 women): CDR **6.1 vs 5.1/1,000**, false-positive rate **1.5% in both**, **44.3% reduction in screen-reading workload.** *(DOI 10.1016/S1470-2045(23)00298-X)*
- *Full secondary outcomes* — **Hernström et al., Lancet Digital Health 2025** (~106,000 women): CDR **6.4 vs 5.0/1,000 = +29% (ratio 1.29, p=0.0021)**, recall and false positives **flat.** Extra cancers were mostly small, node-negative invasive cancers — clinically meaningful, not overdiagnosis. **44.2% workload reduction** confirmed. *(DOI 10.1016/S2589-7500(24)00267-X)*
- *Primary endpoint (the headline) —* **Lancet 2026** (Lång group): **interval-cancer rate 1.55 vs 1.76/1,000 — non-inferior (ratio 0.88, p=0.41)**, fewer invasive / T2+ / non-luminal-A interval cancers. **Sensitivity 80.5% vs 73.8% (p=0.031); specificity 98.5% in both.** *The first RCT to show AI-supported screening doesn't trade away interval-cancer safety — the exact worry, answered with level-1 evidence.* *(DOI 10.1016/S0140-6736(25)02464-X)* **[VERIFY exact 2026 citation/volume]**

**PRAIM — real-world confirmation (Germany).** Eisemann et al., *Nature Medicine* 2025. **463,094 women**, 119 radiologists, 12 sites — largest real-world dataset: AI-supported double reading CDR **6.7 vs 5.7/1,000 = +17.6%** (statistically superior); recall **non-inferior/slightly lower**. *The MASAI signal holds outside a controlled trial.* *(DOI 10.1038/s41591-024-03408-6; observational — selection bias, lower evidence tier.)*

**ScreenTrustCAD — Dembrower et al., Lancet Digital Health 2023** (Sweden, 55,581 women, Lunit): **one radiologist + AI was non-inferior to two radiologists**; two + AI was superior (+8%). *The "AI replaces one of two readers" proof-of-concept.* *(Lunit-funded — disclose.)*

**The next wave (ongoing):** **EDITH (UK NHS, launched April 2025)** — ~700,000 women, 30 sites, **5 AI platforms**, against a ~30% reader shortfall; rollout targeted ~2027 **[VERIFY — press]**. **AI-STREAM (Korea)** for non-Western breadth.

---

## The trials at a glance (slide table)

| Trial | Year/Journal | Country, N | Design | Headline |
|---|---|---|---|---|
| MASAI safety | 2023 Lancet Oncol | Sweden, 80k | RCT | CDR 6.1 vs 5.1; **44% workload cut**; FP flat |
| MASAI secondary | 2025 Lancet Digit Health | Sweden, 106k | RCT | **+29% CDR**; recall/FP flat |
| **MASAI primary** | **2026 Lancet** | Sweden, 106k | RCT | **interval cancer non-inferior (0.88); sens 80.5 vs 73.8%** |
| PRAIM | 2025 Nat Med | Germany, 463k | Real-world | **+17.6% CDR**; recall non-inferior |
| ScreenTrustCAD | 2023 Lancet Digit Health | Sweden, 56k | Prospective | 1 reader + AI = 2 readers |
| EDITH | launched 2025 | UK, ~700k | RCT, 5 platforms | pending |

---

## The reframe (the era's defining idea)

> Old CAD tried to make one radiologist slightly better at one image. Modern screening AI **decouples reading workload from accuracy** — letting a strained workforce read more, miss less, and recall no more often. In Europe, where double-reading is standard and radiologists are short, that's not a marginal gain; it's a different operating model.

> **[VISUAL — build later]** Two-column slide: **CAD** (left, red) — *scaled to 92%, then the 2015 verdict*; **Modern AI** (right, green) — *RCT-validated first: MASAI +29% / 44% workload, PRAIM +17.6%*. The whole front half of the talk in one image. *(Flagged, not built.)*

---

## "What we still don't know" — a deliberate ~45-second honesty beat (don't skip)

*Promote this to the front of the audience's mind before the future half. It's what separates you from a vendor pitch — and it earns the credibility you'll spend in the next 30 minutes.*

> *"Let me be the skeptic for a moment, because this is where intellectual honesty matters most."*

1. **No mortality data yet.** Detection, recall, workload, sensitivity, and now interval cancer (a strong surrogate) — but **not breast-cancer mortality.** Don't overclaim.
2. **Generalizability.** The pivotal RCTs are overwhelmingly **Swedish/European, single-vendor, double-reading.** The **US screens with a single reader** — these workflow gains may not transfer directly. **MASAI did not collect race/ethnicity.**
3. **The reproducibility legacy** (Haibe-Kains): many models still lack open code; "superhuman" retrospective AUCs repeatedly shrink under independent validation.
4. **In-situ / overdiagnosis** isn't fully settled (MASAI in-situ ratio ~1.51).
5. **The honest driver is operational** — workforce capacity — as much as pure diagnostic superiority. Say that out loud; it's a *good* reason, not a weak one.

> **Bridge to Part II:** *"So that's the lesion — finding today's cancer, and finally doing it with evidence. But that same normal mammogram, the one we opened with, holds far more than today's cancer. To see it, we stop asking the image about the lesion — and start asking it about the whole **image**."*

---

## [VERIFY] before podium
- MASAI 2026 primary-endpoint exact citation (Lancet volume/authors).
- EDITH figures (press-sourced); current count of FDA-cleared breast-specific AI devices (≥6 for DBT; verify live FDA list).
- A Spanish "MAIA" RCT could **not** be confirmed — do not cite.
