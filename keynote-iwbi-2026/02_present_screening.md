# Section 2 — The Deep-Learning Turn & the Prospective Evidence
### ~8 minutes · spine stage: **LESION** (modern) · the present-day state of screening AI

**Purpose:** Show how modern AI differs from CAD. The defining feature of this era is that screening AI earned **level-1 randomized evidence** before scaling. This responds to Section 1's cautionary tale, and to the view that AI is a repeat of CAD.

---

## OPENING BEAT (picks up from §1's Kohli & Jha)

*Section 1 closed on the field's own review: skeptics cite CAD as evidence that AI will not deliver. Open Section 2 by addressing that point.*

> *"Here is the fair challenge. When someone says AI is CAD with better marketing, they have a point; CAD earns that skepticism. The burden is on us. The question is not whether the AUC is higher, since CAD had reasonable numbers too. The question CAD never answered before it scaled to 92% of American mammograms is whether AI holds up, prospectively, in a real screening program. We now have evidence on that question."*

> **The frame for the whole section:** CAD was paid first and not prospectively proven. Modern screening AI was proven first.

---

## Acts 1–3 — the path to validation (run as a brief ~2.5-min sequence, one slide each)

**Act 1 — retrospective performance (2020).** McKinney et al., *Nature* 2020 (Google Health/DeepMind, UK + US): false positives down **5.7% (US)/1.2% (UK)**, false negatives down **9.4% (US)/2.7% (UK)**; AI AUC exceeded the average radiologist by **11.5%** absolute; reduced a simulated second reader's workload **88%**. *(Nature 577:89–94)*

**Act 2 — the reproducibility critique (2020).** Haibe-Kains et al., *Nature* 2020: McKinney withheld code and model details, so the result was **not independently reproducible.** *(Nature 586:E14–E16)* This echoes CAD's adoption without verification; here the field raised the concern itself.

**Act 3 — independent validation (2020).**
- Salim et al., *JAMA Oncology* 2020 (Stockholm, 8,805 women): best algorithm **AUC 0.956**; AI + first reader reached **88.6% sensitivity at 93.0% specificity**, exceeding two human readers.
- Schaffter et al., *JAMA Network Open* 2020 — **the DREAM Challenge**: 126 teams / 44 countries; **no single AI exceeded radiologists**, while an **AI + radiologist ensemble reached AUC 0.942.** **[YOUR PAPER — first-person: "I was part of this one."]**

> **Takeaway (one line):** *"By 2020 the evidence showed AI could match or complement readers on retrospective data. CAD could meet that bar too. The relevant question is prospective."*

---

## Act 4 — the prospective and randomized evidence (~5 min — this is what is new)

> **NUMBER LEAD (two slides):**
> **"+29% cancer detection, with no increase in false positives."**
> *(next slide)* **"44% less reading workload."**
> *"Same women, same images, in a randomized trial. This is the level of evidence CAD lacked."*

**MASAI — the central RCT for this section (Sweden, Transpara).**
- *Safety analysis* — **Lång et al., Lancet Oncology 2023** (80,033 women): CDR **6.1 vs 5.1/1,000**, false-positive rate **1.5% in both**, **44.3% reduction in screen-reading workload.** *(DOI 10.1016/S1470-2045(23)00298-X)*
- *Full secondary outcomes* — **Hernström et al., Lancet Digital Health 2025** (~106,000 women): CDR **6.4 vs 5.0/1,000 = +29% (ratio 1.29, p=0.0021)**, recall and false positives **flat.** Extra cancers were mostly small, node-negative invasive cancers — clinically meaningful, not overdiagnosis. **44.2% workload reduction** confirmed. *(DOI 10.1016/S2589-7500(24)00267-X)*
- *Primary endpoint —* **Gommers et al. (…Lång), Lancet 2026;407(10527):505–514**: **interval-cancer rate 1.55 vs 1.76/1,000 — non-inferior (ratio 0.88, p=0.41)**, fewer invasive / T2+ / non-luminal-A interval cancers. **Sensitivity 80.5% vs 73.8% (p=0.031); specificity 98.5% in both.** *The first RCT to show AI-supported screening does not increase the interval-cancer rate, addressing that concern with level-1 evidence.* *(PMID 41620232; DOI 10.1016/S0140-6736(25)02464-X)* **[✓ confirmed — note: lead author Gommers, NOT Hernström/Lång]**

**PRAIM — real-world data (Germany).** Eisemann et al., *Nature Medicine* 2025. **463,094 women**, 119 radiologists, 12 sites; the largest real-world dataset: AI-supported double reading CDR **6.7 vs 5.7/1,000 = +17.6%** (statistically superior); recall **non-inferior/slightly lower**. *The MASAI result holds outside a controlled trial.* *(DOI 10.1038/s41591-024-03408-6; observational — selection bias, lower evidence tier.)*

**ScreenTrustCAD — Dembrower et al., Lancet Digital Health 2023** (Sweden, 55,581 women, Lunit): **one radiologist + AI was non-inferior to two radiologists**; two + AI was superior (+8%). *Evidence for replacing one of two readers with AI.* *(Lunit-funded — disclose.)*

**Ongoing trials:** **EDITH (UK NHS, launched April 2025)** — ~700,000 women, 30 sites, **5 AI platforms**, against a ~30% reader shortfall; rollout targeted ~2027 **[VERIFY — press]**. **AI-STREAM (Korea)** for non-Western data.

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

> CAD aimed to make one radiologist marginally better at one image. Modern screening AI **decouples reading workload from accuracy**, allowing a constrained workforce to read more, miss fewer cancers, and recall no more often. In Europe, where double-reading is standard and radiologists are in short supply, this is a different operating model rather than an incremental gain.

> **[VISUAL — build later]** Two-column slide: **CAD** (left, red) — *scaled to 92%, then the 2015 finding*; **Modern AI** (right, green) — *RCT-validated first: MASAI +29% / 44% workload, PRAIM +17.6%*. Summarizes the first half of the talk in one image. *(Flagged, not built.)*

---

## CAVEAT EMPTOR — the aggregate AUC hides subgroup blind spots (~1.5 min, your own data)

*Follow the "it works" evidence with our own work, which qualifies it. This is the first half of the section's central caution: **explainability and subgroup performance are the two gaps that separate a strong aggregate number from a trustworthy tool.***

> *"A single AUC describes average performance and says nothing about where a model fails. We looked at where it fails."*

- **★ Our DBT audit — [Trivedi group], Subgroup Performance of a Commercial DBT Model, Nature Communications 2026** (EMBED, **163,449 exams**, Lunit INSIGHT DBT). Overall **AUC 0.91**. Stratified, performance falls where the stakes are highest: **in-situ cancers AUC 0.85 / sensitivity 0.55; calcifications 0.80 / 0.66; dense breasts 0.88 / 0.63.** Demographically it was relatively robust; the weak points were **clinical** (the subtle, hard cancers) rather than racial. *(DOI 10.1038/s41467-026-70637-3)* **[your group]**
- **★ A similar pattern in neuro — [Trivedi group], real-world ICH model evaluation, npj Digital Medicine 2025** (Aidoc, **101,944 head CTs**, 17 facilities). Overall **82.2% sensitivity**, with **subacute 45.5%, chronic 54.8%, small ≤10 mm 74.8%, outpatient 72.2%.** Robust across demographics; the misses were the **subtle, small, non-acute** bleeds. *(DOI 10.1038/s41746-025-02244-3)* **[your group]** *Across two modalities, the aggregate number hid the clinically important failures.*
- **The pattern is seen across the field:** the **2023 RSNA Mammography AI Challenge** (Radiology 2024) and **BreastScreen Norway** (99,489 women) both show performance varies by **breast density** (more false positives in dense breasts); the **ARIES** stratified study (306,839 mammograms) reports variation across density/age/region; and a multivariate analysis of screening-mammography AI found **density, not race or age, drove most false-positive variation.** *(corroborating; cite 1–2.)*

> **The takeaway line:** *"Caveat emptor. An aggregate AUC does not tell you what you have until the model is audited by subgroup, and until we can see why it fails. The cases it fails on are the subtle ones where help was most needed."*

> **Explainability hook (sets up §3 and §6):** *"Why does it miss those cases? In most cases we cannot say, and that is the second gap. We will return to it."*

---

## "What we still don't know" — a ~45-second honesty beat (don't skip)

*State the open questions before the future half. This distinguishes the talk from a vendor presentation.*

> *"A few limitations, stated plainly."*

1. **No mortality data yet.** Detection, recall, workload, sensitivity, and now interval cancer (a strong surrogate), but **not breast-cancer mortality.** Do not overclaim.
2. **Generalizability.** The pivotal RCTs are predominantly **Swedish/European, single-vendor, double-reading.** The **US screens with a single reader**, so these workflow gains may not transfer directly. **MASAI did not collect race/ethnicity.**
3. **Reproducibility** (Haibe-Kains): many models still lack open code; retrospective AUCs often decline under independent validation.
4. **In-situ / overdiagnosis** is not fully settled (MASAI in-situ ratio ~1.51).
5. **The main driver is operational**, workforce capacity, as much as diagnostic superiority. This is a legitimate reason and worth stating directly.

> **Bridge to Part II:** *"That is the lesion: finding today's cancer, and now doing it with evidence. The same normal mammogram, the one we opened with, holds more than today's cancer. To see it, we stop asking the image about the lesion and start asking it about the whole **image**."*

---

## ✓ VERIFIED (agent pass) — all confirmed exact
- McKinney, MASAI safety (Lång, Lancet Oncol 2023), MASAI secondary (Hernström, Lancet Digit Health 2025, n=105,934), **MASAI primary (Gommers, Lancet 2026;407:505–514)**, PRAIM (Eisemann, Nat Med 2025), ScreenTrustCAD (Dembrower 2023), Salim (0.956), DREAM/Schaffter (0.942), EDITH (Feb 2025, ~700k/30 sites/5 platforms) — all confirmed.
- **"MAIA" Spanish RCT does NOT exist** (likely confused with MASAI; a product "MIA" by Kheiron/Roche exists but is not a trial). Removed. ✓
- **Caveat-emptor sources confirmed:** DBT subgroup paper (Nat Commun 2026, DOI 10.1038/s41467-026-70637-3, EMBED 163,449, AUC 0.91; in-situ 0.85/0.55, calc 0.80/0.66, dense 0.88/0.63); ICH paper (npj Digit Med 2025, DOI 10.1038/s41746-025-02244-3, 101,944 CTs, 82.2% sens; subacute 45.5%, chronic 54.8%, outpatient 72.2%); RSNA 2023 challenge (Radiology 2024); BreastScreen Norway density (PMC11399294); ARIES (306,839); screening-mammography performance-gaps (arXiv 2305.04422). **Confirm your authorship on the two ★ papers.**

> ⏱️ Adds ~1.5 min to §2 (now ~9–9.5). If time is short, trim a trial from the spoken track and keep it on the table slide.

## [VERIFY — still open]
- Precise current count of FDA-cleared breast-specific AI devices (≥6 for DBT; check live FDA list before stating a number).
