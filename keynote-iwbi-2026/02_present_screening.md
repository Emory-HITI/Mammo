# Section 2 — The Deep-Learning Turn & the Prospective Evidence
### ~8 minutes · spine stage: **LESION** (modern) · the present-day state of screening AI

**Purpose:** Show how modern AI differs from CAD. The defining feature of this era is that AI is being forced to generate evidence before scaling, and we have some of that.

---

The years is 2020, you're a radiologist and you've spent the last 4 years hearing about AI. Maybe you're impressed, maybe you're worried about job replacement. 
---
The problem is the radiologists are still feeling the burn from CAD, some are using CAD and AI in tandem, even adding to the confusion.

So what does mammography AI look like and what is the evidence 




## Acts 1–3 — the path to validation (run as a brief ~2.5-min sequence, one slide each)

**Act 1 — retrospective performance (2020).** McKinney et al., *Nature* 2020 (Google Health/DeepMind, UK + US): false positives down **5.7% (US)/1.2% (UK)**, false negatives down **9.4% (US)/2.7% (UK)**; AI AUC exceeded the average radiologist by **11.5%** absolute; reduced a simulated second reader's workload **88%**. *(Nature 577:89–94)*

**Act 2 — the reproducibility critique (2020).** Haibe-Kains et al., *Nature* 2020: McKinney withheld code and model details, so the result was **not independently reproducible.** *(Nature 586:E14–E16)* This echoes CAD's adoption without verification; here the field raised the concern itself.

**Act 3 — independent validation (2020).**
- Salim et al., *JAMA Oncology* 2020 (Stockholm, 8,805 women): best algorithm **AUC 0.956**; AI + first reader reached **88.6% sensitivity at 93.0% specificity**, exceeding two human readers.

> **Takeaway (one line):** *"Retrospective reader studies have well-known limitations — enriched case sets, lab conditions, no real workflow. They can show promise, but they are not the bar. The bar is prospective deployment in a screening program. That is where the rest of this section lives."*

Put the dream challenge info back into this slide, incldue the year, and first large scale development of AI models for breast cancer. We participated in this, and there was this incredible surge of optimism that this challenge would solve breast cancer AI, and the winning model had something like an AUC of (xx, look up from schaffter dream challenge paper), and there was a company that even spun out of this and was still around today.

---

It's now 2026, AI for breast imaging has been around for 6 years, and where do we stand. A 2024 ESR survey of 572 members found 48% of respondents currently use AI tools in clinical practice (up from 20% in 2018), with mammography cited by 75 respondents (13.1% of the total surveyed) as the modality for which they use certified AI tools. PubMed Central
By contrast, a U.S. report estimated only ~2% of practices use AI today, indicating significant regional variation. IntuitionLabs

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

**Ongoing trials:** **EDITH (UK NHS, launched April 2025)** — ~700,000 women, 30 sites, **5 AI platforms**, against a ~30% reader shortfall; rollout targeted ~2027 **[VERIFY — press]**. **PRISM (USA)** — the *Pragmatic Randomized Trial of Artificial Intelligence for Screening Mammography*, a **$16M PCORI-funded, 7-site RCT announced Sept 2025**, recruiting now; described as the **first large-scale RCT of AI for screening mammography in the United States**, using **Transpara (ScreenPoint)**. Results are years away. *This matters because the pivotal evidence so far is European/double-reading; PRISM tests AI in the US single-reader workflow.* **[press/PCORI — no results yet]**. **AI-STREAM (Korea)** for non-Western data.

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
| **PRISM** | announced 2025 | **USA, 7 sites** | **RCT** (Transpara) | first large US AI screening RCT; recruiting — results years away |

---
Ok so now we have the evidence we need right? Off to the races? Well not quite.
## CAVEAT EMPTOR — the aggregate AUC hides subgroup blind spots (~1.5 min, your own data)

*Follow the "it works" evidence with our own work, which qualifies it. This is the first half of the section's central caution: **explainability and subgroup performance are the two gaps that separate a strong aggregate number from a trustworthy tool.***

> *"A single AUC describes average performance and says nothing about where a model fails. We looked at where it fails."*

- **★ Our DBT audit — [Trivedi group], Subgroup Performance of a Commercial DBT Model, Nature Communications 2026** (EMBED, **163,449 exams**, Lunit INSIGHT DBT). Overall **AUC 0.91**. Stratified, performance falls where the stakes are highest: **in-situ cancers AUC 0.85 / sensitivity 0.55; calcifications 0.80 / 0.66; dense breasts 0.88 / 0.63.** Demographically it was relatively robust; the weak points were **clinical** (the subtle, hard cancers) rather than racial. *(DOI 10.1038/s41467-026-70637-3)* **[your group]**
- **★ A similar pattern in neuro — [Trivedi group], real-world ICH model evaluation, npj Digital Medicine 2025** (Aidoc, **101,944 head CTs**, 17 facilities). Overall **82.2% sensitivity**, with **subacute 45.5%, chronic 54.8%, small ≤10 mm 74.8%, outpatient 72.2%.** Robust across demographics; the misses were the **subtle, small, non-acute** bleeds. *(DOI 10.1038/s41746-025-02244-3)* **[your group]** *Across two modalities, the aggregate number hid the clinically important failures.*
- **★ It doesn't transfer off-distribution — [Trivedi group], Du H et al., "Beyond Screening," 2026** (Emory + NUS; 244,385 screening + 82,643 diagnostic exams; Mammo-CLIP, a CNN, MedImageInsight). On **screening**, AUC 0.78–0.81. Moved to **diagnostic** mammography, the CNN and vision-language model dropped **0.08–0.09 AUC**; the medical foundation model held up (standard 0.78, spot-compression 0.84). On **implants** (~10% of women) all models fell to **AUC 0.69–0.76, specificity 31–54%.** Localization was poor everywhere (mean IoU 0.136). *A model validated on screening is not validated on diagnostic or implant exams — a concrete deployment-safety gap.* **[your group; confirm authorship + final numbers]**
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

## Detection is already becoming risk (bridge to Era III)

*The cleanest hand-off to the future: the detection score itself carries a forward signal.*

- **Hickman et al., Radiology 2026;319(3):e251309** (VAI-B, Sweden; **31,394 individuals / 88,963 exams**; three commercial AI CAD systems — Vara, Lunit INSIGHT MMG, MammoScreen). For people later diagnosed with cancer, the **AI detection score is already elevated up to 10 years before diagnosis and rises steadily toward it**, while cancer-free individuals stay flat. ✓
- **The numbers (proportion of future cancers flagged at the 90th-centile score):** ~13–17% at 10 yr → **19.0–19.7% at 6 yr → 23.3–25.2% at 4 yr → 35.4–39.3% at 2 yr → 42–48% at 1 yr.** Combined predictive AUC **0.63–0.67** across systems vs **0.57 for mammographic density** (all P<.001). *(DOI 10.1148/radiol.251309)*
- *Why it matters:* this is the hinge from Era II to Era III. The model trained to find today's cancer is, in its score, already forecasting tomorrow's — detection blurring into risk. The chart on this slide is **redrawn from the paper's data (Table 2 / Fig 2A)** and cited; we don't reproduce the original figure.

> **Bridge to Era III:** *"That is the lesion: finding today's cancer, now with evidence. But notice what just happened — that detection score was already rising years before the cancer was visible. The same normal mammogram holds more than today's cancer. We stop asking the image about the lesion and start asking it about the whole **image** — and about the patient."*
