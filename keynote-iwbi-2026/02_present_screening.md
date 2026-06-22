# Section 2 — The Deep-Learning Turn & the Prospective Evidence
### ~8 minutes · "Read" · the present-day state of screening AI

**Purpose:** Show that modern AI is not CAD 2.0. The defining feature of this era is not a higher AUC — it's that, for the first time, screening AI has earned **level-1 randomized evidence** before scaling. This is the answer to Section 1's cautionary tale.

---

## Narrative arc (the four acts of the present)

1. **2020 — "superhuman on paper."** Google Health's Nature paper.
2. **2020 — the backlash.** Reproducibility critique; the field gets disciplined.
3. **2020–2023 — rigorous external validation.** Independent, multi-vendor benchmarking.
4. **2023–2026 — the inflection.** Prospective + randomized trials, culminating in MASAI's interval-cancer endpoint in 2026. The frame shifts from "detection aid bolted onto a radiologist" to "AI as a triage engine and quasi-independent reader that decouples workload from accuracy" — the answer to the global radiologist shortage.

---

## Act 1 — "Superhuman on paper" (2020)

**McKinney et al., Nature 2020** — *International evaluation of an AI system for breast cancer screening* (Google Health/DeepMind, UK + US).
- Absolute reduction in **false positives 5.7% (US) / 1.2% (UK)**; **false negatives 9.4% (US) / 2.7% (UK).**
- In a 6-radiologist reader study, AI AUC exceeded the average radiologist by **11.5%** absolute.
- Simulated into UK double-reading: non-inferior while cutting the **second reader's workload by 88%.**
- *Nature 577:89–94; DOI 10.1038/s41586-019-1799-6*

## Act 2 — The backlash that made the field grow up

**Haibe-Kains et al., Nature 2020** — *Transparency and reproducibility in AI.* A formal rebuttal: McKinney et al. withheld code and model details, making the result impossible to independently verify. **The hype-check citation** — and a direct echo of CAD's "trust without verification" error.
- *Nature 586:E14–E16; DOI 10.1038/s41586-020-2766-y*

## Act 3 — Rigorous external validation (2020)

**Salim et al., JAMA Oncology 2020** — first independent external validation of 3 commercial algorithms (Stockholm, 8,805 women, 739 cancers).
- Best algorithm **AUC 0.956**; at radiologists' specificity, its sensitivity **81.9%** exceeded first-reader 77.4% and matched second-reader 80.1%. AI + first reader: **88.6% sensitivity at 93.0% specificity** — beating two human readers. *"AI is ready to be tested as an independent reader."* *(DOI 10.1001/jamaoncol.2020.3321)*

**Schaffter et al., JAMA Network Open 2020 — the DREAM Challenge.** Crowdsourced, 126 teams / 44 countries, ~310,000 exams (US + Sweden).
- Top algorithm **AUC 0.858 (US) / 0.903 (Sweden)**, but **no single AI beat radiologists.** An **AI + radiologist ensemble reached AUC 0.942** at improved specificity.
- *DOI 10.1001/jamanetworkopen.2020.0265* — **[YOUR PAPER — you are a co-author; first-person moment]**

> The Act-3 takeaway: by 2020 we knew AI could match or complement readers retrospectively. The open question was whether it holds up **prospectively, in a real program** — the exact question CAD never answered before scaling.

---

## Act 4 — The prospective + randomized evidence (the heart of "present")

**MASAI — the RCT that anchors the talk (Sweden, Transpara).**
- *Safety analysis* — **Lång et al., Lancet Oncology 2023** (first 80,033 women): CDR **6.1 vs 5.1/1,000**, false-positive rate **1.5% in both**, **44.3% reduction in screen-reading workload.** *(DOI 10.1016/S1470-2045(23)00298-X)*
- *Full secondary outcomes* — **Hernström et al., Lancet Digital Health 2025** (~106,000 women): CDR **6.4 vs 5.0/1,000 = +29% (ratio 1.29, p=0.0021)**, recall and false positives **flat.** Extra cancers were mostly small, node-negative invasive cancers — clinically meaningful, not overdiagnosis. **44.2% workload reduction** confirmed. *(DOI 10.1016/S2589-7500(24)00267-X)*
- *Primary endpoint (the headline) —* **Lancet 2026** (Lång group): **interval-cancer rate 1.55 vs 1.76/1,000 — non-inferior (ratio 0.88, p=0.41)**, with fewer invasive / T2+ / non-luminal-A interval cancers. **Sensitivity 80.5% vs 73.8% (p=0.031); specificity 98.5% in both.** *The first RCT to show AI-supported screening doesn't trade away interval-cancer safety.* *(DOI 10.1016/S0140-6736(25)02464-X)* **[VERIFY exact 2026 citation/volume before podium]**

**PRAIM — the real-world confirmation (Germany).** Eisemann et al., Nature Medicine 2025. **463,094 women**, 119 radiologists, 12 sites — largest real-world dataset.
- AI-supported double reading CDR **6.7 vs 5.7/1,000 = +17.6%** (statistically superior); recall rate **non-inferior / slightly lower**; PPV of recall 17.9% vs 14.9%. The MASAI signal holds outside a controlled trial.
- *DOI 10.1038/s41591-024-03408-6* (observational — note selection bias, lower evidence tier than MASAI).

**ScreenTrustCAD — Dembrower et al., Lancet Digital Health 2023** (Sweden, 55,581 women, Lunit). **One radiologist + AI was non-inferior to two radiologists**; two radiologists + AI was superior (+8%). The "AI can replace one of two readers" proof-of-concept. *(DOI 10.1016/S2589-7500(23)00153-X; Lunit-funded — disclose.)*

**The next wave (ongoing):**
- **EDITH (UK NHS, launched April 2025):** ~700,000 women, 30 sites, **5 AI platforms**, £11M; explicitly testing whether AI can replace one of two readers and recover >100,000 clinician-hours/yr against a ~30% reader shortfall. National rollout targeted ~2027. **[VERIFY — press-sourced]**
- **AI-STREAM (Korea)** — a non-Western prospective cohort if you want geographic breadth.

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

---

## Honest caveats (say these — they are the credibility)

1. **No mortality data yet.** Endpoints to date are detection, recall, workload, sensitivity/specificity, and now interval cancer (a strong surrogate). Not mortality. Don't overclaim.
2. **Generalizability.** Pivotal RCTs are overwhelmingly **Swedish/European, single-vendor, double-reading**. The **US uses single-reading** — gains may not transfer directly. MASAI did not collect race/ethnicity.
3. **The reproducibility legacy** (Haibe-Kains): many models still lack open code; "superhuman" retrospective AUCs repeatedly shrink under independent validation.
4. **In-situ / overdiagnosis** question isn't fully settled (MASAI in-situ detection ratio ~1.51).
5. **The honest driver is operational** — workforce capacity — at least as much as pure diagnostic superiority.

## [VERIFY] before podium
- MASAI 2026 primary-endpoint exact citation (Lancet volume/authors) — confirm against ASCO Post / Lancet.
- EDITH figures (press-sourced); precise current count of FDA-cleared breast-specific AI devices (≥6 for DBT; verify live FDA list).
- A Spanish "MAIA" RCT could **not** be confirmed — do not cite.
