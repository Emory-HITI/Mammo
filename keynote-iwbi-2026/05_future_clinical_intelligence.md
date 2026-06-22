# Section 5 — Clinical Intelligence: Multimodal & Foundation Models
### ~8 minutes · spine stage: **PATIENT** (completes) · the conceptual peak of the talk

> **CALLBACK TO THE HOOK (seed #2):** the cardio-mammography beat (Act 3 below) is where you close the second loop — *"the model reading the calcium in her arteries, the incidental finding we threw away for forty years."* All three seeds from the opening mammogram are now harvested; the audience has felt the loop close three times.

**Purpose:** This is where the title pays off and the **patient** stage completes. We leave the era of **narrow detectors** (one model, one finding) and enter the era of **generalist, multimodal systems** that synthesize all of a patient's data across the care continuum. This is the longest future section — it can be trimmed if running over.

---

## The framing (the title's payoff)

**Anchor — Moor et al., Nature 2023**, *Foundation models for generalist medical AI* (GMAI): self-supervised models that flexibly interpret multiple modalities (imaging, EHR, labs, genomics, text) and produce expressive outputs with little task-specific labeling. *(PMID 37045921)*

> **Vision line:** *"For 30 years, CAD asked one question of one image: is there cancer here? The next decade asks a different question of the whole patient: given everything we know about this woman — her images, her tissue, her genes, her history — what should we do next?"*

---

## Act 1 — Foundation models went generalist, fast

**Medical LLMs crossed the expert threshold:**
- **Singhal et al., Nature 2023** — Med-PaLM: **67.6% on MedQA (USMLE-style).** *(PMID 37438534)*
- **Singhal et al., Nature Medicine 2025** — **Med-PaLM 2: up to 86.5% on MedQA.** *(PMID 39779926; cite the 2025 Nat Med version)*
- **Med-Gemini (Saab et al., Google), 2024** — **91.1% on MedQA**, large gains on multimodal benchmarks. **[PREPRINT — arXiv:2404.18416]**

**The mammography foundation model just arrived (2025):**
- Generalist radiology FMs: **RadFM (Nat Commun 2025)**, CheXagent (Stanford) **[PREPRINT]**.
- **First breast-specific FMs:** VersaMammo and **Mammo-FM** — Mammo-FM pretrained on **821,326 mammograms / 140,677 patients** across 4 US institutions, unifying diagnosis, localization, structured reporting, and risk in one backbone. **[PREPRINT — arXiv 2025]**
- **Don't conflate:** Lunit/commercial products are CAD, not published foundation models.

> **Vision line:** *"The mammography foundation model arrived this year — one pretrained backbone that reads, localizes, reports, and predicts risk, not five separate products."*

---

## Act 2 — Multimodal integration: the patient, not the pixel

Fusing imaging + pathology + genomics + clinical data beats any single modality:
- **Chen et al. (PORPOISE), Cancer Cell 2022** — fused WSI + molecular across 14 cancers; multimodal beat unimodal in **12 of 14** (avg C-index ≈0.64). *(PMID 35944502)*
  - **Honest caveat (pre-empt the experts):** El Nahhas et al. (Cancer Cell 2022) showed some gains were **TCGA site/batch effects** — under site-preserved validation, multimodal won in only 4 of 8. *Use this to demonstrate rigor.*
- **Boehm et al., Nature Reviews Cancer 2022** — the conceptual framing of multimodal precision oncology *(PMID 34663944)*; applied flagship in ovarian (Nature Cancer 2022, n=444). *No breast-specific Boehm paper — frame as "demonstrated in ovarian, architecture applies to breast."*
- **Schmauch et al. (Owkin), Nature Communications 2020** — imaging *predicts* RNA-Seq molecular profile from WSI. *(PMID 32747659)*

**Breast-specific, with hard numbers (treatment response):**
- **Joo et al., Scientific Reports 2021** (NAC response, n=536): multimodal (clinical + MRI) **AUC 0.888** vs 0.827 clinical-only. *(PMID 34552163)*
- **Hong et al., Academic Radiology 2026** (pCR, I-SPY2): internal **AUC 0.888**, external **0.890.** *(PMID 41656135)*

**The digital-twin vision:**
- **Sadée et al. (Gevaert, Stanford), Lancet Digital Health 2025** — *Medical digital twins.* *(PMID 40518342)*

> **Vision line:** *"A digital twin of the patient — imaging, pathology, genomics, and history fused into one longitudinal model that updates with every visit and rehearses the consequences of each decision."*

---

## Act 3 — Opportunistic screening: one image, many predictions

The mammogram is a vastly underused biomarker source — and several of these are **Emory-led**.

- **Future-cancer (recap from Section 3):** Mirai, **C-index 0.76–0.81** (Sci Transl Med 2021).
- **Cardio-mammography (breast arterial calcification → CVD risk):**
  - **Wang et al., IEEE TMI 2017** — original DL-BAC paper. *(PMID 28113340)*
  - **Iribarren et al., Circulation: Cardiovascular Imaging 2022** — 5,059 women: BAC → **HR 1.51** for hard ASCVD, independent of traditional risk factors. *(PMID 35290077)*
  - **Dapamede et al. (Emory + Mayo), European Heart Journal 2026** — **largest to date, 123,762 women:** severe BAC → MACE **HR 3.29**, additive to AHA PREVENT. **[YOUR/Emory headline — VERIFY exact DOI/volume]**
  - Your group: Guo et al. (SCU-Net), Medical Physics 2021. **[your group]**
  - Whole-image CVD: **Barraclough et al., Heart 2026** — 49,196 women, **C-index 0.72.** *(PMID 40957672)*
  - Cross-modality analogy: Pickhardt et al., Lancet Digital Health 2020 (opportunistic CT; aortic calcium HR 4.53 for death).

> **Vision line:** *"Every screening mammogram already contains a cardiovascular risk score and a future-cancer score — we just haven't been reading them. We can give 40 million women a 'free' heart screen with the image they already came for."*

**Honest caveats:** BAC ≠ coronary calcium (modest correlation; Saccenti 2024, AUC ~0.64). **Bone-density-from-mammogram is NOT yet established** — present as an open frontier, not a result.

---

## Act 4 — Generative AI in the workflow (promise + brake)

**Promise:**
- **Li, …Trivedi, Clinical Imaging 2023** — ChatGPT simplifies radiology reports, Flesch-Kincaid **grade 10.4 → 5.8.** **[your group]** *(PMID 37336169)*
- **Rau et al., Radiology 2023** — RAG chatbot matched ACR Appropriateness guidance at **~5 min / €0.19 per case** vs 50 min / €29.99 for radiologists. *(PMID 37489981)*

**The brake (essential for credibility):**
- **Cozzi et al., Radiology 2024** — 2,400 breast reports: GPT-4 BI-RADS agreement only **moderate (AC1 0.52)**; assignments that would **negatively alter management in 10.6% of GPT-4 cases vs 1.5% for humans.** *(PMID 38687216)*
- **Hallucinated references:** ChatGPT medical citations **~47% fabricated, only ~7% accurate** (Cureus 2023); ~64% of radiology references fabricated (CARJ 2024). **[WEB-ONLY — verify]**
- **Ambient documentation** (nearest-term, lowest-risk): Nuance DAX Copilot; Kaiser ~3,400 physicians (NEJM Catalyst 2024) — evidence still observational. **[WEB-ONLY]**

> **Vision line (with the brake):** *"Generative AI will draft your report, translate it for your patient, and answer her questions at midnight. But until it stops inventing half its citations and miscategorizing 1-in-10 BI-RADS, it's a co-pilot, not the pilot."*

---

## Act 5 — Operational AI & the workforce crisis (why this matters now)

- **UK RCR Workforce Census 2023** (pub. 2024): **30% consultant shortfall**, forecast **40% by 2028**; £276M on outsourcing. **[WEB-ONLY]**
- **Christensen et al., JACR 2025** — US radiologist supply +25.7% by 2055, but demand keeps pace; shortage persists. *(PMID 39952776)*
- AI screening evidence as the workforce answer: **MASAI** (44% workload cut), **ScreenTrustCAD** (1 reader + AI = 2), **PRAIM** (+17.6% detection) — see Section 2.
- **Triage is context-dependent:** positive for incidental PE (missed 44.8%→2.6%); **negative** for intracranial hemorrhage turnaround (Savage, AJR 2024). Be balanced.

---

## Closing synthesis of the section (bridge to Section 6)

> *"We started with CAD — a second pair of eyes on one image. We are arriving at clinical intelligence — a system that reads the mammogram for cancer and for cardiovascular risk, fuses it with the patient's pathology and genome into a living digital twin, drafts the report, explains it to the patient, and triages the worklist. The evidence is real — MASAI, PRAIM, Mirai, the cardio-mammography work. So are the failure modes — batch effects, hallucinated citations, 1-in-10 BI-RADS errors. The task for this room is to build the validation, not just the models."*

---

## Slide-ready key numbers
- Med-PaLM 2 **86.5%** USMLE (Nat Med 2025); Med-Gemini **91.1%** [preprint].
- Mammo-FM: **821,326 mammograms / 140,677 patients**, one backbone (2025) [preprint].
- PORPOISE: multimodal beat unimodal in **12/14** cancers (Cancer Cell 2022).
- Severe BAC → MACE **HR 3.29**, 123,762 women (Eur Heart J 2026, Emory).
- GPT-4 BI-RADS: **10.6%** management-altering errors (Radiology 2024).
- ChatGPT references: **~47% fabricated** (Cureus 2023).

## [VERIFY] before podium
Med-Gemini/CheXagent/Mammo-FM are preprints; Dapamede 2026 exact DOI/volume (flagship number — confirm); Haver Radiology 2023; Cureus/CARJ hallucination figures; Kaiser/DAX figures; Barraclough & Hong 2026 details.
