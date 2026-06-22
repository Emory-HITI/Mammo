# Section 3 — From Detection to Risk: The Mammogram as a Biosensor
### ~6 minutes · spine stage: **IMAGE** · opens Part II (the future)

> **CALLBACK TO THE HOOK (seed #1):** *"Remember the first model — the one that knew this woman would develop cancer in about four years? Here it is."* The hook promised it; this section delivers it, and closes that loop on screen.

**Purpose:** The first leap into the future, and the spine's turn from **lesion → image**. AI changes the question a mammogram answers — from *"is there cancer today?"* to *"what is this person's risk, and how hard should we look?"* This is the move from a detection device to a biosensor, and it is no longer hypothetical: it is FDA-cleared.

---

## The opening idea

> *"Every mammogram already contains a five-year forecast. We are only now learning to read it."*

The image carries signal — invisible to the human eye — about *future* cancer, not just current cancer. Deep learning extracts it. That reframes screening from a snapshot into a trajectory.

---

## The leap beyond density and questionnaires

**The old world:** classical risk models use questionnaire factors + density and are weak discriminators — **Tyrer-Cuzick v8 5-year AUC ~0.62**; density-augmented Gail/TC ~0.59–0.61.

**The pivot — MIT + MGH (Yala / Barzilay / Lehman):**
- **Yala et al., Radiology 2019** (88,994 mammograms): hybrid DL (image + risk factors) **5-yr AUC 0.70** vs image-only 0.68, risk-factor logistic 0.67, **Tyrer-Cuzick 0.62** (p<0.001). First clean proof the *pixels* carry risk beyond density.
- **Mirai — Yala et al., Science Translational Medicine 2021.** Designed to predict across time points, tolerate missing risk-factor data, and stay consistent across machines. Externally validated in **three countries**: C-index **0.76 (US) / 0.81 (Sweden) / 0.79 (Taiwan)**, beating Tyrer-Cuzick (all p<0.001). Among women who developed cancer within 5 years, **Mirai flagged 41.5% as high-risk vs 22.9% for Tyrer-Cuzick.** *(DOI 10.1126/scitranslmed.aba4373)*

**Making it intelligible — AsymMirai (Donnelly et al., Radiology 2024; Duke + Emory, EMBED, 210,067 mammograms).** Showed Mirai's "black box" reasoning is largely **local bilateral dissimilarity** (left-vs-right tissue difference). A simplified, interpretable model approximated it: 1-yr AUC 0.79 (Mirai 0.84); in stable-tissue subgroups, **3-yr AUC 0.92.** *(DOI 10.1148/radiol.232780)* **[YOUR PAPER — first-person moment on interpretability]**

**The frontier — long-horizon, multi-institutional (2026).** Eriksson et al., Science Translational Medicine 2026: a 10-year image-derived model (developed KARMA/Sweden; validated Mayo, KARMA, EMBED/Atlanta). **10-yr AUC ~0.72**; in the top-decile risk group it captured **33% of cancers vs 24% (Mirai), 23% (Tyrer-Cuzick), 20% (BCSC)** — next-gen image models now beating Mirai itself for long-range triage. *(DOI 10.1126/scitranslmed.ady7414)* **[VERIFY 2026 cite]**

> Independent replication (it generalizes): Park et al., Diagnostics 2024 (Lunit, Korea), external validation 16,894 mammograms, 1-yr AUC **0.90**, matched Mirai, crushed Tyrer-Cuzick (0.57) and Gail (0.52).

---

## The policy on-ramp: density became a national question

- **FDA national dense-breast notification rule took effect Sept 10, 2024** — every US mammogram report must now state dense / not dense. Density is now a national clinical-action question overnight.
- **Automated density (Volpara, Densitas)** replaces subjective BI-RADS scoring; densest category carries multi-fold higher interval-cancer risk. **[VERIFY exact Brentnall/Volpara figures]**
- Arc: **density → masking + independent risk → supplemental MRI/ultrasound** — but coverage is patchy. Density alone is blunt; this is exactly the gap DL risk models fill.

---

## The future is already FDA-cleared

**Clairity Breast (Constance Lehman's company) — FDA De Novo authorization announced June 2025** — the **first-ever** AI tool to predict 5-year breast cancer risk from a routine screening mammogram alone, validated across **>77,000 mammograms from 5 geographically distinct sites**; subsequently referenced in NCCN screening guidance. *The regulatory bridge from research model (Mirai) to clinical product has been crossed.* **[VERIFY — press-sourced; no primary peer-reviewed validation yet]**

---

## Risk-adapted screening — the evidence arrived in 2025

**WISDOM (US) — Esserman et al., JAMA 2025/2026.** Pragmatic RCT, **28,372 women** 40–74, all 50 states; risk-based vs annual; median follow-up 5.1 yr. Risk engine = 9-gene panel + polygenic score + BCSC model → 4 regimens (6-monthly mammo/MRI for highest risk; down to defer for lowest).
- **Result: risk-based non-inferior for stage ≥IIB cancers** despite **~3,800 fewer mammograms per 100,000 person-yr.**
- **Honest caveat:** it did **NOT** reduce biopsies (the superiority co-primary failed). In the observational arm, **89% of women chose risk-based** — patients want this. *(DOI 10.1001/jama.2025.24784)*

**MyPeBS (Europe) — the larger sibling.** EU RCT, target **85,000 women** across 6 countries (incl. a Greece-adjacent European framing), risk-stratified vs standard screening; **ongoing, results pending.** *(BMC Cancer 2022; DOI 10.1186/s12885-022-09484-6)*

**How AI plugs in (your forward thesis):** WISDOM/MyPeBS triage on *genetics + classical models* today. The obvious next move is to feed the **image-derived AI score** into the engine — high score → MRI; low score → biennial or defer. Bernstein/Yala et al. (PLOS Digital Health 2026) already show Mirai as a 1-year "rule-out" can cut caseload **36–75%** with explicit false-omission tradeoffs — the quantitative scaffold for "who can safely screen less."

---

## Equity & honest caveats (do not skip)

- Mirai's training data was **only ~3.75% African American** — the central fairness worry.
- **But** independent validation is reassuring: Omoleye et al., Radiology: AI 2023 (Univ. of Chicago, **46.4% African American**) found **no significant performance difference by race** (1-yr AUC 0.71). The editorial's point: diverse-cohort validation must be **mandatory, not optional.**
- WISDOM was **77% non-Hispanic White** — generalizability to Black/Hispanic/Asian women unproven, and Black women face higher mortality, younger onset, more triple-negative disease. A system that *defers* screening could **widen disparities** if it under-calls these groups.
- **PRS portability** across ancestries remains unsolved. Real-world AUCs (~0.65–0.72) are good, not oracular.

---

## Slide-ready

**Key numbers**
- Tyrer-Cuzick 5-yr AUC ~0.62 → DL image models 0.70–0.84 (C-index up to 0.81, Sweden).
- Mirai flags **41.5%** of future cancers high-risk vs **22.9%** for TC.
- 2026 long-term model: **33%** of cancers in top decile vs **24%** Mirai, **23%** TC.
- WISDOM: stage ≥IIB non-inferior with ~3,800 fewer mammograms/100k person-yr; **89%** chose risk-based.
- Clairity: first FDA De Novo 5-yr risk-from-mammogram tool (2025).

**Vision lines**
- *"From 'is there cancer today?' to 'when, and how hard should we look?'"*
- *"Risk-adapted screening = more for the few who need it, less for the many who don't — and WISDOM just showed it's safe."*

## [VERIFY] before podium
Brentnall/Volpara density figures; Clairity 77k-image validation + FDA date; MyPeBS enrollment/results (ongoing); Eriksson 2026 STM citation.
