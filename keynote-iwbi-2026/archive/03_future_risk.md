# Section 3 — From Detection to Risk: The Mammogram as a Biosensor

We are now at the brink of Era III where instead of looking for lesions, we are now looking at the whole image or even the whole patient.

Here is the normal mammogram I showed you earlier - there are three additional things we will get from this image - tissue based risk, CVD disease prediction via BAC, and tissue-based risk. 
---
So let's talk about Risk

**The old world:** classical risk models use questionnaire factors + density and are weak discriminators — **Tyrer-Cuzick v8 5-year AUC ~0.62**; density-augmented Gail/TC ~0.59–0.61.

**The pivot — MIT + MGH (Yala / Barzilay / Lehman):**
- **Yala et al., Radiology 2019** (88,994 mammograms): hybrid DL (image + risk factors) **5-yr AUC 0.70** vs image-only 0.68, risk-factor logistic 0.67, **Tyrer-Cuzick 0.62** (p<0.001). Early evidence that the image carries risk information beyond density.
- **Mirai — Yala et al., Science Translational Medicine 2021.** Designed to predict across time points, tolerate missing risk-factor data, and stay consistent across machines. Externally validated in **three countries**: C-index **0.76 (US) / 0.81 (Sweden) / 0.79 (Taiwan)**, exceeding Tyrer-Cuzick (all p<0.001). Among women who developed cancer within 5 years, **Mirai flagged 41.5% as high-risk vs 22.9% for Tyrer-Cuzick.** *(DOI 10.1126/scitranslmed.aba4373)*

**Explainability — AsymMirai (Donnelly et al., Radiology 2024; Duke + Emory, EMBED, 210,067 mammograms).** This addresses the second gap from §2: we usually cannot say why a model fails. AsymMirai examined Mirai's signal and showed it is largely **local bilateral dissimilarity** (left-vs-right tissue difference). A simplified, **interpretable** model nearly matched the original: 1-yr AUC 0.79 (Mirai 0.84); in stable-tissue subgroups, **3-yr AUC 0.92.** Performance that can also be explained is what can be acted on safely and audited for blind spots. *(DOI 10.1148/radiol.232780)* **[YOUR PAPER — first-person moment on explainability]**

**FDA De Novo authorization, early June 2025** — the **first** AI tool to predict 5-year breast cancer risk from a routine screening mammogram alone, validated across **~77,000 mammograms from 5 sites**; subsequently **added to the 2026 NCCN breast-screening guidance (AI-based risk assessment).**  **[✓ FDA date + ~77k/5-sites + NCCN-2026 confirmed by agent; no primary peer-reviewed validation paper yet]**


RISK based screening
Tempo + Mirai achieved better early detection than annual screening while requiring up to 25% fewer mammograms overaly, simulated early detection benefit of roughly 4.5 months

---

## The trap hiding in a good AUC: population performance ≠ individual reliability
### *(~1.5–2 min)*

> *"Every number I just showed you — 0.76, 0.81, 0.84 — is a **population** number. An AUC tells you how a model sorts a million women. It tells you little about the one woman in your clinic."*

**What AUC/C-index actually is.** It is a *ranking* metric: the probability the model scores a random future-cancer patient higher than a random cancer-free one. A C-index of 0.80 is strong at the population level and still misranks a large number of individual pairs. It is not the accuracy of any one woman's predicted risk.

**Discrimination is not calibration.** AUC measures *ranking* (discrimination). Whether a predicted "8% five-year risk" actually means 8% is *calibration*, a separate property, less often reported, and the one that governs an individual decision such as "do you get an MRI?" Image-based models tend to discriminate well; their calibration across sites, scanners, and subpopulations is the under-reported and harder problem.

### The four quadrants (the clinically relevant view)

> **[VISUAL — build later]** A 2×2: classical model (low/high) on one axis, image-based model (low/high) on the other.

| | **Image model: LOW** | **Image model: HIGH** |
|---|---|---|
| **Classical: HIGH** | ⚠️ Discordant: classical high / image low | ✅ Concordant high → MRI, short interval |
| **Classical: LOW** | ✅ Concordant low → standard/de-escalate | ★ Classical low / image high |

1. **Both low / both high** — concordant. Manage accordingly: reassure-and-standardize, or escalate to supplemental MRI.

2. **★ Classical LOW / image HIGH.** The image detects a tissue-state signal the questionnaire does not capture. These are women the questionnaire era classified as "average." The type of risk matters here. Image models perform **best at near-term prediction**, and the field's interpretation (including the Mirai authors' discussion) is that a high short-term score may partly reflect **occult or premalignant change already present**. It is partly a **"look harder now"** signal: short-interval follow-up, supplemental imaging, a second look at *this* mammogram, rather than automatically a lifetime-surveillance decision. Acting on a signal we cannot explain repeats the CAD problem (see §1), which is why interpretability (AsymMirai: the signal is largely bilateral asymmetry) and *prospective* validation (the **MIRAI-MRI trial, NCT05968157**, comparing MRI for Mirai- vs Tyrer-Cuzick–high women) matter before changing management.

3. **⚠️ Classical HIGH / image LOW.** A BRCA carrier, a strong family history, or a high polygenic score, with a quiet-looking mammogram that earns a low image score. A reassuring image read should not override known germline risk. The image model was not trained to detect inherited risk and cannot. De-escalating here would be the most consequential error this technology invites.

### The synthesis: these models measure *different things*

> *"The genome asks what she inherited. The image asks what her tissue is doing now. Discordance between them is two answers to two questions."*

- **Image-based DL** → near-term, tissue-state, occult/masking signal. Horizon: strongest at 1–2 years.
- **Genomic / PRS + family history** → inherited, lifetime, stable.
- **Density** → masking and modest independent risk.

The aim is not to pick a winner but to combine them into a **risk vector** with the appropriate *action* attached to each axis (image-high → look now; genome-high → lifelong surveillance). Substituting one for the other is unsafe; fusion is the goal (early evidence: image-only DL **+ PRS** improves over either alone, Br J Cancer 2026). **Equity caveat:** discordance patterns likely differ by ancestry (PRS portability; image-model training diversity), and the *classical-high/image-low* error could fall hardest on the groups with worse outcomes.

> **One-liner:** *"The clinical question is which model answers what you need for this patient — and whether you would base her management on a number you cannot yet explain."*

One addition worth making: this scenario should also cover high polygenic score (PRS) patients, who are increasingly identifiable and whose risk is similarly invisible to imaging models. The text mentions this briefly, but it deserves the same emphasis as BRCA — PRS-high patients with a quiet mammogram are in an identical position.
---


## Risk-adapted screening — and Europe is running the definitive trial

> *"This is where it becomes a screening program rather than a number on a slide, and the largest trial on this question is running in Europe."*

**WISDOM (US) — first results reported.** Esserman et al., *JAMA* 2026;335(9):763–774. Pragmatic RCT, **28,372 women** 40–74; risk-based (9-gene panel + polygenic score + BCSC model → 4 regimens, from 6-monthly MRI down to *defer*) vs annual.
- **Risk-based was non-inferior for advanced (stage ≥IIB) cancers, with ~3,800 fewer mammograms per 100,000 person-years.** Less screening, no excess advanced cancer.
- **Caveat:** it did **NOT** reduce biopsies (that superiority endpoint failed). **89% of women offered the choice picked risk-based.** *(DOI 10.1001/jama.2025.24784)


Cite UCSF clinical trial - The NCCN 2026 guideline  — recommending consideration of supplemental MRI for women with ≥1.7% five-year AI risk — is supported by CDR data and retrospective validation, but it preceded prospective RCT outcomes evidence. Prospective evidence generation is still ongoing, with MIRAI-MRI evaluating outcomes when AI-identified increased risk triggers supplemental screening MRI. The practice is running ahead of the trial readout, which is both the opportunity and the limitation worth naming in a "promise to practice to prevention" arc. T

---

Show these as a maturity gradient — concept → simulation → RCT:
StudyWhat it personalizesEngineEvidence tierTEMPO (2022)IntervalImage-based (Mirai) + RLRetrospective/simulatedMirai-interval (2025)Supplemental flagImage-basedRetrospectiveScreenTrustMRI (2024)Modality (MRI)Image-based (AISmartDensity)RCT, secondary endpointWISDOM (2025)Interval + start ageClinical + genetic (BCSC+PRS)Pragmatic RCT, primary endpoint
The punchline to say out loud: the largest RCT evidence (WISDOM) uses genetic/clinical risk, while the most sophisticated engines (TEMPO, Mirai) are image-based but haven't been tested in a prospective interval-modification RCT. The two haven't met yet. That's the open frontier.

TEMPO: better early detection than annual screening with 25% fewer mammograms at Karolinska. PubMed
ScreenTrustMRI: AI selection ~4× more efficient than density — 64.4 vs 16.5 cancers per 1,000 MRIs. PubMed
WISDOM: risk-based screening noninferior on stage IIB+ cancers, and ~90% of women chose it when offered. Springer
Optional 4th, the "missed risk" gut-punch: 30% of women who tested positive for high-risk genetic variants had no family history and wouldn't have been offered testing under current guidelines. 
---

## Bridge to Section 4 (image → patient)

> *"The image can tell us **when** to look and **how hard**. Once we find something, the question changes from 'is there cancer, and what is her risk?' to '**what is this cancer, and what will it do?**' That answer is not in radiology. It is across the hospital, on a glass slide. To understand the patient, we move to the pathology lab, where AI does something our images cannot."*

---
