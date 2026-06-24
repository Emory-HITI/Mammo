# Section 3 — From Detection to Risk: The Mammogram as a Biosensor
### ~6 minutes · spine stage: **IMAGE** · opens Part II (the future)

> **CALLBACK TO THE HOOK (seed #1):** *"Remember the first model — the one that knew this woman would develop cancer in about five years? Here it is."* Re-show the **real opening mammogram**, now revealed as a case a risk model flagged. The hook promised it; this section delivers it, and "the image already knew" returns on screen.

**Purpose:** The first leap into the future, and the spine's turn from **lesion → image**. AI changes the question a mammogram answers — from *"is there cancer today?"* to *"what is this person's risk, and how hard should we look?"* This is the move from a detection device to a biosensor, and it is no longer hypothetical: it is FDA-cleared.

---

## The opening — close seed #1 (re-show the mammogram from the hook)

> **[Bring back the EXACT opening mammogram — the real case, still read as normal.]**
>
> *"Let's go back to where we started. This is the mammogram I opened with — the one two radiologists called normal, and they were right. I told you the first model knew this woman would develop cancer in about five years. Here is that model."*
>
> **[Reveal the risk overlay / heatmap on the same image.]**
>
> *"It wasn't looking for a lesion. There wasn't one. It was reading the **tissue** — and it was right. The image already knew. Every mammogram already contains a five-year forecast. We are only now learning to read it."*

The image carries signal — invisible to the human eye — about *future* cancer, not just current cancer. Deep learning extracts it. That reframes screening from a snapshot into a trajectory: from *"is there cancer today?"* to *"when, and how hard should we look?"*

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

## The trap hiding in a good AUC: population performance ≠ individual reliability
### *(the intellectually honest centerpiece of this section — ~1.5–2 min)*

> *"Every number I just showed you — 0.76, 0.81, 0.84 — is a **population** number. And here's the uncomfortable truth: a beautiful AUC tells you how a model sorts a million women. It tells you almost nothing about the one woman sitting in your clinic."*

**What AUC/C-index actually is.** It's a *ranking* metric: the probability the model scores a random future-cancer patient higher than a random cancer-free one. A C-index of 0.80 is excellent at the population level — and *still* misranks a huge number of individual pairs. It is not the accuracy of any one woman's predicted risk.

**Discrimination is not calibration.** AUC measures *ranking* (discrimination). Whether a predicted "8% five-year risk" actually means 8% — that's *calibration*, a separate property, far less often reported, and the one that governs an individual decision like "do you get an MRI?" Image-based models tend to discriminate well; their calibration across sites, scanners, and subpopulations is the under-reported, harder problem.

**Discordance is the rule, not the exception.** Move from old models (Tyrer-Cuzick/Gail/PRS) to new image-based models and **roughly a third of women change risk category** (~35% reclassified) **[VERIFY exact figure — Arasu et al., DL vs traditional models, Radiology 2023]**. Yala 2019: the image model put **31% of future cancers in the top decile vs 18%** for Tyrer-Cuzick. So for *millions* of women, the old number and the new number disagree. That disagreement is not noise to be averaged away — it is the entire individual-level decision.

### The four quadrants (the slide — and the part that actually matters clinically)

> **[VISUAL — build later]** A 2×2: classical model (low/high) on one axis, image-based model (low/high) on the other.

| | **Image model: LOW** | **Image model: HIGH** |
|---|---|---|
| **Classical: HIGH** | ⚠️ **The dangerous quadrant** | ✅ Concordant high → MRI, short interval |
| **Classical: LOW** | ✅ Concordant low → standard/de-escalate | ★ **The high-value quadrant** |

1. **Both low / both high** — concordant. Act with confidence: reassure-and-standardize, or escalate to supplemental MRI.

2. **★ Classical LOW / image HIGH — the high-value quadrant.** The image sees something the questionnaire never could: a tissue-state signal. These are women the questionnaire era called "average." *But what kind of risk is it?* The Mirai paper itself says the model is strongest at **near-term** prediction and that high scores "may harbor occult malignancy or premalignant change" (Yala, STM 2021). So a high image score is partly a **"look harder NOW"** signal — short-interval follow-up, supplemental imaging, a second look at *this* mammogram — not automatically a lifetime-surveillance decision. **And acting on a signal we cannot explain is precisely the CAD trap** (see §1) — which is why interpretability (AsymMirai: the signal is bilateral asymmetry) and *prospective* validation (the **MIRAI-MRI trial, NCT05968157**, randomizing MRI for Mirai- vs Tyrer-Cuzick–high women) matter before we change management.

3. **⚠️ Classical HIGH / image LOW — the dangerous quadrant.** A BRCA carrier, a strong family history, a high polygenic score — and a quiet-looking mammogram that earns a low image score. **Never let a reassuring image read override known germline risk.** The image model was not trained to see inherited risk; it cannot. De-escalating here would be the most consequential error this technology invites.

### The synthesis: these models measure *different things*

> *"The genome asks one question: what did she inherit? The image asks a completely different one: what is her tissue doing right now? Discordance between them isn't a bug to resolve — it's two answers to two questions."*

- **Image-based DL** → near-term, tissue-state, occult/masking signal. Horizon: strongest at 1–2 years.
- **Genomic / PRS + family history** → inherited, lifetime, stable.
- **Density** → masking + modest independent risk.

The future is not picking the winner — it's combining them into a **risk vector** with the right *action* attached to each axis (image-high → look now; genome-high → lifelong surveillance). Naïve substitution of one for the other is dangerous; thoughtful fusion is the goal (early evidence: image-only DL **+ PRS** improves over either alone, Br J Cancer 2026). **Equity caveat:** discordance patterns likely differ by ancestry (PRS portability; image-model training diversity) — the *classical-high/image-low* error could fall hardest on exactly the groups with worse outcomes.

> **One-liner to land it:** *"Don't ask which model is right. Ask which question you're trying to answer for the woman in front of you — and whether you'd bet her management on a number you can't yet explain."*

---

## Density: the crude proxy these models replace (compressed — ~20 sec)

> One line, not a detour: *"For 40 years our only imaging 'risk' marker was breast density — a blunt, subjective category. The US made it a national reporting requirement in 2024; automated tools (Volpara, Densitas) at least made it objective. But density is a crude proxy for what the deep-learning score now reads directly. We're watching density get superseded in real time."* **[VERIFY exact Volpara/Brentnall figures if shown]**

---

## The future is already FDA-cleared (close the hook's "in this year's guidelines")

> *"Remember I said one of those three opening models was already in this year's guidelines? This is the one."*

**Clairity Breast (Constance Lehman's company) — FDA De Novo authorization announced June 2025** — the **first-ever** AI tool to predict 5-year breast cancer risk from a routine screening mammogram alone, validated across **>77,000 mammograms from 5 geographically distinct sites**; subsequently referenced in NCCN screening guidance. *The regulatory bridge from research model (Mirai) to clinical product has been crossed — research model to cleared product to guideline, in under five years.* **[VERIFY — press-sourced; confirm exact 2026 NCCN language; no primary peer-reviewed validation yet]**

---

## Risk-adapted screening — and Europe is running the definitive trial

> *"This is where it stops being a number on a slide and becomes a screening program — and the most important experiment in the world on this question is happening here, in Europe."*

**MyPeBS (Europe) — the definitive trial.** EU-funded RCT, **~85,000 women** across **6 countries** (Belgium, France, Italy, Israel, Spain, UK), risk-stratified vs standard age-based screening, endpoint = stage ≥2 cancer incidence; **ongoing, results pending.** The largest test ever of whether we should screen by *risk* instead of by *age*. *(BMC Cancer 2022; DOI 10.1186/s12885-022-09484-6)*

**WISDOM (US) — the first results are already in.** Esserman et al., *JAMA* 2025. Pragmatic RCT, **28,372 women** 40–74; risk-based (9-gene panel + polygenic score + BCSC model → 4 regimens, from 6-monthly MRI down to *defer*) vs annual.
- **Risk-based was non-inferior for advanced (stage ≥IIB) cancers — with ~3,800 fewer mammograms per 100,000 person-years.** Less screening, no excess advanced cancer.
- **Honest caveat:** it did **NOT** reduce biopsies (that superiority endpoint failed). But **89% of women offered the choice picked risk-based** — patients want this. *(DOI 10.1001/jama.2025.24784)*

**Where the image model plugs in (the forward thesis):** today MyPeBS and WISDOM triage on *genetics + classical models*. The obvious next step — and the synthesis of this whole section — is to add the **image-derived score as a second axis**: image-high → look sooner / MRI; concordant-low → safely extend the interval. Bernstein/Yala et al. (PLOS Digital Health 2026) show Mirai as a 1-year "rule-out" could cut caseload **36–75%** with quantified false-omission tradeoffs — the scaffold for "who can safely screen less." *(Tie back to the four quadrants: the trials currently use only the classical axis; the image axis is the value still on the table.)*

---

## Equity & honest caveats (do not skip)

- Mirai's training data was **only ~3.75% African American** — the central fairness worry.
- **But** independent validation is reassuring: Omoleye et al., Radiology: AI 2023 (Univ. of Chicago, **46.4% African American**) found **no significant performance difference by race** (1-yr AUC 0.71). The editorial's point: diverse-cohort validation must be **mandatory, not optional.**
- WISDOM was **77% non-Hispanic White** — generalizability to Black/Hispanic/Asian women unproven, and Black women face higher mortality, younger onset, more triple-negative disease. A system that *defers* screening could **widen disparities** if it under-calls these groups.
- **PRS portability** across ancestries remains unsolved. Real-world AUCs (~0.65–0.72) are good, not oracular.

---

## Bridge to Section 4 (image → patient)

> *"So the image can tell us **when** to look and **how hard**. But the moment we find something, the question changes completely — from 'is there cancer, and what's her risk?' to '**what is this cancer, and what will it do?**' And that answer isn't in radiology at all. It's across the hospital, on a glass slide. If we're serious about understanding the patient, we have to walk into the pathology lab — because that's where AI is doing something our images can't."*

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
- Brentnall/Volpara density figures; Clairity 77k-image validation + FDA date; MyPeBS enrollment/results (ongoing); Eriksson 2026 STM citation.
- **~35% reclassification figure** — confirm exact number + source (Arasu et al., "DL vs traditional risk models," Radiology 2023, PMC9552206). Yala 2019 "31% vs 18% top-decile" is solid.
- **Occult-malignancy / near-term framing** — ✓ grounded in the Mirai paper (Yala, Sci Transl Med 2021) discussion. **MIRAI-MRI trial = NCT05968157** (verify it's still recruiting/active).
- **Image-DL + PRS** improves over either alone — Br J Cancer 2026 (confirm exact cite before quoting).

## ⏱️ TIME NOTE
The "population vs individual" centerpiece adds ~1.5–2 min. **Density on-ramp now compressed to ~20 sec (done).** Realistic section length ~7–7.5 min vs the 6-min budget. If still long after a run-through, trim WISDOM detail or borrow ~1 min from Section 5 (the longest). Flag for rebalancing once all sections are locked.
