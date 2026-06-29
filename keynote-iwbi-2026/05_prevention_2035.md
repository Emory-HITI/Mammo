# IWBI 2026 — Closing Chapter: A Woman's Breast Journey in 2035
*Five slides for the "Prevention" arc of "From Promise to Practice to Prevention"*
*(Source markdown for `slides/section5alternate.html`)*

---

## Slide 1 — Estimating the Oncotype DX score from the H&E slide. *(moved from §2b — opens this section)*

And this is the capability imaging simply does not have — the third model from the opening. Oncotype DX, the 21-gene recurrence score, **costs about $4,000 and takes 1–2 weeks**. **Orpheus**, trained on **6,172 cases across three institutions**, infers that Recurrence Score directly from the H&E slide — and it not only flags high-risk disease better than a leading nomogram, it predicts recurrence even among low-score patients, beating the score itself: **AUC 0.89 vs 0.73** for flagging high-risk (RS > 25), and **0.75 vs 0.49** for predicting recurrence in low-RS patients. And because it is built on segmented tissue, you can see what it keys on — tumor cells, lymphocytes, stroma. Morphology to molecular identity, read straight off the glass.

*On-slide:* Cell-segmentation micrographs — tumor cells, lymphocytes, stroma (Fig. 4 e–h).

*Citation: Boehm et al. ("Orpheus"), Nat Commun 2025;16:2106 · concept: Kather et al., Nat Cancer 2020.*

---

---

## SLIDE 1
### She knows her risk before she ever lies down

**The risk engine of 2035 isn't built from one signal — it's built from three independent biological layers, and it learns with every scan.**

- **PRS alone is not enough — and the evidence is now unambiguous.** The PRS313 score (313 common SNPs, Mavaddat et al., *Am. J. Hum. Genet.*, 2019) established the genomic baseline, and Chen, Hoffmeister & Brenner *(JNCI Cancer Spectrum*, 2025) showed that women in the top PRS decile reach the average 50-year-old's 5-year breast cancer risk by their mid-30s, making a uniform age-40 start biologically indefensible. But PRS and germline testing are measuring almost entirely different populations. In a secondary analysis of the WISDOM trial, Shieh et al. *(JAMA Oncology*, 2026) found that **only 0.9% of high-penetrance pathogenic variant carriers** — BRCA1, BRCA2, PALB2 — would have been correctly assigned to high-risk screening using clinical risk plus PRS alone. The other 99.1% would have been sent for standard or biennial screening. Among carriers aged 40–49, **63.8% would have been recommended to defer screening until age 50.** PRS catches one population; germline testing catches another; clinical risk catches a third. All three layers are non-negotiable

- **The PRS architecture itself is being rebuilt.** PRS313 is a linear sum — every SNP contributes independently, additively. But cancer genetics is not additive: genes interact, pathways matter, epistasis is real. Li, Zeng, Snyder & Zhang *(Genome Research*, January 2025) published PRS-Net, a graph neural network that maps SNP-level risk onto gene-gene interaction networks and explicitly models those nonlinear relationships. It outperforms conventional PRS methods across multiple complex diseases. This is not a marginal refinement — it is a fundamental architectural shift. The PRS of 2035 will not resemble PRS313 any more than Mirai resembles first-generation CAD

- **At every mammogram, the imaging engine and the genomic engine update each other.** Rothwell et al. *(British Journal of Cancer*, April 2026) demonstrated in a Nurses' Health Study 2 cohort that adding PRS313 to the Mirai image-only model significantly improved AUC beyond either signal alone — confirming they are measuring partially independent biology. In 2035, there is no standalone mammogram risk score and no standalone PRS. The fusion happens at every screen: the imaging AI reads the parenchyma, the PRS reads the genome, and a combined engine outputs a single continuously updated risk trajectory

- **The RCT safety case is closed. The clinical case is building.** WISDOM (Esserman et al., *JAMA*, December 2025; 46,000 women) demonstrated risk-stratified screening was noninferior on Stage IIB+ cancers and produced a **one-third reduction in Stage IIB diagnoses** in the personalized arm. And the objection that risk communication will cause anxiety has been directly tested: PERSPECTIVE I&I (Lambert-Côté et al., *Genetics in Medicine*, May 2025; 3,753 women) found that breast cancer worry and psychological distress remained **low at one-year follow-up** even among women told they were at high risk. Women can hold this information. The barrier to implementation is not patient psychology — it is system inertia

**→ Extrapolated to 2035:** At age 30, a single integrated workup returns three outputs: a non-linear genomic risk score (PRS-Net-class, capturing gene-gene interactions and epistasis); a germline panel across 9+ susceptibility genes; and a BCSC-class clinical risk integration. These three non-overlapping signals assign screening tier, start age, and modality. At every subsequent mammogram, the Mirai-class imaging engine updates the genomic baseline — and sets the date of the next appointment. Not in 12 months. In however many months the current risk trajectory warrants. Women in the top quintile of Mirai-predicted 1-year risk return in 7 months. Women with persistently low image-based risk extend their interval. The schedule is dynamic, personalized, and never again driven by a calendar

*[Visual: Three largely non-overlapping circles — women flagged by PRS alone / by germline testing alone / by clinical risk alone — with the tiny overlap between them annotated. An arrow forward to 2035 shows them fused into a single engine. The point is the non-overlap, not the overlap.]*

**Slide 1 citations:**
- Mavaddat N et al. Polygenic risk scores for prediction of breast cancer and breast cancer subtypes. *Am. J. Hum. Genet.* 2019;104:21–34
- Chen X, Hoffmeister M, Brenner H. Deriving risk-adapted starting ages of breast cancer screening according to polygenic risk score. *JNCI Cancer Spectr.* 2025;9(3):pkaf056
- Shieh Y et al. Impact of population-based pathogenic variant testing on risk-based breast screening recommendations: a secondary analysis of the WISDOM study. *JAMA Oncol.* 2026. PMID: 42218736
- Li H, Zeng J, Snyder MP, Zhang S. Modeling gene interactions in polygenic prediction via geometric deep learning. *Genome Research.* 2025;35(1):178
- Rothwell JWD et al. Performance of an image-only deep learning breast cancer risk model with the addition of a polygenic risk score. *Br J Cancer.* 2026
- Esserman LJ et al. Risk-based vs annual breast cancer screening: the WISDOM randomized clinical trial. *JAMA.* 2025. PMID: 41385349
- Lambert-Côté L et al. Psychological and emotional impacts of communicating breast cancer risk … PERSPECTIVE I&I. *Genet Med.* 2025;27(8):101453. PMID: 40365755

---

## SLIDE 2
### The second reader is now software

**Double reading has been replaced — not by a single study, but by convergent evidence across multiple health systems on four continents. The question is no longer whether AI can do it. The question is why we waited.**

- **The evidence base is now definitive.** MASAI (Sweden, 105,934 women, *The Lancet*, January 2026): 44% radiologist workload reduction, 29% more cancers detected, 12% fewer interval cancers — the first RCT. ScreenTrustCAD (Sweden/Denmark): AI + one radiologist non-inferior to standard double reading. PRAIM (Germany, 463,094 women, *Nature Medicine*, January 2025): 17.6% higher detection rate, lower recall rate, 119 radiologists across 12 sites. Elías-Cabot et al. (Spain, 31,301 women, *Nature Medicine*, March 2026): partially autonomous AI triage across both digital mammography and DBT, CDR up 15.2%, workload reduced 63.6%. AI-STREAM (Korea, *Nat Commun*, 2025). GEMINI (UK, NHS Grampian, *Nature Cancer*, 2026). The same finding, replicated across healthcare systems that could not be more different from each other

- **The mechanism: AI now triages approximately two-thirds of cases away from the reading queue automatically.** In the Elías-Cabot prospective trial, cases classified as low-risk by AI were directly categorised as normal — no radiologist read required — while the remainder were double-read with AI decision support. Radiologist workload fell **63.6%**. At a background prevalence of ~6 per 1,000, the negative predictive value of AI triage for cases classified as normal is approximately **99.9%**

- **Combined with Mirai-class risk stratification, high-risk women are already coming back sooner.** In a UK triennial screening programme (Rothwell et al., *Radiology*, October 2025), the top 20% of Mirai scores captured 42.4% of all interval cancers

- **What this has given back to the radiologist is time and focus.** The radiologist of 2035 reviews the ~30% of cases the AI flagged, performs the biopsies AI localised, consults the high-risk women the system has been watching, and interprets the multimodal fusion studies on recalls. The endgame was never replacement — it was practising at the top of her licence

*[Visual: A before/after reading queue — long double-read backlog on the left; two-thirds auto-cleared by AI to a "normal" channel and the remaining third flowing to a single expert reader, with the highest-risk cases elevated to the top.]*

**Slide 2 citations:**
- Gommers J et al. (MASAI). *The Lancet.* 2026;407:505–514
- Eisemann N et al. (PRAIM). *Nature Medicine.* 2025;31:917–924
- Elías-Cabot E et al. AI-based triage and decision support in mammography and DBT … *Nat Med.* 2026;32:1296–1305
- Kelly J et al. (GEMINI). *Nature Cancer.* 2026
- Chang YW et al. (AI-STREAM). *Nat Commun.* 2025;16:2248
- Rothwell JWD et al. Mammography-based deep learning model for breast cancer risk … *Radiology.* 2025;317:e250391

---

## SLIDE 3
### The mammogram was just the beginning

**Five modalities. Five validated AI layers. In 2035, no single image makes the call — and no modality is ordered without the model already knowing whether it will change the answer.**

- **Ultrasound and MRI each now have their own AI evidence base.** BMU-Net (Yang et al., *Nat Biomed Eng*, December 2024): trimodal US fusion (B-mode, Doppler, elastography) + mammography + clinical metadata in 5,025 patients; matched radiologists, degrades gracefully when modalities are missing. ScreenTrustMRI (Salim et al., *Nat Med*, September 2024): scoring every negative mammogram and offering the top 6.9% an MRI yielded **64 cancers per 1,000 MRI exams vs 16.5 per 1,000** in the density-based comparator. The AI score, not density, decides who gets MRI

- **Pathology is the fifth modality.** Foundation models pretrained on 100,000+ whole-slide images (UNI, Chen et al., *Nat Med*, 2024) underpin tissue-level inference; models on H&E + clinical metadata predict pathological complete response to neoadjuvant chemo at AUC ~0.87–0.90 (Ataraxis AI, clinical validation ongoing). Treatment intelligence off the same slide that makes the diagnosis

- **BINDS unifies it into one architecture.** Wang et al. (*Nat Biomed Eng*, May 2026): 27,048 participants, 8 centres — a two-stage system (US + MG first, MRI when the AI determines it will change the assessment), with radiology–pathology alignment built in. Every fusion modality outperformed any single modality alone

- **Genomics completes the stack.** PRS-Net and the Rothwell BJC 2026 additivity result mean the genomic layer is always present as a prior. A dense breast with high PRS is read differently than a dense breast with average PRS

**→ Extrapolated to 2035:** A unified multimodal foundation model ingests all available inputs at each encounter and outputs a single integrated risk-and-diagnosis recommendation. Modality selection has become a model parameter — informed by, and accountable to, two decades of multimodal AI validation

*[Visual: A five-layer pipeline diagram — genomics (always-on) → MG (entry) → US (conditional) → MRI (conditional) → pathology (at biopsy) — feeding a single unified risk output, with weighted connections.]*

**Slide 3 citations:**
- Yang Z et al. A multimodal ML model for the stratification of breast cancer risk (BMU-Net). *Nat Biomed Eng.* 2024;8:1551–1564
- Salim M et al. AI-based selection for supplemental MRI … ScreenTrustMRI. *Nat Med.* 2024;30(9):2623–2630. PMID: 38977914
- Chen RJ et al. Towards a general-purpose foundation model for computational pathology (UNI). *Nat Med.* 2024;30:850–862
- Wang C et al. A deep learning system for non-invasive breast cancer diagnosis with multimodal data (BINDS). *Nat Biomed Eng.* 2026. DOI 10.1038/s41551-026-01654-2
- Li H et al. Modeling gene interactions in polygenic prediction (PRS-Net). *Genome Research.* 2025;35(1):178
- Rothwell JWD et al. Image-only DL risk model + PRS. *Br J Cancer.* 2026

---

## SLIDE 5
### The mammogram is now a whole-body health visit

**We built a cancer screening system. We discovered it was a window into the entire body. In 2035, that window is open.**

- **The cardiovascular signal was there all along.** AI-quantified breast arterial calcification (BAC) predicts MACE and all-cause mortality beyond the PREVENT score in 123,762 women across Emory and Mayo — severe BAC associated with a **2.8× increased risk of death within five years** (Dapamede et al., *Eur Heart J*, 2026). Barraclough et al. predicted cardiovascular events from mammographic image features alone across 49,196 women over 8.8 years, with concordance comparable to the Pooled Cohort Equations (*Heart*, February 2026)

- **Foundation models read biological age.** Mammo-AGE (Pan et al., *Nat Commun*, December 2025): trained on 95,826 mammograms from 44,497 women; estimates breast age (MAE 4.2–6.1 yr), and the breast age gap stratifies cancer risk independent of density. The mammographic analog of CXR-Age (*Nat Aging*, 2021)

- **Diabetes, kidney, and metabolic risk are the emerging frontier.** BAC correlates with diabetes, hypertension, and hypercholesterolemia; the Emory/Mayo team has named peripheral artery and kidney disease as next applications (*ACC 2025*). Framed honestly as emerging, not established

- **The equity implication is transformative.** ~40 million US mammograms a year, plus mobile units reaching rural communities. For women who will never see a cardiologist, the mammogram is the only preventive health touchpoint the system reliably delivers

*[Visual: A single mammogram with five annotated output streams — cancer risk / cardiovascular risk / biological age / metabolic (dotted) / kidney (dotted).]*

**Slide 5 citations:**
- Dapamede T et al. AI-based quantification of breast arterial calcifications … *Eur Heart J.* 2026:ehag128. [Emory/Mayo, 123,762 women — Hari's paper]
- Barraclough JY et al. Predicting cardiovascular events from routine mammograms using ML. *Heart.* 2026;112(5):261–269. PMID: 40957672
- Pan X et al. Mammo-AGE: deep learning estimation of breast age from mammograms. *Nat Commun.* 2025;16:11157
- Raghu VK et al. Deep learning to estimate biological age from chest radiographs (CXR-Age). *Nat Aging.* 2021;1:1094–1100

---

## SLIDE 6
### The magic wand doesn't exist. The pipeline does.

**Foundation models did not unlock the next era of medical AI by ingesting mountains of dirty data. They unlocked it by making high-quality data curation possible at scale. The model was never the bottleneck. The label was.**

- **Quality over quantity, demonstrated and ignored.** Andrew Ng's data-centric AI paradigm (2021); DataComp-LM discards 99% of raw data through quality filtering; Sambasivan et al. (*CHI 2021*) documented "everyone wants to do the model work, not the data work" in high-stakes medical AI

- **For a decade the rate-limiting step was generating high-quality structured labels from the unstructured archives health systems already possessed.** Every radiology report encodes structured knowledge as prose; extracting it accurately required slow, expensive expert annotation (Zhang et al., *Data-Centric Foundation Models in Computational Healthcare*, arXiv:2401.02458)

- **LLMs crossed the human-level threshold for structured label extraction around 2024–2025.** GPT-based labeling reached average F1 0.90 across 14 thoracic pathologies (Abdullah & Kim, *JMIR Med Inform*, 2025); the Data Scaling Laws for Radiology Foundation Models pipeline (arXiv:2509.12818) built training on GPT-4o-extracted labels including arterial calcification, radiologist-validated

- **The virtuous cycle.** ScaleMAI (arXiv:2501.03410) formalized an EM loop — model trains on current labels, flags divergence for targeted human review, retrains. Mammo-CLIP (Ghosh et al., ECCV 2024) applied this to mammography, improving data efficiency and robustness

**→ The implication:** None of the datasets behind the prior slides (BINDS 27,048; BMU-Net 5,025; Mammo-AGE 95,826; Barraclough 49,196) existed as clean, model-ready training sets before the teams built them — from archives and the unstructured reports that accompanied them. EMBED (3.5M exams; the Emory AI Image Extraction Core) is the local example. In 2035, every mammogram ever performed is in the training set — we just needed a way to read the labels that were always there

*[Visual: Two columns — "What we had" (archive of imaging + unstructured reports) and "What changed" (LLM reads report → extracts label → image-label pair → foundation model → loops back to improve label extraction).]*

**Slide 6 citations:**
- Sambasivan N et al. "Everyone wants to do the model work, not the data work": data cascades in high-stakes AI. *CHI 2021*
- Zhang Y et al. Data-centric foundation models in computational healthcare: a survey. *arXiv:2401.02458*
- Abdullah A, Kim ST. Automated radiology report labeling … LLM framework. *JMIR Med Inform.* 2025;13:e68618
- Goel S et al. Data scaling laws for radiology foundation models. *arXiv:2509.12818*
- Yuan R et al. ScaleMAI. *arXiv:2501.03410*
- Ghosh S et al. Mammo-CLIP. *arXiv:2405.12255 / ECCV 2024*
- Ng A. Data-centric AI. Stanford / DeepLearning.AI, 2021

---

*Deck title: "From Promise to Practice to Prevention: Three Decades of AI in Breast Imaging"*
*Keynote: IWBI 2026, Thessaloniki/Greece*
