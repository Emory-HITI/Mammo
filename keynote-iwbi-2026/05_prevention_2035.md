# IWBI 2026 — Closing Chapter: A Woman's Breast Journey in 2035
*Five slides for the "Prevention" arc of "From Promise to Practice to Prevention"*
*(Source markdown for `slides/section5alternate.html`)*

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

## SLIDE 4 — From one score to a multimodal recurrence — and detection — engine.
*(rebuilt; replaces the earlier "mammogram was just the beginning" slide, which is kept in the file but hidden in the final deck)*

**Subhead:** Here's what each layer does today — and where ten years takes it. By 2035 no single image makes the call, and no modality is acquired unless the model already expects it to change the answer.

### LEFT — input modalities

**Pathology · H&E**
- *Now (2025):* Foundation models pretrained on 100,000+ whole-slide images (UNI — Chen et al., Nat Med 2024;30:850–862) drive tissue-level inference; H&E + clinical predict pCR at AUC ~0.87–0.90 (Ataraxis AI, validation ongoing).
- *2035:* The diagnostic slide is also the treatment slide — one pass yields diagnosis, molecular subtype, and response prediction; path foundation models co-trained with imaging so rad–path agreement is built in, not reconciled after.

**MRI**
- *Now (2025):* ScreenTrustMRI (Salim et al., Nat Med 2024;30(9):2623–2630; PMID 38977914) — an AI score, not density, selects the top 6.9% of negative mammograms for MRI: **64 cancers/1,000 vs 16.5/1,000**.
- *2035:* MRI is never ordered blind — it fires only when the fusion model predicts it changes management; abbreviated and low/zero-contrast protocols routine.

**Ultrasound**
- *Now (2025):* BMU-Net (Yang et al., Nat Biomed Eng 2024;8:1551–1564) — trimodal fusion (B-mode, Doppler, elastography) + mammography + clinical, 5,025 patients; matches radiologists, degrades gracefully when inputs are missing.
- *2035:* AI-guided acquisition turns US into a low-cost first-line in resource-limited settings.

**Clinical**
- *Now (2025):* Structured chart context (age, density, hormonal/family history, prior pathology) fused with imaging for risk stratification and modality selection (NCCN-2026-aligned).
- *2035:* The model reads the longitudinal EHR like a colleague; modality selection becomes a learned parameter, not a fixed protocol.

**Genomics (NEW)**
- *Now (2025):* Germline — PRS-Net (Li et al., Genome Research 2025;35(1):178) and Rothwell et al. (Br J Cancer 2026): image-only DL risk and PRS are additive. Somatic — WGS-powered ctDNA MRD (Garcia-Murillas/Turner et al., Ann Oncol 2025;36(6):673–681) detects relapse **~15 months** ahead of imaging; RaDaR (Lipsyc-Sharf et al.) ~12.4-mo lead, 100% sensitivity for metastatic recurrence.
- *2035:* PRS is a standing prior at every read; post-treatment ctDNA surveillance is routine, and molecular recurrence detected 1–2 years before imaging triggers pre-emptive (interception) therapy.

### CENTER — multimodal fusion
- *Now (2025):* BINDS (Wang et al., Nat Biomed Eng 2026; DOI 10.1038/s41551-026-01654-2) — **27,048 participants, 8 centres**; two-stage (US + MG first, MRI when the AI decides it changes the assessment), rad–path alignment built in. Every fusion modality beat any single modality alone.
- *2035:* One unified multimodal foundation model ingests whatever inputs exist at each encounter, emits a single integrated risk-and-diagnosis recommendation — and decides which modality to acquire next.

### RIGHT — outputs

**Detection (NEW)**
- *Now (2025):* MASAI — first RCT in breast AI (Lång et al., final results Lancet 2026; safety Lancet Oncol 2023;24:936–944) — 105,915 women; **sensitivity 80.5% vs 73.8%, +29% CDR, −12% interval cancers, −44% workload**, no FP increase. PRAIM real-world (Eisemann/Katalinic et al., Nat Med 2025) — 463,094 women; CDR 6.7 vs 5.7/1,000 (+17.6%), non-inferior recall.
- *2035:* Mortality-endpoint readouts have matured; AI is the primary reader with targeted human oversight; interval and modality personalized by risk, not age.

**Recurrence (risk)**
- *Now (2025):* 3D multimodal MRI predicts 2–4-yr DFS at **AUC 0.90 → 0.88**; US + MRI + clinical reaches **c-index ~0.80** (external); multimodal DL correlates with the 21-gene RS / Oncotype DX (Breast Cancer Res 2025; 10.1186/s13058-025-02129-z); ctDNA-MRD adds a molecular signal months ahead of imaging.
- *2035:* A continuously updated recurrence risk — imaging + ctDNA + clinical fused at every encounter — replaces the one-time score.

**Treatment (course)**
- *Now (2025):* H&E + clinical predict pCR at AUC ~0.87–0.90 (Ataraxis, validation ongoing); MRI multimodal models guide adjuvant decisions (PMID 40345352).
- *2035:* The unified model outputs a treatment recommendation with predicted benefit; ctDNA-guided escalation/de-escalation standard; neoadjuvant response forecast before the first cycle.

### Diagram structure
Genomics (germline PRS + ctDNA MRD, always-on / longitudinal prior) → Clinical (chart context) → MG (entry) → US (conditional) → MRI (conditional, AI-gated) → Pathology · H&E (at biopsy) → **Multimodal fusion** → outputs: **Detection · Recurrence · Treatment**. Visual cue: each box in two registers — solid "now" + ghosted "2035".

### Full citations
- Lång K et al. MASAI — clinical safety analysis. *Lancet Oncol.* 2023;24:936–944. (Final results, *Lancet* 2026.)
- Eisemann/Katalinic A et al. Real-world AI for cancer detection (PRAIM). *Nat Med.* 2025.
- Yang Z et al. Multimodal ML for breast cancer risk (BMU-Net). *Nat Biomed Eng.* 2024;8:1551–1564.
- Salim M et al. AI-based selection for supplemental MRI (ScreenTrustMRI). *Nat Med.* 2024;30(9):2623–2630. PMID 38977914.
- Chen RJ et al. Foundation model for computational pathology (UNI). *Nat Med.* 2024;30:850–862.
- Wang C et al. DL system for non-invasive breast cancer diagnosis (BINDS). *Nat Biomed Eng.* 2026. DOI 10.1038/s41551-026-01654-2.
- Li H et al. Gene interactions in polygenic prediction (PRS-Net). *Genome Research.* 2025;35(1):178.
- Rothwell JWD et al. Image-only DL risk model + PRS. *Br J Cancer.* 2026.
- Garcia-Murillas I, Turner NC et al. WGS-powered ctDNA MRD. *Ann Oncol.* 2025;36(6):673–681. DOI 10.1016/j.annonc.2025.01.021.
- Lipsyc-Sharf M et al. Personalized ctDNA (RaDaR) for late recurrence in HR+ breast cancer. *J Clin Oncol.* (verify year/volume.)
- EXActDNA-003 / NSABP B-64. Bespoke ctDNA MRD validation. NCT06401421 (recruiting; est. 2030).
- Multimodal DL ~ Oncotype DX. *Breast Cancer Res.* 2025. DOI 10.1186/s13058-025-02129-z.
- MRI-based multimodal: recurrence + adjuvant therapy. 2025. PMID 40345352.
- Ataraxis AI — H&E + clinical pCR prediction (validation ongoing).

---

## SLIDE 5 — The mammogram is now a whole-body health visit. *(rebuilt; prior version kept but hidden in final)*

*We built a cancer-screening system. We discovered a window into the entire body — and we are not the only ones who found one. The question is what that window shows today, and what it shows if we move it forward ten years.*

- **The cardiovascular signal was there all along.**
  - *Now (2025):* AI-quantified BAC predicts MACE and all-cause mortality beyond the PREVENT score across **123,762 women** (Emory + Mayo); severe BAC → **2.8× higher 5-yr risk of death** (Dapamede et al., *Eur Heart J* 2026). Barraclough et al. predicted CV events from mammographic image features alone in **49,196 women**, concordance rivaling the Pooled Cohort Equations (*Heart* 2026).
  - *2035:* a cardiovascular estimate is an automatic, **reimbursed** output of every screening mammogram, written back to the chart and routing the high-risk woman straight into a primary-prevention pathway.

- **The image also encodes biological age.**
  - *Now (2025):* Mammo-AGE estimates breast age to within **4–6 years**; the breast-age gap stratifies cancer risk independent of density (Pan et al., *Nat Commun* 2025) — the mammographic analog of CXR-Age (Raghu et al., *Nat Aging* 2021).
  - *2035:* an imaging-derived biological age is reported next to chronological age and used to personalize screening intervals and prevention intensity.

- **The mammogram is not alone — every routine image is becoming a whole-body biomarker.**
  - *Now (2025):* chest X-ray estimates 10-yr CV risk (CXR CVD-Risk, Weiss et al., *Ann Intern Med* 2024); abdominal CT yields a longevity model from muscle, fat, bone, aortic plaque (Pickhardt et al., *Nat Commun* 2025); a retinal foundation model reads CV, renal, and neurodegenerative risk from one photo (RETFound, Zhou et al., *Nature* 2023). The 2026 AJR Forum calls opportunistic imaging a maturing discipline (Magudia, Pickhardt et al., *AJR* 2026).
  - *2035:* the modality stops mattering — one multimodal foundation model ingests whatever images a patient already has and returns a single longitudinal systemic-risk profile; the mammogram is one node in a converging field.

- **Metabolic and renal risk are the next frontier — emerging, not yet established.**
  - *Now (2025):* BAC correlates with diabetes, hypertension, and hypercholesterolemia; the Emory/Mayo team has named peripheral-artery and kidney disease as next targets (ACC 2025). Associations, not validated predictors.
  - *2035:* if validation holds, the mammogram opportunistically flags metabolic and kidney risk too — the most cautious bullet, most dependent on prospective evidence we don't yet have.

- **The equity implication is why it matters.**
  - *Now (2025):* ~**40 million** US mammograms/yr, plus mobile units reaching rural communities — for the woman who will never see a cardiologist, the mammogram is often the only preventive touchpoint the system reliably delivers, yet the extra signals mostly go unused.
  - *2035:* the screening visit becomes a genuine whole-body prevention checkpoint that **closes** access gaps rather than widening them — a design choice, since these models can also learn demographic shortcuts.

*[Visual: a single mammogram with five annotated output streams — cancer risk / cardiovascular / biological age / metabolic (dotted) / kidney (dotted); faint in the periphery, chest X-ray · abdominal CT · retinal photo emit the same streams — the mammogram as one node in a converging field.]*

**Speaker note (the frontier — name it so "the modality stops mattering" has evidence):** Whole-body MRI models learn one representation and predict many diseases at once; **Merlin** reads 3D CT as fluently as language; **OCTCube-M** takes one eye scan and predicts seven systemic diseases; **MOSCARD** fuses chest X-ray with ECG. Stop asking what the scan was ordered for; ask what it tells us about the whole patient. *Close:* "Mammography proved the principle at population scale. The next decade is foundation models that pull systemic risk from any routine image — and mammography starts ahead, because it already reaches 40 million women a year, including the ones the rest of the system never sees."

**Slide 5 citations:**
- Dapamede T et al. AI-quantified BAC predicts MACE/mortality beyond PREVENT. *Eur Heart J.* 2026 (Emory/Mayo, 123,762 — our paper).
- Barraclough JY et al. Predicting CV events from routine mammograms. *Heart.* 2026;112(5):261–269. PMID 40957672.
- Pan X et al. Mammo-AGE: breast age from mammograms. *Nat Commun.* 2025;16:11157.
- Raghu VK et al. CXR-Age. *Nat Aging.* 2021;1:1094–1100.
- Weiss J et al. CXR CVD-Risk. *Ann Intern Med.* 2024.
- Pickhardt PJ et al. Abdominal-CT longevity model. *Nat Commun.* 2025.
- Zhou Y et al. RETFound. *Nature.* 2023.
- Magudia K, Pickhardt PJ et al. Opportunistic imaging forum. *AJR.* 2026.
- Frontier (notes only; verify venues): Whole-body MRI representation learning, arXiv 2025 (2508.02307); Merlin CT VLM, *Nature* 2026 (verify); non-contrast-CT breast+lung screening FM, *Nature Health* 2026 (verify); OCTCube-M, arXiv 2024 (2408.11227); MOSCARD, arXiv 2025 (2506.19174).

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
