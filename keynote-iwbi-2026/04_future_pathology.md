# Section 4 — The Other Half of the Slide: Computational Pathology
### ~6 minutes · spine stage: **PATIENT** (begins) · a part a radiology audience rarely sees

> **CALLBACK TO THE HOOK (seed #3):** *"The third model — across town, on the glass slide — about to read a $4,000 recurrence score. This is it."* Close the loop with the Oncotype-from-H&E result (Orpheus, Nat Commun 2025).

**Purpose:** Move from radiology to pathology — the spine's move into the **patient**. The same trajectory (hand-crafted CAD → deep learning → foundation models) is playing out in pathology. Pathology AI also has a capability imaging does not: it can **infer molecular and genomic identity directly from a stained glass slide.** This section closes by connecting the two departments — the bridge to clinical intelligence (Section 5).

---

## THE SCRIPT — foundation models were built on tissue, not radiology

*Lead the section with this fact, then the invasive-lobular example that makes it relevant to an imaging audience.*

> *"A point that is often unexpected for imaging people: some of the largest foundation models in medicine were not built on radiology images. They were built on pathology slides.*
>
> *In 2024, a model called **Virchow** — from Paige and Microsoft — was published in *Nature Medicine*. It was trained on roughly **one and a half million whole-slide images, from about a hundred thousand patients**, in a self-supervised way; the slides were not hand-labeled. The model learned the structure of human tissue on its own. It detects cancer across more than a dozen tissue types at clinical-grade accuracy, **including rare cancers, where training data is limited.***
>
> *This part is directly relevant to breast imaging. These models do more than locate the tumor a pathologist would find. They read features off the H&E that are not visible to the eye. Consider the cancer that is hardest to detect on a mammogram, **invasive lobular carcinoma.** It hides in dense tissue and grows in single file. Lobular cancer is defined molecularly by the loss of one protein, **E-cadherin, the CDH1 gene.** An AI model can predict CDH1 status from the H&E slide alone, with an AUC of about **0.94.** The morphology was always a readout of the molecular biology underneath it, but it was not visible by eye. The model reads it.*
>
> *The model also extended this. Among the cases it called CDH1-lost where the standard test found no mutation, **about three-quarters had some other mechanism switching that gene off** — fusions and noncoding changes the assay did not test for. The model reproduced the known biology and also identified biology the assay had missed."*

> **Two points to carry into the second half** *(say this verbatim — it is the thesis of the convergence):*
> *"First: the mammogram and the H&E slide both contain more than what we currently read off them. Second: radiology and pathology have operated for decades as separate departments, with separate images and separate reports, and AI is the first tool that treats them as **two views of the same patient.** That convergence is where this is heading, and it is where I want to spend most of the remaining time."*

### Source-check for the script
- **Virchow** — Vorontsov et al., *Nature Medicine* 2024 (Paige + Microsoft). **1,488,550 WSIs from 119,629 patients**; self-supervised; **0.95 specimen-level AUC** across **9 common + 7 rare cancers** (published Nature version; the arXiv preprint says 10 + 7 = 17 — say *"more than a dozen tissue types"* to stay safe). *(DOI 10.1038/s41591-024-03141-0)*
- **CDH1 / invasive lobular** — this is a **separate, dedicated paper, NOT Virchow:** *"A Genomics-Driven AI-Based Model Classifies Breast Invasive Lobular Carcinoma and Discovers CDH1 Inactivating Mechanisms,"* **Cancer Research 2024** (Mount Sinai; PMID 39106449). **CDH1 biallelic prediction AUC 0.944** (sens 91.6%, spec 85.9%); lobular-phenotype AUC 0.941; ILC diagnosis accuracy ~0.96; **74%** of AI-flagged "CDH1-lost" cases lacking the classic mutation had alternative inactivating mechanisms. *Use "about 0.94," not 0.97, and don't attribute it to Virchow.*

---

## The foundational landmark — CAMELYON16 (pathology's reader study)

**Ehteshami Bejnordi et al., JAMA 2017** — an "AI vs pathologists" study that parallels radiology reader studies. Lymph-node metastasis detection on H&E; 129-slide test set; 11 pathologists under a 2-hour time constraint.
- **Best algorithm AUC 0.994 vs mean pathologist 0.810** under time pressure (P<0.001); top-5 algorithms (0.960) ≈ expert with unlimited time (0.966).
- **Soundbite:** *Under realistic time pressure, the best AI exceeded the pathologist panel.* The result frames AI as a way to address fatigue and time constraints rather than as a replacement.
- *DOI 10.1001/jama.2017.14585*

---

## A current clinical problem AI addresses: HER2-low

An example of AI applied to a current clinical problem.
- **The problem:** since **HER2-low (IHC 1+ or 2+/ISH−) became treatable** (trastuzumab deruxtecan; DESTINY-Breast04), the decisive call is **HER2 0 vs 1+**, a distinction the assay was not designed to make and on which pathologists disagree.
- **Krishnamurthy et al., JCO Precision Oncology 2024** (120 HER2 IHC slides, 4 sites): baseline pathologist agreement **72.4%**; automated AI **92.1%**. With AI, reader agreement rose **75.0% → 83.7%**; for the **HER2 0 vs 1+** subset, **69.8% → 87.4%**, accuracy **81.9% → 88.8%**. *(DOI 10.1200/PO.24.00353)*
- **Soundbite:** *Pathologists agree on HER2 about 72% of the time, and that call now determines eligibility for a drug that improves survival. AI raises agreement on the hardest call from ~70% to ~87%.*

---

## "Morphology to molecular" — a capability imaging does not have (seed #3 payoff)

> Predict genomics, receptor status, and recurrence risk **directly from an H&E slide.**

- **Kather et al., Nature Cancer 2020** — pan-cancer proof of concept: one DL workflow infers mutations, subtypes, expression signatures, and biomarkers from routine H&E across >5,000 patients. *The concept citation.* *(DOI 10.1038/s43018-020-0087-6)*
- **Oncotype DX recurrence score from H&E — Boehm et al. ("Orpheus"), Nature Communications 2025.** 6,172 cases, 3 institutions: infers the 21-gene Recurrence Score from H&E; flags TAILORx high-risk (RS>25) at **AUC 0.89 vs 0.73** for a leading nomogram; in RS≤25 patients, predicts metastatic recurrence at higher accuracy than the RS itself (0.75 vs 0.49). *(DOI 10.1038/s41467-025-57283-x)*
  - *Connection to the hook (seed #3):* Oncotype DX costs **~$4,000+** and takes 1–2 weeks. The H&E slide is already on the desk. This is the model the hook referred to as reading the glass slide across town.
- Emory tie-in: **Li et al., Frontiers in Medicine 2022** (Emory + Ohio State) — DL features improve the Magee-equation correlation with Recurrence Score. **[your-group flavor]**

---

## The foundation-model wave (2024) — the substrate behind it

**Plain-language frame:** instead of training a new model per task, labs trained **one self-supervised model on millions of unlabeled slides** — a *"GPT for tissue"* — that learns the visual language of pathology and adapts to a given task with minimal fine-tuning. These were published within months of each other in Nature / Nature Medicine:

| Model | Group | Scale | Note |
|---|---|---|---|
| **UNI** (Chen et al., Nat Med 2024) | Mahmood Lab, Harvard/BWH | >100k WSIs / >100M images, ~20 tissues, 34 tasks | open-source tissue encoder |
| **CONCH** (Lu et al., Nat Med 2024) | Mahmood Lab | 1.17M image–text pairs | vision-language; query tissue with text |
| **Virchow** (Vorontsov et al., Nat Med 2024) | Paige + Microsoft | 1.49M WSIs / 119,629 patients (MSK) | 0.95 AUC, pan-cancer incl. rare *(the script's model)* |
| **Prov-GigaPath** (Xu et al., Nature 2024) | Microsoft + Providence + UW | 1.3B tiles / 171k WSIs / 28 centers | first true whole-slide FM; SOTA 25/26 tasks |

**Relevance to breast:** these models reduce the data needed for a breast task (subtyping, HER2, RS prediction, TIL scoring, the CDH1/lobular predictor above) and **generalize across institutions and scanners**, which addresses the reproducibility problem that had limited clinical pathology AI.

---

## Prognostic AI from tissue (brief)

- **TIL quantification — Lu et al., JCO CCI 2020.** DL TIL maps on 1,000 TCGA breast WSIs; density/spatial clustering independently prognostic by subtype. Operationalizes the International TILs Working Group score. *(DOI 10.1200/CCI.19.00126)*
- **Recurrence/prognosis:** the Orpheus result (exceeding the RS in RS≤25) is the main point.

---

## Clinical-deployment reality (the "is this real?" check)

- **FDA precedent — Paige Prostate (Sept 2021):** first FDA-authorized AI in digital pathology (De Novo). Cancer detection +7.3%, false negatives −~70%. It is for prostate, not breast, and serves as the **regulatory template** breast tools follow.
- **Current limitations:** glass slides still dominate, and full digital workflows are in a minority of US labs. The bottleneck is scanning, infrastructure, and reimbursement rather than the algorithm. Most breast pathology AI is decision-support; molecular-from-H&E is not yet replacing the assays.

---

## The bridge to Section 5 (the convergence — already teed up in the script)

> The script's closing line — *"two views of the same patient"* — is the bridge. Section 5 picks it up: radiology AI localizes the lesion, pathology AI reads molecular identity from the same disease, and multimodal models combine them into one representation of the patient. **That convergence is clinical intelligence.**

---

## Slide-ready key numbers
- CAMELYON16: best AI **AUC 0.994** vs pathologist panel **0.810** (JAMA 2017).
- **CDH1 / invasive lobular from H&E: AUC ~0.94; 74% of AI-flagged cases had hidden CDH1 inactivation (Cancer Research 2024).**
- HER2 0-vs-1+ agreement **70% → 87%** with AI (JCO PO 2024).
- Oncotype RS from H&E: high-risk **AUC 0.89** vs nomogram 0.73; exceeds RS in low-RS (0.75 vs 0.49) (Nat Commun 2025).
- Virchow: **1.49M slides / 119,629 patients, 0.95 AUC** pan-cancer incl. rare (Nat Med 2024).
- Paige Prostate: first FDA-authorized pathology AI (2021).

## [VERIFY] before podium
- **Virchow cancer-type count: 16 (9+7, published Nature) vs 17 (10+7, arXiv preprint)** — say "more than a dozen" to be safe.
- **CDH1/lobular = Cancer Research 2024 (Mount Sinai, PMID 39106449), AUC 0.944 — NOT Virchow, NOT 0.97.** ✓ corrected.
- Still open: canonical Ki-67/mitosis/grading paper; definitive breast neoadjuvant-pCR-from-H&E ref; exact PathChat citation; hard % for US digital-pathology adoption.
