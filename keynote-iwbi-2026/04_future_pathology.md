# Section 4 — The Other Half of the Slide: Computational Pathology
### ~6 minutes · "Predict" (tissue) · the part a radiology audience rarely sees

**Purpose:** Cross the room from radiology to pathology. The point: the *same trajectory* (hand-crafted CAD → deep learning → foundation models) is playing out in pathology — but pathology AI can do something imaging cannot: **infer molecular and genomic identity directly from a stained glass slide.** That capability is the bridge to integrated, multimodal decision-making (Section 5).

---

## The one-line frame for radiologists

> *"Radiology AI reads pixels to find and characterize a lesion. Pathology AI does the same — detect, grade, quantify — plus something we can't: it infers the tumor's molecular and genomic identity from a cheap H&E slide."*

---

## The foundational landmark — CAMELYON16 (pathology's reader study)

**Ehteshami Bejnordi et al., JAMA 2017** — the defining "AI vs pathologists" study, a clean parallel to radiology reader studies. Detecting lymph-node metastases on H&E whole-slide images; 129-slide test set; 11 pathologists under a 2-hour time constraint.
- **Best algorithm AUC 0.994 vs mean pathologist 0.810** under time pressure (P<0.001). Top-5 algorithms (mean 0.960) ≈ the expert pathologist with unlimited time (0.966).
- **Soundbite:** *Under realistic time pressure, the best AI beat the human panel; with unlimited time, the expert matched the AI.* Reframed AI as the antidote to fatigue and time, not a replacement.
- *DOI 10.1001/jama.2017.14585*

---

## The current clinical pain point AI is solving: HER2-low

This is the strongest "AI fixes a real, current problem" story for the audience.
- **The problem:** since **HER2-low (IHC 1+ or 2+/ISH−) became treatable** (trastuzumab deruxtecan; DESTINY-Breast04), the clinically decisive call is **HER2 0 vs 1+** — a distinction the assay was never designed to make, and pathologists disagree on.
- **Krishnamurthy et al., JCO Precision Oncology 2024** (120 HER2 IHC slides, 4 sites): baseline pathologist agreement only **72.4%**; automated AI **92.1%** vs high-confidence ground truth. With AI assistance, reader agreement rose **75.0% → 83.7%**, and for the hard **HER2 0 vs 1+** subset, agreement jumped **69.8% → 87.4%**, accuracy **81.9% → 88.8%**. *(DOI 10.1200/PO.24.00353)*
- **Soundbite:** *Pathologists agree on HER2 only ~72% of the time, yet that call now decides eligibility for a drug that improves survival. AI pushes the hardest call from ~70% to ~87% — a concrete precision-oncology fix.*

---

## "Morphology to molecular" — the capability imaging doesn't have

> Predict genomics, receptor status, and recurrence risk **directly from a cheap H&E slide.**

- **Kather et al., Nature Cancer 2020** — the foundational pan-cancer proof: one DL workflow infers mutations, molecular subtypes, expression signatures, and standard biomarkers from routine H&E across >5,000 patients, generalizing across populations. *The concept citation.* *(DOI 10.1038/s43018-020-0087-6)*
- **Oncotype DX recurrence score from H&E — Boehm et al. ("Orpheus"), Nature Communications 2025.** 6,172 cases, 3 institutions: infers the 21-gene Recurrence Score from H&E; identifies TAILORx high-risk (RS>25) with **AUC 0.89 vs 0.73** for a leading clinicopathologic nomogram; in RS≤25 patients it predicted metastatic recurrence **better than the RS itself** (time-dependent AUC 0.75 vs 0.49). *(DOI 10.1038/s41467-025-57283-x)*
  - *Why it lands:* Oncotype DX costs ~$4,000+ and takes 1–2 weeks. An H&E slide is already on every desk. This is **"free, instant genomic triage."**
- Emory tie-in: **Li et al., Frontiers in Medicine 2022** (Emory + Ohio State) — DL features enhance the Magee-equation correlation with Recurrence Score. **[local/your-group flavor]**

---

## The foundation-model wave (2024) — explained simply

**Plain-language frame:** instead of training a new model per task, labs trained **one giant self-supervised model on millions of unlabeled slides** — a *"GPT for tissue"* — that learns the general visual language of pathology and then adapts to any task with minimal fine-tuning. Same shift ImageNet→foundation models brought to vision, but on gigapixel slides. All four landed within months in Nature / Nature Medicine:

| Model | Group | Scale | Note |
|---|---|---|---|
| **UNI** (Chen et al., Nat Med 2024) | Mahmood Lab, Harvard/BWH | >100k WSIs / >100M images, ~20 tissues, 34 tasks | open-source tissue encoder |
| **CONCH** (Lu et al., Nat Med 2024) | Mahmood Lab | 1.17M image–text pairs | vision-language; query tissue with text |
| **Virchow** (Vorontsov et al., Nat Med 2024) | Paige + Microsoft | ~1.5M WSIs (MSK) | 0.95 specimen-level AUC across 16 cancers |
| **Prov-GigaPath** (Xu et al., Nature 2024) | Microsoft + Providence + UW | 1.3B tiles / 171k WSIs / 28 centers | first true whole-slide FM; SOTA 25/26 tasks |

**Why this matters for breast specifically:** these models slash the data needed to build a breast task (subtyping, HER2, RS prediction, TIL scoring) and **generalize across institutions and scanners** — the exact reproducibility problem that has blocked clinical pathology AI. They are the substrate the breast-specific tools are increasingly built on.

---

## Prognostic AI from tissue

- **TIL quantification — Lu et al., JCO Clinical Cancer Informatics 2020.** DL TIL maps on 1,000 TCGA breast WSIs; TIL density/spatial clustering independently prognostic, varying by subtype (ER+, ER−, TNBC). Operationalizes the International TILs Working Group score — making it reproducible. *(DOI 10.1200/CCI.19.00126)*
- **Recurrence/prognosis:** the Orpheus result (beating the RS for metastatic recurrence in RS≤25) is the best single prognostic headline.

---

## Clinical-deployment reality (the "is this real?" slide)

- **FDA precedent — Paige Prostate (Sept 2021):** first FDA-authorized AI in digital pathology (De Novo). Cancer detection +7.3%, false negatives −~70%, false positives −~24%; lifted non-specialists to specialist accuracy. *Prostate, not breast* — use as the **regulatory template** breast tools will follow.
- **The honest gap:** glass slides still dominate; full digital workflows are a minority of US labs — a scanning/infrastructure/reimbursement bottleneck, not an algorithm one. Most breast pathology AI remains decision-support, not autonomous; molecular-from-H&E is not yet replacing the assays.

---

## The bridge to Section 5

> Radiology AI localizes and characterizes the lesion. Pathology AI confirms, grades, and **predicts molecular subtype, recurrence risk, and therapy eligibility from the same slide.** The convergence point is a **shared AI substrate spanning screen-to-treatment** — which is exactly what "clinical intelligence" means.

---

## Slide-ready key numbers
- CAMELYON16: best AI **AUC 0.994** vs pathologist panel **0.810** (JAMA 2017).
- HER2 0-vs-1+ agreement **70% → 87%** with AI (JCO PO 2024).
- Oncotype RS from H&E: high-risk **AUC 0.89** vs nomogram **0.73**; beats RS for recurrence in low-RS (0.75 vs 0.49) (Nat Commun 2025).
- Foundation models (2024): UNI 100M+ images; CONCH 1.17M pairs; Virchow 1.5M slides / 0.95 AUC; Prov-GigaPath 1.3B tiles, SOTA 25/26.
- Paige Prostate: first FDA-authorized pathology AI (2021).

## [VERIFY] before podium
A single canonical Ki-67/mitosis/grading paper; a definitive breast neoadjuvant-pCR-from-H&E reference; exact PathChat citation; a hard % for US digital-pathology adoption.
