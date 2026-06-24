# Section 6b — Frontier Models: Will the Generalists Absorb the Field?
### ~3.5 minutes · placed just before the close · an open strategic question

> **Placement note:** This sits at the tail end, right before the summary. It also works immediately after §5 (which ends on purpose-built medical foundation models); that ordering follows thematically ("…but what about the general frontier models?"). The bridge lines are written for the before-the-close slot.

**Purpose:** Every model in this talk so far was **purpose-built**: Mirai for risk, Transpara for detection, Virchow for tissue, the BAC model for calcium. The general-purpose frontier models from OpenAI, Google, and Anthropic are not built for medicine, yet they improve every few months. The question to end the body of the talk on: **will the generalists absorb all of this?**

---

## The opening (bridges from §6's "who decides")

> *"I said the outcome is decided by us, not by the technology. One factor will shape who 'us' is. I have shown a dozen specialist models, each trained for one job. The frontier models from the large labs — GPT, Gemini, Claude — were not built for radiology, yet they improve faster than any medical model, and each generation handles tasks we assumed required a custom build. So: in five years, do we still build our own models, or do we ask the general one? And if it is the latter, who owns it?"*

---

## Where the generalists already perform well
- **Medical knowledge and text reasoning:** frontier models score **~90%+ on USMLE-style exams** (Med-Gemini 91.1%), across the breadth of medicine.
- **Report drafting (2D):** domain-adapted frontier models perform well. **Med-Gemini-2D chest-X-ray reports were rated "equivalent or better" than the original radiologist's report in 43–96% of cases**, and exceeded prior SOTA on 17 of 20 CXR tasks (Google, 2024).
- **Rapid adaptation:** fine-tuned on a small labeled set, foundation models often exceed bespoke task-specific models, and **generalize better across sites, scanners, and populations** — the brittleness problem that limits narrow models.

## Where they still fall short (for now)
- **Their strength is text, not image content.** General VLMs' edge comes from text knowledge rather than from reading the image. On radiology interpretation, **diagnostic accuracy across general VLMs ranges ~8–29%** (systematic eval, *Life* 2026), and roughly **22% of AI-generated reports contain factual errors / hallucinations.**
- **Specialists still perform better on the image itself:** purpose-built **RadFM outperforms GPT-4V** on radiology tasks.
- **3D and localization lag:** Med-Gemini-**3D** CT reports were only **53% clinically acceptable** — adequate for a 2D chest film, less so for a CT volume.

## A counterintuitive result
> Fine-tuning on medical data does **not** automatically make a model safer. On a hallucination benchmark (Med-HALT, 2025), **general-purpose models were hallucination-free 76.6% of the time vs only 51.3% for medically "specialized" models.** Narrowing a model can erode its calibration. The specialist-versus-generalist question is unsettled, and the procurement instinct to buy the medical one may be wrong.

---

## The thesis to pose: the "bitter lesson"
> Rich Sutton's *bitter lesson* (2019 essay) argues that over decades, **general methods that scale with data and compute have repeatedly outperformed systems built on handcrafted domain knowledge** — in chess, in vision, in language. Radiology AI has spent fifteen years handcrafting. The trajectory of other fields suggests the generalists eventually take over perception as well. **The open question is when, and what we should build in the meantime.**

## A likely answer: orchestration
The probable future is not a generalist replacing the specialist. It is a **stack**:
- a **frontier model as the reasoning and orchestration layer** — it reads the EHR, talks to the patient, writes the report, decides the next step;
- **specialist models as the instruments** it calls — the detector, the risk model, the BAC quantifier, the tissue model — because measurement still rewards purpose-built precision and calibration.
- *"Agentic radiology": the generalist conducts; the specialists play.* Increasingly the generalists are **domain-adapted** (Med-Gemini) rather than raw.

---

## What this means for this room
1. **The locus of value moves.** If the model becomes a commodity, durable value shifts to what the frontier labs do not have: **proprietary, diverse, well-labeled data; rigorous validation; clinical integration; and the care pathway.** Compete on data and evidence, not architecture.
2. **Concentration is a governance problem.** A handful of US tech companies controlling the frontier raises issues of **cost, dependence, data sovereignty, and access**, particularly for Europe and for LMICs (this connects to §6's equity argument). *Who owns the intelligence that reads the world's mammograms?*
3. **The radiologist's role shifts** — from primary detector toward **validator, orchestrator, and the source of clinical context and accountability** the model cannot supply.

> **Bridge to the close:** *"Whether the result is a specialist model or a frontier generalist, built in Palo Alto or in this room, it does not change what makes it safe. The bar is the same one CAD failed and MASAI cleared. So don't bet on the architecture. Bet on the discipline."*

---

## Slide-ready key numbers
- Frontier models: **~91% USMLE** (Med-Gemini); CXR reports **43–96% "equivalent or better"** than radiologist (Med-Gemini-2D, 2024).
- General VLM radiology diagnostic accuracy **~8–29%**; **~22%** of generated reports have factual errors.
- **RadFM > GPT-4V** on radiology tasks (specialist still wins on images).
- Med-HALT: general **76.6%** vs specialized **51.3%** hallucination-free.
- Med-Gemini-**3D** CT: **53%** clinically acceptable (3D still lags).

## ✓ VERIFIED (agent pass) — all confirmed
- **Med-Gemini:** CXR reports "equivalent or better" 43–65% (abnormal) / 57–96% (normal); SOTA on 17/20 tasks; 3D CT 53% acceptable. *(Yang et al., arXiv:2405.03162, 2024 — preprint)*
- **General VLM radiology accuracy 8.1–29.2%; ~22% of reports contain hallucinations.** *(Visual LLMs in Radiology, Life (MDPI) 2026;16(1):66, PMID 41598221)*
- **RadFM > GPT-4V** *(Wu et al., Nat Commun 2025, DOI 10.1038/s41467-025-62385-7; arXiv 2308.02463)*
- **Med-HALT: general 76.6% vs specialized 51.3% hallucination-free** *(MIT Media Lab medical-hallucination eval, 2025; Med-HALT arXiv 2307.15343)*
- **"Bitter lesson"** = Rich Sutton, 2019 essay (attribute as essay/argument, not a study). ✓

## [VERIFY — still open]
- Frame current frontier models (GPT-5 / Gemini 3 / Claude) by the *trajectory*, not by specific unpublished benchmark numbers — don't quote figures you can't cite.

## ⏱️ TIME / PLACEMENT NOTE
Adds ~3.5 min. **This pushes the talk over 45 — see the global time reconciliation.** If cut for time, this section can compress to a **60-second beat** (opening + bitter lesson + "bet on the discipline"). Otherwise run it as a full ~3.5-min section and trim elsewhere.
