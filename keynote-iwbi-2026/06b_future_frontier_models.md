# Section 6b — The Wildcard: Will the Frontier Models Eat Everything?
### ~3.5 minutes · placed just before the close · the field's biggest strategic uncertainty

> **Placement note:** Per your request, this sits at the tail end, right before the summary. It also works *immediately after §5* (which ends on purpose-built medical foundation models) — that ordering flows thematically ("…but what about the *general* frontier models?"). Pick whichever you prefer; the bridge lines are written for the before-the-close slot.

**Purpose:** Every model in this talk so far was **purpose-built** — Mirai for risk, Transpara for detection, Virchow for tissue, the BAC model for calcium. But the most powerful AI on Earth right now isn't built for medicine at all: the general-purpose frontier models from OpenAI, Google, and Anthropic, improving every few months. So the honest question to end the body of the talk on: **will the generalists simply absorb all of this?**

---

## The provocation (open here)

> *"I've shown you a dozen specialist models, each trained for one job. But there's a tidal force offstage. The frontier models from the big labs — GPT, Gemini, Claude — weren't built for radiology, yet they're improving faster than any medical model, and every generation they get better at things we were sure required a custom build. So let me ask the uncomfortable question: in five years, do we still build our own models — or do we just ask the general one?"*

---

## Where the generalists are already winning
- **Medical knowledge / reasoning in text:** frontier models score **~90%+ on USMLE-style exams** (Med-Gemini 91.1%) — they already out-*know* most of us across the breadth of medicine.
- **Report drafting (2D):** domain-adapted frontier models are genuinely good — **Med-Gemini-2D chest-X-ray reports were rated "equivalent or better" than the original radiologist's report in 43–96% of cases**, and beat prior SOTA on 17 of 20 CXR tasks (Google, 2024).
- **Rapid adaptation:** fine-tuned on a *small* labeled set, foundation models often beat bespoke task-specific models — and **generalize better across sites, scanners, and populations** (the exact brittleness problem that sinks narrow models).

## Where they still fall down (the honest brake — for now)
- **Their strength is words, not pixels.** General VLMs' edge comes from text knowledge, **not** from actually reading the image. On real radiology interpretation, **diagnostic accuracy across general VLMs ranges ~8–29%** (systematic eval, *Life* 2026), and roughly **22% of AI-generated reports contain factual errors / hallucinations.**
- **Specialists still win on the image itself:** purpose-built **RadFM outperforms GPT-4V** on radiology tasks.
- **3D and localization lag badly:** Med-Gemini-**3D** CT reports were only **53% clinically acceptable** — fine for a 2D chest film, not for a CT volume.

## The counterintuitive twist (great for a sophisticated room)
> Fine-tuning on medical data does **not** automatically make a model safer. On a hallucination benchmark (Med-HALT, 2025), **general-purpose models were hallucination-free 76.6% of the time vs only 51.3% for medically "specialized" models.** Narrowing a model can *erode* its calibration. "Specialist vs. generalist" is genuinely unsettled — and the procurement instinct ("buy the medical one") may be wrong.

---

## The deep thesis to pose: the "bitter lesson"
> Rich Sutton's *bitter lesson* of AI: over decades, **general methods that scale with data and compute have repeatedly beaten systems built on handcrafted domain knowledge** — in chess, in vision, in language. Radiology AI has spent fifteen years handcrafting. The trajectory of every other field says the generalists eventually win the *perception* too. **The question is not whether, but when — and what we should be building in the meantime.**

## The likely answer: not either/or — orchestration
The probable future isn't "generalist replaces specialist." It's a **stack**:
- a **frontier model as the reasoning and orchestration layer** — it reads the EHR, talks to the patient, writes the report, decides what to do next;
- **specialist models as the instruments** it calls — the detector, the risk model, the BAC quantifier, the tissue model — because measurement still rewards purpose-built precision and calibration.
- *"Agentic radiology": the generalist conducts; the specialists play.* And increasingly the generalists are **domain-adapted** (Med-Gemini) rather than raw.

---

## What this means for THIS room (the strategic payoff)
1. **The moat moves.** If the model becomes a commodity, the durable value shifts to what the frontier labs *don't* have: **proprietary, diverse, well-labeled data; rigorous validation; clinical integration; and the care pathway.** Stop competing on architecture; compete on data and evidence.
2. **Concentration is a governance problem.** A handful of US tech companies controlling the frontier raises issues of **cost, dependence, data sovereignty, and access** — acute for Europe and for LMICs (ties straight back to §6's equity argument). *Who owns the intelligence that reads the world's mammograms?*
3. **The radiologist's role shifts** — from primary detector toward **validator, orchestrator, and the source of clinical context and accountability** the model can't supply.

> **Bridge to the close:** *"Here's the liberating part. Whether the winner is a specialist model or a frontier generalist, built in Palo Alto or in this room — it doesn't change what makes it safe. The bar is the same one CAD failed and MASAI cleared. So don't bet on the architecture. Bet on the discipline."*

---

## Slide-ready key numbers
- Frontier models: **~91% USMLE** (Med-Gemini); CXR reports **43–96% "equivalent or better"** than radiologist (Med-Gemini-2D, 2024).
- General VLM radiology diagnostic accuracy **~8–29%**; **~22%** of generated reports have factual errors.
- **RadFM > GPT-4V** on radiology tasks (specialist still wins on images).
- Med-HALT: general **76.6%** vs specialized **51.3%** hallucination-free.
- Med-Gemini-**3D** CT: **53%** clinically acceptable (3D still lags).

## [VERIFY] before podium
- Med-Gemini figures (arXiv 2405.03162, 2024 — preprint).
- VLM accuracy 8–29% and ~22% report-error figures (systematic reviews, *Life* 2026 / arXiv 2025–26 — confirm exact source + numbers before quoting).
- Med-HALT 76.6% vs 51.3% (2025 benchmark — verify).
- RadFM > GPT-4V (Wu et al., RadFM, Nat Commun 2025).
- "Bitter lesson" = Rich Sutton, 2019 essay (attribute as an essay/argument, not a study).
- Frame current frontier models (GPT-5 / Gemini 3 / Claude) by the *trajectory*, not by specific unpublished benchmark numbers — don't quote figures you can't cite.

## ⏱️ TIME / PLACEMENT NOTE
Adds ~3.5 min. **This pushes the talk over 45 — see the global time reconciliation.** If cut for time, this section can compress to a **60-second "the wildcard" beat** (provocation + bitter lesson + "bet on the discipline") and still earn its place. Strongest as a full ~3.5-min section if we trim elsewhere.
