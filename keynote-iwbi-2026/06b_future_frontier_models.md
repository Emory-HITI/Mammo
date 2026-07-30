# Section 6b — Frontier Models: Will the Generalists Absorb the Field?

> Regenerated from the FINAL slides in `slides/section6b.html` (7 slides). One section per slide, in slide order. Spoken script is the slide `data-note`; bold figures and citations are taken from the on-slide text and `.cite` lines.

---

## Slide 1 — Will the generalists absorb the field?

*Specialist vs Generalist · an open question*

We have seen what purpose-built models can do — across risk, detection, tissue, and the heart. There is one question left before we close. I have shown you a dozen specialist models — Mirai for risk, Transpara for detection, Virchow for tissue, the breast arterial calcium model. **Each was built for one job.** The general-purpose frontier models from the large labs — GPT, Gemini, Claude — were not built for radiology, yet they improve faster than any medical model, and each generation handles tasks we assumed required a custom build. So the question to end the body of the talk on: **in five years, do we still build our own models, or do we ask the general one? And if it is the latter, who owns it?**

---

## Slide 2 — Where the generalists already perform well

*Frontier models*

Where do the generalists already perform well? On medical knowledge and text reasoning: frontier models score **above ~90% on USMLE-style exams — Med-Gemini at 91.1%** — across the breadth of medicine. On report drafting in two dimensions: domain-adapted frontier models do well. **Med-Gemini-2D chest-X-ray reports were rated "equivalent or better" than the original radiologist's report in 43–96% of cases**, depending on whether the film was abnormal or normal, and **exceeded prior state of the art on 17 of 20 chest-X-ray tasks**, in Google's 2024 work. And they adapt rapidly: fine-tuned on a small labeled set, foundation models often exceed bespoke task-specific models, and **generalize better across sites, scanners, and populations** — which is exactly the brittleness problem that limits the narrow models. I'll frame the current generation by trajectory, not by specific unpublished benchmark numbers.

**Key figures:** ~91% USMLE (Med-Gemini); 2D CXR reports "equivalent or better" in 43–96% of cases; exceeded prior SOTA on 17/20 CXR tasks.

*Med-Gemini — Yang et al., arXiv:2405.03162, 2024 (preprint).*

---

## Slide 3 — Where they still fall short — the image itself

*Frontier models*

Where do they still fall short, for now? **Their strength is text, not image content** — the general vision-language models' edge comes from text knowledge rather than from actually reading the image. On radiology interpretation, **diagnostic accuracy across general vision-language models ranges only about 8–29%**, in a systematic evaluation published in *Life* this year, and **roughly 22% of the AI-generated reports contain factual errors or hallucinations.** Specialists still perform better on the image itself: the purpose-built **RadFM model outperforms GPT-4V** on radiology tasks. And three dimensions and localization lag: **Med-Gemini-3D CT reports were only 53% clinically acceptable** — adequate for a 2D chest film, much less so for a CT volume.

**Key figures:** general VLM image diagnostic accuracy ~8–29%; ~22% of generated reports contain factual errors; RadFM > GPT-4V; Med-Gemini-3D CT 53% clinically acceptable.

*Visual LLMs in Radiology, Life (MDPI) 2026;16(1):66 (PMID 41598221) · RadFM — Wu et al., Nat Commun 2025 · Med-Gemini, 2024.*

---

## Slide 4 — Specializing a model does not make it safer

*Frontier models*

And a counterintuitive result that complicates the procurement instinct. First, note that medical hallucination is not a single failure mode — this taxonomy, from Kim, Jeong and colleagues at MIT and Harvard this year, sorts it into **five clusters: plain factual errors, outdated references, spurious correlations, fabricated sources or guidelines, and incomplete chains of reasoning** — playing out across diagnostic, procedural, and research tasks. Now the counterintuitive part: fine-tuning on medical data does not automatically reduce any of this. On a hallucination benchmark, **Med-HALT (2025), general-purpose models were hallucination-free 76.6% of the time, versus only 51.3% for the medically "specialized" models.** Narrowing a model can erode its calibration. So the specialist-versus-generalist question is genuinely unsettled, and the procurement instinct — to **buy the medical one — may be wrong.**

> Figure: A taxonomy of medical hallucination — five clusters across diagnostic, procedural, and research tasks. Kim, Jeong et al., arXiv 2503.05777 (2025).

**Key figures:** Med-HALT — general-purpose 76.6% vs specialized 51.3% hallucination-free.

*Taxonomy: Kim, Jeong et al., "Medical Hallucination in Foundation Models," arXiv 2503.05777 (2025) · Med-HALT, arXiv 2307.15343.*

---

## Slide 5 — The thesis: the "bitter lesson"

*Frontier models*

Here is the thesis to pose, Rich Sutton's bitter lesson, from his 2019 essay — and I attribute it as an argument, not a study. Sutton argues that over decades, **general methods that scale with data and compute have repeatedly outperformed systems built on handcrafted domain knowledge — in chess, in computer vision, in language.** Radiology AI has spent fifteen years handcrafting. The trajectory of those other fields suggests the generalists eventually take over perception as well. **The open question is when, and what we should build in the meantime.**

*Rich Sutton, "The Bitter Lesson," 2019 essay (an argument, not a study).*

---

## Slide 6 — A likely answer: orchestration

*Frontier models*

A likely answer is orchestration. The probable future is not a generalist simply replacing the specialist. It is a **stack**: a **frontier model as the reasoning and orchestration layer** — it reads the EHR, talks to the patient, writes the report, decides the next step — and the **specialist models as the instruments it calls**: the detector, the risk model, the breast-arterial-calcium quantifier, the tissue model. Because measurement still rewards purpose-built precision and calibration. Call it **agentic radiology: the generalist conducts, and the specialists play.** And increasingly the generalists are themselves **domain-adapted, like Med-Gemini**, rather than raw.

---

## Slide 7 — What it means for this room

*Frontier models*

So what does this mean for this room? Three things. First, **the locus of value moves.** If the model becomes a commodity, durable value shifts to what the frontier labs do not have: proprietary, diverse, well-labeled data; rigorous validation; clinical integration; and the care pathway. **Compete on data, validation, and integration — not architecture.** Second, **concentration is a governance problem** — cost, sovereignty, access. Third, **the radiologist's role shifts, from primary detector toward validator and orchestrator**, and the source of clinical context and accountability the model cannot supply. And the bridge to the close: **whether the result is a specialist model or a frontier generalist, built in Palo Alto or in this room, it does not change what makes it safe. The bar is the same one CAD failed and MASAI cleared.**

*Connects to §6 equity · bridges to §7 — the close.*
