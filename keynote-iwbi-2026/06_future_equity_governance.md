# Section 6 — The Hard Part: Equity, Governance, and Trust
### ~6 minutes · spine stage: **POPULATION** · the segment that makes the vision credible

**Purpose:** Bring the room back to earth — deliberately — right after the §5 peak. The technology is now the easy part. This section is **the CAD lesson retold at population scale:** deploy without evidence, monitoring, and discipline, and you don't just waste $400M — you widen a global mortality gap. Single narrative arc: **the gap → three ways AI could betray the promise → the CAD lesson at scale → the choice.** Your first-person credibility moment (Gichoya) lives here; the *principles* live in §7.

---

## The frame (the CAD loop closes)

> *"I started this talk with a confession: this field deployed a technology before it proved it worked, and it cost us twenty years and four hundred million dollars a year. I want to end the future-half with the same warning — because the next time we make that mistake, the stakes won't be a wasted line item. They'll be measured in lives, and in who gets left behind."*

---

## Movement 1 — The gap (lead here; it's the moral center)

- **2.3 million** new female breast-cancer cases and **~670,000 deaths** worldwide in 2022 — the most common cancer in women; projected to **exceed 6 million cases/year by 2050.** *(GLOBOCAN 2022; Bray et al., CA Cancer J Clin 2024)*
- **The headline disparity:** Africa's mortality-to-incidence ratio is **0.51** — roughly **half** the women diagnosed die — versus far lower ratios in high-income Europe/North America. *Same disease, radically different survival — determined by access, not biology.*
- **WHO Global Breast Cancer Initiative (2021):** target **2.5% annual mortality reduction**, averting **2.5M deaths by 2040.**
- **Where AI's promise is genuinely real:** over two-thirds of the world's population lacks access to a radiologist; **14 African countries have zero**; many LMICs have <5 mammography units and <10 radiologists per million. *The honest force-multiplier case is where there is no reader at all.*

> **The double-edged framing (the spine of the section):** *"The very same models that could bring screening to the fourteen countries with no radiologist were trained on data that excludes those countries' women. AI will either be the great equalizer of breast cancer outcomes — or its great amplifier. Nothing about the technology decides which. We do."*

---

## Movement 2 — Three ways we betray the promise

**(a) Bias we cannot see — the first-person beat.**
> *"Here's a finding from my own group that still unsettles me."* **Gichoya et al., Lancet Digital Health 2022** — deep learning predicts a patient's self-reported race from medical images — **AUC 0.81 on mammography**, 0.91–0.99 on chest X-ray — **even from corrupted, cropped, and noised images**, and *not* through any known proxy (density AUC only 0.61). *"No human radiologist can see race on a mammogram. The model can — and we still don't fully know how. If a model can learn that, it can silently learn to act on it. That is the hidden-bias problem in one experiment."* *(DOI 10.1016/S2589-7500(22)00063-2)* **[confirm Trivedi co-authorship]**
- And performance doesn't transfer cleanly — externally validated mammography AI degrades on under-represented groups (worse in Hispanic women, women with prior breast cancer). Models trained on homogeneous high-income data are **brittle**.
- The fix begins with data: **EMBED (Emory, 2022) — 3.5M racially diverse mammograms** **[your group]** — but frameworks still don't compel developers to disclose dataset composition.

**(b) Drift we don't monitor.** Models are not static. **Data drift** (new scanners/protocols), phenotype drift, concept drift — real-world performance decays. **FDA has authorized >1,250 AI/ML devices** (mid-2025, radiology ~75%), and finalized **Predetermined Change Control Plan (PCCP)** guidance (Dec 2024) — yet only ~8% of new AI devices have one. **A clearance or a CE mark is a snapshot, not a guarantee.** This is CAD's "deploy and forget," dressed up.

**(c) Humans we quietly de-skill.** The human-in-the-loop is supposed to be the safeguard — but:
- **Dratsch et al., Radiology 2023** (automation bias): when a (sometimes-wrong) AI was present, and it was wrong, reader accuracy collapsed — even **very experienced** readers fell to **~46%** (from ~80%+). *Everyone, at every level, was dragged down by a confident machine.*
- **Budzyń et al., Lancet Gastroenterology & Hepatology 2025** (deskilling, now measured): after AI was introduced for colonoscopy, endoscopists' *unaided* detection fell **28.4% → 22.4%.** Routine AI exposure erodes the human's own skill.
- **The oversight paradox:** the EU AI Act *mandates* a human in the loop — but the evidence shows the human can be the weak link, and de-skilling corrodes the very judgment oversight depends on. **Oversight is a design problem, not a checkbox.**

---

## Movement 3 — Governance: Europe writes the rules the world inherits

- **EU AI Act — in force 1 Aug 2024.** Medical AI is largely **"high-risk"**: risk management, data governance, transparency, human oversight, robustness, post-market monitoring. **High-risk obligations bite from Aug 2026**; AI medical devices under MDR transition to **Aug 2027.** The most concrete governance reality in this room.
- **Double regulation:** EU medical AI must satisfy **both MDR/IVDR and the AI Act** — real compliance cost, real innovation tension.
- **Reimbursement ≠ clearance.** A cleared model that no one pays for doesn't reach a single woman. (US still lacks a dedicated CMS pathway for most diagnostic AI; the NHS is betting the other way with EDITH against a ~30% reader shortfall. **[VERIFY — press]**)

> **The ownership line (bridge toward the close):** *"This audience — European regulators, clinicians, and scientists — will write the governance model the rest of the world inherits. That is not a burden. It is the most important contribution this field can make. Get it right here, and you set the standard everywhere."*

---

## The one summary slide (keep just this — the keeper from the old "tensions" list)

> **FORCE MULTIPLIER ⟷ DIVIDE MULTIPLIER.** Same models. Same evidence base. The difference is entirely in the choices we make about *data, validation, monitoring, and access.* The technology is neutral; the outcome is not.

---

## Slide-ready key numbers
| Claim | Number | Source |
|---|---|---|
| Global cases / deaths (2022) | 2.3M / ~670,000 | GLOBOCAN 2022; Bray 2024 |
| Africa mortality:incidence ratio | 0.51 | GLOBOCAN 2022 |
| WHO target | 2.5%/yr ↓; 2.5M deaths averted by 2040 | WHO 2021 |
| African countries with no radiologist | 14 | reviews 2024–25 |
| AI predicts race from mammography | AUC 0.81 | Gichoya, Lancet Digit Health 2022 |
| Automation bias: expert accuracy when AI wrong | ~46% (from ~82%) | Dratsch, Radiology 2023 |
| Deskilling: unaided detection drop | 28.4%→22.4% | Budzyń, Lancet GH 2025 |
| FDA AI/ML devices authorized | >1,250 (mid-2025) | FDA list |
| EU AI Act high-risk obligations | Aug 2026 (devices Aug 2027) | EU AI Act 2024 |

## [VERIFY] before podium
NHS/EDITH figures (press-sourced); confirm Trivedi co-authorship on Gichoya 2022 before claiming it; LMIC radiologist/units-per-million figures (review-sourced); GLOBOCAN 2022 exact figures via IARC.

## Structure note
Restructured into 3 movements + 1 summary slide, with the **CAD-at-population-scale** through-line opening and closing the section. Principles/"what to do" deliberately held for §7 so the talk ends on the call to action.
