# Section 6 — The Hard Part: Equity, Governance, and Trust
### ~6 minutes · spine stage: **POPULATION** · the segment that makes the vision credible

**Purpose:** Bring the room back to earth — deliberately. The technology is now the easy part. Whether "clinical intelligence" narrows or widens the world's breast-cancer divide is decided by the *non-algorithmic* parts: equity, governance, implementation science. This is also the segment where you speak in the first person (Gichoya, EMBED) and where the European framing lands hardest.

---

## The frame

> *"CAD failed not because the algorithms were bad, but because we deployed it without the evidence, the monitoring, or the implementation discipline medicine demands. Clinical intelligence becomes real only if we get the parts around the model right. The technology is now the easy part."*

---

## 1. The global stakes (lead here — it lands with an international audience)

- **2.3 million** new female breast-cancer cases and **~670,000 deaths** worldwide in 2022 — the most common cancer in women (23.8% of female cancers). Projected to **exceed 6 million cases/year by 2050.** *(GLOBOCAN 2022; Bray et al., CA Cancer J Clin 2024)*
- **The disparity is the headline:** Africa's mortality-to-incidence ratio is **0.51** — roughly half of women diagnosed die — versus far lower ratios in high-income Europe/North America. *Same disease, radically different survival — determined by access, not biology.*
- **WHO Global Breast Cancer Initiative (2021):** target **2.5% annual mortality reduction**, averting **2.5M deaths by 2040** (≥60% diagnosed at stage I/II; diagnosis within 60 days; ≥80% completing treatment).
- **The workforce gap — where AI's promise is real:** over two-thirds of the world's population lacks access to a radiologist; **14 African countries have zero radiologists**; many LMICs have <5 mammography units and <10 radiologists per million. *The genuine "force-multiplier" case is where there is no reader at all.*

> **Greece/EU framing:** Europe carries the second-highest case burden (557,000 cases, 2022) and is the regulatory pacesetter. *Europe must lead not just on adoption, but on the governance model the rest of the world will inherit.*

---

## 2. Algorithmic equity — the uncomfortable evidence

- **The defining study — Gichoya et al., Lancet Digital Health 2022**, *AI recognition of patient race in medical imaging.* DL predicted self-reported race from images — **AUC 0.81 on mammography** (0.91–0.99 chest X-ray) — **even from corrupted, cropped, noised images**, and *not* via known proxies (density AUC only 0.61). Clinicians can't see this signal; the model can. **Implication:** any model can silently encode a path to discriminate, and we can't easily detect or remove it. *(DOI 10.1016/S2589-7500(22)00063-2)* **[YOUR PAPER — confirm Trivedi co-authorship; first-person]**
- **Performance doesn't transfer cleanly:** externally validated mammography AI degrades on diverse populations (worse in Hispanic women, women with prior breast cancer). Models trained on homogeneous high-income data are brittle.
- **The representation gap:** development concentrated in high-income countries on non-diverse datasets; frameworks don't compel disclosure of dataset composition. Hence the push for diverse data — e.g., **EMBED (Emory, 2022): 3.5M racially diverse mammograms.** **[your group / data thread]**

---

## 3. Regulation & governance (your home-field advantage with this audience)

- **EU AI Act — in force 1 Aug 2024.** Medical AI is largely **"high-risk"**: risk management, data governance, documentation, transparency, **human oversight**, accuracy, robustness, post-market monitoring. **Timeline:** high-risk core obligations from **Aug 2026**; AI medical devices under MDR get transition to **Aug 2027.** *The most concrete near-term governance reality for the room.*
- **Double-regulation challenge:** EU medical AI must satisfy **both MDR/IVDR and the AI Act** — a real compliance/innovation-cost tension.
- **US contrast:** FDA has authorized **>1,250 AI/ML-enabled devices** (mid-2025), **radiology dominant** (~75% historically). **PCCP guidance finalized Dec 2024** lets a device update within a pre-authorized envelope — but uptake is early (~8% of new AI devices had a PCCP by 2025).
- **The continuous-oversight problem:** models drift — **data drift** (new scanners/protocols), phenotype drift, concept drift. Clearance/CE-marking does not guarantee sustained performance. Need post-market surveillance + distribution-shift detection (FUTURE-AI consensus, 2021–2025).

---

## 4. Implementation science & trust — what CAD taught us, what we keep forgetting

- **Automation bias — Dratsch et al., Radiology 2023**, *Automation Bias in Mammography.* 27 radiologists read with a sometimes-wrong purported AI. When AI was correct, even novices were ~80% accurate; **when AI was wrong, accuracy collapsed to ~20% (inexperienced), ~25% (moderate), and even very experienced readers fell to ~46%.** *Everyone, at every level, was dragged down by a wrong machine.* **The single most important honesty slide.** *(DOI 10.1148/radiol.222176)*
- **Deskilling is now measured — Budzyń et al., Lancet Gastroenterology & Hepatology 2025.** After AI was introduced for colonoscopy, endoscopists' *unaided* adenoma detection fell **28.4% → 22.4% (−20% relative).** First real-world evidence that routine AI exposure erodes the human's own skill. Directly transferable to breast imaging.
- **The optimistic counterweight — prospective evidence is now genuinely good:** **MASAI RCT** (Lång 2023; Hernström 2025) — **+29% detection, 44% workload cut, no rise in false positives** — the contrast to CAD's retrospective-only history. *(Interval-cancer primary endpoint now reported in 2026 — non-inferior; see Section 2.)*
- **Reimbursement / business-model gap:** clearance ≠ payment. CMS still lacks a dedicated pathway for most diagnostic AI (the proposed US Health Tech Investment Act, 2025, is **unenacted**). The NHS launched the world's largest AI mammography trial (EDITH) explicitly to address a ~30% reader shortfall, rollout targeted ~2027. **[VERIFY — press-sourced]**
- **Liability / the oversight paradox:** the AI Act mandates a human in the loop — but Dratsch shows the human can be the *weak* link, and deskilling erodes the very judgment oversight depends on. *Oversight is a design problem, not a checkbox.*

---

## 5. Tensions to name out loud (the honesty that earns the vision)

1. **Force multiplier vs. divide multiplier.** The models that could bring screening to the 14 African countries with no radiologists are trained on data that excludes those populations. Without deliberate action, AI defaults to *widening* the gap.
2. **The oversight paradox.** Regulation demands a human in the loop; the evidence shows humans defer to and are deskilled by the machine.
3. **Clearance ≠ sustained safety.** 1,250 cleared devices, but models drift and most are never monitored. We risk repeating CAD's "deploy and forget."
4. **Evidence asymmetry.** MASAI shows AI screening *can* work prospectively — but in homogeneous Sweden, with no race/ethnicity data collected. One excellent trial is not a global mandate.

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
| FDA PCCP guidance finalized | Dec 2024 | FDA |

## [VERIFY] before podium
NHS/EDITH figures (press-sourced); US Health Tech Investment Act is *proposed, not enacted*; confirm Trivedi co-authorship on Gichoya 2022 before claiming it; LMIC radiologist/units-per-million figures (review-sourced).
