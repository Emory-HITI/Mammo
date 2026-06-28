# The Adoption Problem
### Why "One Model to Rule Them All" Is the Wrong Goal

*Placement: after §6b (Frontier models), immediately before §7 (Close) — IWBI 2026, Thessaloniki.*
*Rationale: the practical capstone. By this point the audience has seen purpose-built and frontier models; the natural next question is why a single model doesn't simply get adopted. The "portfolio, not a monolith" reframe pays off the §2 "data isn't enough to drive adoption" thread and lands the audience in the real world right before the close.*

---

## Slide 1 — One Model for Everything Is the Wrong Goal

We chase better detection for three decades. But look at what we ask a *single* model to do at once:

- Hold up across **every scanner and vendor** in the field
- Perform in women of **all ages, all densities, all risk types**
- Serve both the **expert breast radiologist** and the **generalist who reads mammo one day a week**

> We should seriously reframe how we consider **mammography-model deployment**.

**Speaker note:** Pivot from "the technology works" to "working in a paper is not the bar." Asking one model to satisfy all of these at once is the wrong goal; the reframe sets up the next slide.

---

## Slide 2 — How We Should Deploy Instead

Not one model imposed everywhere — three reframes:

**1. Operating points, not one threshold.** Sensitivity vs. false positives is a *value judgment*, not a setting. Let a model run at a range of operating points, **tailored to the site and the individual radiologist** — the expert and the once-a-week reader don't want the same one.

**2. Multimodal — and soon.** A model that operates only on imaging data, or **imaging plus basic clinical data**, will end in user frustration. The model a radiologist will actually trust pulls in **image, clinical context, risk, and history** together. This needs to happen as soon as possible.

**3. The integration tax** *(unchanged).* Even a perfect model dies if it can't get into PACS. Custom one-off integrations *don't scale* (RSNA); overloaded IT means onboarding only happens through a platform or vendor you already run.

> **The reframe:** The winning system isn't the most accurate in a paper; it's the one tuned to the reader, that knows what they know, and that fires inside the workflow.

---

### References

1. **Sharma N, Ng AY, James JJ, et al.** Multi-vendor evaluation of artificial intelligence as an independent reader for double reading in breast cancer screening on 275,900 mammograms. *BMC Cancer.* 2023;23(1):460. doi:10.1186/s12885-023-10890-7
   *(275,900 cases; 4 vendors, 7 sites, 2 countries — multi-vendor generalization evidence "still lacking")*

2. **Eisemann N, Bunk S, Mukama T, et al.** Nationwide real-world implementation of AI for cancer detection in population-based mammography screening. *Nat Med.* 2025;31(3):917–924. doi:10.1038/s41591-024-03408-6
   *(PRAIM: 463,094 women, 119 radiologists, 5 vendors, 12 sites — real-world deployment scale)*

3. **Tejani AS, Cook TS, Hussain M, Sippel Schmidt T, O'Donnell KP.** Integrating and Adopting AI in the Radiology Workflow: A Primer for Standards and Integrating the Healthcare Enterprise (IHE) Profiles. *Radiology.* 2024;311(3):e232653. doi:10.1148/radiol.232653
   *(Custom one-off integrations don't scale; multi-vendor interoperability burden)*

4. **Allen B, Dreyer K, Stibolt R Jr, et al.** Implementing Artificial Intelligence Algorithms in the Radiology Workflow: Challenges and Considerations. *Mayo Clin Proc Digit Health.* 2024;2(4). doi:10.1016/j.mcpdig.2024.10.001
   *(PACS/EHR integration, IT burden, proprietary-format friction)*

> Note: References 3–4 carry the integration/IT-burden argument. Two further claims in Slide 2 — the operating-point/specificity trade-off and the interpretability-trust link (incl. the nipple-shadow anecdote) — are drawn from review-level literature; if you want primary citations for those specific points (e.g. an AJR AI special-series piece on interpretability, or a systematic review quantifying false-positive burden), flag it and I'll pin exact references rather than leave them review-sourced.
