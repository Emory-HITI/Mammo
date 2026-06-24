# Section 1 — The Cautionary Tale of CAD
### ~5 minutes · "Point" · sets up the entire talk's standard of evidence

**Purpose:** Open with humility, not hype. The story of traditional CAD is the reason this audience should believe the modern evidence — because we know what failure looked like, and what it cost. Theme: **history repeats itself — unless we make it not.**

---

## THE HONEST START (picks up directly from the hook's last line)

*The hook (`00_HOOK.md`) ends: "this field has fooled itself before, and I'll start there." Land straight into the numbers — no second cold open.*

> *(Black slide, one number.)*
>
> **"$400 million a year."**
>
> *"That's what the United States spent, every year, on a breast-imaging AI that didn't work. Not a fringe technology — it ran on nearly every screening mammogram in America. And when the definitive study finally looked, radiologists reading with the tool weren't better. Some were measurably **worse**."*
>
> *(Second slide: **OR 0.53**.)*
>
> *"The same radiologist. The same eyes. Worse, with the machine turned on."*
>
> *"That technology was computer-aided detection — CAD. The first machines we built to find the **lesion**. I tell you this story not to make you cynical about AI, but because **history repeats itself unless we understand why it failed the first time.** Everything I show you for the next 40 minutes has to clear the bar that CAD did not."*

**Why this works:** the hook soared (the image already knew); now you earn trust by being the person who names the field's biggest failure first. Two cold numbers keep the energy up, and "history repeats itself" frames the rest of the talk as a test modern AI must pass — not a hype reel. The word **lesion** ties CAD to the first stage of the spine.

---

## The rise (1998–2016): technology outrunning evidence

- **1998 (June):** FDA approves the first CAD system — R2 Technology's **ImageChecker M1000** — for film-screen screening. *(CancerNetwork; SPIE)*
- **2002:** CMS reimburses CAD separately. This — not outcome data — is what drove adoption. *(Lehman et al., 2015)*
- **By 2008: ~74%** of US screening mammograms were read with CAD. **By 2016: ~92%.** *(review, PMC6927034, 2020)*

> **The "Paid first, proven never" gap:** adopted on a billing code in 2002, scaled to nearly every mammogram in America — and the definitive accuracy study didn't land until 2015, concluding it never helped.

> **[VISUAL — to build later]** A timeline bar: 1998 approval → 2002 reimbursement → adoption curve climbing to ~92% → 2015 the verdict lands (in red). One slide that shows adoption racing *ahead* of evidence. *(I'll create this when we move to visuals — flagged, not built yet.)*

---

## The evidence that it didn't work

**Lehman et al., JAMA Internal Medicine 2015** — the landmark, in the **digital** era. 323,973 women, **625,625 digital mammograms**, 271 radiologists, 66 facilities.
- Sensitivity **85.3% with CAD vs 87.3% without.** Specificity **91.6% vs 91.4%.** Cancer detection **4.1/1,000 in both** — identical.
- **The killer finding:** among the 107 radiologists who read both with and without CAD, sensitivity was **significantly *lower* with CAD (OR 0.53; 95% CI 0.29–0.97).** The same eyes missed more cancers with the machine on.
- *DOI: 10.1001/jamainternmed.2015.5231*

> **THE QUOTE (on-screen, verbatim):**
> *"Computer-aided detection does not improve diagnostic accuracy of mammography. These results suggest that insurers pay more for CAD with no established benefit to women."*
> — Lehman CD et al., *JAMA Internal Medicine* 2015

*This was foreshadowed years earlier:* **Fenton et al. (NEJM 2007; JNCI 2011)** had already shown CAD lowered specificity and raised biopsies (NEJM 2007: AUC **0.871 with CAD vs 0.919 without**; biopsy rate **+19.7%**) — with no gain in cancers found. The warning signs were there for a decade; we kept paying anyway.

---

## Why it failed — technically (the conceptual pivot of the talk)

Traditional CAD was a **rule-based, hand-engineered expert system** — humans wrote explicit features and pixel-pattern rules, and a classifier flagged regions.
- **It was built to find what we already find.** CAD was trained to flag the cancers radiologists already catch well — adding marks and noise, not a new class of finding. *(Kohli & Jha, JACR 2018; DOI 10.1016/j.jacr.2017.12.029)* *(Frame the "~84%" as context from the R2 device's original detection claim, not a quotable statistic from this commentary.)*
- **Alert fatigue:** false-positive marks appeared on **~70% of normal cases.** As marks pile up, attention to each one drops. *(PMC3180536)*
- **The human-factors trap:** as a "second reader" anchored on imperfect human-defined features, it could *lower* a good reader's sensitivity by inducing complacency about unmarked regions.

> **The field wrote its own post-mortem:** Kohli & Jha titled their 2018 JACR analysis *"Why CAD Failed in Mammography"* — and noted that AI skeptics now cite exactly this story. The honest answer to them is in the *next* section.

---

## The turn (the bridge to Section 2)

- **2012 — AlexNet** wins ImageNet; the deep-learning era begins.
- **~2016–2017** — CNNs reach mammography research in earnest.
- **The conceptual difference (the heart of the talk):** Old CAD was *programmed* — humans specified what cancer "looks like." Modern deep learning is *trained end-to-end* — the network learns the discriminative features directly from labeled images. Old CAD **pointed** at boxes for a human to adjudicate; new AI can **read** the image as an independent reader or triage tool.

> **Transition line:** *"We spent two decades and billions teaching computers to **point** at mammograms. The breakthrough was teaching them to **read** them — and this time, validating it before we billed for it."*

---

## Key numbers (slide-ready)

| Fact | Number | Source |
|---|---|---|
| First CAD FDA approval (R2 ImageChecker) | June 1998 | CancerNetwork/SPIE |
| CMS reimbursement for CAD | 2002 | Lehman 2015 |
| CAD use of US screening mammograms | ~74% (2008) → ~92% (2016) | PMC6927034 |
| Annual US CAD cost | >$400M/yr | Lehman 2015 |
| JAMA IM 2015 within-reader sensitivity | **OR 0.53** (worse with CAD) | Lehman 2015 |
| NEJM 2007 accuracy | AUC 0.871 (CAD) vs 0.919 | Fenton 2007 |
| CAD aimed at cancers radiologists already find well | (context, not a hard stat) | Kohli & Jha 2018 |
| CAD false-positive marks on normal cases | ~70% | PMC3180536 |

## ✓ VERIFIED (agent pass) — all confirmed exact
- Lehman 2015 (sens 85.3 vs 87.3%, spec 91.6 vs 91.4%, within-reader OR 0.53, CDR 4.1/1000, >$400M/yr; PMID 26414882) + verbatim conclusion quote — confirmed.
- Fenton NEJM 2007 (spec 90.2→87.2%, biopsy +19.7%, AUC 0.871 vs 0.919, PPV 4.1→3.2%; PMID 17409321) — confirmed.
- Kohli & Jha JACR 2018;15(3 Pt B):535–537 (PMID 29398499; **DOI added: 10.1016/j.jacr.2017.12.029**) — confirmed.
- R2 ImageChecker FDA June 1998 (P970058); CMS 2002; CAD ~74% (2008) → ~92% (2016) per Keen, JACR 2018 — confirmed.

## [VERIFY — still open]
- The "~84%" is **context** (echoes the R2 device's original ~80→88/100 claim), not a quotable stat — keep as framing, no hard number on a slide.
- AlexNet "2012" / DL-reaches-mammography "2016–17" are framing dates, anchored by McKinney 2020.
