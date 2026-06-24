# Section 1 — The Cautionary Tale of CAD
### ~5 minutes · spine stage: **Lesion** (Era I) · sets the talk's standard of evidence

**Purpose:** Open plainly. CAD is the reason to weigh the modern evidence carefully — we know what failure looked like, and what it cost. The standard set here (prove it before scaling it) is the standard the rest of the talk is held to.

---

## The start (picks up from the hook's last line)

*The hook ends: "this field has overpromised before, and I'll start there." Go straight to the numbers.*

> *(Slide: one number.)*
>
> **"$400 million a year."**
>
> *"That's roughly what the U.S. spent each year on computer-aided detection for mammography — about one dollar of every ten thousand spent on U.S. health care. It ran on most screening mammograms in the country. When the definitive study was finally done, radiologists using CAD were no more accurate, and by one measure less."*
>
> *(Slide: **OR 0.53**.)*
>
> *"Among radiologists who read both with and without it, sensitivity was lower with CAD on. Same readers, same images, fewer cancers caught."*
>
> *"That tool was computer-aided detection — the first systems we built to find the **lesion**. I start here for a simple reason: the field has overpromised before. Everything in the next 40 minutes should clear the bar CAD did not."*

**Note:** name the field's clearest failure first; it earns the credibility to be optimistic later. The word **lesion** ties CAD to the first stage of the spine.

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
- Among the 107 radiologists who read both with and without CAD, sensitivity was **lower with CAD on (OR 0.53; 95% CI 0.29–0.97).**
- *DOI: 10.1001/jamainternmed.2015.5231*

> **Quote (on-screen, verbatim):**
> *"Computer-aided detection does not improve diagnostic accuracy of mammography. These results suggest that insurers pay more for CAD with no established benefit to women."*
> — Lehman CD et al., *JAMA Internal Medicine* 2015

The authors offered a plausible mechanism, worth quoting because it foreshadows automation bias (§6): *"radiologists reading with CAD are overly dependent on the technology and ignore suspicious lesions if they are not marked by CAD"* (Lehman, 2015). The companion editorial was titled *"Is It Time to Stop Paying for Computer-Aided Mammography?"* (JAMA Intern Med 2015).

The signal had been there for years. **Fenton et al. (NEJM 2007; JNCI 2011)** found CAD lowered specificity (90.2%→87.2%) and raised the biopsy rate (+19.7%) with no gain in cancers detected; the small, non-significant sensitivity change in 2007 was driven mostly by extra DCIS. *Honest caveat to state:* the 2007 study has been criticized because CAD was actually used at only 7 of 43 facilities — which is why the 2015 digital-era study (where CAD was near-universal) is the stronger evidence.

---

## Why it failed (the technical reasons)

Traditional CAD was a rule-based system: humans specified the image features and pixel patterns, and a classifier flagged regions. Three reasons it didn't help:
- **It targeted the cancers we already find well.** So it added marks, not new information. *(Kohli & Jha, JACR 2018; DOI 10.1016/j.jacr.2017.12.029. Treat the "~84%" as context from the original device claim, not a quotable statistic.)*
- **Readers learned to ignore it.** False-positive marks appeared on ~70% of normal studies; after enough false alarms, trust in the marks erodes. *(PMC3180536)*
- **It could lower a good reader's sensitivity** — the over-dependence Lehman described.

> Kohli & Jha titled their 2018 JACR review *"Why CAD Failed in Mammography,"* and noted that AI skeptics cite this history. Section 2 is the response.

---

## The turn (the bridge to Section 2)

- **2012 — AlexNet** wins ImageNet; the deep-learning era begins.
- **~2016–2017** — CNNs reach mammography research in earnest.
- **The difference:** old CAD was *programmed* — humans specified what cancer looks like. Modern deep learning is *trained* — the model learns the features from labeled images. Old CAD **pointed** at regions for a human to check; modern AI can **read** the image, as a second reader or on its own.

> **Transition line:** *"For two decades we taught computers to point at mammograms. The change was teaching them to read — and this time, validating it before billing for it."*

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
