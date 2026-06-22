# Section 1 — The Cautionary Tale of CAD
### ~5 minutes · "Point" · sets up the entire talk's standard of evidence

**Purpose:** Open with humility, not hype. The story of traditional CAD is the reason this audience should believe the modern evidence — because we know what failure looked like, and what it cost.

---

## The narrative beat

> *"Before we talk about where AI is going, we have to talk about the last time we were sure it had arrived."*

Traditional computer-aided detection is the field's cautionary tale: a technology adopted at massive scale on the strength of a billing code, not a clinical trial — and which, when the definitive study finally ran, turned out not to work, and may have made good readers worse.

---

## The rise (1998–2016): technology outrunning evidence

- **1998 (June):** FDA approves the first CAD system — R2 Technology's **ImageChecker M1000** — for film-screen screening. *(CancerNetwork; SPIE)*
- **2002:** CMS provides separate/increased reimbursement for CAD. This — not outcome data — is what drove adoption. *(Lehman et al., 2015)*
- **By 2008: ~74%** of US screening mammograms were read with CAD. **By 2016: ~92%.** *(review, PMC6927034, 2020)*

> **Hook — "Paid first, proven never":** The market adopted CAD on a reimbursement code in 2002, scaled it to nearly every mammogram in America, and the definitive accuracy study didn't land until 2015 — concluding it never helped.

---

## The evidence that it didn't work (a clean three-study arc)

**Fenton et al., NEJM 2007** — *Influence of CAD on Performance of Screening Mammography.* 222,135 women, 429,345 mammograms, 43 facilities.
- Specificity **fell 90.2% → 87.2%** (P<0.001); PPV **fell 4.1% → 3.2%** (P=0.01); **biopsy rate +19.7%.**
- Overall accuracy **lower** with CAD: **AUC 0.871 vs 0.919** (P=0.005).
- Sensitivity gain not significant; cancer detection rate essentially unchanged.
- *DOI: 10.1056/NEJMoa066099*

**Fenton et al., JNCI 2011** — community practice, 684,956 women, >1.6M film-screen mammograms.
- Lower specificity (OR 0.87) and PPV (OR 0.89); **no improvement** in cancer detection, stage, size, or node status. More false alarms, no better cancers.
- *DOI: 10.1093/jnci/djr206*

**Lehman et al., JAMA Internal Medicine 2015** — the landmark, now in the **digital** era. 323,973 women, **625,625 digital mammograms**, 271 radiologists, 66 facilities, 3,159 cancers.
- Sensitivity **85.3% with CAD vs 87.3% without.** Specificity **91.6% vs 91.4%.** Cancer detection **4.1/1,000 in both** — identical.
- **The killer finding:** among the 107 radiologists who read both with and without CAD, sensitivity was **significantly *lower* with CAD (OR 0.53; 95% CI 0.29–0.97).** The same eyes missed more cancers with the machine on.
- Quotable conclusion: *"CAD does not improve diagnostic accuracy of mammography... insurers pay more for CAD with no established benefit to women."*
- Cost: CAD **"costs over $400 million a year."**
- *DOI: 10.1001/jamainternmed.2015.5231*

> **Hook — the single most arresting data point:** *"The same radiologist, the same eyes — measurably worse with the machine on."* (Lehman 2015, OR 0.53)

---

## Why it failed — technically (the conceptual pivot of the talk)

Traditional CAD was a **rule-based, hand-engineered expert system** — humans wrote explicit features and pixel-pattern rules, and a classifier flagged regions.
- **Uneven by lesion type:** good on microcalcifications (~99%), mediocre on masses (75–89%), poor on architectural distortion (~38%) — weakest where humans also struggle. *(PMC6927034)*
- **Alert fatigue:** false-positive marks appeared on **~70% of normal cases.** As marks pile up, attention to each one drops. *(PMC3180536)*
- **The human-factors trap:** as a "second reader" anchored on imperfect human-defined features, it could *lower* a good reader's sensitivity by inducing complacency about unmarked regions.

---

## The turn (the bridge to Section 2)

- **2012 — AlexNet** wins ImageNet; the deep-learning era begins.
- **~2016–2017** — CNNs reach mammography research in earnest.
- **The conceptual difference (the heart of the talk):** Old CAD was *programmed* — humans specified what cancer "looks like." Modern deep learning is *trained end-to-end* — the network learns the discriminative features directly from labeled images. Old CAD pointed at boxes for a human to adjudicate; new AI can *read* the image as an independent reader or triage tool.

> **Transition line:** *"We spent two decades and billions teaching computers to point at mammograms. The breakthrough was teaching them to read them — and this time, validating it before we billed for it."*

---

## Key numbers (slide-ready)

| Fact | Number | Source |
|---|---|---|
| First CAD FDA approval (R2 ImageChecker) | June 1998 | CancerNetwork/SPIE |
| CMS reimbursement for CAD | 2002 | Lehman 2015 |
| CAD use of US screening mammograms | ~74% (2008) → ~92% (2016) | PMC6927034 |
| Annual US CAD cost | >$400M/yr | Lehman 2015 |
| NEJM 2007 accuracy | AUC 0.871 (CAD) vs 0.919 | Fenton 2007 |
| NEJM 2007 biopsy rate | +19.7% | Fenton 2007 |
| JAMA IM 2015 within-reader sensitivity | OR 0.53 (worse with CAD) | Lehman 2015 |
| CAD false-positive marks on normals | ~70% | PMC3180536 |

## [VERIFY] before podium
- Fenton 2014 (JAMA IM) exact dollar figure — cite the **$400M/yr to Lehman 2015**; use Fenton 2014 only as the "downstream costs" reference.
- 74%/92% adoption come from a 2020 review citing Medicare data — fine as field stats, optional one-line primary-source check.
- AlexNet "2012" / DL-reaches-mammography "2016–17" are framing dates, anchored by McKinney 2020 as the named landmark.
