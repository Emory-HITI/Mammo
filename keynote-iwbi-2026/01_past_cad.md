# Section 1 — Era I: The Cautionary Tale of CAD

### ~5 minutes · spine stage: **Lesion** (Era I) · sets the talk's standard of evidence

**Purpose:** Open plainly. CAD is the reason to weigh the modern evidence carefully — we know what failure looked like, and what it cost. The standard set here (prove it before scaling it) is the standard the rest of the talk is held to.

---

## Slide 1 — The year is 1998.

The year is **1998** — the year the FDA cleared the first CAD system for mammography. Computer-aided detection has just arrived: software that puts a mark on every mammogram to flag a possible cancer. A second set of eyes, automatically. It feels like the future.

*On-slide:* "Computer-aided detection has just arrived — a mark on every mammogram flagging a possible cancer. A second set of eyes, automatically. It feels like the future."

*Source: Intelerad InteleViewer documentation (inteleviewer.documentation.intelerad.com)*

---

## Slide 2 — Within a decade, CAD was used on most US mammograms.

And within a decade, it is everywhere. A reimbursement code in 2002 — not an outcomes trial — drove adoption from nothing to about **74% of US screening mammograms by 2008, and 92% by 2016.** We adopted it on a billing code, not a clinical trial.

- **1998:** FDA approval
- **2002:** CMS reimbursement
- **74%** by 2008 → **92%** by 2016
- **2015 — the verdict: no benefit.** The definitive study landed in 2015, after CAD already ran on ~9 in 10 mammograms.

*A 2002 reimbursement code drove adoption — not evidence of benefit.*

---

## Slide 3 — Conventional CAD put a false-positive mark on most normal exams.

And the reading room is not convinced — because conventional CAD produced false positives constantly. Across independent series, **seventy to eighty-five percent of normal mammograms carried at least one false-positive mark:** Kim seventy, Mahoney up to eighty-five, Watanabe eighty-three — versus fifty-two percent for an AI-based CAD. In per-image terms that is roughly **half a mark to one and a half false marks on every image** — about 2–4 per 4-view case. Mahoney measured about two to three false marks on a normal case — and more on non-dense breasts, where there is more to flag. After enough false alarms, radiologists learned to tune the marks out.

- **70–85%** of normal mammograms had ≥1 false positive
- **0.5–1.5** false positives per image — about 2–4 per 4-view case

**Mean false-positive marks per case · ImageChecker v7.2 (Mahoney & Meganathan, Table 4):**

| | Non-dense | Dense |
|---|---|---|
| All marks | 2.6 | 1.8 |
| Masses | 2.0 | 1.5 |
| Calcifications | 0.6 | 0.3 |

*~2–3 false marks on a normal case; more on non-dense breasts.*

*Citations: Kim 2009 · Mahoney 2011 · Watanabe 2019 · Leon 2009. Mahoney MC & Meganathan K, J Digit Imaging 2011, PMID 21547517 (Table 4, PMC3180536); Kim PMID 19863409; Leon SM et al., J Digit Imaging 2009, PMID 18704581; Watanabe et al., J Digit Imaging 2019, PMC6646646.*

---

## Slide 4 — $400M/yr; −47% odds with CAD on.

Then the data came in. The United States was spending over **four hundred million dollars a year** — roughly one dollar of every ten thousand in US healthcare. And among radiologists who read both with and without CAD, the **odds of detecting a malignant lesion were 47 percent lower with CAD on** — same readers, same images. That is quite remarkable. For the radiologists in the audience: every breast radiologist I have spoken with essentially ignores the CAD marks, or finds them disruptive — so this paper was not much of a surprise.

- **$400M/yr** — U.S. spend ≈ $1 of every $10,000 in health care
- **−47% odds** of detecting a malignant lesion — CAD on vs off, same readers (**OR 0.53**)

*When the definitive study finally ran, CAD added cost and, for the same readers, lowered the odds of catching a cancer.*

*Citation: Lehman et al., JAMA Internal Medicine 2015 · 323,973 women · 271 radiologists. DOI: 10.1001/jamainternmed.2015.5231; PMID 26414882.*

---

## Slide 5 — The verdict, verbatim.

The conclusion is worth reading verbatim. It had been foreshadowed years earlier — **Fenton, in the New England Journal in 2007,** already showed CAD lowered accuracy and raised the biopsy rate.

> "Computer-aided detection does not improve diagnostic accuracy of mammography. These results suggest that insurers pay more for CAD with no established benefit to women."
> — Lehman et al., JAMA Internal Medicine 2015

- **85.3%** vs **87.3%** sensitivity — with vs without CAD
- **4.1 = 4.1** /1,000 cancer detection — identical
- **0.871** vs **0.919** AUC — with vs without (Fenton, NEJM 2007; PMID 17409321)

---

## Slide 6 — What went wrong.

Why did it fail? Three reasons. First, it could only flag what we already know how to describe — the features were hand-tuned, so it was limited to what we could design, and it added marks rather than new information. Second, it marked most normal studies — 70–85% carried at least one false-positive mark — so readers learned to ignore it, and trust eroded. And third, the deeper problem: it was deployed at scale, reimbursed and run on nearly every screening mammogram in the country, before it was ever validated. The definitive accuracy study didn't arrive until 2015 — and found no benefit.

1. **It flagged only what we can already describe.** **Hand-tuned features** are limited to what we can design — so it added marks, not new information.
2. **Readers learned to ignore it.** **70–85% of normal exams** carried at least one false-positive mark — eroding trust.
3. **It was deployed at scale before it was ever validated.** Reimbursed in 2002 and run on **~92% of US screening mammograms** by 2016; the definitive accuracy study didn't arrive until 2015 — and found no benefit.

*Citations: Lehman CD, et al. JAMA Intern Med 2015;175(11):1828–1837 (definitive accuracy study; PMID 26414882) · Keen JD, et al. J Am Coll Radiol 2018 (adoption ~92%) · Kohli & Jha, "Why CAD Failed in Mammography," JACR 2018;15(3 Pt B):535–537; PMID 29398499; DOI 10.1016/j.jacr.2017.12.029.*

---

## Slide 7 — A paradigm shift.

The lesson from Era I is that CAD was reimbursed before it was proven, and was never shown to help. Around **2012** the technology changed. Traditional CAD was programmed — we wrote the rules. Modern deep learning is trained — it learns the patterns from the images. Traditional CAD pointed at spots; modern models can read the image. Sixteen years later.

On-slide title: **A paradigm shift.** (bottom blurb removed; the two boxes remain)

- **Traditional CAD — Programmed:** We wrote the rules for what cancer looks like. It **pointed** at spots for a human to check.
- **Modern deep learning — Trained:** The model learns the patterns from the images themselves. It can **read** the image — as a second reader, or on its own.
