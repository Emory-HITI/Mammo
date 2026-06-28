# Section 2 — Era II: The Screening Era Opens (2016)

### spine stage: **LESION** (modern) · the present-day state of screening AI

---

## Slide 1 — The year is 2016

The year is 2016. You're a radiologist, you're just beginning to hear about AI for breast imaging — and the reading room is still feeling the burn from CAD. So what does mammography AI actually look like, and what is the evidence?

---

## Slide 2 — 2016: the first large-scale push — and a surge of optimism

The optimism wasn't new in 2020. Back in 2016 came the first large-scale effort to build breast-cancer AI: the Digital Mammography DREAM Challenge. Over a thousand participants, 126 teams from 44 countries, on roughly 640,000 mammograms; we took part. There was an incredible surge of optimism that this would simply solve breast AI.

On the held-out evaluation set, the best single model reached an **AUC of 0.858**; the **ensemble** of the eight best models reached **0.895**; and the **ensemble plus a single radiologist** reached **0.942**. The winner, Therapixel, spun out a company whose product, MammoScreen, is still in clinical use today.

- **1,000+ participants · 126 teams · 44 countries · ~640,000 mammograms**
- Best model **AUC 0.858** → ensemble **0.895** → ensemble + a radiologist **0.942**

**Citation:** Schaffter et al., *JAMA Network Open* 2020;3(3):e200265.

---

## Slide 3 — 2020: Google publishes its breast-cancer AI model

So what was the early evidence? In 2020, McKinney and colleagues at Google Health, in *Nature*, reported a deep-learning system that read screening mammograms stand-alone. On US data it cut false positives by 5.7% and false negatives by 9.4%, exceeded the average radiologist's AUC by over 11 points, and could cut a second reader's workload by 88%. The figure shows example cases the model localized. On paper, superhuman — but again, only on clean, curated data.

- **False positives −5.7% US / −1.2% UK**
- **False negatives −9.4% US / −2.7% UK**
- **AUC +11.5% vs average radiologist**
- **2nd-reader workload −88%**

**Citation:** McKinney et al., *Nature* 2020 (Google Health, UK + US).

---

## Slide 4 — 2020: three commercial models, independently validated on 8,800 women

And it held up independently. Salim and colleagues, in *JAMA Oncology*, took three commercial AI algorithms and tested them on an external Stockholm cohort of about 8,800 women. The best — Algorithm 1 — reached an **AUC of 0.956**, and combined with a first reader it reached **88.6% sensitivity at 93% specificity**, exceeding two human readers. Three different vendors, external data — the evidence is looking good.

- Best (Algorithm 1) **AUC 0.956**
- AI + first reader **88.6% sensitivity @ 93% specificity** — exceeded two human readers
- **n = 8,805**

**Citation:** Salim et al., *JAMA Oncology* 2020.

---

## Slide 5 — 2023: MASAI publishes its safety analysis

In 2023, MASAI published its clinical safety analysis — a randomized trial of about eighty thousand women, roughly forty thousand per arm, AI-supported reading versus standard double reading. Cancer detection was **6.1 per 1,000 with AI versus 5.1 in controls, a ratio of 1.2 (95% CI 1.0–1.5)**, above the lowest acceptable detection limit for safety. Recall was essentially identical, 2.2 versus 2.0 percent; the false-positive rate was 1.5 percent in both; and the positive predictive value of recall was higher with AI, 28 versus 25 percent. Three-quarters of detected cancers were invasive. So: safe, and detecting more — the green light to continue.

| Measure | AI | Control |
|---|---|---|
| CDR per 1,000 | 6.1 | 5.1 (ratio 1.2, 95% CI 1.0–1.5) |
| Recall rate | 2.2% | 2.0% |
| False-positive rate | 1.5% | 1.5% |
| PPV of recall | 28.3% | 24.8% |
| Invasive cancers | 75% | 81% |

**Citation:** Lång et al., *Lancet Oncology* 2023.

---

## Slide 6 — Data isn't enough to drive adoption

But here is the discipline this talk runs on. Reader studies have well-known limits: enriched case sets, lab conditions, no real workflow. CAD had good numbers too. That kind of data is not enough to drive adoption. The bar is prospective deployment, and real-world uptake has been slow.

And it shows: a 2024 European survey found about **48% of radiologists use AI** (up from 20% in 2018), but of those only **13.7% use it for breast imaging — about one in eight**. In the US, by one estimate, only around **2% of practices** use it at all.

- **48% of European radiologists use AI (2024)** — up from 20% in 2018
- **13.7% of them, for breast imaging** — about 1 in 8
- **~2% of US practices**, by one estimate

**Citation:** ESR EuroAIM/EuSoMII survey, *Insights Imaging* 2024 (n=572) · US estimate: industry report.
