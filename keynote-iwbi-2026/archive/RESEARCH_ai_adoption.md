# Research note — Breast AI adoption, uptake, satisfaction (for the talk)
*Compiled June 2026. All facts PubMed/peer-reviewed unless flagged. Distinguish breast-specific from radiology-overall, and "currently using" from "plan to / attitude."*

---

## BOTTOM LINE (quote-ready)
- **Best breast-specific number:** in a **2024 Society of Breast Imaging (SBI) survey, 55.6% (90/162) of responding breast radiologists reported using AI-aided CAD.** Caveat: n=162, 7.2% response rate → self-selection toward AI-engaged readers likely inflates this. *(Nenow et al., J Breast Imaging 2026; PMID 41728897; DOI 10.1093/jbi/wbaf079)*
- **US radiology overall:** ~**30%** using AI in clinical practice (mammographic abnormalities named a top-3 use case). *(ACR Data Science Institute 2020 survey; Allen et al., JACR 2021;18(8):1153–9; PMID 33891859)*
- **Europe, with a real time trend:** active clinical AI use rose **20.4% (2018) → 47.9% (2024)**; breast imaging rated the subspecialty **most** impacted by AI. *(ESR EuroAIM/EuSoMII; Zanardo et al., Insights Imaging 2024;15:240; PMID 39373853; earlier 2018: PMID 31673823)*

**Defensible framing for the slide:** "Roughly **half** of breast radiologists now use AI in some form (US ~30–56%, Europe ~48%), up from ~20% in 2018 — but routine, trusted, reimbursed use is far lower."

---

## 1. Adoption / uptake — current-use surveys
| Survey | Population | n | Current AI use | Source |
|---|---|---|---|---|
| **SBI 2024** (breast-specific) | US breast radiologists | 162 | **55.6%** use AI-aided CAD | Nenow, J Breast Imaging 2026 (PMID 41728897) |
| ACR DSI 2020 | US radiologists (all) | 1,427 | **~30%**; +20% plan to buy in 1–5 yr | JACR 2021 (PMID 33891859) |
| ESR EuroAIM 2024 | European (all) | 572 | **47.9%**; 25.3% planning | Insights Imaging 2024 (PMID 39373853) |
| ESR 2022 practical-experience | European (all) | 690 | **40%** had clinical AI experience; only **13.3%** intend to acquire | Insights Imaging 2022 (PMID 35727355) |
| SIRM Lombardy 2024 (breast subgroup) | Italian | 232 | **36.2%** daily AI use | Eur J Radiol 2024 (PMID 38959557) |
| Netherlands dept. tracking | NL departments | ~33–43 | 20% (2020) → 28% (2021) → **33%** (2022); breast NOT a top app | Eur Radiol 2023 (van Leeuwen) |

**FDA/CE product growth (supply side):** CE-marked radiology AI products **100 (2020) → 173 (2023)** *(van Leeuwen Eur Radiol 2021 PMID 33856519; Antonissen Eur Radiol 2025 PMID 40707732)*. FDA cumulative AI-enabled devices ~**1,350–1,450** by late 2025, **~75% radiology** (analyst trackers, not peer-reviewed). Of the 2021 set of 100, **12 were breast products**.

---

## 2. Satisfaction / attitudes / trust
- **Generally positive, oversight insisted on.** Sweden: **80.8%** of breast radiologists positive toward AI in screening *(Högberg, BMJ Health Care Inform 2023; PMID 37217249)*. Norway: **82%** positive *(Martiniussen, Eur J Radiol 2023; PMID 37657381)*. Italy: 61% see AI as opportunity, **84% say radiologist's final assessment still essential** *(Cè 2024)*.
- **SBI 2024 perceived benefits:** efficiency 71.0%, increased detection 65.2%, reduced recall 39.1%. **Barriers:** cost 71.5%, software integration 62.0%, lack of trust 63.3%. Only 27.5% felt AI threatens job security.
- **Trust is moderate, autonomy rejected.** ESR 2022: 75.7% of users found AI reliable, but **69.8% saw no workload reduction**. Croatia/Slovenia: mean trust **3.5/5**; only **13% would use AI without physician oversight** *(Orešković, Croat Med J 2023)*.
- **Knowledge ↔ fear:** basic AI knowledge associated with fear of replacement (OR 1.56); advanced knowledge inversely associated *(Huisman, Eur Radiol 2021; PMID 33744991/33974148)*.

---

## 3. The automation-bias warning (breast-specific, KEY for the "caveat emptor" thread)
**Dratsch et al., Radiology 2023 (PMID 37129490; DOI 10.1148/radiol.222176).** 27 radiologists read mammograms with a (sometimes wrong) AI BI-RADS suggestion. When the AI was **wrong**, correct ratings collapsed: inexperienced 79.7%→**19.8%**; moderately experienced 81.3%→24.8%; very experienced 82.3%→**45.5%**. **Radiologists at every experience level are prone to automation bias.** → ties directly to the §1 CAD "made good readers worse" (OR 0.53) and §6 governance.

---

## 4. Real-world deployments at scale (the "it's actually happening" slide)
| Country / program | N women/exams | Design | Headline | Cite |
|---|---|---|---|---|
| **Sweden — MASAI** | 105,934 | **RCT** | CDR **+29%**, workload **−44%**, interval-cancer **non-inferior** | Lång Lancet Oncol 2023 (37541274); Hernström Lancet Digital Health 2025 (39904652); Gommers Lancet 2026 (41620232) |
| **Germany — PRAIM** | 463,094 | Prospective real-world | CDR **+17.6%**, recall non-inferior | Eisemann, Nat Med 2025 (39775040) |
| **Denmark — Capital Region** | ~119,000 | Deployed before/after | recall **−20.5%**, workload −33.5% | Lauritzen, Radiology 2024 (38832880) |
| **Hungary — Kheiron Mia** | ~25,065 | Prospective (3-phase) | +0.7–1.6 cancers/1,000 (extra reader) | Ng, Nat Med 2023 (37973948) |
| **US — ASSURE (DeepHealth)** | >579,000 | Retrospective real-world | CDR **+21.6%**, no subgroup disparity | Nat Health 2025 (DOI 10.1038/s44360-025-00001-0) |
| **UK — EDITH** | ~700,000 (planned) | RCT, 5 platforms, ~30 sites | ongoing, no results | GOV.UK/NIHR 2025 (press) |
| **US — PRISM (UCLA/UC Davis)** | announced | RCT | no results | PCORI 2025 (press) |

Most common vendor across Sweden/Denmark/Netherlands/Norway evaluations: ScreenPoint **Transpara**. Two deployment models: **triage/workload-reduction** (MASAI, Denmark) vs **extra-reader/detection-boost** (Hungary, Netherlands single+AI).

---

## 5. The cleared-vs-adopted GAP (ties to the §1 CAD lesson)
- **Potnis et al., JAMA Intern Med 2022 (PMID 36342705)** — breast-specific anchor: of **9 FDA-cleared breast-screening AI products**, **all** cleared on retrospective data; 7/9 used enriched datasets; **none reported clinically meaningful outcomes** (stage, interval cancer).
- **van Leeuwen 2021:** **64/100** CE-marked radiology AI products had **no peer-reviewed evidence**; only 18/100 at efficacy level ≥3. By 2023, 34% still had none; only 31% had higher-level evidence.
- **Reimbursement bottleneck:** almost no radiology AI has a Category I CPT code (only coronary-CT AI does, 2024). Breast/mammography AI has none; ACR's "AI risk scoring" CPT was sent back by the AMA panel. *(Razavian/Topol, NEJM AI 2025, PMID 41695240; trade press for CPT specifics.)*
- **"AI chasm":** Aristidou, Jena, Topol, Lancet 2022 (PMID 35151388) — development outpaces demonstrated clinical value.
- **The cautionary precedent:** conventional CAD reached **~92% of US mammography facilities by 2016** despite the 2015 evidence it didn't help (Lehman). The adoption curve ran ahead of the evidence — the exact mistake to avoid.

---

## Where this can go in the talk
1. **§2 (Era II) or a short new "where we actually are" beat** — the ~50% adoption number + the deployment table = "this is no longer hypothetical."
2. **§6 (equity/governance)** — Dratsch automation bias + Potnis cleared-without-outcomes + CPT/reimbursement gap = the governance argument.
3. **§1 callback** — CAD hit 92% adoption before the evidence; don't repeat it.

## Gaps (do not invent)
- No peer-reviewed study gives a precise "% of cleared breast AI devices actually in clinical use."
- No dedicated EUSOBI breast-radiologist AI-attitude survey exists; breast attitude data are subgroups within broader surveys.
- SBI 55.6% has a small, self-selected sample — state the caveat.
