# Narrative Restructure — "Thirty Years, Three Eras" as a time-travel journey
### Proposal + new research to fold in. Backup branch: `keynote-v2-backup-2026-06-24`.

The ask: recast the talk as a first-person journey. After the hook, the audience *travels* through the three eras — feeling the excitement, the adoption, the disillusionment or validation, and the lesson of each — rather than hearing a survey. The device: **"You are a radiologist in [year]…"** at each era opening, then live the arc.

The three eras already map onto the content; this reframes the delivery and resequences a few pieces.

---

## The frame story

- **Title + Hook (unchanged):** the normal mammogram, the three models, "the information was already there." Ends: *"To understand where this is going, we have to travel through three eras. Set your clock back thirty years."*
- The recurring device at each era: a **dated waypoint** ("1998… 2016… today"), the **mood** (excitement → everyone adopts → the reckoning), and the **lesson** carried forward.
- Era I and Era II deliberately **rhyme** — both open with excitement and a wave of adoption. The difference is the reckoning: Era I failed its test; Era II passed it. That rhyme is the spine of the argument.

---

## ERA I — 1998–2015 · "You are a radiologist in 1998." (CAD)  [~6–7 min]
*Mood: excitement → ubiquity → quiet disappointment.*
1. **1998 — it arrives.** CAD is FDA-approved. A second set of eyes, automatically. You're excited.
2. **2002–2008 — everyone's using it.** A reimbursement code lands; adoption races to 74%, then 92%. It's just *there* now.
3. **…but the room is grumbling.** Marks on every other normal study; you learn to tune it out (83% of cases marked with conventional CAD). Colleagues don't trust it.
4. **The reckoning (the data).** Fenton 2007 → Lehman 2015: no benefit; OR 0.53 — the same readers did *worse*. $400M/yr, ≈$1 of every $10,000 in U.S. healthcare.
5. **The lesson.** Paid first, proven never. *(carry forward: demand prospective evidence)*
6. **The turn.** Programmed → trained. Point → read.

## ERA II — 2016–today · "Fast-forward. You're a radiologist in 2016." (modern AI)  [~10 min]
*Mood: excitement again — but this time the field demands the evidence.*
1. **Primer.** What modern mammography AI is and how it works (deep network → suspicion score + localization, Lunit output image); deployment modes; the vendor landscape.
2. **2020 — déjà vu.** "Superhuman on paper" (McKinney) — and the field checks itself (Haibe-Kains reproducibility). Retrospective ≠ the bar.
3. **The reckoning, this time positive.** The RCTs: MASAI +29% / 44% workload / interval-cancer non-inferior; PRAIM +17.6%. It actually works. Era II passed the test Era I failed.
4. **Caveat emptor — the new grumbling.** A single AUC hides where a model fails:
   - **Subgroup blind spots** — our DBT audit (AUC 0.91 overall → 0.80 calcifications).
   - **It doesn't transfer** — *NEW:* Du Hao et al., "Beyond Screening" (Emory+NUS): screening-trained models AUC 0.78–0.81 on screening, but drop 0.08–0.09 on diagnostic; on **implants** AUC 0.69–0.76, specificity 31–54%. Even foundation models localize poorly (IoU 0.136).
   - **The two gaps:** explainability + subgroup/transferability.
5. **The lesson.** It works *where it was validated*. Hold the next era to the same bar.

## ERA III — today → the future · "Now step forward into the next decade." (clinical intelligence)  [~24 min]
*Mood: what the image, the patient, and the population could become — grounded, not breathless.*
Internal progression = the spine: **image → patient → population.**
1. **Image.** The mammogram as biosensor: risk (Mirai → Clairity), the population-vs-individual reckoning, the four quadrants; opportunistic cardio (BAC).
2. **Patient.** Pathology (morphology→molecular, CDH1/Virchow). **NEW: multimodal radiology fusion** — combining MG + US + MRI (+ clinical/path) into one prediction (see fusion notes below). Then convergence / clinical intelligence + the frontier-model wildcard.
3. **Population.** Equity, governance, the global gap — the CAD lesson at population scale.
4. **Close.** The discipline carried from all three eras. Caveat emptor; prove it before scaling.

---

## NEW CONTENT — multimodal radiology AI fusion (Era III · Patient)
The point: today almost all deployed AI reads **one modality**. The near future fuses them.
- **MG + US:** a multimodal model reached **AUC ~0.88**, +13–16 points over single-modality. *(verify exact study)*
- **MG + US + MRI + clinical:** three-tier fusion frameworks (e.g., TransFusion-BCNet 2026) — intra-modality (across MG views), inter-modality, and decision-level fusion. *(verify)*
- **MRI fusion:** ultrafast DCE-MRI + clinical data for lesion classification (ScienceDirect 2025). *(verify)*
- **Treatment response:** your group's RSNA26 abstract — *"Predicting pCR in Breast Cancer: do mammographic and MRI findings complement pathology-derived features"* (MG+MRI+path fusion). **[your group]**
- **"A Multi-Modal AI System for Screening Mammography"** (NYU/Wu-style; PDF in lab Slack) — multimodal screening precedent.
- Framing line: *"Each modality answers a different question — mammography for calcifications and architecture, ultrasound for the dense breast, MRI for extent and enhancement. Fusing them is how a model starts to see the way a breast radiologist already thinks."*
- **[VERIFY]** all fusion citations before a numbers slide — these are mostly recent/algorithmic papers, not large clinical trials; present as an emerging direction.

## NEW CONTENT — Du Hao "Beyond Screening" (Era II · caveat)
Du H, Brown-Mulry B, Jeon YS, Issac RS, Dapamede T, Li F, Mansuri A, Wang AX, Hartman M, Feng M, Gichoya JW, Trivedi H. **[your group]**
- 244,385 screening + 82,643 diagnostic exams, single institution, 2013–2020.
- Models: Mammo-CLIP (vision-language FM), standard CNN, MedImageInsight (medical FM). Specificity at fixed sensitivity 68.1%.
- Screening AUC 0.78–0.81 (all). Diagnostic: CNN/VLM −0.08 to −0.09; MedImageInsight held (standard 0.78, spot-compression 0.84).
- **Implants: AUC 0.69–0.76; specificity 31–54% (−21 to −51 pts).** ~10% of women.
- Localization poor (mean IoU 0.136); pointing accuracy implant 45.2% vs non-implant 30.7%.
- Consistent across race/ethnicity. NIH OT2OD032581.
- **Use:** the cleanest "modern AI doesn't generalize off-distribution" evidence — screening→diagnostic→implant. Pairs with the DBT subgroup audit; foundation models are more robust than CNNs but still not safe to deploy off-label.

---

## Decisions I need before rebuilding slides
1. **Approve the time-travel framing** ("You are a radiologist in 1998 / 2016 / today")? It's a strong stylistic device — good for engagement, but commits the whole deck. Yes / tone it down / no.
2. **Era II grew** (primer + positive RCTs + the two caveats incl. Du Hao). With Era III at ~24 min, total is tight for 45 — Era I may need to compress to ~6. OK?
3. Rebuild order: I'd redo the **opening hook hand-off + Era I divider** first (small), confirm the device lands, then roll Era II/III.
