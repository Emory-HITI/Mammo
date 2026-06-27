#!/usr/bin/env python3
"""Era II — 2026 (§2b). 3 slides: time-machine divider, two-prospective-studies + PRISM, subgroup Fig 6."""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pptxlib import *
from bs4 import BeautifulSoup
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

SLIDES="/home/user/Mammo/keynote-iwbi-2026/slides"
RAW=open(SLIDES+"/section2b.html").read()
soup=BeautifulSoup(RAW,'html.parser')
secs=soup.select('section.slide')
notes=[s.get('data-note','') for s in secs]
_svgm=re.search(r'<svg[\s\S]*?</svg>', RAW); svg=_svgm.group(0) if _svgm else None
fig6=None
for img in soup.select('img'):
    if img.get('src','').startswith('data:'): fig6=img['src']; break

prs=new_prs()

# ---- 1 time machine -> 2026 ----
s=slide(prs); note(s,notes[0])
_,tf=box(s,0.85,0.62,9,0.4); run(para(tf,True),"ERA II · 2026",11,AMBER,bold=True,font=MONO)
_,tf=box(s,0.85,2.5,6.6,1.5)
p=para(tf,True); run(p,"The year is ",50,INK,bold=True); run(p,"2026.",50,AMBER,bold=True)
_,tf=box(s,0.85,4.35,6.3,1.8)
p=para(tf,True,line=1.3); run(p,"Ten years on, the early promise has been put to the test — ",19,INK2); run(p,"prospectively, in real screening programs.",19,INK,bold=True)
if svg is not None: embed_svg(s, svg, 7.6, 1.3, 5.2, 5.2)

# ---- 2 two prospective studies + PRISM ----
s=slide(prs); note(s,notes[1]); eyebrow(s,"Era II · 2026")
_,tf=box(s,0.85,1.15,11.7,1.1)
p=para(tf,True,line=1.1); run(p,"2026: two prospective studies show ",28,INK,bold=True); run(p,"AI screening works.",28,AMBER,bold=True)

def study_card(l, accent, label, big_runs, sub_runs, cite):
    card(s,l,2.55,5.55,2.5,fill=CARD,edge=LINE,edge_w=1.0)
    rect(s,l,2.55,0.06,2.5,accent)
    _,tf=box(s,l+0.28,2.78,5.0,0.35); run(para(tf,True),label,10.5,accent,bold=True,font=MONO)
    _,tf=box(s,l+0.28,3.18,5.0,0.7); p=para(tf,True,line=1.1)
    for t,c,b in big_runs: run(p,t,19,c,bold=b)
    _,tf=box(s,l+0.28,3.95,5.0,0.95); p=para(tf,True,line=1.35)
    for t,c,b in sub_runs: run(p,t,12.5,c,bold=b)
    _,tf=box(s,l+0.28,4.72,5.0,0.3); run(para(tf,True),cite,9.5,INK3,font=MONO)

study_card(0.85, AMBER, "MASAI · SWEDEN · RCT (~105,000)",
    [("Interval cancers non-inferior ",INK,True),("(ratio 0.88)",INK3,False)],
    [("Sensitivity ",INK2,False),("80.5%",INK,True),(" vs 73.8% at matched specificity · ",INK2,False),("~44%",INK,True),(" less reading workload.",INK2,False)],
    "Gommers/Lång et al., Lancet 2026;407:505–514")
study_card(6.95, CYAN, "PRAIM · GERMANY · REAL-WORLD (463,000)",
    [("+17.6%",AMBER,True),(" cancers detected",INK,True)],
    [("Recall ",INK2,False),("non-inferior",INK,True),(" · 6.7 vs 5.7 CDR per 1,000 · 12 sites, 119 radiologists.",INK2,False)],
    "Eisemann et al., Nature Medicine 2025")

_,tf=box(s,0.85,5.35,11.7,1.5)
p=para(tf,True,line=1.35,align=PP_ALIGN.LEFT)
run(p,"A third is underway: ",15,INK2); run(p,"PRISM",15,AMBER,bold=True)
run(p," — the first large randomized trial of screening AI in the United States (Transpara; $16M PCORI; 7 sites; announced Sept 2025, recruiting). Results are years away, but it tests the ",15,INK2)
run(p,"US single-reader workflow",15,INK,bold=True); run(p," directly.",15,INK2)

# ---- 3 subgroup Fig 6 ----
s=slide(prs); note(s,notes[2]); eyebrow(s,"Era II · 2026")
_,tf=box(s,0.85,1.1,11.7,0.9)
p=para(tf,True,line=1.1); run(p,"The averages look strong — so where do these models ",24,INK,bold=True); run(p,"still fail?",24,AMBER,bold=True)
_,tf=box(s,0.85,2.05,11.7,0.8)
p=para(tf,True,line=1.35); run(p,"Overall AUC ",13.5,INK2); run(p,"0.91",13.5,INK,bold=True); run(p," — but performance depends on the ",13.5,INK2); run(p,"imaging feature",13.5,INK,bold=True); run(p,". Masses, asymmetries, calcifications are called negative in 60–81% of exams; ",13.5,INK2); run(p,"architectural distortions behave erratically.",13.5,INK,bold=True)
if fig6: embed_img(s,fig6,2.7,2.95,7.9,4.0,card_bg='white')
_,tf=box(s,0.85,6.95,11.7,0.4); run(para(tf,True,align=PP_ALIGN.CENTER),"Exam-level model-score distributions by imaging feature · [your group], Nat Commun 2026 (DOI 10.1038/s41467-026-70637-3), Fig. 6.",9,INK3,font=MONO)

save(prs, "/home/user/Mammo/keynote-iwbi-2026/pptx/IWBI2026_Trivedi_02b_EraII_2026.pptx")
