#!/usr/bin/env python3
"""Era II (2016) — editable PPTX. 6 slides. Figures embedded; tables native; one text box per blurb."""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pptxlib import *
from bs4 import BeautifulSoup
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

SLIDES="/home/user/Mammo/keynote-iwbi-2026/slides"
H=open(SLIDES+"/section2.html").read()
soup=BeautifulSoup(H,'html.parser')
secs=soup.select('section.slide')
notes=[s.get('data-note','') for s in secs]
imgs=[img['src'] for img in soup.select('section.slide img') if img.get('src','').startswith('data:')]
DREAM, MCK, SALIM = (imgs+[None,None,None])[:3]

prs=new_prs()

# ---- 1 divider 2016 ----
s=slide(prs); note(s, notes[0])
_,tf=box(s,0.85,0.62,9,0.4); run(para(tf,True),"ERA II",11,AMBER,bold=True,font=MONO)
_,tf=box(s,0.85,2.6,11.5,1.6)
p=para(tf,True); run(p,"The year is ",52,INK,bold=True); run(p,"2016.",52,AMBER,bold=True)
_,tf=box(s,0.85,4.5,9.5,1.4)
p=para(tf,True,line=1.3); run(p,"You're beginning to hear about AI — and still ",20,INK2); run(p,"feeling the burn from CAD.",20,INK,bold=True)

# ---- 2 DREAM ----
s=slide(prs); note(s,notes[1]); eyebrow(s,"Era II")
_,tf=box(s,0.85,1.15,6.4,1.5)
p=para(tf,True,line=1.12); run(p,"2016: the first large-scale push — and a ",26,INK,bold=True); run(p,"surge of optimism.",26,AMBER,bold=True)
_,tf=box(s,0.85,3.0,6.4,1.2)
run(para(tf,True,line=1.35),"The Digital Mammography DREAM Challenge was the first large-scale effort to build breast-cancer AI — and we took part. The hope: a crowdsourced competition would simply solve it.",14,INK2)
chip(s,0.85,4.45,6.2,"1,000+ participants · 126 teams · 44 countries")
chip(s,0.85,4.95,6.2,"~640,000 mammograms")
_,tf=box(s,0.85,5.65,6.4,1.0)
p=para(tf,True,line=1.35); run(p,"Best model ",13.5,INK2); run(p,"AUC 0.858",13.5,INK,bold=True); run(p," → ensemble ",13.5,INK2); run(p,"0.895",13.5,INK,bold=True); run(p," → + a radiologist ",13.5,INK2); run(p,"0.942.",13.5,INK,bold=True); run(p," Winner Therapixel spun out MammoScreen — still in use.",13.5,INK2)
_,tf=box(s,0.85,6.7,6.4,0.4); run(para(tf,True),"Schaffter et al., JAMA Network Open 2020;3(3):e200265.",10,INK3,font=MONO)
if DREAM: embed_img(s,DREAM,7.7,1.2,5.0,5.6,card_bg='white')

# ---- 3 McKinney ----
s=slide(prs); note(s,notes[2]); eyebrow(s,"Era II")
_,tf=box(s,0.85,1.15,6.3,1.4)
p=para(tf,True,line=1.12); run(p,"2020: Google publishes its ",26,INK,bold=True); run(p,"breast-cancer AI model.",26,AMBER,bold=True)
stats=[("−5.7% / −1.2%","false positives (US / UK)"),("−9.4% / −2.7%","false negatives (US / UK)"),
       ("+11.5%","AUC over the average radiologist"),("−88%","simulated second-reader workload")]
sy=2.9
for big,lab in stats:
    _,tf=box(s,0.85,sy,6.2,0.5); run(para(tf,True),big,21,AMBER,bold=True)
    _,tf=box(s,0.85,sy+0.46,6.2,0.4); run(para(tf,True),lab,11.5,INK2,font=MONO); sy+=0.95
_,tf=box(s,0.85,6.75,6.3,0.5); run(para(tf,True),"McKinney et al., Nature 2020;577:89–94 — promising, but only on clean, curated data.",10,INK3,font=MONO)
if MCK: embed_img(s,MCK,7.7,1.5,5.0,5.0,card_bg='white')

# ---- 4 Salim ----
s=slide(prs); note(s,notes[3]); eyebrow(s,"Era II")
_,tf=box(s,0.85,1.15,6.4,1.6)
p=para(tf,True,line=1.12); run(p,"2020: three commercial models, independently validated on ",24,INK,bold=True); run(p,"8,800 women.",24,AMBER,bold=True)
_,tf=box(s,0.85,3.2,6.4,1.6)
p=para(tf,True,line=1.35); run(p,"Best algorithm ",15,INK2); run(p,"AUC 0.956",15,INK,bold=True); run(p,". AI + first reader reached ",15,INK2); run(p,"88.6% sensitivity at 93.0% specificity",15,INK,bold=True); run(p," — exceeding two human readers.",15,INK2)
_,tf=box(s,0.85,5.2,6.4,0.5); run(para(tf,True),"Salim et al., JAMA Oncology 2020 (Stockholm, 8,805 women).",10,INK3,font=MONO)
if SALIM: embed_img(s,SALIM,7.7,1.4,5.0,5.2,card_bg='white')

# ---- 5 MASAI 2023 safety ----
s=slide(prs); note(s,notes[4]); eyebrow(s,"Era II")
_,tf=box(s,0.85,1.15,11.6,1.0)
p=para(tf,True,line=1.1); run(p,"2023: MASAI publishes its ",30,INK,bold=True); run(p,"safety analysis.",30,AMBER,bold=True)
_,tf=box(s,0.85,2.35,11.4,0.7)
p=para(tf,True,line=1.3); run(p,"A randomized trial, ",16,INK2); run(p,"~80,000 women",16,INK,bold=True); run(p," (≈40,000 per arm) — AI-supported reading vs standard double reading.",16,INK2)
_,tf=box(s,0.85,3.25,11.4,1.1)
p=para(tf,True); run(p,"6.1 ",54,AMBER,bold=True); run(p,"vs ",30,INK3); run(p,"5.1",54,AMBER,bold=True); run(p,"   CDR per 1,000 · ratio 1.2 (95% CI 1.0–1.5)",13,INK,font=MONO)
rows=[("Measure","AI","Control"),("Recall rate","2.2%","2.0%"),("False-positive rate","1.5%","1.5%"),
      ("PPV of recall","28.3%","24.8%"),("Invasive cancers","75%","81%")]
native_table(s,rows,0.85,4.7,7.6,col0=3.6,fs=12.5)
_,tf=box(s,0.85,6.95,11.5,0.4); run(para(tf,True),"Lång et al., Lancet Oncology 2023 — above the lowest acceptable detection limit for safety.",10,INK3,font=MONO)

# ---- 6 adoption ----
s=slide(prs); note(s,notes[5]); eyebrow(s,"Era II")
_,tf=box(s,0.85,1.25,11.6,1.0)
p=para(tf,True,line=1.1); run(p,"Data isn't enough to ",30,INK,bold=True); run(p,"drive adoption.",30,AMBER,bold=True)
_,tf=box(s,0.85,2.5,11.4,0.7)
run(para(tf,True,line=1.3),"Enriched cases, lab conditions, no real workflow — CAD had good numbers too. The bar is prospective deployment, and real-world uptake has been slow.",15,INK2)
trip=[("48%","of European radiologists use AI (2024)"),("13.7%","of them, for breast imaging"),("~2%","of US practices, by one estimate")]
tx=0.85
for big,lab in trip:
    _,tf=box(s,tx,3.7,3.6,1.0); run(para(tf,True),big,46,AMBER,bold=True)
    _,tf=box(s,tx,4.85,3.6,1.0); run(para(tf,True,line=1.25),lab,11.5,INK,font=MONO); tx+=4.0
_,tf=box(s,0.85,6.6,11.5,0.4); run(para(tf,True),"ESR EuroAIM/EuSoMII survey, Insights Imaging 2024 (n=572) · US estimate: industry report.",10,INK3,font=MONO)

save(prs, "/home/user/Mammo/keynote-iwbi-2026/pptx/IWBI2026_Trivedi_02_EraII.pptx")
