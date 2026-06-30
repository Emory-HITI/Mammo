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

def chiprun(s, x, y, runs, fs=10.5):
    """cyan-accent chip sized to content; runs = [(text, bold)], bold→cyan."""
    chars=sum(len(t) for t,_ in runs)
    w=chars*fs*0.0082+0.42
    c=card(s,x,y,w,0.42,fill=CARD,edge=LINE,edge_w=0.75)
    rect(s,x,y,0.045,0.42,CYAN)
    tf=c.text_frame; tf.word_wrap=False; tf.vertical_anchor=MSO_ANCHOR.MIDDLE
    tf.margin_left=Inches(0.15); tf.margin_right=Inches(0.10); tf.margin_top=Pt(0); tf.margin_bottom=Pt(0)
    p=para(tf,True)
    for t,bold in runs: run(p,t,fs,(CYAN if bold else INK),bold=bold,font=MONO)
    return 0.42

def cmp_table(s, x, y, rows, w_meas=4.2, w_val=1.6):
    """MASAI-style comparison table: transparent, thin row borders, AI col amber-bold.
    rows[0]=header; each row=(measure, ai, control)."""
    rowh=0.52; aix=x+w_meas; ctx=aix+w_val
    for ri,(m,ai,ct) in enumerate(rows):
        ry=y+ri*rowh
        head=(ri==0)
        _,tf=box(s,x,ry,w_meas,rowh,anchor=MSO_ANCHOR.MIDDLE)
        run(para(tf,True),m if head else m,11 if head else 13,INK2,font=(MONO if head else SANS))
        _,tf=box(s,aix,ry,w_val,rowh,anchor=MSO_ANCHOR.MIDDLE)
        run(para(tf,True,align=PP_ALIGN.RIGHT),ai,10.5 if head else 14,INK2 if head else AMBER,bold=(not head),font=(MONO if head else SANS))
        _,tf=box(s,ctx,ry,w_val,rowh,anchor=MSO_ANCHOR.MIDDLE)
        run(para(tf,True,align=PP_ALIGN.RIGHT),ct,10.5 if head else 13,INK2 if head else INK,font=(MONO if head else SANS))
        ln=rect(s,x,ry+rowh-0.01,w_meas+2*w_val,0.01,LINE); ln.height=Pt(1)
    return len(rows)*rowh

# ---- 1 divider 2016 (with time-machine graphic) ----
s=slide(prs); note(s, notes[0])
_,tf=box(s,0.85,2.45,9,0.34); run(para(tf,True),"ERA II",11,AMBER,bold=True,font=MONO)
_,tf=box(s,0.85,2.85,6.9,0.95)
p=para(tf,True,line=1.0); run(p,"The year is ",44,INK,bold=True); run(p,"2016.",44,AMBER,bold=True)
_,tf=box(s,0.85,3.95,6.3,1.3)
p=para(tf,True,line=1.4); run(p,"You're beginning to hear about AI — but still ",17,INK2); run(p,"feeling the burn from CAD.",17,INK,bold=True)
_m=re.search(r'<svg[\s\S]*?</svg>', H)
if _m: embed_svg(s, _m.group(0), 7.8, 1.05, 4.9, 4.7)

# ---- 2 DREAM ----
s=slide(prs); note(s,notes[1]); eyebrow(s,"Era II")
_,tf=box(s,0.85,1.15,6.4,1.5)
p=para(tf,True,line=1.12); run(p,"2016: the first large-scale push — and a ",26,INK,bold=True); run(p,"surge of optimism.",26,AMBER,bold=True)
_,tf=box(s,0.85,3.1,6.4,1.6)
run(para(tf,True,line=1.4),"The Digital Mammography DREAM Challenge was the first large-scale effort to build breast-cancer AI — and we took part. The hope was that a crowdsourced competition would simply solve it.",14.5,INK2)
chiprun(s,0.85,5.05,[("1,000+",True),(" participants · 126 teams · 44 countries",False)])
chiprun(s,0.85,5.6,[("~640,000",True),(" mammograms",False)])
_,tf=box(s,0.85,6.4,6.4,0.4); run(para(tf,True),"Schaffter et al., JAMA Network Open 2020;3(3):e200265.",10,INK3,font=MONO)
if DREAM: embed_img(s,DREAM,7.7,1.2,5.0,5.6,card_bg='white')

# ---- 3 McKinney ----
s=slide(prs); note(s,notes[2]); eyebrow(s,"Era II")
_,tf=box(s,0.85,1.15,6.3,1.4)
p=para(tf,True,line=1.12); run(p,"2020: Google publishes its ",26,INK,bold=True); run(p,"breast-cancer AI model.",26,AMBER,bold=True)
_,tf=box(s,0.85,2.55,6.4,0.9)
p=para(tf,True,line=1.35); run(p,"McKinney et al., Nature 2020",14,INK,bold=True); run(p," (Google Health, UK + US) — a deep-learning system read screening mammograms stand-alone.",14,INK2)
mck_chips=[[("false positives  ",False),("−5.7% US / −1.2% UK",True)],
           [("false negatives  ",False),("−9.4% US / −2.7% UK",True)],
           [("AUC ",False),("+11.5%",True),(" vs avg radiologist",False)],
           [("2nd-reader workload  ",False),("−88%",True)]]
sy=3.7
for runs in mck_chips:
    chiprun(s,0.85,sy,runs); sy+=0.58
_,tf=box(s,0.85,sy+0.06,6.3,0.5); run(para(tf,True,line=1.3),"Results are promising — but again, only on clean, curated data.",13,INK2)
_,tf=box(s,0.85,6.82,6.3,0.4); run(para(tf,True),"McKinney et al., Nature 2020;577:89–94.",9.5,INK3,font=MONO)
if MCK: embed_img(s,MCK,7.7,1.5,5.0,5.0,card_bg='white')

# ---- 4 Salim ----
s=slide(prs); note(s,notes[3]); eyebrow(s,"Era II")
_,tf=box(s,0.85,1.15,6.4,1.6)
p=para(tf,True,line=1.12); run(p,"2020: three commercial models, independently validated on ",24,INK,bold=True); run(p,"8,800 women.",24,AMBER,bold=True)
_,tf=box(s,0.85,3.0,6.4,0.8)
p=para(tf,True,line=1.35); run(p,"Salim et al., JAMA Oncology 2020",14,INK,bold=True); run(p," tested three commercial algorithms on an external Stockholm cohort.",14,INK2)
sal_chips=[[("AUC — AI-1 ",False),("0.956",True),(" · AI-2 ",False),("0.922",True),(" · AI-3 ",False),("0.920",True)],
           [("at readers' specificity (",False),("96.6%",True),("): AI-1 ",False),("81.9%",True),(" sens vs readers ",False),("77.4%",True)],
           [("AI + reader: ",False),("+77%",True),(" abnormal calls for ",False),("+8%",True),(" detection",False)]]
sy=3.95
for runs in sal_chips:
    chiprun(s,0.85,sy,runs); sy+=0.58
_,tf=box(s,0.85,sy+0.06,6.4,0.6); run(para(tf,True,line=1.3),"The combination's recall cost is the operating-point tension in miniature.",13,INK2)
_,tf=box(s,0.85,6.78,6.4,0.4); run(para(tf,True),"Salim et al., JAMA Oncology 2020 (Stockholm, 8,805 women).",10,INK3,font=MONO)
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
cmp_table(s,0.85,4.55,rows,w_meas=4.2,w_val=1.6)
_,tf=box(s,0.85,6.95,11.5,0.4); run(para(tf,True),"Lång et al., Lancet Oncology 2023 — above the lowest acceptable detection limit for safety.",10,INK3,font=MONO)

# ---- 6 adoption ----
s=slide(prs); note(s,notes[5]); eyebrow(s,"Era II")
_,tf=box(s,0.85,1.25,11.6,1.0)
p=para(tf,True,line=1.1); run(p,"Data isn't enough to ",30,INK,bold=True); run(p,"drive adoption.",30,AMBER,bold=True)
_,tf=box(s,0.85,2.5,11.4,0.7)
run(para(tf,True,line=1.3),"Enriched cases, lab conditions, no real workflow — CAD had good numbers too. The bar is prospective deployment, and real-world uptake has been slow.",15,INK2)
trip=[("48%","European radiologists use AI (2024) — up from 20% in 2018"),("13.7%","of them, for breast imaging — about 1 in 8"),("~2%","of US practices, by one estimate")]
tx=0.85
for big,lab in trip:
    _,tf=box(s,tx,3.7,3.6,1.0); run(para(tf,True),big,46,AMBER,bold=True)
    _,tf=box(s,tx,4.85,3.6,1.4); run(para(tf,True,line=1.3),lab,11.5,INK,font=MONO); tx+=4.0
_,tf=box(s,0.85,6.6,11.5,0.4); run(para(tf,True),"ESR EuroAIM/EuSoMII survey, Insights Imaging 2024 (n=572) · US estimate: industry report.",10,INK3,font=MONO)

import os
out="/home/user/Mammo/keynote-iwbi-2026/pptx/sections/IWBI2026_Trivedi_02_EraII.pptx"
os.makedirs(os.path.dirname(out), exist_ok=True)
save(prs, out)
