import os, glob, sys
from playwright.sync_api import sync_playwright
from pptx import Presentation
from pptx.util import Inches

SLIDES="/home/user/Mammo/keynote-iwbi-2026/slides"
OUT="/home/user/Mammo/keynote-iwbi-2026/pptx"
TMP="/tmp/claude-0/-home-user-Mammo/6e3f6d6c-3e45-5480-8f5e-a0e6f5f1309e/scratchpad/render"
os.makedirs(TMP, exist_ok=True)
CHROME="/opt/pw-browsers/chromium-1194/chrome-linux/chrome"

DECKS=[("hook","IWBI2026_Trivedi_00_hook"),
       ("section1","IWBI2026_Trivedi_01_CAD"),
       ("section2","IWBI2026_Trivedi_02_EraII"),
       ("section3","IWBI2026_Trivedi_03_risk"),
       ("section4","IWBI2026_Trivedi_04_pathology"),
       ("section5","IWBI2026_Trivedi_05_multimodal"),
       ("section5b","IWBI2026_Trivedi_05b_BAC_CVD"),
       ("section6","IWBI2026_Trivedi_06_governance"),
       ("section6b","IWBI2026_Trivedi_06b_frontier"),
       ("section7","IWBI2026_Trivedi_07_close"),
       ("section_era3_combined","IWBI2026_Trivedi_EraIII_combined")]

def render(pw, name):
    b=pw.chromium.launch(executable_path=CHROME, args=["--no-sandbox","--font-render-hinting=none"])
    ctx=b.new_context(viewport={"width":1300,"height":820}, device_scale_factor=2)
    pg=ctx.new_page()
    pg.goto("file://%s/%s.html"%(SLIDES,name))
    pg.wait_for_timeout(800)
    n=pg.eval_on_selector_all(".slide","els=>els.length")
    items=[]
    for i in range(n):
        note=pg.evaluate("""(idx)=>{const s=[...document.querySelectorAll('.slide')];s.forEach((el,k)=>el.classList.toggle('active',k===idx));return s[idx].getAttribute('data-note')||'';}""", i)
        pg.wait_for_timeout(240)
        p="%s/%s_%02d.png"%(TMP,name,i)
        fr=pg.query_selector("#frame")
        fr.screenshot(path=p)
        items.append((p,note))
    b.close()
    return items

def build(items, out):
    prs=Presentation(); prs.slide_width=Inches(13.333); prs.slide_height=Inches(7.5)
    blank=prs.slide_layouts[6]
    for png,note in items:
        s=prs.slides.add_slide(blank)
        s.shapes.add_picture(png,0,0,width=prs.slide_width,height=prs.slide_height)
        s.notes_slide.notes_text_frame.text=note or ""
    prs.save(out)

with sync_playwright() as pw:
    for name,outbase in DECKS:
        if not os.path.exists("%s/%s.html"%(SLIDES,name)):
            print("skip (missing):",name); continue
        try:
            items=render(pw,name)
            build(items, "%s/%s.pptx"%(OUT,outbase))
            print("OK %-26s %2d slides -> %s.pptx"%(name,len(items),outbase))
        except Exception as e:
            print("FAIL",name,repr(e)[:200])
