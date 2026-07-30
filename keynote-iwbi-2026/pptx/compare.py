#!/usr/bin/env python3
"""Self-check tool: render the HTML deck section and its generated .pptx and
stack them SIDE BY SIDE per slide (HTML left, PPTX right) into one image, so a
single view shows every mismatch in font size, spacing, color, and position.

Usage:  python3 compare.py <section> [out.png]
        section = slides/<section>.html (hidden slides dropped to match the pptx)

Workflow: build the section's pptx, run this, VIEW the output, fix the build to
match the HTML, rebuild, re-run — repeat until the two columns are identical.
No need to ask the user for screenshots; drive the loop from this image.
"""
import sys, os, pathlib
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright
import render_pptx
from deckpaths import SLIDES, SECTIONS

CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
MAP = {"hook": "IWBI2026_Trivedi_00_hook.pptx", "section1": "IWBI2026_Trivedi_01_CAD.pptx",
       "section2": "IWBI2026_Trivedi_02_EraII.pptx", "section2b": "IWBI2026_Trivedi_02b_EraII_2026.pptx",
       "section5alternate": "IWBI2026_Trivedi_05_EraIII_prevention.pptx", "section6b": "IWBI2026_Trivedi_06b_frontier.pptx",
       "section_adoption": "IWBI2026_Trivedi_06c_adoption.pptx", "section7": "IWBI2026_Trivedi_07_close.pptx",
       "slide_imagenet_2012": "IWBI2026_Trivedi_ImageNet2012.pptx",
       "slide_sbi_survey": "IWBI2026_Trivedi_SBI_survey.pptx"}

def html_slides(name):
    """Per-slide HTML frame screenshots, hidden slides dropped (match the pptx)."""
    url = "file://" + str((SLIDES / (name + ".html")).resolve())
    out = []
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME, args=["--no-sandbox"])
        pg = b.new_page(viewport={"width": 1280, "height": 760}, device_scale_factor=2)
        pg.goto(url); pg.wait_for_timeout(600)
        n = pg.eval_on_selector_all(".slide", "e=>e.length")
        for i in range(n):
            hidden = pg.evaluate("""(i)=>{const s=[...document.querySelectorAll('.slide')];
                s.forEach((el,k)=>el.classList.toggle('active',k===i));
                const a=s[i]; return (a.outerHTML.match(/hide in final|⚑/)!=null)||a.hasAttribute('data-hide');}""", i)
            if hidden: continue
            pg.wait_for_timeout(150)
            png = "/tmp/_cmp_%s_%02d.png" % (name, i)
            pg.query_selector("#frame").screenshot(path=png)
            out.append(Image.open(png).convert("RGB"))
        b.close()
    return out

def main(name, out=None):
    out = out or "compare_%s.png" % name
    pptx = str(SECTIONS / MAP[name])
    html = html_slides(name)
    ppt = render_pptx.render_slides(pptx)
    n = max(len(html), len(ppt))
    cw = 760; ch = int(cw * 9 / 16); pad = 10; lab = 22
    sheet = Image.new("RGB", (2 * cw + 3 * pad, n * (ch + lab + pad) + pad), (8, 8, 10))
    d = ImageDraw.Draw(sheet)
    try: f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except Exception: f = ImageFont.load_default()
    for i in range(n):
        y = pad + i * (ch + lab + pad)
        d.text((pad, y), "slide %d — HTML (reference)" % i, fill=(150, 170, 140), font=f)
        d.text((cw + 2 * pad, y), "slide %d — PPTX (proxy; Arial≈narrower in real PPT)" % i, fill=(180, 150, 90), font=f)
        if i < len(html): sheet.paste(html[i].resize((cw, ch)), (pad, y + lab))
        if i < len(ppt): sheet.paste(ppt[i].resize((cw, ch)), (cw + 2 * pad, y + lab))
    sheet.save(out)
    print("wrote %s — HTML %d slides vs PPTX %d slides" % (out, len(html), len(ppt)))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
