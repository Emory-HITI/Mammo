#!/usr/bin/env python3
"""QA: render every slide of an HTML deck section to a 2-col contact sheet PNG.
Usage:  python3 render_html.py <section-name> [out_dir]
        (section-name = file in ../slides without .html; out_dir defaults to cwd)
Also writes per-slide frames h_<name>_NN.png for full-size inspection."""
from playwright.sync_api import sync_playwright
import sys, os, pathlib
from PIL import Image

CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
name = sys.argv[1]
out_dir = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()
SLIDES = pathlib.Path(__file__).resolve().parent.parent / "slides"
url = "file://" + str((SLIDES / (name + ".html")).resolve())
os.makedirs(out_dir, exist_ok=True)
frames = []
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=CHROME, args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": 1280, "height": 760}, device_scale_factor=2)
    pg.goto(url); pg.wait_for_timeout(600)
    n = pg.eval_on_selector_all(".slide", "e=>e.length")
    for i in range(n):
        pg.evaluate("(i)=>{const s=[...document.querySelectorAll('.slide')];s.forEach((el,k)=>el.classList.toggle('active',k===i));}", i)
        pg.wait_for_timeout(180)
        pth = os.path.join(out_dir, "h_%s_%02d.png" % (name, i))
        pg.query_selector("#frame").screenshot(path=pth); frames.append(pth)
    b.close()
ims = [Image.open(o) for o in frames]
w, h = ims[0].size; pad = 8; rows = (len(ims) + 1) // 2
sheet = Image.new("RGB", (2 * (w // 2 + pad) + pad, rows * (h // 2 + pad) + pad), (0, 0, 0))
for i, im in enumerate(ims):
    t = im.resize((w // 2, h // 2)); sheet.paste(t, (pad + (i % 2) * (w // 2 + pad), pad + (i // 2) * (h // 2 + pad)))
dest = os.path.join(out_dir, "HTML_%s.png" % name)
sheet.save(dest)
print("rendered", name, n, "slides ->", dest)
