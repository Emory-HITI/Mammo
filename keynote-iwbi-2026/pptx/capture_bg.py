from playwright.sync_api import sync_playwright
import pathlib
url="file://"+str(pathlib.Path("/home/user/Mammo/keynote-iwbi-2026/slides/section1.html").resolve())
with sync_playwright() as p:
    b=p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome",args=["--no-sandbox"])
    pg=b.new_page(viewport={"width":1320,"height":860}, device_scale_factor=2)
    pg.goto(url); pg.wait_for_timeout(700)
    # strip everything but the gradient + motif: hide slides, accent hairline, rounded corners
    pg.add_style_tag(content=".slide{opacity:0!important;visibility:hidden!important}"
                     " .frame{border-radius:0!important;border:0!important;box-shadow:none!important}"
                     " .frame::after{display:none!important}")
    pg.wait_for_timeout(500)
    fr=pg.query_selector("#frame")
    fr.screenshot(path="/home/user/Mammo/keynote-iwbi-2026/pptx/assets/bg_motif.png")
    b.close()
from PIL import Image
im=Image.open("/home/user/Mammo/keynote-iwbi-2026/pptx/assets/bg_motif.png")
print("bg size:", im.size)
