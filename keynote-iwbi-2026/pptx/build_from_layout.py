#!/usr/bin/env python3
"""Build a PPTX from extracted HTML layout JSON — exact positions, font sizes,
colors, and figures. Adds the gradient/motif background; rasterizes SVGs
transparently; embeds base64 images at their exact box."""
import sys, os, re, json, base64, tempfile, hashlib
import cairosvg
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR, MSO_AUTO_SIZE
from pptx.enum.shapes import MSO_SHAPE

SW, SH = 13.333, 7.5
BG = RGBColor(0x0F, 0x14, 0x1A)
TMP = tempfile.mkdtemp()
BGIMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "bg_motif.png")
ALIGN = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT,
         "justify": PP_ALIGN.JUSTIFY, "start": PP_ALIGN.LEFT, "end": PP_ALIGN.RIGHT}

def rgb(css):
    m = re.match(r'rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)', css or "")
    if not m: return RGBColor(0xEC, 0xEF, 0xF3)
    return RGBColor(int(float(m.group(1))), int(float(m.group(2))), int(float(m.group(3))))

def svg_png(svg):
    for a, b in [('viewbox=', 'viewBox='), ('radialgradient', 'radialGradient'), ('lineargradient', 'linearGradient'),
                 ('gradientunits', 'gradientUnits'), ('gradienttransform', 'gradientTransform'), ('markerwidth', 'markerWidth'),
                 ('markerheight', 'markerHeight'), ('refx', 'refX'), ('refy', 'refY'), ('markerunits', 'markerUnits'),
                 ('preserveaspectratio', 'preserveAspectRatio')]:
        svg = svg.replace(a, b)
    svg = re.sub(r'(<svg\b[^>]*?)\s+style="[^"]*"', r'\1', svg, count=1)
    if 'xmlns' not in svg[:80]: svg = svg.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"', 1)
    m = re.search(r'viewBox="([\d.\- ]+)"', svg, re.I); vb = m.group(1).split() if m else ['0', '0', '600', '400']
    key = hashlib.md5(svg.encode()).hexdigest()[:10]; p = os.path.join(TMP, 's_%s.png' % key)
    cairosvg.svg2png(bytestring=svg.encode(), write_to=p, output_width=int(float(vb[2]) * 3), output_height=int(float(vb[3]) * 3), background_color=None)
    return p

def data_png(src):
    m = re.match(r'data:image/(\w+);base64,(.*)$', src, re.S)
    if not m: return None
    ext = 'png' if m.group(1) == 'png' else 'jpg'; raw = base64.b64decode(m.group(2))
    key = hashlib.md5(raw).hexdigest()[:10]; p = os.path.join(TMP, 'i_%s.%s' % (key, ext)); open(p, 'wb').write(raw)
    return p

def build(name):
    data = json.load(open("/home/user/Mammo/keynote-iwbi-2026/pptx/layout/%s.json" % name))
    prs = Presentation(); prs.slide_width = Inches(SW); prs.slide_height = Inches(SH)
    blank = prs.slide_layouts[6]
    nvis = 0
    for slide in data:
        if slide["hidden"]: continue
        nvis += 1
        s = prs.slides.add_slide(blank)
        s.background.fill.solid(); s.background.fill.fore_color.rgb = BG
        if os.path.exists(BGIMG):
            s.shapes.add_picture(BGIMG, 0, 0, Inches(SW), Inches(SH))
        # amber accent bar (HTML ::after, 2px)
        bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(6), Pt(3))
        bar.fill.solid(); bar.fill.fore_color.rgb = RGBColor(0xE7, 0xAC, 0x51); bar.line.fill.background(); bar.shadow.inherit = False
        for u in slide["units"]:
            r = u["rect"]; x = r["x"] * SW; y = r["y"] * SH; w = max(0.1, r["w"] * SW); h = max(0.05, r["h"] * SH)
            if u["type"] == "svg":
                try:
                    p = svg_png(u["html"]); s.shapes.add_picture(p, Inches(x), Inches(y), Inches(w), Inches(h))
                except Exception as e: print("svg fail", e)
            elif u["type"] == "img":
                p = data_png(u["src"])
                if p: s.shapes.add_picture(p, Inches(x), Inches(y), Inches(w), Inches(h))
            elif u["type"] == "text":
                pt = u["fontPx"] * 960.0 / slide["W"]
                lh = (u.get("linePx") or u["fontPx"] * 1.2) / u["fontPx"]
                pxin = SW / slide["W"]          # inches per CSS px
                bordL = u.get("bordL", 0) * pxin
                padL = u.get("padL", 0) * pxin
                if bordL > 0.005:               # left-border accent (callouts / turn-lines)
                    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(max(bordL, 0.03)), Inches(h))
                    bar.fill.solid(); bar.fill.fore_color.rgb = rgb(u.get("bordC")); bar.line.fill.background(); bar.shadow.inherit = False
                x += bordL + padL; w = max(0.1, w - bordL - padL)
                # single-line elements: widen so a slightly different font can't force a wrap
                line_in = pt * lh / 72.0
                nlines = max(1, round(h / line_in)) if line_in > 0 else 1
                if pt >= 22:                       # titles: give full width so they stay on one line
                    w = SW - x - 0.2
                elif nlines <= 1:                  # other single-line text: small headroom
                    w = min(SW - x - 0.2, w + 1.0)
                tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h + 0.05)); tf = tb.text_frame
                tf.word_wrap = True
                try: tf.auto_size = MSO_AUTO_SIZE.NONE
                except Exception: pass
                tf.margin_left = 0; tf.margin_right = 0; tf.margin_top = 0; tf.margin_bottom = 0
                tt = u.get("transform", "none")
                paras = [[]]
                for rn in u["runs"]:
                    txt = rn["t"]
                    if tt == "uppercase": txt = txt.upper()
                    elif tt == "lowercase": txt = txt.lower()
                    rn = dict(rn); rn["t"] = txt
                    seg = rn["t"].split("\n")
                    for j, piece in enumerate(seg):
                        if j > 0: paras.append([])
                        if piece: paras[-1].append((piece, rn))
                first = True
                for pr in paras:
                    p = tf.paragraphs[0] if first else tf.add_paragraph(); first = False
                    p.alignment = ALIGN.get(u["align"], PP_ALIGN.LEFT)
                    try: p.line_spacing = lh
                    except Exception: pass
                    for piece, rn in pr:
                        run = p.add_run(); run.text = piece; f = run.font
                        f.size = Pt(pt); f.bold = rn.get("bold", False); f.italic = rn.get("italic", False)
                        f.name = "Consolas" if "mono" in (u.get("family", "").lower()) or "consol" in u.get("family", "").lower() or "sfmono" in u.get("family", "").lower() else "Arial"
                        f.color.rgb = rgb(rn.get("color"))
    out = "/home/user/Mammo/keynote-iwbi-2026/pptx/sections/%s" % MAP[name]
    os.makedirs(os.path.dirname(out), exist_ok=True); prs.save(out)
    print("%s -> %s (%d slides)" % (name, MAP[name], nvis))

MAP = {"hook": "IWBI2026_Trivedi_00_hook.pptx", "section1": "IWBI2026_Trivedi_01_CAD.pptx",
       "section2": "IWBI2026_Trivedi_02_EraII.pptx", "section2b": "IWBI2026_Trivedi_02b_EraII_2026.pptx",
       "section5alternate": "IWBI2026_Trivedi_05_EraIII_prevention.pptx", "section6b": "IWBI2026_Trivedi_06b_frontier.pptx",
       "section_adoption": "IWBI2026_Trivedi_06c_adoption.pptx", "section7": "IWBI2026_Trivedi_07_close.pptx"}

if __name__ == "__main__":
    build(sys.argv[1])
