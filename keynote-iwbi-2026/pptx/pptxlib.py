#!/usr/bin/env python3
"""Shared helpers for building IWBI-2026 editable PPTX decks (dark clinical theme).
One text box per blurb; native tables; SVG charts rasterized + embedded; base64 <img> embedded.
13.333 x 7.5 in (16:9)."""
import os, re, base64, tempfile, hashlib
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from bs4 import BeautifulSoup, NavigableString
import cairosvg
from PIL import Image

BG    = RGBColor(0x0F,0x14,0x1A); PANEL = RGBColor(0x06,0x09,0x0C)
INK   = RGBColor(0xEC,0xEF,0xF3); INK2 = RGBColor(0x9D,0xA9,0xB5); INK3 = RGBColor(0x5C,0x69,0x75)
AMBER = RGBColor(0xE7,0xAC,0x51); AMBERDP = RGBColor(0xC9,0x8A,0x2E); CYAN = RGBColor(0x5F,0xB7,0xC9)
WARN  = RGBColor(0xD9,0x78,0x5B); CARD = RGBColor(0x1B,0x23,0x2D); LINE = RGBColor(0x2A,0x33,0x3D)
WHITE = RGBColor(0xFF,0xFF,0xFF); BLACK = RGBColor(0x00,0x00,0x00)
SANS, MONO = "Arial", "Consolas"
SW, SH = 13.333, 7.5
TMP = tempfile.mkdtemp()

def new_prs():
    prs = Presentation(); prs.slide_width = Inches(SW); prs.slide_height = Inches(SH)
    return prs

def slide(prs, hidden=False, barw=6.0):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid(); s.background.fill.fore_color.rgb = BG
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(barw), Pt(3))
    bar.fill.solid(); bar.fill.fore_color.rgb = AMBER; bar.line.fill.background(); bar.shadow.inherit = False
    if hidden:
        try: s._element.set('show','0')
        except Exception: pass
    return s

def box(s, l, t, w, h, anchor=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tb, tf

def para(tf, first=False, align=PP_ALIGN.LEFT, before=0, after=0, line=1.0):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    if before: p.space_before = Pt(before)
    if after:  p.space_after = Pt(after)
    try: p.line_spacing = line
    except Exception: pass
    return p

def run(p, text, size, color, bold=False, font=SANS, italic=False):
    r = p.add_run(); r.text = text; f = r.font
    f.size = Pt(size); f.bold = bold; f.italic = italic; f.name = font; f.color.rgb = color
    return r

def eyebrow(s, label):
    _, tf = box(s, 0.85, 0.55, 9, 0.35)
    run(para(tf, True), label.upper(), 11, AMBER, bold=True, font=MONO)

def card(s, l, t, w, h, fill=CARD, edge=AMBER, edge_w=1.0):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(l), Inches(t), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = edge; sh.line.width = Pt(edge_w); sh.shadow.inherit = False
    return sh

def rect(s, l, t, w, h, color):
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(l), Inches(t), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = color; sh.line.fill.background(); sh.shadow.inherit = False
    return sh

def chip(s, l, t, w, text):
    """cyan-left-border rounded chip with text; returns height used."""
    c = card(s, l, t, w, 0.42, fill=CARD, edge=LINE, edge_w=0.75)
    accent = rect(s, l, t, 0.04, 0.42, CYAN)
    tf=c.text_frame; tf.word_wrap=True; tf.vertical_anchor=MSO_ANCHOR.MIDDLE
    tf.margin_left=Inches(0.14); tf.margin_right=Inches(0.1)
    run(para(tf,True), text, 11, INK, font=MONO)
    return 0.42

def note(s, text):
    s.notes_slide.notes_text_frame.text = text or ""

def native_table(s, rows, l, t, w, col0=None, fs=12):
    nr=len(rows); nc=max(len(r) for r in rows)
    gf=s.shapes.add_table(nr,nc,Inches(l),Inches(t),Inches(w),Inches(0.36*nr))
    tbl=gf.table
    if col0:
        tbl.columns[0].width=Inches(col0)
        rest=(w-col0)/(nc-1)
        for c in range(1,nc): tbl.columns[c].width=Inches(rest)
    for ri,row in enumerate(rows):
        for ci in range(nc):
            cell=tbl.cell(ri,ci); cell.fill.solid(); cell.fill.fore_color.rgb=BG
            cell.vertical_anchor=MSO_ANCHOR.MIDDLE
            cell.margin_left=Inches(0.08); cell.margin_right=Inches(0.08); cell.margin_top=Pt(2); cell.margin_bottom=Pt(2)
            val=row[ci] if ci<len(row) else ""
            p=cell.text_frame.paragraphs[0]; p.alignment=PP_ALIGN.LEFT if ci==0 else PP_ALIGN.CENTER
            r=p.add_run(); r.text=val; f=r.font; f.size=Pt(fs); f.name=SANS
            f.bold=(ri==0); f.color.rgb=AMBER if ri==0 else (INK if ci==0 else INK2)
    return 0.36*nr

def _fit(ar, maxw, maxh):
    w=maxw; h=w/ar
    if h>maxh: h=maxh; w=h*ar
    return w,h

def embed_svg(s, svg_el_or_str, l, t, maxw, maxh, bg='#0F141A', center=True):
    svg=str(svg_el_or_str)
    if 'xmlns' not in svg[:80]:
        svg=svg.replace('<svg','<svg xmlns="http://www.w3.org/2000/svg"',1)
    m=re.search(r'viewBox="([\d.\- ]+)"', svg)
    if m:
        vb=m.group(1).split(); ar=float(vb[2])/float(vb[3])
    else: ar=1.6
    key=hashlib.md5(svg.encode()).hexdigest()[:10]
    p=os.path.join(TMP,'svg_%s.png'%key)
    cairosvg.svg2png(bytestring=svg.encode(), write_to=p,
                     output_width=int(float(vb[2])*3), output_height=int(float(vb[3])*3),
                     background_color=bg)
    w,h=_fit(ar,maxw,maxh)
    x=l+(maxw-w)/2 if center else l; y=t+(maxh-h)/2 if center else t
    s.shapes.add_picture(p, Inches(x), Inches(y), Inches(w), Inches(h))
    return p

def embed_img(s, datauri, l, t, maxw, maxh, card_bg=None, center=True):
    m=re.match(r'data:image/(\w+);base64,(.*)$', datauri, re.S)
    if not m: return None
    ext='png' if m.group(1)=='png' else 'jpg'
    raw=base64.b64decode(m.group(2))
    key=hashlib.md5(raw).hexdigest()[:10]
    p=os.path.join(TMP,'img_%s.%s'%(key,ext)); open(p,'wb').write(raw)
    try:
        im=Image.open(p); ar=im.width/im.height
    except Exception: ar=1.4
    w,h=_fit(ar,maxw,maxh)
    x=l+(maxw-w)/2 if center else l; y=t+(maxh-h)/2 if center else t
    if card_bg:
        pad=0.1
        cd=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,Inches(x-pad),Inches(y-pad),Inches(w+2*pad),Inches(h+2*pad))
        cd.fill.solid(); cd.fill.fore_color.rgb=WHITE if card_bg=='white' else BLACK
        cd.line.color.rgb=AMBERDP; cd.line.width=Pt(0.75); cd.shadow.inherit=False
    s.shapes.add_picture(p, Inches(x), Inches(y), Inches(w), Inches(h))
    return p

# ---------- HTML parsing ----------
def runs_of(el):
    out=[]
    def walk(node,bold,color):
        if isinstance(node,NavigableString):
            t=re.sub(r'\s+',' ',str(node))
            if t: out.append((t,bold,color))
            return
        b=bold or node.name in ('b','strong')
        col=color
        st=node.get('style','') if hasattr(node,'get') else ''
        if 'var(--amber)' in st or 'am' in (node.get('class',[]) if hasattr(node,'get') else []): col=AMBER
        if 'var(--cyan)' in st: col=CYAN
        if 'var(--warn)' in st: col=WARN
        for c in node.children: walk(c,b,col)
    walk(el,False,None)
    return [r for r in out if r[0].strip()]

def parse_slides(html_path):
    soup=BeautifulSoup(open(html_path).read(),'html.parser')
    brand=soup.select_one('.brand'); brand=brand.get_text(' ',strip=True) if brand else ''
    out=[]
    for sec in soup.select('section.slide'):
        out.append({'classes':sec.get('class',[]), 'note':sec.get('data-note',''),
                    'eyebrow':(sec.select_one('.eyebrow').get_text(' ',strip=True) if sec.select_one('.eyebrow') else ''),
                    'el':sec})
    return brand, out

def save(prs, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prs.save(path); print("saved", path, "slides:", len(prs.slides._sldIdLst))
