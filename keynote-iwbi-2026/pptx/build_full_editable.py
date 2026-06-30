#!/usr/bin/env python3
"""Build ONE combined, editable PPTX of the full keynote, straight from the
current section HTML (the source of truth). Every slide => real text boxes
(one paragraph per block element, amber/cyan emphasis preserved), native
tables, and SVG/`<img>` figures rasterized + embedded. Hidden slides
(data-hide / 'hide in final' / ⚑) are dropped so the file mirrors the
59-slide combined deck. Speaker notes are carried on every slide.

13.333 x 7.5 in (16:9). Output: IWBI2026_Trivedi_FULL_keynote.pptx
"""
import os, re, base64, tempfile, hashlib
from bs4 import BeautifulSoup, NavigableString, Tag
import cairosvg
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR, MSO_AUTO_SIZE
from pptx.enum.shapes import MSO_SHAPE

BG   = RGBColor(0x0F,0x14,0x1A); PANEL=RGBColor(0x06,0x09,0x0C); CARD=RGBColor(0x1B,0x23,0x2D)
INK  = RGBColor(0xEC,0xEF,0xF3); INK2 = RGBColor(0x9D,0xA9,0xB5); INK3 = RGBColor(0x5C,0x69,0x75)
AMBER= RGBColor(0xE7,0xAC,0x51); AMBERDP=RGBColor(0xC9,0x8A,0x2E); CYAN = RGBColor(0x5F,0xB7,0xC9)
WARN = RGBColor(0xD9,0x78,0x5B); VIOLET=RGBColor(0x8C,0x7B,0xD8); LINE=RGBColor(0x2A,0x33,0x3E)
WHITE= RGBColor(0xFF,0xFF,0xFF); BLACK=RGBColor(0x00,0x00,0x00)
SANS, MONO = "Arial", "Consolas"
SW, SH = 13.333, 7.5
TMP = tempfile.mkdtemp()
SLIDES = "/home/user/Mammo/keynote-iwbi-2026/slides"

SECTIONS = ["hook","section1","section2","section2b",
            "section5alternate","section6b","section_adoption","section7"]

INLINE = {'b','strong','span','br','i','em','a','sub','sup','code','u','small'}
HIDE_RE = re.compile(r'hide in final|⚑')

# ---------- inline runs with colour ----------
def _colour(node):
    if not isinstance(node, Tag): return None
    st = node.get('style','') or ''
    cls = node.get('class',[]) or []
    if 'am' in cls or 'q' in cls or 'var(--amber)' in st or '#E7AC51' in st or '#C98A2E' in st: return AMBER
    if 'var(--cyan)' in st or '#5FB7C9' in st: return CYAN
    if 'var(--warn)' in st or '#D9785B' in st: return WARN
    if '#8C7BD8' in st: return VIOLET
    return None

def runs_of(el):
    out=[]
    def walk(node, bold, color):
        if isinstance(node, NavigableString):
            t=re.sub(r'\s+',' ',str(node))
            if t: out.append([t,bold,color])
            return
        if node.name in ('svg','img','table'): return
        if node.name=='br': out.append(['\n',bold,color]); return
        b = bold or node.name in ('b','strong')
        c = _colour(node) or color
        for ch in node.children: walk(ch,b,c)
    walk(el,False,None)
    # merge whitespace artifacts
    res=[r for r in out if r[0]]
    return res

# ---------- figures ----------
def svg_png(svg_tag, bg='#0F141A'):
    s=str(svg_tag)
    # html.parser lowercases attribute names; cairosvg is case-sensitive — restore camelCase
    for a,b in [('viewbox=','viewBox='),('radialgradient','radialGradient'),
                ('lineargradient','linearGradient'),('gradientunits','gradientUnits'),
                ('gradienttransform','gradientTransform'),('markerwidth','markerWidth'),
                ('markerheight','markerHeight'),('refx','refX'),('refy','refY'),
                ('markerunits','markerUnits'),('preserveaspectratio','preserveAspectRatio'),
                ('stroke-dasharray','stroke-dasharray')]:
        s=s.replace(a,b)
    s=re.sub(r'(<svg\b[^>]*?)\s+style="[^"]*"', r'\1', s, count=1)
    if 'xmlns' not in s[:80]: s=s.replace('<svg','<svg xmlns="http://www.w3.org/2000/svg"',1)
    m=re.search(r'viewBox="([\d.\- ]+)"', s, re.I)
    vb=m.group(1).split() if m else ['0','0','560','320']
    ar=float(vb[2])/float(vb[3])
    key=hashlib.md5(s.encode()).hexdigest()[:10]; p=os.path.join(TMP,'svg_%s.png'%key)
    cairosvg.svg2png(bytestring=s.encode(), write_to=p,
                     output_width=int(float(vb[2])*3), output_height=int(float(vb[3])*3),
                     background_color=bg)
    return p, ar

def data_png(img):
    m=re.match(r'data:image/(\w+);base64,(.*)$', img.get('src','') ,re.S)
    if not m: return None,1.4
    ext='png' if m.group(1)=='png' else 'jpg'; raw=base64.b64decode(m.group(2))
    key=hashlib.md5(raw).hexdigest()[:10]; p=os.path.join(TMP,'img_%s.%s'%(key,ext)); open(p,'wb').write(raw)
    try: im=Image.open(p); ar=im.width/im.height
    except Exception: ar=1.4
    return p, ar

def place_fig(s, path, ar, x, y, maxw, maxh, card=None):
    w=maxw; h=w/ar
    if h>maxh: h=maxh; w=h*ar
    px=x+(maxw-w)/2; py=y+(maxh-h)/2
    if card:
        pad=0.09
        cd=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,Inches(px-pad),Inches(py-pad),Inches(w+2*pad),Inches(h+2*pad))
        cd.fill.solid(); cd.fill.fore_color.rgb=WHITE if card=='white' else BLACK
        cd.line.color.rgb=AMBERDP; cd.line.width=Pt(0.75); cd.shadow.inherit=False
    s.shapes.add_picture(path,Inches(px),Inches(py),Inches(w),Inches(h))

# ---------- block-level text extraction ----------
SKIP_CLASS = re.compile(r'right|viewport|vp-|figure|eyebrow|dots|controls|btns')

def has_block_child(el):
    for c in el.children:
        if isinstance(c, Tag) and c.name not in INLINE and c.name not in ('svg','img','table'):
            return True
    return False

def classify(el):
    cls=' '.join(el.get('class',[]) or [])
    st=el.get('style','') or ''
    if 'cite' in cls or 'attr' in cls: return 'cite'
    if 'eyebrow' in cls: return 'skip'
    if 'turn-line' in cls or 'spine-hand' in cls: return 'emph'
    if 'chip' in cls and 'chips' not in cls: return 'chip'
    if el.name in ('h1','h2'): return 'title'
    # big stat number: bold + large font in inline style
    if re.search(r'font-weight:\s*7', st) and re.search(r'font-size:clamp\((?:1\.[2-9]|[2-9])', st): return 'stat'
    return 'body'

def gather(root, skip_first_title=True):
    """Return ordered list of (kind, runs). Recurses into block containers."""
    out=[]
    seen_title=[False]
    def rec(el):
        for c in el.children:
            if not isinstance(c, Tag): continue
            if c.name in ('script','style','svg','img'): continue
            cls=c.get('class',[]) or []
            if any(SKIP_CLASS.search(x) for x in cls): continue
            if c.name=='table':
                out.append(('table', c)); continue
            if has_block_child(c):
                rec(c); continue
            kind=classify(c)
            if kind=='skip': continue
            r=runs_of(c)
            if not r: continue
            if kind=='title':
                if skip_first_title and not seen_title[0]:
                    seen_title[0]=True; out.append(('maintitle', r)); continue
                kind='body'
            out.append((kind, r))
    rec(root)
    return out

def add_text(s, x,y,w,h, runs, size, default=INK2, align=PP_ALIGN.LEFT, leading=1.14, space=4, font=SANS):
    tb=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h)); tf=tb.text_frame
    tf.word_wrap=True
    try: tf.auto_size=MSO_AUTO_SIZE.NONE
    except Exception: pass
    tf.margin_left=Pt(2); tf.margin_right=Pt(2); tf.margin_top=Pt(1); tf.margin_bottom=Pt(1)
    # support \n within a run-list as paragraph breaks
    paras=[[]]
    for t,b,c in runs:
        seg=t.split('\n')
        for i,piece in enumerate(seg):
            if i>0: paras.append([])
            if piece: paras[-1].append((piece,b,c))
    first=True
    for pr in paras:
        p=tf.paragraphs[0] if first else tf.add_paragraph(); first=False
        p.alignment=align; p.line_spacing=leading; p.space_after=Pt(space)
        for t,b,c in pr:
            r=p.add_run(); r.text=t; f=r.font; f.size=Pt(size); f.name=font
            f.bold=bool(b); f.color.rgb=(c if c is not None else (INK if b else default))
    return tb

def est_lines(runs, w_in, size):
    cpl=max(8,int(w_in/(size*0.0095))); txt=''.join(t for t,_,_ in runs)
    return max(1,-(-len(txt)//cpl)) + txt.count('\n')

def add_table(s,x,y,w,tbl):
    rows=[]
    for tr in tbl.find_all('tr'):
        cells=[c.get_text(' ',strip=True) for c in tr.find_all(['td','th'])]
        if cells: rows.append(cells)
    if not rows: return 0
    nr=len(rows); nc=max(len(r) for r in rows); h=0.34*nr
    g=s.shapes.add_table(nr,nc,Inches(x),Inches(y),Inches(w),Inches(h)).table
    for ri,row in enumerate(rows):
        for ci in range(nc):
            cell=g.cell(ri,ci); cell.margin_left=Pt(4); cell.margin_right=Pt(4); cell.margin_top=Pt(1); cell.margin_bottom=Pt(1)
            cell.fill.solid(); cell.fill.fore_color.rgb=CARD if ri==0 else BG
            cell.text=row[ci] if ci<len(row) else ''
            for p in cell.text_frame.paragraphs:
                p.alignment=PP_ALIGN.LEFT if ci==0 else PP_ALIGN.CENTER
                for r in p.runs:
                    r.font.size=Pt(10); r.font.name=SANS; r.font.bold=(ri==0)
                    r.font.color.rgb=AMBER if ri==0 else (INK if ci==0 else INK2)
    return h

# ---------- slide builder ----------
def build_slide(prs, sec):
    s=prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid(); s.background.fill.fore_color.rgb=BG
    bar=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,0,0,Inches(SW),Pt(3))
    bar.fill.solid(); bar.fill.fore_color.rgb=AMBER; bar.line.fill.background(); bar.shadow.inherit=False
    s.notes_slide.notes_text_frame.text=sec.get('data-note','') or ''
    classes=sec.get('class',[]) or []

    eb=sec.select_one('.eyebrow')
    if eb:
        add_text(s,0.6,0.34,9,0.4,[(eb.get_text(' ',strip=True).upper(),True,AMBER)],11,default=AMBER,font=MONO)

    # figure (right side) — first svg or data-img not inside skip container
    fig=None; ar=1.4; card=None
    for img in sec.find_all('img'):
        if img.get('src','').startswith('data:'): fig,ar=data_png(img); card='white'; break
    if not fig:
        svg=sec.find('svg')
        if svg is not None and (svg.get('viewbox') or svg.get('viewBox')): fig,ar=svg_png(svg)

    root = sec.select_one('.left') or sec.select_one('.dv-copy') or sec.select_one('.discl-grid') or sec
    blocks=gather(root)
    # title may live outside .left for some layouts
    if not any(k=='maintitle' for k,_ in blocks):
        h=sec.find(['h1','h2'])
        if h: blocks.insert(0,('maintitle',runs_of(h)))

    is_divider='s-divider' in classes or 's-turn' in classes
    two_col = fig is not None and ('s-caveat' in classes or 's-fig' in classes or 's-divider' in classes
                                   or sec.find('svg') is not None or card=='white')
    textw = 6.7 if two_col else 11.9
    tx=0.6; cy=0.95

    for kind,payload in blocks:
        if kind=='table':
            cy+=add_table(s,tx,min(cy,6.6),min(textw,7.6),payload)+0.18; continue
        runs=payload
        if kind=='maintitle':
            sz=32 if is_divider else 26
            ln=est_lines(runs,textw,sz); add_text(s,tx,cy,textw,ln*sz*1.2/72+0.25,runs,sz,default=INK,leading=1.06)
            cy+=ln*sz*1.2/72+0.22
        elif kind=='stat':
            ln=est_lines(runs,textw,22); add_text(s,tx,cy,textw,ln*22*1.2/72+0.12,runs,22,default=AMBER)
            cy+=ln*22*1.2/72+0.12
        elif kind=='emph':
            ln=est_lines(runs,textw,17); add_text(s,tx,cy,textw,ln*17*1.3/72+0.12,runs,17,default=INK,leading=1.3)
            cy+=ln*17*1.3/72+0.14
        elif kind=='chip':
            ln=est_lines(runs,textw,12); add_text(s,tx,cy,textw,ln*12*1.3/72+0.08,[('▸ ',False,CYAN)]+runs,12,default=INK)
            cy+=ln*12*1.3/72+0.08
        elif kind=='cite':
            ln=est_lines(runs,textw,10); add_text(s,tx,cy,textw,ln*10*1.3/72+0.06,runs,10,default=INK3,font=MONO)
            cy+=ln*10*1.3/72+0.08
        else:
            sz=16 if is_divider else 14
            ln=est_lines(runs,textw,sz); add_text(s,tx,cy,textw,ln*sz*1.25/72+0.12,runs,sz,default=INK2,leading=1.25)
            cy+=ln*sz*1.25/72+0.13

    if fig:
        if two_col: place_fig(s,fig,ar,7.5,1.0,5.25,5.6,card=card)
        else:       place_fig(s,fig,ar,3.0,min(cy+0.15,3.2),7.3,min(6.9-cy,3.6) if cy<3 else 3.4,card=card)
    return s

def main():
    prs=Presentation(); prs.slide_width=Inches(SW); prs.slide_height=Inches(SH)
    total=0; per=[]
    for name in SECTIONS:
        html=open(os.path.join(SLIDES,name+'.html')).read()
        soup=BeautifulSoup(html,'html.parser')
        n=0
        for sec in soup.select('section.slide'):
            blob=str(sec)
            if HIDE_RE.search(blob) or sec.get('data-hide'):
                continue
            build_slide(prs, sec); n+=1
        per.append((name,n)); total+=n
        print("%-22s %2d slides"%(name,n))
    out="/home/user/Mammo/keynote-iwbi-2026/pptx/IWBI2026_Trivedi_FULL_keynote.pptx"
    prs.save(out)
    print("TOTAL", total, "->", out)

if __name__=='__main__':
    main()
