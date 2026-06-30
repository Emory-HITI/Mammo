#!/usr/bin/env python3
"""Editable PPTX, built per-section from the current section HTML (source of
truth), using the font sizes/spacing of the approved hand-tuned decks.

Typography (pt) matches the approved §2 deck:
  eyebrow 11 mono · divider/title hero 44 · s-turn 30 · section title 26
  body 14 · turn-line/emph 19 · stat 30 / caption 11 mono · chip 11 mono · cite 10 mono

Layout: eyebrow pinned top-left; the text column is measured and VERTICALLY
CENTERED (as the HTML flexbox does); figures (inline SVG rasterized, or base64
<img>) placed in the right column for two-column slides, else centered below.
Hidden slides (data-hide / 'hide in final' / ⚑) are dropped.

Emits one file per section into pptx/sections/ AND a concatenated full deck.
"""
import os, re, base64, tempfile, hashlib
from bs4 import BeautifulSoup, NavigableString, Tag
import cairosvg
from PIL import Image, ImageDraw, ImageFont
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
LX = 0.85                      # left margin
TWO_TW = 6.55                  # text width, two-column
ONE_TW = 11.6                  # text width, single column
FIG_X, FIG_W, FIG_MAXH = 7.65, 5.05, 5.5
TOP, BOT = 1.02, 7.18          # vertical band for the centred text column
TMP = tempfile.mkdtemp()
SLIDES = "/home/user/Mammo/keynote-iwbi-2026/slides"
SECTIONS = ["hook","section1","section2","section2b",
            "section5alternate","section6b","section_adoption","section7"]
NAMEMAP = {"hook":"00_hook","section1":"01_CAD","section2":"02_EraII","section2b":"02b_EraII_2026",
           "section5alternate":"05_EraIII_prevention","section6b":"06b_frontier",
           "section_adoption":"06c_adoption","section7":"07_close"}

INLINE = {'b','strong','span','br','i','em','a','sub','sup','code','u','small'}
HIDE_RE = re.compile(r'hide in final|⚑')
SKIP_CLASS = re.compile(r'right|viewport|vp-|figure|eyebrow|dots|controls|btns|speaker')

# sizes
SZ = dict(eyebrow=11, hero=44, turn=30, title=26, body=14, emph=19,
          stat=30, statcap=11, chip=11, cite=10)

# ---------- inline runs with colour ----------
def _colour(node):
    if not isinstance(node, Tag): return None
    st=node.get('style','') or ''; cls=node.get('class',[]) or []
    if 'am' in cls or 'q' in cls or 'var(--amber)' in st or '#E7AC51' in st or '#C98A2E' in st: return AMBER
    if 'var(--cyan)' in st or '#5FB7C9' in st: return CYAN
    if 'var(--warn)' in st or '#D9785B' in st: return WARN
    if '#8C7BD8' in st: return VIOLET
    return None

def runs_of(el):
    out=[]
    def walk(node,bold,color):
        if isinstance(node,NavigableString):
            t=re.sub(r'\s+',' ',str(node))
            if t: out.append([t,bold,color]); return
            return
        if node.name in ('svg','img','table'): return
        if node.name=='br': out.append(['\n',bold,color]); return
        b=bold or node.name in ('b','strong'); c=_colour(node) or color
        for ch in node.children: walk(ch,b,c)
    walk(el,False,None)
    return [r for r in out if r[0]]

# ---------- figures ----------
def svg_png(svg_tag, bg='#0F141A'):
    s=str(svg_tag)
    for a,b in [('viewbox=','viewBox='),('radialgradient','radialGradient'),('lineargradient','linearGradient'),
                ('gradientunits','gradientUnits'),('gradienttransform','gradientTransform'),
                ('markerwidth','markerWidth'),('markerheight','markerHeight'),('refx','refX'),('refy','refY'),
                ('markerunits','markerUnits'),('preserveaspectratio','preserveAspectRatio')]:
        s=s.replace(a,b)
    s=re.sub(r'(<svg\b[^>]*?)\s+style="[^"]*"', r'\1', s, count=1)
    if 'xmlns' not in s[:80]: s=s.replace('<svg','<svg xmlns="http://www.w3.org/2000/svg"',1)
    m=re.search(r'viewBox="([\d.\- ]+)"', s, re.I); vb=m.group(1).split() if m else ['0','0','560','320']
    ar=float(vb[2])/float(vb[3]); key=hashlib.md5(s.encode()).hexdigest()[:10]; p=os.path.join(TMP,'s_%s.png'%key)
    cairosvg.svg2png(bytestring=s.encode(), write_to=p, output_width=int(float(vb[2])*3),
                     output_height=int(float(vb[3])*3), background_color=bg)
    return p, ar

def data_png(img):
    m=re.match(r'data:image/(\w+);base64,(.*)$', img.get('src','') ,re.S)
    if not m: return None,1.4
    ext='png' if m.group(1)=='png' else 'jpg'; raw=base64.b64decode(m.group(2))
    key=hashlib.md5(raw).hexdigest()[:10]; p=os.path.join(TMP,'i_%s.%s'%(key,ext)); open(p,'wb').write(raw)
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

# ---------- block extraction ----------
def has_block_child(el):
    for c in el.children:
        if isinstance(c,Tag) and c.name not in INLINE and c.name not in ('svg','img','table'): return True
    return False

def classify(el):
    cls=' '.join(el.get('class',[]) or []); st=el.get('style','') or ''
    if 'cite' in cls or 'attr' in cls: return 'cite'
    if 'eyebrow' in cls: return 'skip'
    if 'turn-line' in cls or 'spine-hand' in cls: return 'emph'
    if ('chip' in cls and 'chips' not in cls) or 'dchip' in cls: return 'chip'
    if 'drole-k' in cls or 'dsupp-k' in cls: return 'rolek'
    if el.name in ('h1','h2'): return 'title'
    if re.search(r'font-weight:\s*7', st) and re.search(r'font-size:clamp\((?:1\.[2-9]|[2-9])', st): return 'stat'
    return 'body'

def _has(el,name): return name in (el.get('class',[]) or [])

def gather(root, skip_first_title=True):
    out=[]; seen=[False]
    def rec(el):
        for c in el.children:
            if not isinstance(c,Tag): continue
            if c.name in ('script','style','svg','img'): continue
            cls=c.get('class',[]) or []
            if any(SKIP_CLASS.search(x) for x in cls): continue
            if c.name=='table': out.append(('table',c)); continue
            # chip CONTAINER -> emit one chip per item (don't merge)
            if _has(c,'chips') or _has(c,'dchips'):
                for item in c.find_all(lambda t: isinstance(t,Tag) and (_has(t,'chip') or _has(t,'dchip')), recursive=False):
                    r=runs_of(item)
                    if r: out.append(('chip',r))
                continue
            if has_block_child(c): rec(c); continue
            kind=classify(c)
            if kind=='skip': continue
            r=runs_of(c)
            if not r: continue
            if kind=='title' and skip_first_title and not seen[0]:
                seen[0]=True; out.append(('maintitle',r)); continue
            if kind=='title': kind='body'
            out.append((kind,r))
    rec(root)
    return out

# chip rows: collect consecutive chips into grouped lines
def coalesce_chips(blocks):
    out=[]; buf=[]
    for k,p in blocks:
        if k=='chip': buf.append(p); continue
        if buf: out.append(('chips',buf)); buf=[]
        out.append((k,p))
    if buf: out.append(('chips',buf))
    return out

# ---------- text measurement (real font metrics) ----------
_MF={}; _MD=ImageDraw.Draw(Image.new("RGB",(8,8)))
def _font(size, bold=False, mono=False):
    key=(round(size),bold,mono)
    if key in _MF: return _MF[key]
    if mono: path="/usr/share/fonts/truetype/dejavu/DejaVuSansMono%s.ttf"%("-Bold" if bold else "")
    else:    path="/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf"%("-Bold" if bold else "")
    try: f=ImageFont.truetype(path, max(6,round(size)))
    except Exception:
        try: f=ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",max(6,round(size)))
        except Exception: f=ImageFont.load_default()
    _MF[key]=f; return f

def n_lines(runs, w_in, size, mono=False, bold=False):
    # font px == size pt; compare against width expressed in points (w_in*72)
    maxw=w_in*72*0.97
    f=_font(size, bold, mono); lines=0
    txt=''.join(t for t,_,_ in runs)
    for hard in txt.split('\n'):
        words=hard.split(' '); cur=''
        for wd in words:
            t=(cur+' '+wd).strip()
            if _MD.textlength(t,font=f)<=maxw or not cur: cur=t
            else: lines+=1; cur=wd
        lines+=1
    return max(1,lines)

def blk_height(kind, runs, w):
    if kind=='maintitle': sz=SZ['_titlesize']; lead=1.12; pad=0.18; nl=n_lines(runs,w,sz,bold=True)
    elif kind=='stat': sz=SZ['stat']; lead=1.14; pad=0.10; nl=n_lines(runs,w,sz,bold=True)
    elif kind=='emph': sz=SZ['emph']; lead=1.3; pad=0.16; nl=n_lines(runs,w,sz,bold=True)
    elif kind=='cite': sz=SZ['cite']; lead=1.3; pad=0.10; nl=n_lines(runs,w,sz,mono=True)
    elif kind=='rolek': sz=SZ['eyebrow']; lead=1.2; pad=0.06; nl=n_lines(runs,w,sz,mono=True)
    elif kind=='chips': return 0.40*_chip_rows(runs,w)+0.12
    else: sz=SZ['body']; lead=1.28; pad=0.13; nl=n_lines(runs,w,sz)
    return nl*sz*lead/72 + pad

def runs_size(kind, runs):
    # title size depends on layout, set later; default section title
    return SZ.get('_titlesize', SZ['title'])

def chip_w(ch):
    txt=''.join(t for t,_,_ in ch)
    return _MD.textlength(txt,font=_font(SZ['chip'],False,True))/72 + 0.34

def _chip_rows(chips, w):
    x=0.0; rows=1
    for ch in chips:
        cw=chip_w(ch)
        if x+cw>w and x>0: rows+=1; x=cw+0.10
        else: x+=cw+0.10
    return rows

# ---------- adders ----------
def textbox(s,x,y,w,h,runs,size,default=INK2,align=PP_ALIGN.LEFT,leading=1.28,space=4,font=SANS):
    tb=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h)); tf=tb.text_frame
    tf.word_wrap=True
    try: tf.auto_size=MSO_AUTO_SIZE.NONE
    except Exception: pass
    tf.margin_left=Pt(2); tf.margin_right=Pt(2); tf.margin_top=Pt(1); tf.margin_bottom=Pt(1)
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

def chiprow(s,x,y,w,chips):
    """render chips as small rounded cards; returns height used."""
    cx=x; cy=y; rowh=0.40; gap=0.10
    for ch in chips:
        cw=chip_w(ch)
        if cx+cw>x+w and cx>x: cx=x; cy+=rowh+0.08
        c=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,Inches(cx),Inches(cy),Inches(cw),Inches(rowh))
        c.fill.solid(); c.fill.fore_color.rgb=CARD; c.line.color.rgb=LINE; c.line.width=Pt(0.75); c.shadow.inherit=False
        ac=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(cx),Inches(cy),Inches(0.04),Inches(rowh))
        ac.fill.solid(); ac.fill.fore_color.rgb=CYAN; ac.line.fill.background(); ac.shadow.inherit=False
        tf=c.text_frame; tf.word_wrap=False; tf.vertical_anchor=MSO_ANCHOR.MIDDLE
        tf.margin_left=Inches(0.12); tf.margin_right=Inches(0.08); tf.margin_top=Pt(0); tf.margin_bottom=Pt(0)
        p=tf.paragraphs[0]
        for t,b,cc in ch:
            r=p.add_run(); r.text=t; r.font.size=Pt(SZ['chip']); r.font.name=MONO; r.font.color.rgb=INK
        cx+=cw+gap
    return (cy-y)+rowh

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

# ---------- slide ----------
def build_slide(prs, sec):
    s=prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid(); s.background.fill.fore_color.rgb=BG
    bar=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,0,0,Inches(SW),Pt(3))
    bar.fill.solid(); bar.fill.fore_color.rgb=AMBER; bar.line.fill.background(); bar.shadow.inherit=False
    s.notes_slide.notes_text_frame.text=sec.get('data-note','') or ''
    classes=sec.get('class',[]) or []

    eb=sec.select_one('.eyebrow')
    if eb:
        textbox(s,LX,0.45,9.5,0.4,[(eb.get_text(' ',strip=True).upper(),True,AMBER)],SZ['eyebrow'],default=AMBER,font=MONO)

    # figure
    fig=None; ar=1.4; card=None
    for img in sec.find_all('img'):
        if img.get('src','').startswith('data:'): fig,ar=data_png(img); card='white'; break
    if not fig:
        svg=sec.find('svg')
        if svg is not None and (svg.get('viewbox') or svg.get('viewBox')): fig,ar=svg_png(svg)

    two_col = fig is not None
    TW = TWO_TW if two_col else ONE_TW

    # title size by layout
    if 's-title' in classes: SZ['_titlesize']=50
    elif 's-divider' in classes: SZ['_titlesize']=44
    elif 's-turn' in classes: SZ['_titlesize']=30
    elif 's-primer' in classes: SZ['_titlesize']=26
    else: SZ['_titlesize']=26

    root = sec.select_one('.left') or sec.select_one('.dv-copy') or sec.select_one('.discl-grid') or sec
    blocks=gather(root)
    if not any(k=='maintitle' for k,_ in blocks):
        h=sec.find(['h1','h2'])
        if h: blocks.insert(0,('maintitle',runs_of(h)))
    blocks=coalesce_chips(blocks)

    # measure total height for vertical centering
    heights=[]
    for k,p in blocks:
        if k=='table':
            rows=sum(1 for _ in p.find_all('tr')); heights.append(0.34*max(1,rows)+0.16)
        elif k=='chips':
            heights.append(chip_block_h(p,TW))
        else:
            heights.append(blk_height(k,p,TW))
    gaps=0.0
    total=sum(heights)
    band_top, band_bot = (0.95 if eb else 0.75), 7.15
    y0 = max(band_top, band_top + ((band_bot-band_top) - total)/2)
    if eb: y0=max(y0, 1.18)

    cy=y0
    for (k,p),hh in zip(blocks,heights):
        if k=='table':
            cy+=add_table(s,LX,cy,min(TW,7.6),p)+0.18; continue
        if k=='chips':
            cy+=chiprow(s,LX,cy,TW,p)+0.14; continue
        if k=='maintitle':
            textbox(s,LX,cy,TW,hh,p,SZ['_titlesize'],default=INK,leading=1.06)
        elif k=='stat':
            textbox(s,LX,cy,TW,hh,p,SZ['stat'],default=AMBER,leading=1.12)
        elif k=='emph':
            textbox(s,LX,cy,TW,hh,p,SZ['emph'],default=INK,leading=1.3)
        elif k=='cite':
            textbox(s,LX,cy,TW,hh,p,SZ['cite'],default=INK3,font=MONO,leading=1.3)
        elif k=='rolek':
            textbox(s,LX,cy,TW,hh,p,SZ['eyebrow'],default=AMBER,font=MONO,leading=1.2)
        else:
            textbox(s,LX,cy,TW,hh,p,SZ['body'],default=INK2,leading=1.28)
        cy+=hh
    if fig:
        if two_col: place_fig(s,fig,ar,FIG_X,1.0,FIG_W,FIG_MAXH,card=card)
        else:       place_fig(s,fig,ar,3.0,min(cy+0.2,3.3),7.3,3.4,card=card)
    return s

def chip_block_h(chips,w):
    return 0.40*_chip_rows(chips,w) + 0.10

# ---------- main ----------
def build_section(name):
    soup=BeautifulSoup(open(os.path.join(SLIDES,name+'.html')).read(),'html.parser')
    prs=Presentation(); prs.slide_width=Inches(SW); prs.slide_height=Inches(SH)
    n=0
    for sec in soup.select('section.slide'):
        if HIDE_RE.search(str(sec)) or sec.get('data-hide'): continue
        build_slide(prs, sec); n+=1
    out=os.path.join(os.path.dirname(os.path.abspath(__file__)),'sections','IWBI2026_Trivedi_%s.pptx'%NAMEMAP[name])
    os.makedirs(os.path.dirname(out), exist_ok=True); prs.save(out)
    print("%-22s %2d slides -> %s"%(name,n,os.path.basename(out)))
    return n

def build_full():
    prs=Presentation(); prs.slide_width=Inches(SW); prs.slide_height=Inches(SH); n=0
    for name in SECTIONS:
        soup=BeautifulSoup(open(os.path.join(SLIDES,name+'.html')).read(),'html.parser')
        for sec in soup.select('section.slide'):
            if HIDE_RE.search(str(sec)) or sec.get('data-hide'): continue
            build_slide(prs, sec); n+=1
    out=os.path.join(os.path.dirname(os.path.abspath(__file__)),'IWBI2026_Trivedi_FULL_keynote.pptx')
    prs.save(out); print("FULL %d slides -> %s"%(n,os.path.basename(out)))

if __name__=='__main__':
    import sys
    args=sys.argv[1:]
    if args==['--full-only']:
        build_full()
    else:
        tgt=[a for a in args if not a.startswith('--')] or SECTIONS
        for nm in tgt: build_section(nm)
        if not [a for a in args if not a.startswith('--')]:  # built all -> also concat
            build_full()
