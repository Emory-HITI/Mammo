#!/usr/bin/env python3
import sys, os, re, base64, tempfile
from bs4 import BeautifulSoup, NavigableString, Tag
import cairosvg
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR, MSO_AUTO_SIZE
from pptx.enum.shapes import MSO_SHAPE

FLOOR=RGBColor(0x0B,0x0F,0x14); BG=RGBColor(0x0F,0x14,0x1A); BG3=RGBColor(0x1B,0x23,0x2D)
INK=RGBColor(0xEC,0xEF,0xF3); INK2=RGBColor(0x9D,0xA9,0xB5); INK3=RGBColor(0x5C,0x69,0x75)
AMBER=RGBColor(0xE7,0xAC,0x51); CYAN=RGBColor(0x5F,0xB7,0xC9); WARN=RGBColor(0xD9,0x78,0x5B)
LINE=RGBColor(0x2A,0x33,0x3E)
EMUIN=914400
SW, SH = 13.333, 7.5
TMP=tempfile.mkdtemp()

def runs_of(el):
    """flatten inline content -> [(text, bold, color)]"""
    out=[]
    def walk(node, bold):
        if isinstance(node, NavigableString):
            t=str(node)
            if t: out.append([t, bold, None])
            return
        b = bold or node.name in ('b','strong')
        for c in node.children: walk(c, b)
    walk(el, False)
    # collapse whitespace
    res=[]
    for t,b,c in out:
        t=re.sub(r'\s+',' ',t)
        if t.strip()=='' and (not res or res[-1][0].endswith(' ')): continue
        res.append([t,b,c])
    return [r for r in res if r[0]]

def add_text(slide, x,y,w,h, blocks, size, color=INK2, bold_color=INK, align=PP_ALIGN.LEFT, title_pt=None, leading=1.12):
    tb=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
    tf=tb.text_frame; tf.word_wrap=True
    try: tf.auto_size=MSO_AUTO_SIZE.NONE
    except: pass
    tf.margin_left=Pt(2); tf.margin_right=Pt(2); tf.margin_top=Pt(1); tf.margin_bottom=Pt(1)
    first=True
    for blk in blocks:  # blk = list of runs
        p=tf.paragraphs[0] if first else tf.add_paragraph()
        first=False; p.alignment=align; p.line_spacing=leading; p.space_after=Pt(4)
        for t,b,c in blk:
            r=p.add_run(); r.text=t; f=r.font; f.size=Pt(size); f.name="Arial"
            f.bold=bool(b)
            f.color.rgb = (bold_color if b else color) if c is None else c
    return tb

def est_h(blocks, w_in, size):
    cpl=max(8, int(w_in / (size*0.0089)))
    lines=0
    for blk in blocks:
        txt=''.join(t for t,_,_ in blk)
        lines += max(1, -(-len(txt)//cpl))
    return lines*(size*1.18/72)+0.12

def svg_png(svg_tag):
    s=str(svg_tag)
    if 'xmlns' not in s[:60]:
        s=s.replace('<svg','<svg xmlns="http://www.w3.org/2000/svg"',1)
    vb=svg_tag.get('viewBox','0 0 16 9').split()
    w,h=float(vb[2]),float(vb[3])
    p=os.path.join(TMP,'svg_%d.png'%abs(hash(s))%10**8)
    cairosvg.svg2png(bytestring=s.encode(), write_to=p, output_width=int(w*2.5), output_height=int(h*2.5), background_color='#0F141A')
    return p, w/h

def data_img(img):
    src=img.get('src','')
    m=re.match(r'data:image/(\w+);base64,(.*)$', src, re.S)
    if not m: return None,None
    ext=m.group(1); raw=base64.b64decode(m.group(2))
    p=os.path.join(TMP,'img_%d.%s'%(abs(hash(src))%10**8, 'png' if ext=='png' else 'jpg'))
    open(p,'wb').write(raw)
    from PIL import Image
    try:
        im=Image.open(p); ar=im.width/im.height
    except: ar=1.4
    return p, ar

def place_fig(slide, path, ar, x, y, maxw, maxh, card=None):
    w=maxw; h=w/ar
    if h>maxh: h=maxh; w=h*ar
    px=x+(maxw-w)/2; py=y+(maxh-h)/2
    if card:
        cd=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,Inches(px-0.08),Inches(py-0.08),Inches(w+0.16),Inches(h+0.16))
        cd.fill.solid(); cd.fill.fore_color.rgb=RGBColor(0xFF,0xFF,0xFF) if card=='white' else RGBColor(0,0,0)
        cd.line.color.rgb=AMBER; cd.line.width=Pt(0.75); cd.shadow.inherit=False
    slide.shapes.add_picture(path,Inches(px),Inches(py),Inches(w),Inches(h))

def table_blocks(tbl):
    rows=[]
    for tr in tbl.find_all('tr'):
        cells=[c.get_text(' ',strip=True) for c in tr.find_all(['td','th'])]
        if cells: rows.append(cells)
    return rows

def add_table(slide,x,y,w,rows):
    if not rows: return
    nr=len(rows); nc=max(len(r) for r in rows)
    h=0.32*nr
    gtbl=slide.shapes.add_table(nr,nc,Inches(x),Inches(y),Inches(w),Inches(h)).table
    for ri,row in enumerate(rows):
        for ci in range(nc):
            cell=gtbl.cell(ri,ci); cell.margin_left=Pt(4); cell.margin_right=Pt(4); cell.margin_top=Pt(1); cell.margin_bottom=Pt(1)
            cell.fill.solid(); cell.fill.fore_color.rgb=BG3 if ri==0 else BG
            txt=row[ci] if ci<len(row) else ''
            cell.text=txt
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size=Pt(10); r.font.name="Arial"; r.font.color.rgb=AMBER if ri==0 else INK2; r.font.bold=(ri==0)
    return h

def build(html_path, out_path):
    soup=BeautifulSoup(open(html_path).read(),'html.parser')
    brand=soup.select_one('.brand'); brandtxt=brand.get_text(' ',strip=True) if brand else ''
    prs=Presentation(); prs.slide_width=Inches(SW); prs.slide_height=Inches(SH)
    blank=prs.slide_layouts[6]
    slides=soup.select('section.slide')
    for sec in slides:
        s=prs.slides.add_slide(blank)
        bg=s.background; bg.fill.solid(); bg.fill.fore_color.rgb=BG
        # accent bar
        bar=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,0,0,Inches(SW),Pt(3))
        bar.fill.solid(); bar.fill.fore_color.rgb=AMBER; bar.line.fill.background(); bar.shadow.inherit=False
        note=sec.get('data-note','')
        s.notes_slide.notes_text_frame.text=note
        classes=sec.get('class',[])
        # eyebrow
        eb=sec.select_one('.eyebrow')
        if eb:
            add_text(s,0.6,0.32,8,0.4,[[(eb.get_text(' ',strip=True).upper(),False,AMBER)]],11,color=AMBER,bold_color=AMBER)
        # title
        h=sec.find(['h1','h2'])
        is_div='s-divider' in classes
        # figure detection
        img=sec.find('img'); svg=sec.find('svg')
        fig=None; fig_ar=1.4; card=None
        if img and img.get('src','').startswith('data:'):
            fig,fig_ar=data_img(img)
            # white card for light figures (km, taxonomy, seg=black)
            card='white'
        elif svg and svg.get('viewBox'):
            fig,fig_ar=svg_png(svg)
        two_col = fig is not None
        textw = 6.7 if two_col else 11.8
        tx=0.6; ty=0.95
        if h:
            tr=runs_of(h)
            # color .am/.amber spans amber
            for sp in h.find_all('span'):
                cls=sp.get('class',[]); 
            # recolor: any run that came from a span with am/amber -> amber (approx: bold the am)
            tsize = 34 if is_div else 26
            th=est_h([tr],textw,tsize)
            add_text(s,tx,ty,textw,th+0.2,[tr],tsize,color=INK,bold_color=AMBER); 
            # amber emphasis: redo title to color am spans
            ty2=ty+th+0.18
        else:
            ty2=ty
        # body blocks: collect top-level content elements
        skip={'eyebrow'}
        body_parents = sec.select_one('.left') or sec.select_one('.dv-copy') or sec
        blocks_boxes=[]
        tables=[]
        # gather paragraphs, chips, stats, cites, turn-line, table
        for el in sec.find_all(['p','table'], recursive=True):
            # skip if inside figure/right/figcaption
            if el.find_parent(class_=re.compile(r'right|viewport|vp-|figure')): continue
            if el.name=='table':
                tables.append(table_blocks(el)); continue
            cls=' '.join(el.get('class',[]))
            r=runs_of(el)
            if not r: continue
            txt=''.join(t for t,_,_ in r)
            if 'cite' in cls:
                blocks_boxes.append(('cite',[r]))
            elif 'statnum' in cls or 'num' in cls.split():
                blocks_boxes.append(('stat',[r]))
            else:
                blocks_boxes.append(('p',[r]))
        # chips
        for ch in sec.select('.chips'):
            chips=[[('▸ '+c.get_text(' ',strip=True),False,CYAN)] for c in ch.select('.chip')]
            if chips: blocks_boxes.append(('chips',chips))
        # stat big numbers (statnum/twostat .num)
        # render boxes stacked in left col
        cy=ty2+0.05
        for kind,blks in blocks_boxes:
            if kind=='stat':
                sz=30; col=AMBER; bh=est_h(blks,textw,sz)
                add_text(s,tx,cy,textw,bh+0.1,blks,sz,color=AMBER,bold_color=AMBER); cy+=bh+0.12
            elif kind=='cite':
                bh=est_h(blks,textw,10); add_text(s,tx,cy,textw,bh+0.05,blks,10,color=INK3,bold_color=INK3); cy+=bh+0.08
            elif kind=='chips':
                sz=12; bh=est_h(blks,textw,sz)*1.1
                add_text(s,tx,cy,textw,bh+0.1,blks,sz,color=INK,bold_color=CYAN); cy+=bh+0.12
            else:
                sz=15 if not is_div else 17; bh=est_h(blks,textw,sz)
                add_text(s,tx,cy,textw,bh+0.1,blks,sz,color=INK2,bold_color=INK); cy+=bh+0.14
        # tables
        for rows in tables:
            th=add_table(s,tx,min(cy,6.6),min(textw,7.5),rows) or 0; cy+=th+0.2
        # figure
        if two_col:
            place_fig(s,fig,fig_ar,7.55,1.05,5.2,5.4,card=card)
        elif fig:
            place_fig(s,fig,fig_ar,3.2,min(cy+0.1,3.4),6.9,3.4,card=card)
    prs.save(out_path)
    return len(slides)

if __name__=='__main__':
    n=build(sys.argv[1], sys.argv[2]); print("slides:",n,"->",sys.argv[2])
