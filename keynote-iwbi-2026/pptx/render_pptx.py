#!/usr/bin/env python3
"""Lightweight PPTX -> PNG renderer (PIL) for layout QA without LibreOffice.
Reads shape geometry, text runs (size/color/bold), tables, and pictures
straight from the .pptx and draws them at scale. Good enough to verify font
sizes, spacing, overlaps, and overflow."""
import sys, io
from pptx import Presentation
from pptx.util import Emu
from PIL import Image, ImageDraw, ImageFont

EMU=914400
def fpt(emu): return emu/EMU  # inches
SCALE=96  # px per inch -> 1280x720
def px(inch): return int(round(inch*SCALE))

FONTS={}
def font(size_pt, bold=False, mono=False):
    key=(round(size_pt),bold,mono)
    if key in FONTS: return FONTS[key]
    px_sz=int(round(size_pt*SCALE/72))
    cands=[]
    if mono: cands=["/usr/share/fonts/truetype/dejavu/DejaVuSansMono%s.ttf"%("-Bold" if bold else "")]
    else: cands=["/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf"%("-Bold" if bold else "")]
    cands+=["/usr/share/fonts/truetype/liberation/LiberationSans%s.ttf"%("-Bold" if bold else "")]
    f=None
    for c in cands:
        try: f=ImageFont.truetype(c,px_sz); break
        except Exception: pass
    if f is None: f=ImageFont.load_default()
    FONTS[key]=f; return f

def rgb(c):
    try: return (c[0],c[1],c[2])
    except Exception: return None

def wrap(draw, text, fnt, maxw):
    words=text.split(' '); lines=[]; cur=""
    for w in words:
        t=(cur+" "+w).strip()
        if draw.textlength(t,font=fnt)<=maxw or not cur: cur=t
        else: lines.append(cur); cur=w
    if cur: lines.append(cur)
    return lines

def render_slides(path):
    """Return a list of per-slide PIL images (proxy render of the .pptx)."""
    prs=Presentation(path)
    W=px(prs.slide_width/EMU); H=px(prs.slide_height/EMU)
    imgs=[]
    for s in prs.slides:
        im=Image.new("RGB",(W,H),(15,20,26)); d=ImageDraw.Draw(im)
        for sh in s.shapes:
            try: L=px(sh.left/EMU); T=px(sh.top/EMU); Wd=px(sh.width/EMU); Hd=px(sh.height/EMU)
            except Exception: continue
            st=sh.shape_type
            # picture
            if st==13:
                try:
                    pic=Image.open(io.BytesIO(sh.image.blob)).convert("RGBA")
                    pic.thumbnail((Wd,Hd))
                    im.paste(pic,(L+(Wd-pic.width)//2,T+(Hd-pic.height)//2), pic)  # alpha mask
                except Exception: d.rectangle([L,T,L+Wd,T+Hd],outline=(90,100,115))
                continue
            # shape with solid fill (cards/bars/chips)
            try:
                if sh.fill.type is not None and sh.fill.type==1:
                    fc=rgb(sh.fill.fore_color.rgb)
                    if fc: d.rectangle([L,T,L+Wd,T+Hd],fill=fc)
            except Exception: pass
            # table
            if sh.has_table:
                tb=sh.table; rows=len(tb.rows); colN=len(tb.columns)
                ch=Hd/max(1,rows); cw=Wd/max(1,colN)
                for ri in range(rows):
                    for ci in range(colN):
                        cx=L+ci*cw; cy=T+ri*ch
                        cell=tb.cell(ri,ci)
                        fc=None
                        try: fc=rgb(cell.fill.fore_color.rgb)
                        except Exception: pass
                        if fc: d.rectangle([cx,cy,cx+cw,cy+ch],fill=fc,outline=(42,51,62))
                        else: d.rectangle([cx,cy,cx+cw,cy+ch],outline=(42,51,62))
                        txt=cell.text_frame.text
                        col=(236,239,243); sz=10; bold=False
                        for p in cell.text_frame.paragraphs:
                            for r in p.runs:
                                if r.font.size: sz=r.font.size.pt
                                if r.font.color and r.font.color.type is not None:
                                    cc=rgb(r.font.color.rgb);  col=cc or col
                                bold=bool(r.font.bold)
                        if txt: d.text((cx+4,cy+3),txt[:40],fill=col,font=font(sz,bold))
                continue
            # text frame
            if sh.has_text_frame:
                y=T+2
                for p in sh.text_frame.paragraphs:
                    runs=[(r.text, (r.font.size.pt if r.font.size else 14),
                           rgb(r.font.color.rgb) if (r.font.color and r.font.color.type is not None) else (157,169,181),
                           bool(r.font.bold), (r.font.name=="Consolas"))
                          for r in p.runs if r.text]
                    if not runs:
                        y+=px(0.12); continue
                    sz=runs[0][1]; mono=runs[0][4]
                    full="".join(t for t,_,_,_,_ in runs)
                    lines=wrap(d,full,font(sz,runs[0][3],mono),Wd-4)
                    lh=int(sz*SCALE/72*1.25)
                    # simple: draw wrapped full text in first run's color/size (approx; emphasis lost in proxy)
                    # better: draw run by run on one advancing x with wrap
                    x=L+2; lineh=lh
                    for t,rsz,rcol,rb,rmono in runs:
                        for piece in _split(t):
                            ww=d.textlength(piece,font=font(rsz,rb,rmono))
                            if x+ww>L+Wd-2 and x>L+2:
                                x=L+2; y+=int(rsz*SCALE/72*1.28)
                            d.text((x,y),piece,fill=(rcol or (157,169,181)),font=font(rsz,rb,rmono))
                            x+=ww
                    y+=int(sz*SCALE/72*1.28)
        # frame border
        d.rectangle([0,0,W-1,H-1],outline=(38,48,59))
        imgs.append(im)
    return imgs

def render(path, out):
    imgs=render_slides(path)
    if not imgs: print("no slides"); return
    W,H=imgs[0].size
    perrow=2; rows=(len(imgs)+perrow-1)//perrow
    pad=10; tw=W//2; th=H//2
    sheet=Image.new("RGB",(perrow*(tw+pad)+pad, rows*(th+pad)+pad),(0,0,0))
    for i,im in enumerate(imgs):
        t=im.resize((tw,th)); r=i//perrow; c=i%perrow
        sheet.paste(t,(pad+c*(tw+pad), pad+r*(th+pad)))
    sheet.save(out); print("wrote",out,"slides",len(imgs))

def _split(t):
    # keep words+trailing spaces as tokens for wrapping
    import re
    return re.findall(r'\S+\s*', t) or [t]

if __name__=='__main__':
    render(sys.argv[1], sys.argv[2])
