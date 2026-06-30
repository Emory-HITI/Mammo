#!/usr/bin/env python3
"""Extract the EXACT computed layout of every slide in an HTML deck (positions,
font sizes, colors, runs, figures) via Playwright, and emit JSON the PPTX
builder consumes. Frame px → inches by frame width; font px → pt by the same
scale. This makes the PPTX a faithful match of the HTML render."""
import sys, json, pathlib
from playwright.sync_api import sync_playwright

CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
JS = r"""
() => {
  const INLINE = new Set(['SPAN','B','I','EM','A','BR','SUP','SUB','SMALL','CODE','STRONG','U','MARK']);
  const fr = document.querySelector('#frame').getBoundingClientRect();
  const W = fr.width, H = fr.height;
  function vis(el){ const s=getComputedStyle(el); if(s.display==='none'||s.visibility==='hidden'||+s.opacity===0) return false; return true; }
  function runsOf(el){
    const runs=[];
    (function walk(node){
      if(node.nodeType===3){ const t=node.textContent; if(t.replace(/\s/g,'')!==''){ const p=node.parentElement; const cs=getComputedStyle(p); runs.push({t:t, color:cs.color, bold:(+cs.fontWeight)>=600, italic:cs.fontStyle==='italic'}); } }
      else if(node.nodeType===1){ if(node.tagName==='BR'){runs.push({t:'\n'});return;} for(const c of node.childNodes) walk(c); }
    })(el);
    return runs;
  }
  function isLeafTextBlock(el){
    // has text, and all element children are inline
    if(el.textContent.replace(/\s/g,'')==='') return false;
    for(const c of el.children){ if(!INLINE.has(c.tagName)) return false; }
    return true;
  }
  const units=[];
  function rectOf(el){ const r=el.getBoundingClientRect(); return {x:(r.left-fr.left)/W, y:(r.top-fr.top)/H, w:r.width/W, h:r.height/H}; }
  (function rec(el){
    for(const c of el.children){
      if(!vis(c)) continue;
      const tag=c.tagName;
      if(tag==='SVG'||tag==='svg'){ const r=rectOf(c); units.push({type:'svg', rect:r, html:c.outerHTML}); continue; }
      if(tag==='IMG'){ const src=c.getAttribute('src')||''; if(src.startsWith('data:')){ const r=rectOf(c); units.push({type:'img', rect:r, src:src}); } continue; }
      if(tag==='CANVAS') continue;
      if(isLeafTextBlock(c)){
        const cs=getComputedStyle(c); const r=rectOf(c);
        units.push({type:'text', rect:r, fontPx:parseFloat(cs.fontSize), linePx:parseFloat(cs.lineHeight)||parseFloat(cs.fontSize)*1.2,
                    align:cs.textAlign, family:cs.fontFamily, runs:runsOf(c)});
      } else { rec(c); }
    }
  })(document.querySelector('.slide.active') || document.querySelector('.slide'));
  return {W:W, H:H, units:units};
}
"""

def extract(name):
    url = "file://" + str(pathlib.Path("/home/user/Mammo/keynote-iwbi-2026/slides/%s.html" % name).resolve())
    out = []
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME, args=["--no-sandbox"])
        pg = b.new_page(viewport={"width": 1400, "height": 900}, device_scale_factor=1)
        pg.goto(url); pg.wait_for_timeout(700)
        n = pg.eval_on_selector_all(".slide", "e=>e.length")
        for i in range(n):
            info = pg.evaluate("""(i)=>{const s=[...document.querySelectorAll('.slide')];s.forEach((el,k)=>el.classList.toggle('active',k===i));return null;}""", i)
            pg.wait_for_timeout(120)
            hidden = pg.evaluate("""()=>{const a=document.querySelector('.slide.active'); const h=(a.outerHTML.match(/hide in final|⚑/)!=null)|| a.hasAttribute('data-hide'); return !!h;}""")
            data = pg.evaluate(JS)
            data["hidden"] = hidden
            out.append(data)
        b.close()
    return out

if __name__ == "__main__":
    name = sys.argv[1]
    data = extract(name)
    dest = "/home/user/Mammo/keynote-iwbi-2026/pptx/layout/%s.json" % name
    pathlib.Path(dest).parent.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(dest, "w"))
    vis = sum(1 for s in data if not s["hidden"])
    print("%s: %d slides (%d visible) -> %s" % (name, len(data), vis, dest))
