#!/usr/bin/env python3
"""Build slides/all_slides.html — a continuity-review deck that embeds every
section in an isolated iframe (srcdoc => inherits parent origin, CSS sandboxed),
driven by one global controller. Strips hide-flagged slides. Overwrites nothing
else; writes a brand-new file."""
import re, json, os

SL = "/home/user/Mammo/keynote-iwbi-2026/slides"
SECTIONS = [
    ("hook.html",            "Opening"),
    ("section1.html",        "Era I · 1998 · CAD"),
    ("section2.html",        "Era II · 2016"),
    ("section2b.html",       "Era II · 2026"),
    ("section5alternate.html","Era III · 2035 · Prevention"),
    ("section6b.html",       "Frontier models"),
    ("section_adoption.html","Adoption"),
    ("section7.html",        "Close"),
]

SLIDE_RE = re.compile(r'<section class="slide[^"]*"[\s\S]*?</section>')

def strip_hidden(html):
    """Remove any <section class=slide> whose text carries a hide marker."""
    dropped = 0
    def repl(m):
        nonlocal dropped
        block = m.group(0)
        if '⚑' in block or 'hide in final' in block.lower():
            dropped += 1
            return ''
        return block
    out = SLIDE_RE.sub(repl, html)
    n = len(SLIDE_RE.findall(out))
    return out, n, dropped

def prep(html):
    """Hide the in-iframe nav bar (parent provides global nav)."""
    inject = "<style>.nav{display:none!important}</style>"
    if "</head>" in html:
        return html.replace("</head>", inject + "</head>", 1)
    return inject + html

data = []
counts = []
for fn, label in SECTIONS:
    raw = open(os.path.join(SL, fn)).read()
    stripped, n, dropped = strip_hidden(raw)
    stripped = prep(stripped)
    data.append({"file": fn, "label": label, "n": n, "html": stripped})
    counts.append(n)
    print(f"{fn:24s} slides={n:2d}  dropped_hidden={dropped}")

total = sum(counts)
print("TOTAL FINAL SLIDES:", total)

payload = json.dumps(data).replace("</", "<\\/")

PAGE = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Thirty Years, Three Eras — full talk (continuity review)</title>
<style>
:root{--floor:#080B0F;--bg:#0F141A;--bg3:#1B232D;--ink:#ECEFF3;--ink2:#9DA9B5;--ink3:#5C6975;--amber:#E7AC51;--cyan:#5FB7C9;--line:#26303B}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:#000;color:var(--ink);
font-family:"Helvetica Neue",Arial,sans-serif;overflow:hidden}
#top{position:fixed;top:0;left:0;right:0;height:46px;z-index:50;display:flex;
align-items:center;gap:14px;padding:0 14px;background:rgba(8,11,15,.92);
border-bottom:1px solid var(--line);backdrop-filter:blur(6px)}
#chips{display:flex;gap:6px;flex:1;overflow-x:auto;scrollbar-width:none}
#chips::-webkit-scrollbar{display:none}
.chip{white-space:nowrap;font-size:11px;letter-spacing:.04em;color:var(--ink2);
background:var(--bg3);border:1px solid var(--line);border-radius:20px;
padding:5px 11px;cursor:pointer;font-family:"SFMono-Regular",ui-monospace,monospace}
.chip.on{color:#1A1206;background:var(--amber);border-color:var(--amber);font-weight:700}
.ct{font-family:"SFMono-Regular",ui-monospace,monospace;font-size:12px;color:var(--ink3);white-space:nowrap}
.nb{font-family:"SFMono-Regular",ui-monospace,monospace;font-size:13px;color:var(--ink);
background:var(--bg3);border:1px solid var(--line);border-radius:7px;
width:34px;height:30px;cursor:pointer;display:flex;align-items:center;justify-content:center}
.nb:hover{border-color:var(--amber);color:var(--amber)}
#stage{position:fixed;top:46px;bottom:74px;left:0;right:0;background:#000}
iframe{position:absolute;inset:0;width:100%;height:100%;border:0;background:#000;
opacity:0;visibility:hidden;pointer-events:none}
iframe.on{opacity:1;visibility:visible;pointer-events:auto}
#foot{position:fixed;bottom:0;left:0;right:0;height:74px;z-index:50;
display:flex;align-items:center;gap:12px;padding:8px 14px;
background:rgba(8,11,15,.92);border-top:1px solid var(--line);backdrop-filter:blur(6px)}
#secname{font-family:"SFMono-Regular",ui-monospace,monospace;font-size:11px;
color:var(--amber);letter-spacing:.05em;white-space:nowrap;min-width:150px}
#note{flex:1;font-size:12px;line-height:1.4;color:var(--ink2);max-height:58px;overflow:auto}
.click{position:fixed;top:46px;bottom:74px;width:9%;z-index:40;cursor:pointer}
#cl{left:0}#cr{right:0}
</style></head><body>
<div id=top>
  <button class=nb id=prev>&#8249;</button>
  <button class=nb id=next>&#8250;</button>
  <div id=chips></div>
  <div class=ct id=ctr>1 / __TOTAL__</div>
</div>
<div id=stage></div>
<div class=click id=cl></div><div class=click id=cr></div>
<div id=foot>
  <div id=secname></div>
  <div id=note></div>
</div>
<script id=data type="application/json">__PAYLOAD__</script>
<script>
(function(){
var DATA=JSON.parse(document.getElementById('data').textContent);
var stage=document.getElementById('stage');
// flat map: global index -> [section, local]
var flat=[];DATA.forEach(function(s,si){for(var k=0;k<s.n;k++)flat.push([si,k]);});
var TOTAL=flat.length;document.getElementById('ctr').textContent='1 / '+TOTAL;
var frames=DATA.map(function(s,si){
  var f=document.createElement('iframe');f.srcdoc=s.html;f.dataset.si=si;f.loaded=false;
  f.addEventListener('load',function(){f.loaded=true;});stage.appendChild(f);return f;});
// chips
var chips=document.getElementById('chips');
DATA.forEach(function(s,si){var c=document.createElement('div');c.className='chip';
  c.textContent=s.label;c.addEventListener('click',function(){var g=flat.findIndex(function(p){return p[0]===si;});go(g);});
  chips.appendChild(c);});
var G=0;
function drive(f,local,tries){
  // click the local dot to jump; retry until iframe DOM ready
  try{var d=f.contentDocument.querySelectorAll('.dot');
    if(d&&d[local]){d[local].click();showNote(f);return;}}catch(e){}
  if((tries||0)<40)setTimeout(function(){drive(f,local,(tries||0)+1);},60);
}
function showNote(f){try{var a=f.contentDocument.querySelector('.slide.active')||
  f.contentDocument.querySelectorAll('.slide')[0];
  document.getElementById('note').textContent=a?(a.getAttribute('data-note')||''):'';}catch(e){}}
function go(g){
  G=Math.max(0,Math.min(TOTAL-1,g));
  var pair=flat[G],si=pair[0],local=pair[1];
  frames.forEach(function(f,k){f.classList.toggle('on',k===si);});
  drive(frames[si],local,0);
  document.getElementById('ctr').textContent=(G+1)+' / '+TOTAL;
  document.getElementById('secname').textContent=DATA[si].label;
  [].forEach.call(chips.children,function(c,k){c.classList.toggle('on',k===si);});
}
document.getElementById('next').onclick=function(){go(G+1);};
document.getElementById('prev').onclick=function(){go(G-1);};
document.getElementById('cr').onclick=function(){go(G+1);};
document.getElementById('cl').onclick=function(){go(G-1);};
document.addEventListener('keydown',function(e){
  if(e.key==='ArrowRight'||e.key===' '||e.key==='PageDown'){e.preventDefault();go(G+1);}
  else if(e.key==='ArrowLeft'||e.key==='PageUp'){e.preventDefault();go(G-1);}
  else if(e.key==='Home'){go(0);}else if(e.key==='End'){go(TOTAL-1);}});
go(0);
})();
</script></body></html>"""

out = PAGE.replace("__PAYLOAD__", payload).replace("__TOTAL__", str(total))
dest = os.path.join(SL, "all_slides.html")
open(dest, "w").write(out)
print("WROTE", dest, "bytes", len(out))
