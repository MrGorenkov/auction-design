import json, sys, time, urllib.request, urllib.parse
UA={"User-Agent":"papers-work-litcheck/1.0"}
def get(u):
    for i in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=30) as r: return json.load(r)
        except Exception as e:
            if getattr(e,'code',0)==404: return None
            time.sleep(2**i)
    return None
out={}
try: out=json.load(open('crossref.json'))
except: pass
for d in [l.strip() for l in open(sys.argv[1]) if l.strip()]:
    if d in out: continue
    j=get("https://api.crossref.org/works/"+urllib.parse.quote(d))
    out[d]=j["message"] if j else None
    m=out[d]
    if m: print("OK ",d,"|",m.get("issued",{}).get("date-parts",[[None]])[0][0],"|",(m.get("container-title") or [""])[0][:40],"|",m.get("title",[""])[0][:90],"|",";".join(a.get("family","?") for a in m.get("author",[]))[:60])
    else: print("MISS",d)
    time.sleep(0.4)
json.dump(out,open('crossref.json','w'),ensure_ascii=False)
