import json,time,urllib.request,urllib.parse,re,html
UA={"User-Agent":"papers-work-litcheck/1.0"}
def get(u, js=True):
    for i in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=40) as r:
                b=r.read(); return json.loads(b) if js else b.decode("utf-8","ignore")
        except Exception as e:
            if getattr(e,'code',0)==404: return None
            time.sleep(2**i)
db={}
try: db=json.load(open('meta.json'))
except: pass
for line in open('final.tsv'):
    key,src,tags=line.rstrip('\n').split('\t')
    if key in db: continue
    if src.startswith('arXiv:'):
        a=get("https://api.datacite.org/dois/10.48550/arxiv."+src[6:])
        db[key]={"src":"datacite","tags":tags,"m":a["data"]["attributes"] if a else None}
    elif src.startswith('IDEAS:'):
        h=get("https://ideas.repec.org/a/"+src[6:]+".html",js=False)
        meta={k.lower():html.unescape(v) for k,v in re.findall(r'<META NAME="citation_([a-z_]+)" content="([^"]*)"',h,re.I)}
        db[key]={"src":"ideas","tags":tags,"m":meta,"url":"https://ideas.repec.org/a/"+src[6:]+".html"}
    else:
        j=get("https://api.crossref.org/works/"+urllib.parse.quote(src))
        db[key]={"src":"crossref","tags":tags,"m":j["message"] if j else None}
    m=db[key]["m"]; print("OK " if m else "MISS", key)
    time.sleep(0.4)
json.dump(db,open('meta.json','w'),ensure_ascii=False,indent=0)
