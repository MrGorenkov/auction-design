import json, sys, time, urllib.request, urllib.parse
UA={"User-Agent":"papers-work-litcheck/1.0"}
def get(u):
    for i in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=30) as r: return json.load(r)
        except Exception as e: time.sleep(2**i)
for q in sys.argv[1:]:
    j=get("https://api.crossref.org/works?rows=3&query.bibliographic="+urllib.parse.quote(q))
    print("##",q)
    for m in (j or {}).get("message",{}).get("items",[]):
        print("   ",m.get("DOI"),"|",m.get("issued",{}).get("date-parts",[[None]])[0][0],"|",(m.get("container-title") or [""])[0][:35],"|",m.get("title",[""])[0][:85],"|",";".join(a.get("family","?") for a in m.get("author",[]))[:50])
    time.sleep(0.5)
