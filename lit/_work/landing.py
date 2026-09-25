import json,re,html,subprocess,sys
d=json.load(open('meta.json')); s2=json.load(open('s2.json'))
extra={}
try: extra=json.load(open('landing.json'))
except: pass
alt={"horton2023large":"https://www.nber.org/papers/w31122","manning2024automated":"https://www.nber.org/papers/w32381",
     "roth2002last":"https://www.aeaweb.org/articles?id=10.1257/00028280260344632","einav2018auctions":"https://www.journals.uchicago.edu/doi/10.1086/695529",
     "backus2015sniping":"https://www.nber.org/papers/w20942"}
for k in s2:
    if s2[k]["abstract"] or k in extra and extra[k]: continue
    url=alt.get(k,"https://doi.org/"+d[k]['m']['DOI'])
    h=subprocess.run(["curl","-sL","--max-time","40","-A","Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36",url],capture_output=True,text=True,errors="ignore").stdout
    cands=re.findall(r'<meta[^>]+(?:name|property)="(?:citation_abstract|dc\.description|DC\.Description|og:description|description|twitter:description)"[^>]+content="([^"]{150,})"',h,re.I)
    cands+=re.findall(r'<meta[^>]+content="([^"]{150,})"[^>]+(?:name|property)="(?:citation_abstract|dc\.description|og:description|description)"',h,re.I)
    a=html.unescape(max(cands,key=len)) if cands else None
    extra[k]=a; print(k,len(h),(a or "")[:120])
json.dump(extra,open('landing.json','w'),ensure_ascii=False,indent=0)
