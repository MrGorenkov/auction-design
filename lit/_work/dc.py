import sys,json,urllib.request,urllib.parse,time
UA={"User-Agent":"papers-work-litcheck/1.0"}
for q in sys.argv[1:]:
    u="https://api.datacite.org/dois?page[size]=5&query="+urllib.parse.quote(q+" AND client_id:arxiv.content")
    j=json.load(urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=60))
    print("##",q)
    for d in j["data"]:
        a=d["attributes"]; print("   ",d["id"],a.get("publicationYear"),"|",a["titles"][0]["title"][:100],"|",";".join(c.get("familyName","?") for c in a["creators"])[:60])
    time.sleep(1)
