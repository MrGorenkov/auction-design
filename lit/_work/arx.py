import sys,urllib.request,urllib.parse,re,time
for q in sys.argv[1:]:
    u="https://export.arxiv.org/api/query?max_results=5&search_query="+urllib.parse.quote(q)
    x=urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"papers-work-litcheck/1.0"}),timeout=60).read().decode()
    print("##",q)
    for e in x.split("<entry>")[1:]:
        t=re.sub(r"\s+"," ",re.search(r"<title>(.*?)</title>",e,re.S).group(1))
        i=re.search(r"<id>http://arxiv.org/abs/(.*?)</id>",e).group(1)
        a=re.findall(r"<name>(.*?)</name>",e); d=re.search(r"<published>(\d{4})",e).group(1)
        print("   ",i,d,"|",t[:100],"|",";".join(n.split()[-1] for n in a)[:60])
    time.sleep(3)
