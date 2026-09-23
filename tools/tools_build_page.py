import json, pathlib
data = json.load(open("web/mb-data.json"))
html = pathlib.Path("web/_template.html").read_text()
html = html.replace("/*__DATA__*/", json.dumps(data, separators=(",", ":")))
pathlib.Path("web/mushroom-body-hash.html").write_text(html)
print("bytes:", len(html))
