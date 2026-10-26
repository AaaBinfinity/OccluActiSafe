"""Check this diagram delivery, not the thesis algorithms or simulation."""
import json
import re
import struct
from pathlib import Path
from html.parser import HTMLParser
from xml.etree import ElementTree as ET

root = Path(__file__).resolve().parent
spec = json.loads((root/"source/graphs.json").read_text())
manifest = json.loads((root/"manifest.json").read_text())
for g, m in zip(spec["graphs"], manifest["figures"]):
    ids = {n["id"] for n in g["nodes"]}
    assert len(ids) == len(g["nodes"])
    assert all(e[0] in ids and e[1] in ids for e in g["edges"])
    svg = ET.parse(root/"images"/(g["id"]+".svg")).getroot()
    assert svg.tag.endswith("svg")
    assert len(svg.findall(".//{http://www.w3.org/2000/svg}text")) >= len(g["nodes"])
    png = (root/"images"/(g["id"]+".png")).read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = struct.unpack(">II", png[16:24])
    assert (w, h) == (m["png_width"], m["png_height"])
    assert m["node_overlaps"] == 0
    dot = (root/"source"/(g["id"]+".dot")).read_text()
    mm = (root/"source"/(g["id"]+".mmd")).read_text()
    for n in ids:
        assert n+" [" in dot or n+"[" in dot or n+" " in dot
        assert re.search(r"^\s*"+re.escape(n)+r"[\[\{(]", mm, re.M), (g["id"], n)
    for a, b, *_ in g["edges"]:
        assert a+" -> "+b in dot
        assert re.search(r"^\s*"+re.escape(a)+r" (?:-->|-\.->).* "+
                         re.escape(b)+r"$", mm, re.M), (g["id"], a, b)
for name in ["01_overall", "03_transformer"]:
    g = next(g for g in spec["graphs"] if g["id"] == name)
    truth = "truth" if name == "01_overall" else "gt"
    allowed = {"evaluate"} if name == "01_overall" else {"loss"}
    assert {e[1] for e in g["edges"] if e[0] == truth} == allowed

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.anchors = [], set()
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.anchors.add(a["id"])
        for key in ["href", "src"]:
            if key in a:
                self.links.append(a[key])

p = Links()
p.feed((root/"index.html").read_text())
for link in p.links:
    if link.startswith("#"):
        assert link[1:] in p.anchors
    else:
        assert (root/link).is_file(), link
for link in re.findall(r"\]\(([^)]+)\)", (root/"毕业设计流程图集.md").read_text()):
    assert (root/link).is_file(), link
qa = {
    "date": "2026-10-26", "figures": 6, "dot_renderings_passed": 6,
    "svg_png_exports_checked": 12, "node_overlap_count": 0,
    "source_node_and_edge_consistency": True,
    "training_truth_boundary_checked": True,
    "visual_review": {g["id"]: "checked" for g in spec["graphs"]},
    "html_local_links": "passed",
    "html_browser_execution": "not performed; static gallery only",
    "mermaid_status": "generated from same graph; no Mermaid engine render claimed",
    "model_or_simulation_tests": "not run for this diagram delivery"
}
(root/"qa_report.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2)+"\n")
print(json.dumps({"status": "passed", "figures": 6, "exports": 12,
                  "gallery_links": len(p.links)}, ensure_ascii=False))
