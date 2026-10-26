"""Statically inspect existing diagram exports without rendering or visual review."""
import argparse
import json
import re
import struct
from datetime import datetime, timezone
from pathlib import Path
from html.parser import HTMLParser
from xml.etree import ElementTree as ET

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, help="将静态检查结果另存为新 JSON 文件；默认只输出到终端")
args = parser.parse_args()
if args.output and args.output.resolve() == (root/"qa_report.json").resolve():
    parser.error("不能覆盖历史 qa_report.json；请使用新的报告文件名")
if args.output and args.output.exists():
    parser.error("输出文件已存在；请使用新的报告文件名")
spec = json.loads((root/"source/graphs.json").read_text())
manifest = json.loads((root/"manifest.json").read_text())
assert spec["graphs"], "No source graphs"
assert len(spec["graphs"]) == len(manifest["figures"]), "Source/manifest figure count differs"
for g, m in zip(spec["graphs"], manifest["figures"]):
    assert g["id"] == m["id"], "Source/manifest figure order differs"
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
    "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "passed",
    "scope": "static checks of existing source and exports only",
    "figures": len(spec["graphs"]),
    "svg_png_exports_checked": 2 * len(spec["graphs"]),
    "source_node_and_edge_consistency": True,
    "training_truth_boundary_checked": True,
    "html_local_links": "passed",
    "new_rendering_performed": False,
    "visual_review_performed": False,
    "browser_execution_performed": False,
    "model_or_simulation_tests_performed": False,
    "historical_metadata": {
        "manifest_date": manifest.get("date"),
        "manifest_reported_node_overlap_count": sum(m["node_overlaps"] for m in manifest["figures"]),
        "note": "Overlap values are read from the archived manifest, not remeasured in this check."
    }
}
report = json.dumps(qa, ensure_ascii=False, indent=2)+"\n"
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(report)
print(report, end="")
