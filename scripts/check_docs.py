#!/usr/bin/env python3
"""仅用标准库静态检查文档链接、JSON 及已有图集清单；不会联网。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR_NAMES = {".git", "node_modules", ".venv", "venv", "__pycache__",
                   ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cache", ".tox", ".nox"}
LOCAL_OUTPUT_DIR_NAMES = {"data", "artifacts", "runs", "outputs", "checkpoints"}


class Links(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links = []
        self.anchors = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for key in ("id", "name" if tag == "a" else "id"):
            if attrs.get(key):
                self.anchors.add(attrs[key])
        for key in ("href", "src", "poster"):
            if attrs.get(key):
                self.links.append((attrs[key], self.getpos()[0]))

    handle_startendtag = handle_starttag


def blank(match):
    return re.sub(r"[^\n]", " ", match.group())


def reject_constant(value):
    raise ValueError(f"JSON 不允许非有限数值常量：{value}")


def markdown_links(content):
    """支持当前文档使用的行内链接、引用定义及嵌入 HTML；忽略代码。"""
    lines = content.splitlines(keepends=True)
    fence = None
    for i, line in enumerate(lines):
        match = re.match(r"^[ \t]{0,3}(`{3,}|~{3,})", line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            lines[i] = re.sub(r"[^\n]", " ", line)
        elif fence:
            lines[i] = re.sub(r"[^\n]", " ", line)
    text = "".join(lines)
    text = re.sub(r"<!--.*?-->", blank, text, flags=re.S)
    text = re.sub(r"(`+)(?!`)(.*?)(?<!`)\1(?!`)", blank, text, flags=re.S)
    found = []
    for match in re.finditer(r"(?<!\\)\]\(", text):
        start = match.end()
        end, depth = start, 0
        angled = text[start:start + 1] == "<"
        if angled:
            end = text.find(">", start + 1)
            if end < 0:
                continue
            destination = text[start + 1:end]
        else:
            while end < len(text):
                char = text[end]
                if char == "\\":
                    end += 2
                    continue
                if char == "(":
                    depth += 1
                elif char == ")":
                    if depth == 0:
                        break
                    depth -= 1
                elif char.isspace() and depth == 0:
                    break
                end += 1
            destination = text[start:end]
        if destination:
            found.append((re.sub(r"\\([\\() ])", r"\1", destination),
                          text.count("\n", 0, start) + 1))
    for match in re.finditer(r"^[ ]{0,3}\[[^\]\n]+\]:[ \t]*(?:<([^>]+)>|(\S+))", text, re.M):
        found.append((match.group(1) or match.group(2),
                      text.count("\n", 0, match.start()) + 1))
    html = Links()
    html.feed(text)
    return found + html.links


class Check:
    def __init__(self, root):
        self.root = root
        self.counts = dict(markdown_files=0, html_files=0, json_files=0,
                           local_links=0, remote_urls_skipped=0,
                           other_urls_skipped=0, absolute_paths_skipped=0,
                           html_anchors=0, non_html_anchors_skipped=0,
                           manifests=0, sha256_checks=0, png_dimension_checks=0)
        self.errors = []
        self.json_data = {}
        self.html_cache = {}

    def relative(self, path):
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    def error(self, path, message, line=None):
        item = {"file": self.relative(path), "message": message}
        if line is not None:
            item["line"] = line
        self.errors.append(item)

    def html(self, path):
        if path not in self.html_cache:
            parser = Links()
            parser.feed(path.read_text(encoding="utf-8"))
            self.html_cache[path] = parser
        return self.html_cache[path]

    def link(self, source, value, line):
        try:
            url = urlsplit(value.strip())
        except ValueError as exc:
            self.error(source, f"无法解析链接 {value!r}：{exc}", line)
            return
        if url.scheme in ("http", "https") or url.netloc:
            self.counts["remote_urls_skipped"] += 1
            return
        if url.scheme:
            self.counts["other_urls_skipped"] += 1
            return
        if url.path.startswith("/"):
            self.counts["absolute_paths_skipped"] += 1
            return
        target = (source.parent / unquote(url.path)).resolve() if url.path else source
        if url.path:
            self.counts["local_links"] += 1
            if not target.is_relative_to(self.root):
                self.error(source, f"相对链接越出仓库：{value}", line)
                return
            if not target.exists():
                self.error(source, f"相对链接目标不存在：{value}", line)
                return
        if url.fragment:
            if target.suffix.lower() in (".html", ".htm") and target.is_file():
                self.counts["html_anchors"] += 1
                try:
                    if unquote(url.fragment) not in self.html(target).anchors:
                        self.error(source, f"HTML 锚点不存在：{value}", line)
                except (OSError, UnicodeError) as exc:
                    self.error(source, f"无法读取锚点目标 {value}：{exc}", line)
            else:
                self.counts["non_html_anchors_skipped"] += 1

    def digest(self, manifest, path, expected):
        self.counts["sha256_checks"] += 1
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            self.error(manifest, f"SHA-256 不匹配：{self.relative(path)}")

    def png(self, manifest, path, width, height):
        self.counts["png_dimension_checks"] += 1
        with path.open("rb") as handle:
            header = handle.read(24)
        if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
            self.error(manifest, f"无效 PNG 文件头：{self.relative(path)}")
        elif struct.unpack(">II", header[16:24]) != (width, height):
            self.error(manifest, f"PNG 尺寸不匹配：{self.relative(path)}，清单为 {width}×{height}")

    def manifest(self, path, data):
        self.counts["manifests"] += 1
        try:
            if "images" in data:
                for image in data["images"]:
                    target = (self.root / image["path"]).resolve()
                    if not target.is_relative_to(self.root):
                        raise ValueError("图片路径越出仓库")
                    self.digest(path, target, image["sha256"])
                    self.png(path, target, image["width"], image["height"])
            elif "figures" in data:
                for figure in data["figures"]:
                    figure_id = figure["id"]
                    if not isinstance(figure_id, str) or not re.fullmatch(r"[\w-]+", figure_id):
                        raise ValueError("图编号含不支持的字符")
                    if "source_sha256" in figure:
                        self.digest(path, path.parent / "source" / f"{figure_id}.dot", figure["source_sha256"])
                    if "svg_sha256" in figure:
                        self.digest(path, path.parent / "images" / f"{figure_id}.svg", figure["svg_sha256"])
                    if not any(key in figure for key in ("source_sha256", "svg_sha256")):
                        raise ValueError(f"{figure_id} 缺少受支持的哈希字段")
                    self.png(path, path.parent / "images" / f"{figure_id}.png",
                             figure["png_width"], figure["png_height"])
            else:
                raise ValueError("不支持的清单结构；需要 images 或 figures")
        except (KeyError, TypeError, ValueError, OSError) as exc:
            self.error(path, f"无法完成清单检查：{exc}")

    def run(self):
        for directory, subdirs, names in os.walk(self.root):
            current = Path(directory)
            subdirs[:] = sorted(name for name in subdirs if name not in CACHE_DIR_NAMES
                                and not (current == self.root and name in LOCAL_OUTPUT_DIR_NAMES)
                                and current / name != self.root / "docs" / "history"
                                and not (current / name).is_symlink())
            for name in sorted(names):
                path = current / name
                suffix = path.suffix.lower()
                if suffix not in (".md", ".html", ".htm", ".json"):
                    continue
                key = {".md": "markdown_files", ".html": "html_files", ".htm": "html_files", ".json": "json_files"}[suffix]
                self.counts[key] += 1
                try:
                    content = path.read_text(encoding="utf-8")
                    if suffix == ".json":
                        self.json_data[path] = json.loads(content, parse_constant=reject_constant)
                    else:
                        links = markdown_links(content) if suffix == ".md" else self.html(path).links
                        for value, line in links:
                            self.link(path, value, line)
                except (OSError, UnicodeError, ValueError) as exc:
                    self.error(path, f"读取或解析失败：{exc}")
        for path in sorted((self.root / "docs" / "figures").glob("*/manifest.json")):
            if path in self.json_data:
                self.manifest(path, self.json_data[path])
        return {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "passed" if not self.errors else "failed",
                "counts": self.counts, "errors": self.errors,
                "scope": "静态检查当前仓库 MD/HTML 相对链接和 HTML 锚点、JSON 语法、四类既有图集清单的哈希与 PNG 文件头尺寸。",
                "limits": ["不联网；远程 URL 仅计数，未验证可访问性或内容。",
                           "跳过 docs/history、符号链接目录，以及任意层级依赖/缓存目录：" + "、".join(sorted(CACHE_DIR_NAMES)) + "。",
                           "跳过仓库根本地产物目录：" + "、".join(sorted(LOCAL_OUTPUT_DIR_NAMES)) + "。",
                           "Markdown 提取器覆盖当前仓库使用的链接形式，不是完整 CommonMark 解析器。",
                           "不验证 Markdown 标题锚点、其他格式文件锚点、绝对路径或带协议的非 HTTP URL。",
                           "不检查 CSS url()/srcset/动态生成链接；不解码 PNG 像素或检查图形排版。",
                           "不证明算法、模型、前端或 CARLA 仿真运行正确。"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="可选：将本次结果写入指定新 JSON；拒绝覆盖已有文件，路径相对当前工作目录")
    args = parser.parse_args()
    report = Check(ROOT).run()
    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        except FileExistsError:
            print(f"报告保存失败：目标文件已存在，拒绝覆盖：{args.output.resolve()}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"报告保存失败：{exc}", file=sys.stderr)
            return 2
    counts = report["counts"]
    print(f"文档静态检查：{'通过' if report['status'] == 'passed' else '失败'}")
    print(f"Markdown {counts['markdown_files']} 份，HTML {counts['html_files']} 份，JSON {counts['json_files']} 份；相对链接 {counts['local_links']} 个。")
    print(f"图集清单 {counts['manifests']} 份，SHA-256 {counts['sha256_checks']} 项，PNG 尺寸 {counts['png_dimension_checks']} 项。")
    print(f"未联网：跳过远程 URL {counts['remote_urls_skipped']} 个；其他协议 {counts['other_urls_skipped']} 个、绝对路径 {counts['absolute_paths_skipped']} 个、非 HTML 锚点 {counts['non_html_anchors_skipped']} 个。")
    print("范围：相对文件/图片链接、HTML 锚点、JSON 语法及图集清单；不验证外网、渲染或算法运行。")
    for error in report["errors"]:
        line = f":{error['line']}" if "line" in error else ""
        print(f"错误 {error['file']}{line}：{error['message']}")
    if args.output:
        print(f"本次报告：{args.output.resolve()}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
