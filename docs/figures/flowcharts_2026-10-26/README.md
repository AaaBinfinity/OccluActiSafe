# 彩色流程图（历史留档）

日期：2026-10-26。依据主方案 V0.9、详细技术路线 V1.1；最新论文插图请使用 [黑白版](../flowcharts_paper_2026-10-26/README.md)。

- `index.html`：离线预览与下载入口。
- `images/`：6 张 SVG 与 6 张 PNG。
- `source/graphs.json`：节点与关系的统一可编辑源。
- `source/*.mmd`、`source/*.dot`：Mermaid 与 Graphviz 源码。
- `generate.cjs`：重新生成脚本；依赖 `@viz-js/viz` 与 `sharp`。
- `manifest.json`、`qa_report.json`：原导出清单与历史核查记录。

## 重新生成

在仓库根目录安装依赖并运行：

```sh
npm install --no-save --package-lock=false @viz-js/viz sharp
node docs/figures/flowcharts_2026-10-26/generate.cjs
```

默认使用 Node.js 的普通模块解析。如已有依赖目录，可显式指定 `FLOWCHART_NODE_MODULES`，例如：

```sh
FLOWCHART_NODE_MODULES=/absolute/path/to/node_modules node docs/figures/flowcharts_2026-10-26/generate.cjs
```

渲染使用系统中文字体 `PingFang SC`；字体和包版本变化可能影响布局，重新生成后应检查差异与视觉效果。脚本会更新本目录的图像、图源、清单和展示文档。

## 静态检查已有导出

```sh
python3 docs/figures/flowcharts_2026-10-26/check_exports.py
python3 docs/figures/flowcharts_2026-10-26/check_exports.py --output docs/validation/彩色流程图静态检查_新记录.json
```

默认仅向终端输出 JSON；`--output` 只允许写入新文件，拒绝覆盖历史 `qa_report.json`。检查范围为已有 SVG/PNG、节点与边、真值用途及本地链接；重叠数量取自旧清单，不重新渲染或进行视觉复核，也不运行模型与仿真。

这是研究方案的可视化资料，不是已实现系统或已完成实验的证据。
