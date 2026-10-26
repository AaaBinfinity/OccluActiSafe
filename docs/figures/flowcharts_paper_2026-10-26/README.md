# 论文流程图（黑白版）

`index.html` 为离线预览入口；`images/` 中为 SVG、600 dpi PNG 及预览图；`论文流程图.drawio` 可编辑全部 6 张图；`source/figures.json` 与 `render.cjs` 用于重新生成。正文图题与说明不嵌入图片。

尺寸单位为 pt，图宽 420 pt（148.17 mm）、字号 10.5 pt、线宽 0.75 pt。黑线、白底、无装饰标题与脚注。图集表示拟实施设计，不是训练或 CARLA 实验结果。

## 重新生成

在仓库根目录安装依赖并运行：

```sh
npm install --no-save --package-lock=false sharp
node docs/figures/flowcharts_paper_2026-10-26/render.cjs
```

默认使用 Node.js 的普通模块解析。如已有依赖目录，可显式指定 `FLOWCHART_NODE_MODULES`：

```sh
FLOWCHART_NODE_MODULES=/absolute/path/to/node_modules node docs/figures/flowcharts_paper_2026-10-26/render.cjs
```

字体依赖系统中的 `Songti SC` 与 `Times New Roman`。字体和包版本变化可能影响布局，重新生成后应检查差异与视觉效果。脚本会更新本目录的 SVG、PNG、draw.io 文件、清单和展示文档；历史 `qa_report.json` 保留原核查时间与范围，不能替代新导出的复核。
