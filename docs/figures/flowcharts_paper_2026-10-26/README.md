# 论文流程图（黑白版）

- [图集正文](../../论文流程图集.md)：唯一正文入口，直接预览SVG。
- `images/`：6张SVG原图。
- [合并draw.io图源](论文流程图.drawio)：一个文件编辑全部6张图。
- `source/figures.json`：节点位置、连线与文字的统一来源。
- `manifest.json`：当前6张SVG的SHA-256清单。

图宽148.17 mm，字号10.5 pt，线宽0.75 pt。流程图表示拟实施设计，不是模型或CARLA运行证据。

既有图源按早期方案保留：YOLO11m不是最终选型，当前至少三族对比（N17离线预选、N26闭环选型）；车辆SAT图不替代行人圆/骑行者几何。旧的到达/校准标签以usable_time和专用风险校准集解释，最新执行约束见 [节点计划](../../项目节点实施与验收计划.md)。本次只补说明，不重绘SVG或改图源/清单。

## 重新生成

```sh
npm install --no-save --package-lock=false sharp
node docs/figures/flowcharts_paper_2026-10-26/render.cjs
```

脚本仅更新SVG、合并draw.io与清单，不再生成PNG、预览图、逐图draw.io、HTML或重复Markdown。需要栅格格式时按用途临时导出。

默认使用Node模块解析；已有依赖时可设置 `FLOWCHART_NODE_MODULES`。文本测量使用 `sharp`，系统字体为 `Songti SC` 与 `Times New Roman`。字体或依赖变化后须检查差异与实际视觉效果。
