# 流程图导出文件

日期：2026-10-26

- index.html：离线预览与下载入口。
- images/：6 张 SVG 与 6 张 PNG。
- source/graphs.json：节点与关系的统一可编辑源。
- source/*.mmd：Mermaid；source/*.dot：Graphviz。
- generate.cjs：再生成脚本；依赖 @viz-js/viz 与 sharp。
- manifest.json：导出尺寸、节点/边数与布局检查。

这是既有毕业设计方案的可视化交付，不是已实现系统或已完成实验的证据。
