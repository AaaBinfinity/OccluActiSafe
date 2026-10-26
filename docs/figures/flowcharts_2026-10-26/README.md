# 初版彩色流程图

保留6张原始SVG与统一图源 `source/graphs.json`。依据为主方案V0.9、技术路线V1.1，表示当时的拟议设计。

旧图不作为当前完整规格：YOLO11m是示例，现行要求至少比较三族并在N17/N26分别预选/定选；“已到达”不能代替重组/解码/校验完成的usable_time，“验证集校准”现已拆为模型验证与专用风险校准。SAT仅覆盖车辆OBB分支，异构目标另验。以 [节点计划](../../项目节点实施与验收计划.md) 和当前专题接口为准，SVG和图源保留不改。

- [01_overall](images/01_overall.svg)
- [02_cooperative_perception](images/02_cooperative_perception.svg)
- [03_transformer](images/03_transformer.svg)
- [04_risk_decision](images/04_risk_decision.svg)
- [05_mpc](images/05_mpc.svg)
- [06_testing](images/06_testing.svg)

默认论文插图见 [黑白图集](../../论文流程图集.md)。旧PNG、重复页面、DOT/Mermaid导出及旧生成器已清理，必要时从Git历史提取。
