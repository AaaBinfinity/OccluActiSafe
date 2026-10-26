# OccluActiSafe

**Occlusion-Aware Active Safety for Cooperative Driving**

中文课题：基于 Transformer 的超视野车路协同主动避撞方案研究。

用户确定项目名为OccluActiSafe：Occlu取自Occlusion（遮挡），Acti取自Active（主动），Safe表示安全。仓库名和Python包名统一约定为 `occluactisafe`；当前目录和模型工作名RC-STGT继续保留。

核心成品确定为可脱离界面运行的车路协同主动避撞算法包，最终决策与控制在车端。必做盲区来车和行人/骑行者鬼探头，以车内视角展示；CARLA和管理页面负责验证与观察。

- [算法成品与盲区、鬼探头方案（含车内视角图）](docs/算法成品与盲区鬼探头方案.md)
- [车内视角原图预览](docs/figures/driver_view_2026-10-26/index.html)
- [研究方案 Markdown](docs/毕业设计研究方案.md)
- [详细技术路线 Markdown](docs/详细技术路线.md)
- [车路通信方案：链路、消息、实现与测试](docs/车路通信方案.md)
- [成品形态与辅助界面概念图](docs/成品形态与界面效果图.md)
- [成品界面效果图离线预览](docs/figures/product_mockups_2026-10-26/index.html)
- [论文流程图集（黑白版）](docs/论文流程图集.md)
- [流程图离线预览与下载](docs/figures/flowcharts_paper_2026-10-26/index.html)
- [技术路线参考示例核查](docs/validation/技术路线参考示例核查_2026-10-14.json)
- [更新记录](docs/更新记录.md)
- [原始设想](./docs/随手记.txt)



建议路线：公开真实轨迹数据验证，加 CARLA 车路双视角感知与主动避撞闭环。建议由路侧边缘计算处理广域感知与协同预测，车端负责消息检查、最终避撞决策和失联安全回退。核心研究点为信息可靠性建模、多模态碰撞风险和风险驱动决策。



本课题每次实质讨论后更新方案和记录，涉及技术实施时同步维护独立技术路线。保留旧版本，不把计划写成已完成实验。
