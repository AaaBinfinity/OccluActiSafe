# OccluActiSafe

**Occlusion-Aware Active Safety for Cooperative Driving**

毕业设计课题：基于 Transformer 的超视野车路协同主动避撞方案研究。

核心交付是可脱离界面运行的车路协同主动避撞算法包。路侧补充盲区观测并进行协同预测，车端负责最终风险复核、AEB/AES 决策和控制；CARLA 与车内视角、辅助页面用于验证和展示。建筑遮挡来车及行人/骑行者“鬼探头”均为必做场景。

项目与远程仓库名为 `OccluActiSafe`，Python 包名为 `occluactisafe`，预测模型工作名为 `RC-STGT`。

## 当前阶段

用户已授权 N01，仓库新增轻量可安装包与环境检查入口；其测试、证据和验收状态以[节点计划](docs/项目节点实施与验收计划.md)为准。算法实现、模型训练、通信服务及 CARLA 闭环仍待对应节点授权。检测器须统一比较YOLO26、YOLOv8、YOLO11及有必要的其他候选，按本项目实测结果选用，不预设胜者。53 项正式测试与 B01～B08 补充测试是规格；概念图中的数值不能作为实验结果。

## 阅读入口

1. [项目节点实施与验收计划](docs/项目节点实施与验收计划.md)：33个核心节点、2个条件扩展；逐项范围、测试、标准、证据和结果回填。
2. [工程架构与文件结构设计](docs/工程架构与文件结构设计.md)：代码框架、技术栈与接口。
3. [文档导航](docs/README.md)：阅读顺序与各资料用途。
4. [项目现状与开发顺序](docs/项目现状与开发顺序.md)：已有证据与开发依赖。
5. [毕业设计研究方案](docs/毕业设计研究方案.md)：研究目标、范围与实验设计。
6. [详细技术路线](docs/详细技术路线.md)：算法、接口、参数和测试规格。
7. [开发与 Git 协作规范](docs/开发与Git协作规范.md)：中文提交、小步管理及文档维护。

以后每次编码先读节点计划，只做用户明确授权的一个节点；完成必需测试后在同一文件回填真实结果并提交，然后停止等待用户验收。用户验收通过也不自动授权下一节点。

设计补充：[数据来源与验证方案](docs/数据来源与验证方案.md)。后续Python训练/评测脚本支持IDE直接点运行，命令行保留作可选批量入口；首次环境和数据准备需先完成。

## N01 首次安装与环境检查

包元数据要求 Python 3.10 或以上，主开发环境目标为 Python 3.11；使用本机已安装的对应解释器创建虚拟环境。当前轻量包没有运行时第三方依赖，构建安装仍需要 Python 打包工具。下面的命令在仓库根目录执行；实际验证过的系统和解释器版本见节点计划。

macOS / Linux：

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/check_environment.py
```

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts\check_environment.py
```

首次安装后，IDE 选择 `.venv` 中的 Python 解释器，打开 `scripts/check_environment.py` 并直接运行，无需填写参数；检查按脚本位置确定仓库根目录，不依赖工作目录。若指定版本的解释器命令不可用，请先安装或选择已有的受支持解释器。检查仅报告当前轻量工程和可选模块发现情况，不自动安装、下载或验证 GPU/CARLA 兼容性；缺少必要项目包或 Python 版本过低会失败并提示缺项。

专题入口：[算法成品与鬼探头](docs/算法成品与盲区鬼探头方案.md) · [车路通信](docs/车路通信方案.md) · [论文流程图](docs/论文流程图集.md) · [辅助界面概念图](docs/成品形态与界面效果图.md)。

## 资料维护

后续由助手承担代码与Git日常管理，按步骤、按模块完成小改动后检查并提交。提交使用标准小写英文type，scope、描述、正文与页脚以中文为主，结构和示例见 [Git协作规范](docs/开发与Git协作规范.md)。节点实际结果回填计划；版本历史统一由Git追溯，不维护通用历史记录或备份目录。`thesis/`保留，其他约定见 [AGENTS.md](AGENTS.md)。

文档静态检查：`python3 scripts/check_docs.py`。该检查覆盖本地资源、JSON 和图资产清单，不代表算法测试、外链可用性或 CARLA 已验证。
