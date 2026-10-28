# OccluActiSafe

**Occlusion-Aware Active Safety for Cooperative Driving**

毕业设计课题：基于 Transformer 的超视野车路协同主动避撞方案研究。

核心交付是可脱离界面运行的车路协同主动避撞算法包。路侧补充盲区观测并进行协同预测，车端负责最终风险复核、AEB/AES 决策和控制；CARLA 与车内视角、辅助页面用于验证和展示。建筑遮挡来车及行人/骑行者“鬼探头”均为必做场景。

项目与远程仓库名为 `OccluActiSafe`，Python 包名为 `occluactisafe`，预测模型工作名为 `RC-STGT`。

## 当前阶段

N01、N02 已经用户验收；N03 已审计用户提供的 TFD 样例，但正式数据尚未就绪，节点仍受阻，详情见[节点计划](docs/项目节点实施与验收计划.md)。预测与控制算法、模型训练、通信服务及 CARLA 闭环仍待对应节点授权。检测器须统一比较YOLO26、YOLOv8、YOLO11及有必要的其他候选，按本项目实测结果选用，不预设胜者。53 项正式测试与 B01～B08 补充测试是规格；概念图中的数值不能作为实验结果。

## 阅读入口

1. [项目节点实施与验收计划](docs/项目节点实施与验收计划.md)：33个核心节点、2个条件扩展；逐项范围、测试、标准、证据和结果回填。
2. [工程架构与文件结构设计](docs/工程架构与文件结构设计.md)：代码框架、技术栈与接口。
3. [文档导航](docs/README.md)：阅读顺序与各资料用途。
4. [项目现状与开发顺序](docs/项目现状与开发顺序.md)：已有证据与开发依赖。
5. [毕业设计研究方案](docs/毕业设计研究方案.md)：研究目标、范围与实验设计。
6. [详细技术路线](docs/详细技术路线.md)：算法、接口、参数和测试规格。
7. [开发与 Git 协作规范](docs/开发与Git协作规范.md)：中文提交、小步管理及文档维护。
8. [国内外研究调研报告](docs/国内外研究调研报告.md)：协同感知、轨迹预测、遮挡风险与主动避撞的国内外文献和出处。

以后每次编码先读节点计划，只做用户明确授权的一个节点；完成必需测试后在同一文件回填真实结果并提交，然后停止等待用户验收。用户验收通过也不自动授权下一节点。

数据准备统一见 [数据来源与验证方案](docs/数据来源与验证方案.md)：第1.1节为官方入口，第4.1～4.3节为SPD下载清单、图像/点云区别和完整性核验，第4.4节对应各实验节点。SPD按RGB路线准备，点云分卷暂不需要，标注与标定仍须保留并核查。后续Python训练/评测脚本支持IDE直接点运行，命令行保留作可选批量入口；首次环境和数据准备需先完成。

## 使用 uv 安装与检查

项目使用 `pyproject.toml` 声明包和依赖，`uv.lock` 保存当前依赖解析结果，`.python-version` 选择默认 Python 3.11。共享包支持 Python 3.10 起；N02 使用 NumPy 和 PyYAML，运行时版本由锁文件按 Python 版本解析。构建依赖 `setuptools>=61` 不由这个锁文件固定。

首次准备时，先确保已安装 `uv`，然后在仓库根目录执行：

```sh
uv sync --locked
```

该命令按锁文件创建或同步 `.venv`，以可编辑方式安装本项目；必要时可能下载 Python 或构建依赖。`uv sync` 默认会移除环境中不属于当前项目的额外包，因此请不要将已有的其他项目环境当成本项目 `.venv` 使用。

准备完成后可在仓库根目录运行：

```sh
uv run --no-sync python scripts/check_environment.py
uv run --no-sync python -B -m unittest discover -s tests/unit -v
```

也可以在 IDE 中选择 `.venv/bin/python`（Windows 为 `.venv/Scripts/python.exe`），打开 `scripts/check_environment.py` 直接运行，无需填写参数。入口按脚本位置确定仓库根目录，不依赖工作目录，检查过程不会自动安装或下载；它只报告当前轻量工程和可选模块发现情况，不验证 GPU/CARLA 兼容性。缺少项目包或 Python 版本过低会失败并提示缺项。上述 uv 流程已在 macOS 验证；其他系统的实际验证状态见节点计划。

N02 的字段、消息时间和调用约定见[数据契约字典](docs/N02数据契约字典.md)。基础配置在 `configs/base.yaml`，由 `load_config()` 依次合并包默认、基础配置、选定任务、脚本覆盖与显式 CLI。

## 审计已取得的 TFD 数据

可在 IDE 选择 `.venv` 解释器直接运行 `scripts/prepare_data.py`；默认读取用户提供的 `data/raw/v2x_seq_tfd/V2X-Seq-TFD-Example`，缺数据时明确失败，不会自动下载。也可在仓库根目录运行：

```sh
uv run --no-sync python scripts/prepare_data.py
# 数据在其他目录时，明确指定已有目录
uv run --no-sync python scripts/prepare_data.py --data-root /path/to/TFD-Example
```

正式数据需明确指定目录并使用 `--scope full`；脚本不会仅凭该标记宣称正式数据已就绪。报告的原始时间和坐标单位在核实前保持未确认；官方样例、许可及运行结果见 [N03 数据审计说明](docs/N03数据审计说明.md)。当前没有训练入口。

专题入口：[算法成品与鬼探头](docs/算法成品与盲区鬼探头方案.md) · [车路通信](docs/车路通信方案.md) · [论文流程图](docs/论文流程图集.md) · [辅助界面概念图](docs/成品形态与界面效果图.md)。

## 资料维护

后续由助手承担代码与Git日常管理，按步骤、按模块完成小改动后检查并提交。提交使用标准小写英文type，scope、描述、正文与页脚以中文为主，结构和示例见 [Git协作规范](docs/开发与Git协作规范.md)。节点实际结果回填计划；版本历史统一由Git追溯，不维护通用历史记录或备份目录。`thesis/`保留，其他约定见 [AGENTS.md](AGENTS.md)。

文档静态检查：`python3 scripts/check_docs.py`。该检查覆盖本地资源、JSON 和图资产清单，不代表算法测试、外链可用性或 CARLA 已验证。
