# N17单型号三seed连续训练

默认队列为 **YOLOv8n：17→29→43**。一次选参数并确认，后台串行执行；每项完成后自动接下一项，三项处理完退出。用户于2026-11-05明确选择“仅YOLOv8n，三个seed共3项”；本次交付脚本，不由助手启动正式训练，N18未授权。

## 使用

在项目终端运行，或在IDE直接运行同一脚本：

```powershell
.\.venv\Scripts\python.exe .\scripts\train_n17_seeds.py
```

菜单依次选择模型、学习率、batch、操作。全部保持默认即YOLOv8n、lr0=0.001、batch32；操作选择 **3 / train**，最后输入 **y** 确认一次。seed固定为17、29、43，不逐项提问。也可直接执行：

```powershell
# 仅预览，没有任何训练或后台进程
.\.venv\Scripts\python.exe scripts/train_n17_seeds.py --model YOLOv8n --action train --dry-run

# 用户显式启动完整三seed队列
.\.venv\Scripts\python.exe scripts/train_n17_seeds.py --model YOLOv8n --action train

# 若只想补29和43，可显式缩小队列；不声称已完成共同配方三seed
.\.venv\Scripts\python.exe scripts/train_n17_seeds.py --model YOLOv8n --seeds 29 43 --action train
```

无参数菜单默认check，须明确选择train才训练。队列的check/plan只核对冻结索引、输入清单和所选官方初始权重SHA，并保存队列身份；不执行CUDA或全量40,381资产预检。每项真实训练开始前仍调用已有完整独立预检，不能以队列计划替代训练环境验收。路径相对项目根解析，启动目录不影响行为。

默认配置在[队列选项](../configs/experiments/n17_seed_queue.py)，资源参数继承[日常新任务配置](../configs/experiments/n17_local_fast.py)：最多300轮、最少15轮、早停耐心10轮、n档batch=nbs32、训练worker4/每worker预取2、验证worker2/batch≤4、CPU4、AMP/960、lr0=.001。2026-11-09按本机32GB内存调整供数，当前配置与Windows机制通过，GPU吞吐仍待短测确认，不自动缩batch；详见[32GB资源适配](N17本机训练吞吐优化.md)。

旧YOLOv8n/seed17为50/30、worker1；不能自动拼入当前15/10、worker4矩阵。默认三项均按本次冻结配方**从官方预训练权重新建**，不覆盖旧训练。模型菜单允许选择既有15种型号中的一个；切换尺寸仍按32/4/2/1/1建议batch起点，其他尺寸容量未实测。

## 自动运行和查看

启动后打印队列绝对路径和状态命令，目录位于H盘N17/02_运行产物/n17_seed_queue_<uuid>。Windows后台宿主及每项训练worker用CREATE_NO_WINDOW启动，宿主和实际模型worker核验控制台句柄；关掉启动终端后仍可接续。重启/注销/结束进程/拔盘会中断，脚本不自动重试或重放未保存曝光。

```powershell
.\.venv\Scripts\python.exe scripts/train_n17_seeds.py --status 'H:\实际队列绝对目录'
```

实时查看队列目录的`队列状态.json`：pending→waiting_for_gpu→running→completed/failed。GPU被当前训练占用时等待其正常释放锁，默认每15秒检查、每项最多24小时；不会终止现有进程。每项目录的`队列控制台.txt`记录其实际训练run地址，`训练任务启动身份.json`给出原任务参数位置；完整逐批训练进度在该run的`控制台.txt`。关闭终端后可用编辑器打开这些文件查看，没有新增实时网页。

某项失败会保存原异常、退出码和已有产物，继续下一seed；不自动重试、改变配方或降低门槛。最后状态为completed或completed_with_failures；后者不能当三seed完成。源码/输入身份变化或等待超时会阻断剩余任务。重复调用同一宿主/单项被票据和锁拒绝；中断后的模型续训沿用原单任务入口及同源码/配方/GPU保护，先核实曝光预算，不提供无条件整队列重跑。

## 输入、方法、输出与断言

| 模块/函数 | 输入和处理 | 输出/必需断言 |
| --- | --- | --- |
| queue_options / choose_queue | 单型号及17→29→43有序不重复子集、现有资源配置字节；菜单一次确认 | 固定源字节和所有实际覆盖；非法seed/资源/动作拒绝，取消或dry-run不启动进程 |
| create_queue / verify_identity | 已提交源码、输入清单/冻结索引、固定官方权重SHA；冻结同一配置供全部seed | 队列计划/源指纹/输入身份/配置SHA；每项启动前及返回后复核；3×300上限为15,302,700曝光，实际按早停另记 |
| start_background / run_queue | 单后台宿主、队列锁、GPU锁；每项独立子进程，结束后再执行下一项 | 启动票据、宿主身份、状态JSON、追加事件JSONL和中文结果汇总CSV；无重试或自动恢复；失败/未执行指标留空 |
| worker / completed_result | 复用run_local的真实输入/CUDA预检、训练/保存/图集与完整训练审计 | status=completed、正确型号/seed/源码、配置和审计SHA均正确才记成功；三个实际配方SHA相同，否则记失败 |
| execute_training_worker / model_worker | 只在新队列独立进程内替换启动器，保持原CUDA_VISIBLE_DEVICES映射 | 宿主与模型worker无控制台，任务路径/SHA与输出边界复核；既有训练源码和活动任务不修改 |

各seed仍在独立原始run中保存完整逐批次/实际更新、逐epoch验证、原生参考YOLO图集、分页PNG/SVG、诊断样例和来源SHA；weights只含last.pt、best.pt。成功项的`单项结果.json`链接真实run/审计/双权重；失败项保留异常与原始控制台，已进入训练时另有`训练任务启动身份.json`追溯实际run。最终`队列结果汇总.csv`按实际完成轮数和曝光抄录，失败不写假零指标，其已消耗曝光须从原始run复核。媒体/权重/大记录均留H盘，不推普通Git。完成结果按既有流程归档中文图、核对最终run及last入口，计划/失败不能当训练最终产物。

本次机制测试覆盖串行成功、失败继续、缺回执/配方不一致拒绝、源码/输入/选项篡改阻断、不可重复执行、GPU等待超时、菜单取消和参数真实绑定、Windows真实无控制台负例及异地CLI预览；补跑N17及N16回归。机制夹具不是真实模型训练。本次无GPU训练/推理，三项真实训练、统一FP32/ByteTrack评分与模型准入等待用户实际运行；不可用队列completed代替完整N17比较。被测提交、实际环境、全部检查数量与证据回填[同一节点计划](项目节点实施与验收计划.md)。
