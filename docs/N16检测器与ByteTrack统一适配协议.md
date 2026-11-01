# N16 检测器与 ByteTrack 统一适配协议

更新时间：2026年11月01日。

用户于2026-11-01回复“验收通过 继续执行下一个节点”，据此验收N16并确认本协议提交的门槛和排序建议。N16必需检查现已全部满足；下文保留确认前的实测和限制，旧配置/源码/运行哈希不改写。已确认门槛抄录至N17冻结配置，当前只授权N17。

## 1 范围与输入隔离

N16为检测器提供统一原图像素框契约，实现配置注册、Ultralytics推理结果适配、两阶段ByteTrack风格跟踪、检测/跟踪评分与SPD开发验证样本读取。本节点不下载权重、不运行YOLO正式比较矩阵、不训练、不选型，也不进入N17。测试夹具中的人工检测框只验证接口和指标实现，不作为任何检测器实测结果。

SPD在线输入仅为RGB及其采集时间、图像尺寸和显式相机/episode命名空间。相机内参的路径与哈希只登记供N18定位使用，不传入检测器。类别、二维框、track_id及遮挡/截断字段仅进入离线评分或人工叠加。在线检测输出不得读取SPD标注、point cloud、未来帧或GT ID。

## 2 接口、数据与类别

| 接口 | 输入 | 输出与错误行为 |
| --- | --- | --- |
| `Detector.detect(frame, frame_id, timestamp_s, namespace)` | 原始H×W RGB帧、采集时间、命名空间 | `Detections`，框统一为原图`xyxy`像素、置信度`[0,1]`、规范类名；保留空帧/错误状态，越界和尺寸不符拒绝 |
| `create_detector(name, **config)` | 显式注册名及参数 | 构造适配器；未注册模型拒绝，未提供本地权重时不下载 |
| `ByteTrack.track(detections, timestamp)` | 一路一episode当前帧检测与采集时间 | 命名空间隔离的`PixelTrack`；时间不递增拒绝，低分孤立检测不建立确认轨迹 |
| `score_detections(frames, iou_threshold=.5)` | 完整评测帧、GT、检测 | TP/FP/FN、分母、precision/recall及mAP@IoU阈值；缺预测帧保留GT并计FN |
| `score_tracking(...)` | GT track_id及预测轨迹 | 全局IDF1、IDTP/IDFP/IDFN、ID切换、轨迹覆盖 |
| `score_detection_strata(...)` | 上述评测帧、相机侧及GT尺度/官方遮挡字段 | 按相机侧检测指标；按类别×目标尺度×官方遮挡状态报告GT分母和召回 |
| `load_spd_perception_frame(root, split_row)` | N15冻结分组清单中的一行 | 校验RGB哈希/尺寸，读取采集微秒、相机内参来源、原始类名、可见框及GT ID；返回GT评分对象与未支持类别明细 |

规范类别映射固定为：Car/Truck/Van/Bus→`vehicle`；Pedestrian→`pedestrian`；Cyclist/Tricyclist/Motorcyclist→`cyclist`；Barrowlist→`barrowlist`；TrafficCone→`traffic_cone`。原始类别始终保留，禁止把手推车目标静默归并为骑行者。命名空间格式为`source_id:sensor_id:episode_id`；相机分别建状态，episode变化后使用新滤波状态，即使局部轨迹号都为1也不串轨。

Ultralytics适配要求输入为RGB `H×W×3`，送入Ultralytics前明确转为其numpy输入约定的BGR；`Results.orig_shape`须与输入帧一致，并使用Ultralytics定义中已恢复至原图的`boxes.xyxy`，不会把模型letterbox内部坐标传给后续定位。空预测返回`empty`，推理异常/多结果/类映射缺失返回携带错误说明的`error`状态；原图尺寸/框越界等契约错误拒绝输出。

## 3 ByteTrack与评分细节

ByteTrack按当前配置执行高低分两阶段IoU关联：高分阈值0.5、低分范围[0.1,0.5)、新轨迹阈值0.5；高分关联IoU阈值0.3，随后未匹配轨迹与低分框以IoU阈值0.2恢复。轨迹框状态使用XYAH（中心、宽高比、高度）八维常速度Kalman状态及协方差预测；measurement更新后记录当前检测框。至少连续命中2帧才确认；最多保留30个漏帧或1.0秒，先到者为准。配置位于`configs/perception/n16.yaml`，未来公平比较须冻结同一配置哈希，不能对某个候选模型单独换跟踪策略。

检测匹配使用同帧、同类别、IoU≥0.5的一对一最大IoU分配。没有输出的预测行/帧不从GT分母移除；不存在检测时AP按0计入有GT的类别。IDF1按完整评测帧的GT ID与预测ID全局分配；ID switch按同一GT的时序匹配ID变化计。按像素面积划分small≤32 px、medium 33–96 px、large>96 px；框像素尺寸只用于评测分层，不改变检测坐标。

## 4 入口、配置与产物

IDE直接运行`python scripts/adapt_n16_spd.py`，默认配置`configs/perception/n16.yaml`，路径均由配置绝对路径解析；`--config`为可选覆盖。入口校验N15冻结split和完整审计报告内部/外部哈希，且逐帧核对RGB、相机标签、内参三类原件的N15不可变文件清单哈希；只遍历冻结SPD `model_val`角色，逐帧读取RGB与GT/内参来源，产出图像/类别/时间映射、未支持类别明细、分层统计与中文GT叠加图。输出保存在移动硬盘`artifacts/节点资料/N16检测器与ByteTrack统一适配/02_运行产物/<run_id>/`，包括冻结配置、来源哈希、JSON映射/统计、产物清单及PNG原图尺寸叠加图。GT叠加图仅作标注可视化，不代表预测结果。

## 5 测试与可确认门槛

| 规格 | 测试输入/预期 | 实测与产物 |
| --- | --- | --- |
| P01 | 100×60非正方形RGB原图及模型内部缩放过程；显式RGB→BGR，输出仍为原图xyxy；错尺寸拒绝、推理异常保留状态 | `tests.unit.test_n16_perception.DetectorAdapterTests`；另对真实SPD车端/路侧1920×1080 GT叠加图目测各1张 |
| P02 | 单目标10帧，第5帧score=.15且运动一致；保持同一确认ID；孤立低分不确认 | `ByteTrackTests.test_low_score_frame_five_recovers_existing_identity`、`test_isolated_low_score_does_not_create_confirmed_track` |
| P03 | 两相机均从局部ID=1开始并开始新episode；命名空间不同且reset后不沿用旧状态 | `ByteTrackTests.test_camera_and_episode_namespaces_never_share_ids_or_state` |
| P04 | 人工匹配片段含2个GT与1个正确检测；应TP=1/FP=0/FN=1/recall=.5，且缺失预测保留；验证IDF1/switch与尺度/遮挡分层分母 | `MetricTests`及实际SPD映射审计；没有运行真实检测模型，不报告真实mAP/召回/IDF1 |
| SPD适配 | 冻结split、N15报告与逐文件manifest哈希均核对；model_val真实RGB、标签、标定来源、时间、相机/episode适配 | 最终run `n16_spd_d8e26cd2ff324fa0a96b1f2a760dd481`；2,885帧/27,301目标，时间非递增0、未映射类别0，详见计划及硬盘JSON |
| 自动检查 | N16专项单元测试、全量单元测试、compileall和文档静态检查 | N16专项13/13通过；全量303/303通过（失败0、跳过0）；`compileall -q src scripts tests`通过；文档静态检查38份Markdown、18份JSON及352个本地链接通过。 |

N17入围需要数值门槛在验证结果开封前冻结并由用户确认。本次配置中的建议值为：IoU50召回vehicle≥0.95、pedestrian≥0.90、cyclist≥0.90；空帧FP≤0.10/帧；检测含预处理/推理/后处理p95≤100 ms；投影误差p95在0–20 m≤0.5 m、20–40 m≤1.0 m。该值是基于10 Hz 100 ms预算、N13实测制动和已审计类别分母给出的待确认协议建议，不是实测门槛或模型性能结论。建议候选排序为：先同时满足三类召回、空帧误警和端到端延迟硬门槛；合格候选按vehicle/pedestrian/cyclist宏平均召回由高到低排序，再依次按更低误警、更低p95延迟、更少参数量破同分。若验证集没有负帧，空帧FP记为不可估计，候选不能因此宣称通过此门槛；投影门槛留N18实测，不用作N17检测器排序。按类别×车路侧×small/medium/large×官方遮挡状态报告各自GT分母及召回，不丢弃稀有/失败分层；测试集不得进入达标判断、排序或破同分。以上均需用户确认后才成为冻结规则。

SPD model_val当前2,885帧中空目标帧为0，因此SPD域空帧误警率不可估计；N17须保留为不可估计或另行在预先冻结的CARLA验证负例测量，不能写成0。用户确认前N17不得据此选择模型。

## 6 实际结果与边界

N16代码提交为`b0d5d9c0308b879046722828683608d105ab4278`，待确认门槛/排序配置提交为`8fbb96669e7137b41fc95667cbcc2b08d84b7d11`。源代码、配置、机制测试和实际model_val读取均已实现。最终SPD运行报告status为passed，run_id `n16_spd_d8e26cd2ff324fa0a96b1f2a760dd481`，2026-11-01 19:40:27～19:41:40；配置SHA-256 `cd925cb1c3a76a0caed5d545d4ba632c56dac1d635275e9c4335a2b8f9bf65b4`，N15 split文件SHA-256 `8413b27634c834479931d3203a6275f6e286972e030c1104b1df492932e5431d`（内部canonical SHA为`0a4d165a5345a68d13a9ad10dd4b7b8dfd62ebd784ab15113f8483c10395825d`），N15完整审计报告文件SHA-256 `29b14d33de790575b4c7aa0384a5d94d1791e1343d71394df501484f9d75e94a`，N15输入manifest SHA-256 `f40be2562ebaa01901d4987458b0f006edecaba08bee27f815013d5e22dbeaa9`。2,885帧/两视角共计，包含27,301个目标（vehicle 19,903、pedestrian 1,367、cyclist 6,031；Barrowlist/TrafficCone在本split实际为0），官方遮挡字段计数0/1/2分别15,212/10,004/2,085，small/medium/large分别5,295/12,902/9,104，标注截断状态非零目标2,475，负帧0，24个相机episode命名空间；无时间倒序、无未映射标签。12张GT叠加图（车端/路侧各6）已保存并核验中文独立图例；人工目测检查车端与路侧各1张，框与可见目标总体对齐。该目测不是全量标注正确性审查。映射JSON SHA-256为`fa30b4c9c102db181f82d215727445b10abd4aa7b5c8cb756b61fbdc0aa1611e`，汇总报告SHA-256为`fafbe570f2150b2592cb84d63389a34726f4f8aaab41d7432a684a7a7e05a62c`，产物清单SHA-256为`fdd290013fa922718016014e4affaa10fc3aafc59edd473ba42c2091cd132c60`。N16专项测试13/13、全量单元测试303/303通过，失败/跳过均0；`compileall`及文档静态检查通过。环境及逐文件源码哈希见硬盘`03_验收证据/n16_20261101/`。

首次适配启动使用错误的数据根目录，保留失败run目录及失败说明；修正路径后完整重跑。第二次适配的数据读取成功，但首版图内逐框长文本重叠；该图版本保留并标记被替代。加强N15报告、split和RGB/标注/内参逐文件哈希核验后再次全量重跑，最终映射与图像可视化一致。单元测试开发失败均保留：包括API异常类型/IDF1预期、确认轨迹只推进1帧、SPD来源负例断言文本不匹配；修正测试输入和期望后13项全部通过，未降低算法或数据完整性标准。

**未完成和限制：** N16未下载权重或运行Ultralytics真实推理，未测mAP、真实召回、FP、IDF1或端到端延迟；预训练/微调公平比较留N17。SPD model_val来自单一留出路口，且没有空目标帧，不能据此估计跨路口泛化或SPD空帧误警。N15用户已验收，但其CARLA遮挡代理与SPD人工复核限制不因此消失。IoU50召回vehicle≥0.95、pedestrian/cyclist≥0.90，空帧FP≤0.10/帧，检测链p95≤100 ms，两个距离段投影误差p95≤0.5m/≤1.0m为N16协议建议值，尚未获用户确认；因此规格⑥未执行、整体N16自测不通过。用户确认前不得按该数值在N17筛选模型；本节点交付后停止，不进入N17。
