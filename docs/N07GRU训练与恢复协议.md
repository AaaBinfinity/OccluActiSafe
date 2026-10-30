# N07 GRU训练、验证与恢复协议

本节点实施事实、不可变代码版本、必需测试和真实训练结果统一回填[节点计划N07](项目节点实施与验收计划.md#n07-gru训练验证与检查点恢复)。N06已获用户验收。当前N04原始坐标一致性疑点与稀少类别监督限制继续保留；本节点允许在原样标注上验证真实训练管线，不把数值收敛、GPU可运行或CV误差解释为数据已物理核验或避撞有效。

## 输入、模型与监督

`prediction/train.py:prepare_data()`只打开N04的train/model_val输入和标签，先校验完成状态、split/scaler规范哈希、索引与原train地理分组角色、时域及完整监督定义。实际默认H/F=50/50、dt=.1s；原val与项目risk_calibration/test标签不打开。同N05，监督对象为历史TARGET_AGENT且完整H/F；不完整对象计入排除分母，原始样本不删。普通GRU按目标独立建模，不使用其他目标上下文，数据缓存只保留这类监督对象，不能称已实现交互或车路融合。

归一化只读取N04已有train-only scaler，N07不重拟合、不裁剪、不按结果改组。每个实际窗口的原输入内部哈希与输入/标签字节SHA-256另存inventory，目标来源、组、文件内ID及类别另存索引。缓存、统计副本与输入身份写到H盘当前run，既有N04与原始CSV保持只读。无数据、哈希冲突、无合法监督或缺标签明确失败，不构造真实数据替代品。

`baselines/gru.py:MultiModalGRU`输入白名单为features[M,H,10]、feature_mask、observed_mask、actual_dt、history_times、categories和anchor_position。M是当前批次的文件内目标，不是场景独立数。x/y/vx/vy/ax/ay/sin/cos/长/宽来自N04，拼接逐字段mask、真实dt和相对历史时刻；两个GRUCell(hidden=128)仅在位置与观测有效时更新隐状态，缺格点保持状态。末状态与4类one-hot拼接，经Linear/ReLU及dropout=.1，输出6模式的逐步米制增量及logits，增量累积后加最后可用历史位置。锚与最终位置保留float64，GRU隐状态/权重为float32，避免大偏移坐标相消的精度损失；这不修正原始物理坐标问题。

空历史prediction_valid=False、权重和为0，其他有效目标权重和为1；无效特征/时间槽可有NaN且不会更新状态。模型不接收未来、风险标签、图、地图或消息队列；本节点只覆盖已存在单源标注历史，不提前实现N08/N09。结构参考[PyTorch 2.11 GRUCell](https://docs.pytorch.org/docs/2.11/generated/torch.nn.GRUCell.html)，本项目具体掩码与解码由上述代码实现。

`losses.py:multimodal_loss()`以有效未来点的米制ADE选择赢家，仅用赢家计算SmoothL1(beta=1)回归，按每目标有效点数及二维坐标取均值，再按目标等权平均；模式交叉熵权重1。先替换无效GT，避免NaN×0；有效GT非有限则拒绝，无有效监督返回None而非伪造零loss。赢家是离线训练匹配，不允许在线读取真值选择模式。

## 入口、预算与结果

IDE选择仓库`.venv/Scripts/python.exe`后直接运行`scripts/train_prediction.py`。默认`configs/experiments/tfd_gru.yaml`引用`configs/models/gru.yaml`与N06预算，`OVERRIDES={}`，可选CLI支持config、resume、seed、output-root、device、preflight-only。配置→脚本覆盖→显式CLI，False保留；所有路径从项目根解析，import不训练，不安装依赖或下载数据。新run默认不续训。Windows有主入口保护，DataLoader workers=0，实际采用有界预读缓存和主进程批处理，不启动训练子worker。

本轮在看真实训练结果前固定：seed[17,29,43]分别完整训练3epoch、batch64、AdamW(lr=.0003, weight_decay=.01)、clip_grad_norm=1、patience10，以完整model_val的minADE选择best；默认点击运行会执行全部这些工作，不把少量batch冒充全量。3epoch是N07训练/恢复管线验收预算，不是正式模型收敛或选优结论。早停逻辑已实现，但3epoch任务不可能触发patience10；不能称本次真实训练已触发早停。

N06每seed的训练/验证/最终评分受7200秒、20000更新、最多60epoch、GPU分配器12288MiB及主机工作集8192MiB约束，先到者停止；预算异常保留interrupted检查点和失败，不偷偷换CPU。数据准备阶段单独受墙钟/内存/剩余空间检查；运行时空间检查用实际输出所在盘，不固定假设H盘。显式CPU只作机制诊断，默认正式任务要求cuda:0。缓存准备与三个seed的训练时间分列，不外推长期稳定性。

每epoch写curve.json与epoch_metrics.csv，保存训练批次平均loss、最大梯度范数、完整验证目标数、ADE/FDE/minADE/minFDE/MissRate。K=6、top-K=6，最高logit模式用于top1，min指标按N05统一定义分别取最小，Miss阈值2m、未来5s。真实验证只评分完整H/F，分类别分母如实报告；行人/BICYCLE极少，不作类别泛化推断。原始轨迹/不完整目标分母见data_identity，不能仅展示最好seed。

## 检查点与恢复

`save_checkpoint()`原子写best.pt/latest.pt，包含模型、AdamW状态、epoch/下一batch/优化步数、最佳验证值与未改善epoch数、已用墙钟、既有曲线、完整契约及Python/NumPy/Torch/CUDA RNG状态。契约含数据集合/输入清单/split/scaler哈希、H/F、模型结构、seed、优化器参数、batch、epoch/patience/clip和预算；训练归一化数值另以normalization.json保存，checkpoint保留其不可变身份。序列化参考[PyTorch 2.11说明](https://docs.pytorch.org/docs/2.11/notes/serialization.html)，仅加载本项目产生的tensor/基础类型检查点，`weights_only=True`。

每200步及每epoch保存latest，验证minADE改善保存best。异常时另保存interrupted.pt及failure.json。恢复显式指定一个原seed，不兼容的契约拒绝；恢复到新run，保留旧run，既有best/latest复制到新目录。批次排列由seed+epoch生成，下一batch游标避免重复更新，恢复包含优化器和随机状态，不只是重新读权重。若从周期中间恢复，该epoch曲线的训练loss/梯度统计只包含恢复后实际执行的批次；已保存的旧epoch曲线保留，不能把这段统计称完整epoch均值。

最终加载best，在实际模型验证输入上重新建模并要求重载前后输出逐值相同，再完整输出validation_predictions.npz、per_agent_metrics.csv、result.json及检查点哈希。恢复到已完成任务也应能重新输出结果，不要求额外更新。该设计不声称旧失败run消失或自测等于用户验收。

## 必需验证与证据

`tests/unit/test_n07.py`检查M07的K=6赢家2、logit±20、无效GT的NaN/9999、不含监督的目标及有效GT非有限拒绝；检查空历史、缺测、权重和、未来字段隔离与目标置换。M08检查一次反向有限且参数改变、完整保存/恢复、Python/NumPy/Torch RNG、恢复后下一更新精确一致、归一化/模式/优化器不兼容拒绝。

`tests/unit/test_n07_pipeline.py`使用明确构造的8个原始场景走N03/N04→N07训练/验证/完成任务恢复，检查缺标签失败、三种cwd无必填CLI、主机工作集与墙钟拒绝；构造数据不计真实TFD实验。`tests/gpu/test_n07_gru.py`在真实GPU上拟合8条匀速目标至minADE<.1m，保存诊断曲线；这个断言与真实TFD全范围、三seed训练分别留证。既有N01～N06回归独立执行。开发失败、修复版本及最终计数见节点计划。

实际产物按唯一run_id写H盘，包括manifest/task/environment、输入哈希清单、train/model_val目标索引与缓存、normalization/data_identity、各seed曲线、best/latest、验证预测/逐目标表、result、失败状态及evidence_index。既有数据与权重不入普通Git；本节点不创建后续节点目录、源码、任务或训练结果。
