# 训练与开发过程图存放说明

按用户2026年10月31日最新要求，全部280张现存训练/开发过程图片已从仓库移回移动硬盘各自的原运行目录，中文文件名保持不变，原始字节和SHA-256不变。以后过程图继续保存在硬盘，不提交或推送Git；本目录只保存说明与小型位置/哈希摘要。既有概念图和论文流程SVG仍保留在docs/figures，不属于本次从硬盘迁入的过程图。

## 实际位置与复核

- 本机已核实移动硬盘WD P40 Game Drive、序列号2335NJ401238、卷标bin-data，当前产物根为`H:/OccluActiSafeDataset/artifacts`。换电脑先核对硬盘与实际盘符，不把H当作所有电脑固定盘符。
- [图片位置与哈希清单](图片来源与哈希.json)中，`artifact_path`相对上述实际产物根；`original_artifact_path`记录最早英文或中文原地址，`legacy_repository_path`记录旧仓库地址。旧run的plot_manifest/evidence_index不改写，图片条目按映射定位后验证原哈希，其他原始数据仍按原位置验证。
- N11最终图：`H:/OccluActiSafeDataset/artifacts/n11/bc03ed4f74484ac1ba857b863194e84b/解析分支配对轨迹.png`；其他节点同样回到各自nXX/运行ID/seed或重建目录。N11共162张，包含开发/失败核查/重建/入口测试，不能当162个独立实验。
- 原来入库时的完整来源清单整体移回`H:/OccluActiSafeDataset/artifacts/仓库归档时图片来源与哈希.json`，保留旧不可变证据索引所引用的原始字节及绘图来源记录；它不是新的运行结果。仓库清单更新为轻量的当前位置摘要。
- 对旧N11总索引中的`repository/docs/过程图/...图片`，按legacy_repository_path映射到硬盘；对旧`repository/docs/过程图/图片来源与哈希.json`，使用上述硬盘原始清单核验原哈希。仓库历史版本仍由Git保留，不改写已发布历史，本次正常提交仅移除当前分支的过程图片。
- [既有示意图命名映射](既有示意图中文命名映射.json)仍用于17张已有概念图/流程SVG，继续保留其中文文件名和仓库引用。

## 后续交付

IDE直接运行`scripts/archive_process_images.py`；默认核对硬盘、把误留仓库的已登记过程图移回硬盘、对硬盘nXX目录中新图按中文原位命名，保存位置/哈希清单。不会把图片搬入仓库，不扫描原始数据或依赖缓存。CLI可选`--artifacts-root <实际硬盘产物根>`；缺盘、同名不同内容、源哈希异常明确失败，不覆盖。

`python -B scripts/archive_process_images.py --verify-only`只读核对硬盘当前300张登记图片，必须接入硬盘。操作先验证源/目标和范围，目标哈希及原子写入清单成功后才移除确切源文件；写入遇Windows临时访问冲突有限重试，未完成时保留源图供续传。新增图类须补充脚本NAMES中的中文名；不同运行/seed/重建目录不混合，不用重绘覆盖旧图。

每次交付保留PNG/SVG、原始CSV/JSON、失败与恢复记录、图表来源/哈希；先在硬盘归档并复核，再检查文档和Git差异，提交代码/文档/位置摘要。`.gitignore`排除本目录过程图片，禁止再次将它们加入Git。历史节点的代码版本、指标和用户验收不因位置变化而改变。

本次迁回核对报告位于`H:/OccluActiSafeDataset/artifacts/图片硬盘归档核对.json`：9项文件机制/实际位置检查通过，280张图片哈希一致、仓库过程图为0；旧N10的186条及N11的1004条不可变证据按映射逐项哈希核对通过。未重新训练、重跑算法或改变节点验收；迁移中首次清单替换访问冲突已续传处理，源图与原清单保留后再完成核验。

N12实施、失联排查与恢复验收累计新增20张PNG/SVG（含失败检查的诊断图），当前300张已原位中文归档复核。排查和恢复原始报告、图表及来源记录分别位于 `H:/OccluActiSafeDataset/artifacts/n12_disconnect_diagnosis/` 和 `H:/OccluActiSafeDataset/artifacts/n12_recovery/` 的相应运行目录。原有280张迁移事实及不可变来源不变，不将故障episode或失败图计作成功实验。
