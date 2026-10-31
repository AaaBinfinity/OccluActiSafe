# 训练与开发过程图

2026年10月31日按用户要求，将已生成的全部项目过程图移入仓库并使用中文文件名，后续同样执行。共118张（含PNG/SVG、不同运行及重建图），原始字节与SHA-256不变，全部随Git保存；另17张既有概念图/流程SVG已统一中文文件名并修复引用。未迁入依赖缓存附带的97张图标/示例图。

本目录保留真实训练、构造诊断、开发运行和图表重建的各自来源，不将它们混算为独立实验。N10首轮05649db7624f487c9a7ec04220ca6ecb含网格绘图坐标警告，其图仅作开发问题留存；节点是否通过仍以[节点计划](../项目节点实施与验收计划.md)的被测版本和证据为准。本次仅整理图片，不重新训练、不改变指标、不代用户验收N10，不开展N11。

## 保存与复核方法

- [图片来源与哈希](图片来源与哈希.json)：每张图的仓库相对路径、原产物相对路径、字节数、SHA-256，以及已有plot_manifest的来源记录/原清单哈希。各节点、运行ID、seed及重建目录保持独立，不覆盖或合并不同运行。
- [既有示意图命名映射](既有示意图中文命名映射.json)：既有17张概念PNG/流程SVG的改名前后路径与相同内容哈希，图片原件保留；概念图不冒充实验截图。
- 原始CSV/JSON、权重、大型日志/视频继续在移动硬盘，未搬入普通Git。2026年10月31日核实旧产物根为`H:/OccluActiSafeDataset/artifacts`；迁移清单original_artifact_path相对此根，换电脑需使用实际挂载根。
- 原run的plot_manifest/evidence_index等不可变记录不改写。复核旧清单的图片条目时，先用original_artifact_path在迁移清单中定位新path，再验证原图哈希；非图片仍按原硬盘路径验证。只按旧图片地址检查会报告已迁移文件缺失，应使用映射，不把它算作实验失败。
- IDE可直接运行`scripts/archive_process_images.py`，默认扫描移动硬盘各nXX产物目录，先核对已有图片，按中文名称归档；核对目标SHA-256并落盘映射后才移除对应源图片，不删除原日志/清单。默认路径从脚本所在仓库解析，硬盘未接入明确失败；新图类须在NAMES中声明中文名。CLI可选`--artifacts-root <本机实际产物根>`。
- `python -B scripts/archive_process_images.py --verify-only`只复核仓库已归档图片，无需接硬盘。每次交付必须执行归档、复核、文档检查、明确暂存图片与清单后中文提交，按授权推送。若重绘内容发生变化，使用新运行/重建目录，不覆盖已归档同一来源的图片。
- `.gitattributes`对本目录SVG关闭行尾转换，保留Windows生成的原始字节；提交前同时核对Git暂存对象的SHA-256，确保换电脑检出仍能按原清单复核。

## 图片目录

| 原始节点/运行/文件 | 中文图片 | 大小（字节） |
| --- | --- | --- |
| n03_20261029/full_audit/80e4390cb71442498769c622eebf06e9/trajectory_preview.svg | [轨迹预览](n03_20261029/full_audit/80e4390cb71442498769c622eebf06e9/轨迹预览.svg) | 1832 |
| n03_20261029/inspection_run/3ef2efb3670549d5bdbd6b00fe9bd942/trajectory_preview.svg | [轨迹预览](n03_20261029/inspection_run/3ef2efb3670549d5bdbd6b00fe9bd942/轨迹预览.svg) | 1832 |
| n07/fcb1259dedf24738a9265586e9f4df38/seed_17/epoch_validation.png | [逐轮验证指标](n07/fcb1259dedf24738a9265586e9f4df38/seed_17/逐轮验证指标.png) | 82955 |
| n07/fcb1259dedf24738a9265586e9f4df38/seed_17/epoch_validation.svg | [逐轮验证指标](n07/fcb1259dedf24738a9265586e9f4df38/seed_17/逐轮验证指标.svg) | 67423 |
| n07/fcb1259dedf24738a9265586e9f4df38/seed_17/step_optimization.png | [逐步优化指标](n07/fcb1259dedf24738a9265586e9f4df38/seed_17/逐步优化指标.png) | 197466 |
| n07/fcb1259dedf24738a9265586e9f4df38/seed_17/step_optimization.svg | [逐步优化指标](n07/fcb1259dedf24738a9265586e9f4df38/seed_17/逐步优化指标.svg) | 221110 |
| n07/fcb1259dedf24738a9265586e9f4df38/seed_17/step_resources.png | [逐步资源占用](n07/fcb1259dedf24738a9265586e9f4df38/seed_17/逐步资源占用.png) | 87268 |
| n07/fcb1259dedf24738a9265586e9f4df38/seed_17/step_resources.svg | [逐步资源占用](n07/fcb1259dedf24738a9265586e9f4df38/seed_17/逐步资源占用.svg) | 164695 |
| n07/fcb1259dedf24738a9265586e9f4df38/seed_17/step_training.png | [逐步训练指标](n07/fcb1259dedf24738a9265586e9f4df38/seed_17/逐步训练指标.png) | 221340 |
| n07/fcb1259dedf24738a9265586e9f4df38/seed_17/step_training.svg | [逐步训练指标](n07/fcb1259dedf24738a9265586e9f4df38/seed_17/逐步训练指标.svg) | 192561 |
| n07/fcb1259dedf24738a9265586e9f4df38/seed_29/epoch_validation.png | [逐轮验证指标](n07/fcb1259dedf24738a9265586e9f4df38/seed_29/逐轮验证指标.png) | 87969 |
| n07/fcb1259dedf24738a9265586e9f4df38/seed_29/epoch_validation.svg | [逐轮验证指标](n07/fcb1259dedf24738a9265586e9f4df38/seed_29/逐轮验证指标.svg) | 67410 |
| n07/fcb1259dedf24738a9265586e9f4df38/seed_29/step_optimization.png | [逐步优化指标](n07/fcb1259dedf24738a9265586e9f4df38/seed_29/逐步优化指标.png) | 232037 |
| n07/fcb1259dedf24738a9265586e9f4df38/seed_29/step_optimization.svg | [逐步优化指标](n07/fcb1259dedf24738a9265586e9f4df38/seed_29/逐步优化指标.svg) | 224216 |
| n07/fcb1259dedf24738a9265586e9f4df38/seed_29/step_resources.png | [逐步资源占用](n07/fcb1259dedf24738a9265586e9f4df38/seed_29/逐步资源占用.png) | 87342 |
| n07/fcb1259dedf24738a9265586e9f4df38/seed_29/step_resources.svg | [逐步资源占用](n07/fcb1259dedf24738a9265586e9f4df38/seed_29/逐步资源占用.svg) | 164682 |
| n07/fcb1259dedf24738a9265586e9f4df38/seed_29/step_training.png | [逐步训练指标](n07/fcb1259dedf24738a9265586e9f4df38/seed_29/逐步训练指标.png) | 226488 |
| n07/fcb1259dedf24738a9265586e9f4df38/seed_29/step_training.svg | [逐步训练指标](n07/fcb1259dedf24738a9265586e9f4df38/seed_29/逐步训练指标.svg) | 192612 |
| n07/fcb1259dedf24738a9265586e9f4df38/seed_43/epoch_validation.png | [逐轮验证指标](n07/fcb1259dedf24738a9265586e9f4df38/seed_43/逐轮验证指标.png) | 90587 |
| n07/fcb1259dedf24738a9265586e9f4df38/seed_43/epoch_validation.svg | [逐轮验证指标](n07/fcb1259dedf24738a9265586e9f4df38/seed_43/逐轮验证指标.svg) | 68068 |
| n07/fcb1259dedf24738a9265586e9f4df38/seed_43/step_optimization.png | [逐步优化指标](n07/fcb1259dedf24738a9265586e9f4df38/seed_43/逐步优化指标.png) | 207926 |
| n07/fcb1259dedf24738a9265586e9f4df38/seed_43/step_optimization.svg | [逐步优化指标](n07/fcb1259dedf24738a9265586e9f4df38/seed_43/逐步优化指标.svg) | 227611 |
| n07/fcb1259dedf24738a9265586e9f4df38/seed_43/step_resources.png | [逐步资源占用](n07/fcb1259dedf24738a9265586e9f4df38/seed_43/逐步资源占用.png) | 87211 |
| n07/fcb1259dedf24738a9265586e9f4df38/seed_43/step_resources.svg | [逐步资源占用](n07/fcb1259dedf24738a9265586e9f4df38/seed_43/逐步资源占用.svg) | 164704 |
| n07/fcb1259dedf24738a9265586e9f4df38/seed_43/step_training.png | [逐步训练指标](n07/fcb1259dedf24738a9265586e9f4df38/seed_43/逐步训练指标.png) | 223576 |
| n07/fcb1259dedf24738a9265586e9f4df38/seed_43/step_training.svg | [逐步训练指标](n07/fcb1259dedf24738a9265586e9f4df38/seed_43/逐步训练指标.svg) | 193983 |
| n07_20261030/loss_validation_curves.png | [训练损失与验证曲线](n07_20261030/训练损失与验证曲线.png) | 143048 |
| n07_20261030/loss_validation_curves.svg | [训练损失与验证曲线](n07_20261030/训练损失与验证曲线.svg) | 60455 |
| n07_step_20261030/gpu_fullsteps/overfit_step_curves.png | [匀速拟合逐步诊断](n07_step_20261030/gpu_fullsteps/匀速拟合逐步诊断.png) | 92727 |
| n07_step_20261030/gpu_fullsteps/overfit_step_curves.svg | [匀速拟合逐步诊断](n07_step_20261030/gpu_fullsteps/匀速拟合逐步诊断.svg) | 91747 |
| n07_step_20261030/real_completed_resume/seed_17/epoch_validation.png | [逐轮验证指标](n07_step_20261030/real_completed_resume/seed_17/逐轮验证指标.png) | 80885 |
| n07_step_20261030/real_completed_resume/seed_17/epoch_validation.svg | [逐轮验证指标](n07_step_20261030/real_completed_resume/seed_17/逐轮验证指标.svg) | 65925 |
| n07_step_20261030/real_completed_resume/seed_17/step_optimization.png | [逐步优化指标](n07_step_20261030/real_completed_resume/seed_17/逐步优化指标.png) | 195455 |
| n07_step_20261030/real_completed_resume/seed_17/step_optimization.svg | [逐步优化指标](n07_step_20261030/real_completed_resume/seed_17/逐步优化指标.svg) | 220273 |
| n07_step_20261030/real_completed_resume/seed_17/step_resources.png | [逐步资源占用](n07_step_20261030/real_completed_resume/seed_17/逐步资源占用.png) | 85276 |
| n07_step_20261030/real_completed_resume/seed_17/step_resources.svg | [逐步资源占用](n07_step_20261030/real_completed_resume/seed_17/逐步资源占用.svg) | 163472 |
| n07_step_20261030/real_completed_resume/seed_17/step_training.png | [逐步训练指标](n07_step_20261030/real_completed_resume/seed_17/逐步训练指标.png) | 219346 |
| n07_step_20261030/real_completed_resume/seed_17/step_training.svg | [逐步训练指标](n07_step_20261030/real_completed_resume/seed_17/逐步训练指标.svg) | 189447 |
| n07_step_20261030/regenerated_seed_17/epoch_validation.png | [逐轮验证指标](n07_step_20261030/regenerated_seed_17/逐轮验证指标.png) | 82955 |
| n07_step_20261030/regenerated_seed_17/epoch_validation.svg | [逐轮验证指标](n07_step_20261030/regenerated_seed_17/逐轮验证指标.svg) | 67423 |
| n07_step_20261030/regenerated_seed_17/step_optimization.png | [逐步优化指标](n07_step_20261030/regenerated_seed_17/逐步优化指标.png) | 197466 |
| n07_step_20261030/regenerated_seed_17/step_optimization.svg | [逐步优化指标](n07_step_20261030/regenerated_seed_17/逐步优化指标.svg) | 221110 |
| n07_step_20261030/regenerated_seed_17/step_resources.png | [逐步资源占用](n07_step_20261030/regenerated_seed_17/逐步资源占用.png) | 87268 |
| n07_step_20261030/regenerated_seed_17/step_resources.svg | [逐步资源占用](n07_step_20261030/regenerated_seed_17/逐步资源占用.svg) | 164695 |
| n07_step_20261030/regenerated_seed_17/step_training.png | [逐步训练指标](n07_step_20261030/regenerated_seed_17/逐步训练指标.png) | 221340 |
| n07_step_20261030/regenerated_seed_17/step_training.svg | [逐步训练指标](n07_step_20261030/regenerated_seed_17/逐步训练指标.svg) | 192561 |
| n08/8e38c0e1a9494c8e8202798eaccc19c1/capacity_queue.png | [通信容量与排队](n08/8e38c0e1a9494c8e8202798eaccc19c1/通信容量与排队.png) | 97312 |
| n08/8e38c0e1a9494c8e8202798eaccc19c1/capacity_queue.svg | [通信容量与排队](n08/8e38c0e1a9494c8e8202798eaccc19c1/通信容量与排队.svg) | 71418 |
| n08/8e38c0e1a9494c8e8202798eaccc19c1/duplex_timing.png | [双向通信时序](n08/8e38c0e1a9494c8e8202798eaccc19c1/双向通信时序.png) | 43103 |
| n08/8e38c0e1a9494c8e8202798eaccc19c1/duplex_timing.svg | [双向通信时序](n08/8e38c0e1a9494c8e8202798eaccc19c1/双向通信时序.svg) | 49766 |
| n08/a5ca5752d9b04d619c80aa7e5d34d9c6/capacity_queue.png | [通信容量与排队](n08/a5ca5752d9b04d619c80aa7e5d34d9c6/通信容量与排队.png) | 97353 |
| n08/a5ca5752d9b04d619c80aa7e5d34d9c6/capacity_queue.svg | [通信容量与排队](n08/a5ca5752d9b04d619c80aa7e5d34d9c6/通信容量与排队.svg) | 71410 |
| n08/a5ca5752d9b04d619c80aa7e5d34d9c6/duplex_timing.png | [双向通信时序](n08/a5ca5752d9b04d619c80aa7e5d34d9c6/双向通信时序.png) | 43020 |
| n08/a5ca5752d9b04d619c80aa7e5d34d9c6/duplex_timing.svg | [双向通信时序](n08/a5ca5752d9b04d619c80aa7e5d34d9c6/双向通信时序.svg) | 49938 |
| n08/dcd44df987de47b4bbbd28d370ff51ea/capacity_queue.png | [通信容量与排队](n08/dcd44df987de47b4bbbd28d370ff51ea/通信容量与排队.png) | 90179 |
| n08/dcd44df987de47b4bbbd28d370ff51ea/capacity_queue.svg | [通信容量与排队](n08/dcd44df987de47b4bbbd28d370ff51ea/通信容量与排队.svg) | 68901 |
| n08/dcd44df987de47b4bbbd28d370ff51ea/duplex_timing.png | [双向通信时序](n08/dcd44df987de47b4bbbd28d370ff51ea/双向通信时序.png) | 42319 |
| n08/dcd44df987de47b4bbbd28d370ff51ea/duplex_timing.svg | [双向通信时序](n08/dcd44df987de47b4bbbd28d370ff51ea/双向通信时序.svg) | 51037 |
| n08_20261030/regenerated_plots/capacity_queue.png | [通信容量与排队](n08_20261030/regenerated_plots/通信容量与排队.png) | 90179 |
| n08_20261030/regenerated_plots/capacity_queue.svg | [通信容量与排队](n08_20261030/regenerated_plots/通信容量与排队.svg) | 68901 |
| n08_20261030/regenerated_plots/duplex_timing.png | [双向通信时序](n08_20261030/regenerated_plots/双向通信时序.png) | 42319 |
| n08_20261030/regenerated_plots/duplex_timing.svg | [双向通信时序](n08_20261030/regenerated_plots/双向通信时序.svg) | 51037 |
| n09/a23b334aa9fd4cb38de5a24daec8826e/association_gates.png | [跨源关联与门控](n09/a23b334aa9fd4cb38de5a24daec8826e/跨源关联与门控.png) | 39810 |
| n09/a23b334aa9fd4cb38de5a24daec8826e/association_gates.svg | [跨源关联与门控](n09/a23b334aa9fd4cb38de5a24daec8826e/跨源关联与门控.svg) | 41693 |
| n09/a23b334aa9fd4cb38de5a24daec8826e/source_filter_history.png | [来源滤波与历史状态](n09/a23b334aa9fd4cb38de5a24daec8826e/来源滤波与历史状态.png) | 134610 |
| n09/a23b334aa9fd4cb38de5a24daec8826e/source_filter_history.svg | [来源滤波与历史状态](n09/a23b334aa9fd4cb38de5a24daec8826e/来源滤波与历史状态.svg) | 106900 |
| n09/e1287da500014a6f9d6135ae643bc0d8/association_gates.png | [跨源关联与门控](n09/e1287da500014a6f9d6135ae643bc0d8/跨源关联与门控.png) | 39925 |
| n09/e1287da500014a6f9d6135ae643bc0d8/association_gates.svg | [跨源关联与门控](n09/e1287da500014a6f9d6135ae643bc0d8/跨源关联与门控.svg) | 42473 |
| n09/e1287da500014a6f9d6135ae643bc0d8/source_filter_history.png | [来源滤波与历史状态](n09/e1287da500014a6f9d6135ae643bc0d8/来源滤波与历史状态.png) | 134730 |
| n09/e1287da500014a6f9d6135ae643bc0d8/source_filter_history.svg | [来源滤波与历史状态](n09/e1287da500014a6f9d6135ae643bc0d8/来源滤波与历史状态.svg) | 107481 |
| n09_20261031/development_runs/771fc1a3a9204cbdaf1c8ba41e39d988/association_gates.png | [跨源关联与门控](n09_20261031/development_runs/771fc1a3a9204cbdaf1c8ba41e39d988/跨源关联与门控.png) | 39968 |
| n09_20261031/development_runs/771fc1a3a9204cbdaf1c8ba41e39d988/association_gates.svg | [跨源关联与门控](n09_20261031/development_runs/771fc1a3a9204cbdaf1c8ba41e39d988/跨源关联与门控.svg) | 41762 |
| n09_20261031/development_runs/771fc1a3a9204cbdaf1c8ba41e39d988/source_filter_history.png | [来源滤波与历史状态](n09_20261031/development_runs/771fc1a3a9204cbdaf1c8ba41e39d988/来源滤波与历史状态.png) | 134791 |
| n09_20261031/development_runs/771fc1a3a9204cbdaf1c8ba41e39d988/source_filter_history.svg | [来源滤波与历史状态](n09_20261031/development_runs/771fc1a3a9204cbdaf1c8ba41e39d988/来源滤波与历史状态.svg) | 106604 |
| n09_20261031/gpu_n07/overfit_step_curves.png | [匀速拟合逐步诊断](n09_20261031/gpu_n07/匀速拟合逐步诊断.png) | 92727 |
| n09_20261031/gpu_n07/overfit_step_curves.svg | [匀速拟合逐步诊断](n09_20261031/gpu_n07/匀速拟合逐步诊断.svg) | 91747 |
| n09_20261031/gpu_n07_final/overfit_step_curves.png | [匀速拟合逐步诊断](n09_20261031/gpu_n07_final/匀速拟合逐步诊断.png) | 92727 |
| n09_20261031/gpu_n07_final/overfit_step_curves.svg | [匀速拟合逐步诊断](n09_20261031/gpu_n07_final/匀速拟合逐步诊断.svg) | 91747 |
| n09_20261031/regenerated_plots/association_gates.png | [跨源关联与门控](n09_20261031/regenerated_plots/跨源关联与门控.png) | 39810 |
| n09_20261031/regenerated_plots/association_gates.svg | [跨源关联与门控](n09_20261031/regenerated_plots/跨源关联与门控.svg) | 41693 |
| n09_20261031/regenerated_plots/source_filter_history.png | [来源滤波与历史状态](n09_20261031/regenerated_plots/来源滤波与历史状态.png) | 134610 |
| n09_20261031/regenerated_plots/source_filter_history.svg | [来源滤波与历史状态](n09_20261031/regenerated_plots/来源滤波与历史状态.svg) | 106900 |
| n10/05649db7624f487c9a7ec04220ca6ecb/geometry_intervals.png | [车体几何与区间细分](n10/05649db7624f487c9a7ec04220ca6ecb/车体几何与区间细分.png) | 64487 |
| n10/05649db7624f487c9a7ec04220ca6ecb/geometry_intervals.svg | [车体几何与区间细分](n10/05649db7624f487c9a7ec04220ca6ecb/车体几何与区间细分.svg) | 56710 |
| n10/05649db7624f487c9a7ec04220ca6ecb/raw_scores_coverage.png | [原始风险评分与时域覆盖](n10/05649db7624f487c9a7ec04220ca6ecb/原始风险评分与时域覆盖.png) | 102669 |
| n10/05649db7624f487c9a7ec04220ca6ecb/raw_scores_coverage.svg | [原始风险评分与时域覆盖](n10/05649db7624f487c9a7ec04220ca6ecb/原始风险评分与时域覆盖.svg) | 79605 |
| n10/05649db7624f487c9a7ec04220ca6ecb/visibility_constraints.png | [可见性与未知区域约束](n10/05649db7624f487c9a7ec04220ca6ecb/可见性与未知区域约束.png) | 83220 |
| n10/05649db7624f487c9a7ec04220ca6ecb/visibility_constraints.svg | [可见性与未知区域约束](n10/05649db7624f487c9a7ec04220ca6ecb/可见性与未知区域约束.svg) | 72702 |
| n10/241fa9bc4f1b4981af9dae8547b00c8e/geometry_intervals.png | [车体几何与区间细分](n10/241fa9bc4f1b4981af9dae8547b00c8e/车体几何与区间细分.png) | 64026 |
| n10/241fa9bc4f1b4981af9dae8547b00c8e/geometry_intervals.svg | [车体几何与区间细分](n10/241fa9bc4f1b4981af9dae8547b00c8e/车体几何与区间细分.svg) | 57060 |
| n10/241fa9bc4f1b4981af9dae8547b00c8e/raw_scores_coverage.png | [原始风险评分与时域覆盖](n10/241fa9bc4f1b4981af9dae8547b00c8e/原始风险评分与时域覆盖.png) | 108830 |
| n10/241fa9bc4f1b4981af9dae8547b00c8e/raw_scores_coverage.svg | [原始风险评分与时域覆盖](n10/241fa9bc4f1b4981af9dae8547b00c8e/原始风险评分与时域覆盖.svg) | 85958 |
| n10/241fa9bc4f1b4981af9dae8547b00c8e/visibility_constraints.png | [可见性与未知区域约束](n10/241fa9bc4f1b4981af9dae8547b00c8e/可见性与未知区域约束.png) | 73748 |
| n10/241fa9bc4f1b4981af9dae8547b00c8e/visibility_constraints.svg | [可见性与未知区域约束](n10/241fa9bc4f1b4981af9dae8547b00c8e/可见性与未知区域约束.svg) | 82001 |
| n10/affe1910746940b98a33adb5ddeefb22/geometry_intervals.png | [车体几何与区间细分](n10/affe1910746940b98a33adb5ddeefb22/车体几何与区间细分.png) | 64487 |
| n10/affe1910746940b98a33adb5ddeefb22/geometry_intervals.svg | [车体几何与区间细分](n10/affe1910746940b98a33adb5ddeefb22/车体几何与区间细分.svg) | 56710 |
| n10/affe1910746940b98a33adb5ddeefb22/raw_scores_coverage.png | [原始风险评分与时域覆盖](n10/affe1910746940b98a33adb5ddeefb22/原始风险评分与时域覆盖.png) | 102669 |
| n10/affe1910746940b98a33adb5ddeefb22/raw_scores_coverage.svg | [原始风险评分与时域覆盖](n10/affe1910746940b98a33adb5ddeefb22/原始风险评分与时域覆盖.svg) | 79605 |
| n10/affe1910746940b98a33adb5ddeefb22/visibility_constraints.png | [可见性与未知区域约束](n10/affe1910746940b98a33adb5ddeefb22/可见性与未知区域约束.png) | 73748 |
| n10/affe1910746940b98a33adb5ddeefb22/visibility_constraints.svg | [可见性与未知区域约束](n10/affe1910746940b98a33adb5ddeefb22/可见性与未知区域约束.svg) | 82001 |
| n10/e6622b71e3ae4397893c0a9cfae3e3b2/geometry_intervals.png | [车体几何与区间细分](n10/e6622b71e3ae4397893c0a9cfae3e3b2/车体几何与区间细分.png) | 64026 |
| n10/e6622b71e3ae4397893c0a9cfae3e3b2/geometry_intervals.svg | [车体几何与区间细分](n10/e6622b71e3ae4397893c0a9cfae3e3b2/车体几何与区间细分.svg) | 57060 |
| n10/e6622b71e3ae4397893c0a9cfae3e3b2/raw_scores_coverage.png | [原始风险评分与时域覆盖](n10/e6622b71e3ae4397893c0a9cfae3e3b2/原始风险评分与时域覆盖.png) | 108830 |
| n10/e6622b71e3ae4397893c0a9cfae3e3b2/raw_scores_coverage.svg | [原始风险评分与时域覆盖](n10/e6622b71e3ae4397893c0a9cfae3e3b2/原始风险评分与时域覆盖.svg) | 85958 |
| n10/e6622b71e3ae4397893c0a9cfae3e3b2/visibility_constraints.png | [可见性与未知区域约束](n10/e6622b71e3ae4397893c0a9cfae3e3b2/可见性与未知区域约束.png) | 73748 |
| n10/e6622b71e3ae4397893c0a9cfae3e3b2/visibility_constraints.svg | [可见性与未知区域约束](n10/e6622b71e3ae4397893c0a9cfae3e3b2/可见性与未知区域约束.svg) | 82001 |
| n10_20261031/delivery_review/plot_rebuild/geometry_intervals.png | [车体几何与区间细分](n10_20261031/delivery_review/plot_rebuild/车体几何与区间细分.png) | 64026 |
| n10_20261031/delivery_review/plot_rebuild/geometry_intervals.svg | [车体几何与区间细分](n10_20261031/delivery_review/plot_rebuild/车体几何与区间细分.svg) | 57060 |
| n10_20261031/delivery_review/plot_rebuild/raw_scores_coverage.png | [原始风险评分与时域覆盖](n10_20261031/delivery_review/plot_rebuild/原始风险评分与时域覆盖.png) | 108830 |
| n10_20261031/delivery_review/plot_rebuild/raw_scores_coverage.svg | [原始风险评分与时域覆盖](n10_20261031/delivery_review/plot_rebuild/原始风险评分与时域覆盖.svg) | 85958 |
| n10_20261031/delivery_review/plot_rebuild/visibility_constraints.png | [可见性与未知区域约束](n10_20261031/delivery_review/plot_rebuild/可见性与未知区域约束.png) | 73748 |
| n10_20261031/delivery_review/plot_rebuild/visibility_constraints.svg | [可见性与未知区域约束](n10_20261031/delivery_review/plot_rebuild/可见性与未知区域约束.svg) | 82001 |
| n10_20261031/plot_rebuild/geometry_intervals.png | [车体几何与区间细分](n10_20261031/plot_rebuild/车体几何与区间细分.png) | 64026 |
| n10_20261031/plot_rebuild/geometry_intervals.svg | [车体几何与区间细分](n10_20261031/plot_rebuild/车体几何与区间细分.svg) | 57060 |
| n10_20261031/plot_rebuild/raw_scores_coverage.png | [原始风险评分与时域覆盖](n10_20261031/plot_rebuild/原始风险评分与时域覆盖.png) | 108830 |
| n10_20261031/plot_rebuild/raw_scores_coverage.svg | [原始风险评分与时域覆盖](n10_20261031/plot_rebuild/原始风险评分与时域覆盖.svg) | 85958 |
| n10_20261031/plot_rebuild/visibility_constraints.png | [可见性与未知区域约束](n10_20261031/plot_rebuild/可见性与未知区域约束.png) | 73748 |
| n10_20261031/plot_rebuild/visibility_constraints.svg | [可见性与未知区域约束](n10_20261031/plot_rebuild/可见性与未知区域约束.svg) | 82001 |
