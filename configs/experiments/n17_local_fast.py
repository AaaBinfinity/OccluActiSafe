"""N17终端交互默认值：直接IDE或终端运行后选择任务，确认前不执行。"""

# 以下是菜单默认值，无须为每次训练修改文件；CLI也可覆盖。
MODEL = 'YOLO26n'
SEED = 17
ACTION = 'check'  # 菜单默认检查；在终端选择train并确认后才开始
LR0 = 0.001  # 终端可输入0.0003等数值，实际写入训练配置

MAX_EPOCHS = 300
MIN_EPOCHS = 15
PATIENCE = 10
# 本机32GB内存、16GB显存。YOLOv8n的batch32已做资源短测。
BATCH_BY_SCALE = {'n': 32, 's': 4, 'm': 2, 'l': 1, 'x': 1}
BATCH = BATCH_BY_SCALE[MODEL[-1].lower()]
WORKERS = 2  # 11月26日实测，4和6个进程的稳态GPU利用率增幅均低于5个百分点
PREFETCH_FACTOR = 1  # 每个训练进程预取1批
VALIDATION_WORKERS = 0  # 逐轮验证和best验证均由主进程读图
CPU_THREADS = 4
VALIDATION_BATCH = 4  # 原生逐epoch/best验证上限；统一准入评测仍FP32/batch1
AMP = True
DEVICE = 0
INPUT_ROOT = r'H:\OccluActiSafeDataset\服务器输入\N17\n17_input_385ce282b4444f1bb6fea051b3c0423b'
OUTPUT_ROOT = r'H:\OccluActiSafeDataset\artifacts\节点资料\N17_检测器比较'
RESUME_RUN = ''  # 只恢复同源码/模型/seed/配方/GPU的新配置任务
PAUSE_AFTER_EPOCH = None
PLOT_POINTS_PER_PAGE = 20000

if __name__ == '__main__':
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
    from occluactisafe.experiments.detector_local import main
    sys.argv[1:1] = ['--config', str(Path(__file__).resolve())]
    main()
