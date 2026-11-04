"""N17本机吞吐配置：直接IDE运行；默认只检查，不启动长训。"""

# 与原n17_local.py独立；修改这三个选项选择下一次新任务。
MODEL = 'YOLO26n'
SEED = 17
ACTION = 'check'  # check/plan/probe/train/resume；正式训练显式改train

MAX_EPOCHS = 300
MIN_EPOCHS = 50
PATIENCE = 30
# 首轮候选；以本机测速报告最终冻结值为准。大尺寸容量未实测。
BATCH = {'n': 8, 's': 4, 'm': 2, 'l': 1, 'x': 1}[MODEL[-1].lower()]
WORKERS = 1  # 锁定框架验证loader使用2倍；16GB RAM不直接套用8进程
CPU_THREADS = 4
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
