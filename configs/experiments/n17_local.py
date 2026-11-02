"""N17本机任务选项：修改MODEL/SEED/ACTION后可直接在IDE运行本文件。"""

# 可选：YOLO26n/s/m/l/x、YOLOv8n/s/m/l/x、YOLO11n/s/m/l/x（例如"YOLO11m"）
MODEL = 'YOLO26n'
SEED = 17  # 正式比较建议17/29/43依次独立运行；单次不算完整三seed比较
ACTION = 'check'  # check检查；plan仅计划；probe短探针；train正式；resume续训

MAX_EPOCHS = 300
MIN_EPOCHS = 50
PATIENCE = 30
BATCH = 2
WORKERS = 0
AMP = True
DEVICE = 0
INPUT_ROOT = r'H:\OccluActiSafeDataset\服务器输入\N17\n17_input_385ce282b4444f1bb6fea051b3c0423b'
OUTPUT_ROOT = r'H:\OccluActiSafeDataset\artifacts\节点资料\N17_检测器比较'
RESUME_RUN = ''  # 填原中断任务的绝对根目录；同模型/seed/配方/源码/GPU才能恢复
PAUSE_AFTER_EPOCH = None  # 可填写某轮结束后暂停；last/best写完后退出，旧记录保留
PLOT_POINTS_PER_PAGE = 20000  # 全部逐步原值分页绘图，限制单页内存；不平滑不删原值

if __name__ == '__main__':
    from pathlib import Path
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
    from occluactisafe.experiments.detector_local import main
    main()
