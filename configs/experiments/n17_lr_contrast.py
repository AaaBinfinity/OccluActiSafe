"""已批准的N17六项学习率对照；IDE默认只检查，ACTION='run'才排队。"""
ACTION = 'check'  # check只读检查；plan冻结计划；run等待seed29后串行执行/接续同一计划
INPUT_ROOT = r'H:\OccluActiSafeDataset\服务器输入\N17\n17_input_385ce282b4444f1bb6fea051b3c0423b'
OUTPUT_ROOT = r'H:\OccluActiSafeDataset\artifacts\节点资料\N17_检测器比较'
WAIT_FOR_RUN = r'H:\OccluActiSafeDataset\artifacts\节点资料\N17_检测器比较\02_运行产物\n17_local_yolo26n_seed29_d7e758557291461a89e3f0a7b826cc17'
RUN_DIR = ''  # 空值复用本次授权指针，不会自动另开六项实验
WAIT_TIMEOUT_HOURS = 24
DEVICE = 0

if __name__ == '__main__':
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
    from occluactisafe.experiments.detector_lr_contrast import main
    main()
