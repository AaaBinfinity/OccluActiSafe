"""N17单型号三seed队列；IDE运行打开菜单，确认一次后自动串行。"""

MODEL = 'YOLOv8n'
SEEDS = (17, 29, 43)
ACTION = 'check'  # check仅核队列输入身份；plan冻结队列；train启动后台队列
LOCAL_CONFIG = 'configs/experiments/n17_local_fast.py'
WAIT_HOURS = 24  # GPU锁占用时最多等待；不结束其他训练

if __name__ == '__main__':
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
    from occluactisafe.experiments.detector_seed_queue import main
    sys.argv[1:1] = ['--config', str(Path(__file__).resolve())]
    main()
