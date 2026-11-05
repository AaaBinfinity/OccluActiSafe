"""N17单型号seed队列；无参数打开终端菜单，确认一次自动接续。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from occluactisafe.experiments.detector_seed_queue import main

if __name__ == '__main__':
    main()
