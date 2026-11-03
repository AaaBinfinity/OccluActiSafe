"""IDE无必填参数；默认检查已授权六项对照，运行需显式选择run。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from occluactisafe.experiments.detector_lr_contrast import main

if __name__ == '__main__':
    main()
