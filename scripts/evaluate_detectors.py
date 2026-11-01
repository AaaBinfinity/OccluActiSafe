"""N17真实SPD/CARLA模型验证；IDE默认评测三族预训练及最近完整训练矩阵。"""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from occluactisafe.experiments.detector_data import load_config, DEFAULT_CONFIG
from occluactisafe.experiments.detector_evaluation import run_evaluation_matrix
from occluactisafe.experiments.run_store import read_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--training-matrix', type=Path)
    p.add_argument('--pretrained-only', action='store_true')
    args = p.parse_args()
    cfg = load_config(args.config)
    matrix = args.training_matrix
    if matrix is None and not args.pretrained_only:
        options = [p for p in (Path(cfg['output_root'])/'02_运行产物').glob('n17_train_*/训练矩阵.json')
                   if read_json(p)['status'] == 'completed']
        if not options:
            p.error('缺完整训练矩阵；先运行train_detector.py。单独零样本需显式--pretrained-only')
        matrix = max(options, key=lambda p:p.stat().st_mtime)
    print(run_evaluation_matrix(cfg, matrix), flush=True)


if __name__ == '__main__':
    main()
