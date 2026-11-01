"""N17完整三族三seed GPU微调矩阵；IDE默认直接运行全部九次。"""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from occluactisafe.experiments.detector_data import load_config, DEFAULT_CONFIG
from occluactisafe.experiments.detector_matrix import run_training_matrix, train_one


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--family', choices=['YOLO26', 'YOLOv8', 'YOLO11'])
    p.add_argument('--seed', type=int)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.family is not None:
        if args.seed not in cfg['seeds'] or args.output is None:
            p.error('单次子任务须指定冻结seed及output')
        train_one(cfg, args.family, args.seed, args.output)
    else:
        print(run_training_matrix(cfg), flush=True)


if __name__ == '__main__':
    main()
