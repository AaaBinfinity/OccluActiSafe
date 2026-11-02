"""N17 A100全尺寸入口；IDE直接运行默认只检查，须先填真实路径。"""
from pathlib import Path
from run_n17_server import main

if __name__ == '__main__':
    main(Path(__file__).resolve().parents[1]/'configs/experiments/n17_server_a100.yaml')
