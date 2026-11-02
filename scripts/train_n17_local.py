"""IDE默认检查；在configs/experiments/n17_local.py选择本次模型、seed与train动作。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from occluactisafe.experiments.detector_local import main

if __name__=='__main__':main()
