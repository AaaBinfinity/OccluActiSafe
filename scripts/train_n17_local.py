"""IDE/终端直接运行显示任务菜单；显式CLI保留批处理，不自动开始训练。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from occluactisafe.experiments.detector_local import main

if __name__=='__main__':main()
