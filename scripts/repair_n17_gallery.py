"""IDE可直接运行：仅补齐指定N17旧任务图集，不重新训练。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from occluactisafe.experiments.detector_gallery_repair import main

if __name__=='__main__':main()
