"""IDE直接运行：先启动CARLA，再观看20秒同步诊断和实时摄像头。"""
from run_carla_bridge import main


if __name__=='__main__':
    main(visualize_default=True,default_ticks=400)
