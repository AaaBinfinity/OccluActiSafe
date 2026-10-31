"""准备 C09 开发复核配置；仅引用已有实测候选表，不启动仿真。"""
import argparse
import json
from pathlib import Path

ROOT=Path('H:/OccluActiSafeDataset/artifacts')


def prepare(folder):
    folder=Path(folder)
    source=json.loads((folder/'运行配置.json').read_text(encoding='utf-8'))
    fitted=json.loads((folder/'候选执行器标定.json').read_text(encoding='utf-8'))
    source['calibration']=fitted['calibration']
    source['surface_id']=fitted['calibration']['surface_id']
    # 控制器增益是开发配置，不是实测车辆物理能力。
    source['controller_parameters']={'kp':.5,'ki':.2,'steering_rate_per_s':1.0}
    source['repetitions']=3
    source['response_ticks']=20
    def check(name,accel,angle,target=8.,ticks=20):
        return {'name':name,'initial_speed_mps':8.,'target_speed_mps':target,
                'acceleration_mps2':accel,'angle_rad':angle,'response_ticks':ticks}
    source['controller_checks']=[check('零输入',0.,0.),check('正加速度',.5,0.),check('负加速度',-.5,0.),
        check('左转阶跃',0.,-.1),check('右转阶跃',0.,.1),check('饱和抗积分',1.,0.,12.,10)]
    source['calibration_source']=str(folder/'候选执行器标定.json')
    out=folder/'控制复核配置.json'
    out.write_text(json.dumps(source,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'控制复核配置：{out}；启动由 calibrate_vehicle.py 单独执行')
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,default=None)
    args=parser.parse_args()
    candidates=[p for p in ROOT.glob('n13_*') if (p/'候选执行器标定.json').is_file()]
    if args.run_dir is None and not candidates: raise FileNotFoundError('H盘没有 N13 候选标定')
    prepare(args.run_dir or max(candidates,key=lambda p:p.stat().st_mtime))


if __name__=='__main__':main()
