"""由已完成的 N13 扫值拟合速度分档表；数据不足或非单调时拒绝生成。"""
from collections import defaultdict
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path('H:/OccluActiSafeDataset/artifacts')


def fit(folder):
    folder=Path(folder)
    config=json.loads((folder/'运行配置.json').read_text(encoding='utf-8'))
    states=[json.loads(x) for x in (folder/'工况状态.jsonl').read_text(encoding='utf-8').splitlines()]
    completed=[s for s in states if s['status']=='completed']
    expected=config['repetitions']*len(config['speeds_mps'])*(2*len(config['pedals'])+2*len(config.get('steer_values',[-1.,-.3,.3,1.])))
    if (len(completed)!=expected or {s['case_id'] for s in completed}!=set(range(expected))
            or any(s['status']=='failed' for s in states)):
        raise ValueError(f'扫值未完整完成：{len(completed)}/{expected}')
    groups=defaultdict(list)
    with (folder/'执行器原始采样.csv').open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            if row['phase']=='response':
                groups[(float(row['target_initial_speed_mps']),row['kind'],float(row['value']),int(row['repeat']))].append(row)
    physics=json.loads((folder/'车辆与环境.json').read_text(encoding='utf-8'))
    acceleration=[]; steering=[]; uncertainty=[]
    for speed in sorted(config['speeds_mps']):
        pedal_points=[]
        for kind in ('brake','throttle'):
            for pedal in config['pedals']:
                measures=[]
                for repeat in range(config['repetitions']):
                    rows=groups[(speed,kind,pedal,repeat)]
                    if len(rows)<15: raise ValueError('响应记录不足15帧')
                    measures.append(statistics.mean(float(r['longitudinal_acceleration_mps2']) for r in rows[5:15]))
                value=statistics.mean(measures);out=-pedal if kind=='brake' else pedal
                pedal_points.append((value,out))
                uncertainty.append({'speed_mps':speed,'kind':kind,'command':pedal,'metric':'acceleration_mps2',
                                    'repeat_values':measures,'range':max(measures)-min(measures)})
        pedal_points.sort()
        if any(a[1]>=b[1] for a,b in zip(pedal_points,pedal_points[1:])):
            raise ValueError(f'{speed} m/s 加速度测量与踏板非严格单调，须检查原始数据，不能自动掩盖')
        acceleration.append({'speed_mps':speed,'points':pedal_points})
        zero_values=[]
        for repeat in range(config['repetitions']):
            zero_rows=groups[(speed,'throttle',config['pedals'][0],repeat)]
            zero_values.append(math.radians(statistics.mean(float(r['wheel_angle_deg']) for r in zero_rows[2:5])))
        wheel_points=[(statistics.mean(zero_values),0.)]
        uncertainty.append({'speed_mps':speed,'kind':'zero_steer','command':0.,'metric':'wheel_angle_rad',
                            'repeat_values':zero_values,'range':max(zero_values)-min(zero_values)})
        for steer in config.get('steer_values',[-1.,-.3,.3,1.]):
            measures=[]
            for repeat in range(config['repetitions']):
                rows=groups[(speed,'steer_step',steer,repeat)]
                if len(rows)<5: raise ValueError('转向响应记录不足5帧')
                measures.append(math.radians(statistics.mean(float(r['wheel_angle_deg']) for r in rows[2:5])))
            wheel_points.append((statistics.mean(measures),steer))
            uncertainty.append({'speed_mps':speed,'kind':'steer_step','command':steer,'metric':'wheel_angle_rad',
                                'repeat_values':measures,'range':max(measures)-min(measures)})
        wheel_points.sort()
        if any(a[1]>=b[1] for a,b in zip(wheel_points,wheel_points[1:])):
            raise ValueError(f'{speed} m/s 车轮角与转向命令非严格单调')
        steering.append({'speed_mps':speed,'points':wheel_points})
    low=min(config['speeds_mps']); high=max(config['speeds_mps'])
    low_observed=max(float(rows[0]['speed_mps']) for key,rows in groups.items() if key[0]==low and rows)
    high_observed=min(float(rows[0]['speed_mps']) for key,rows in groups.items() if key[0]==high and rows)
    # 收窄名义范围，避免把从未达到的最低/最高初速当作实测边界。
    minimum=max(low,low_observed);maximum=min(high,high_observed)
    if minimum>=maximum: raise ValueError('实际初速范围无法构成有效标定区间')
    context={k:physics[k] for k in ('map','weather','mass_kg','drag_coefficient','wheels')}
    surface_id=hashlib.sha256(json.dumps(context,sort_keys=True,ensure_ascii=False).encode('utf-8')).hexdigest()
    calibration={'vehicle_blueprint':physics['vehicle_blueprint'],'surface_id':surface_id,
        'speed_min_mps':minimum,'speed_max_mps':maximum,
        'acceleration_by_speed':acceleration,'steering_by_speed':steering}
    source=folder/'执行器原始采样.csv'
    result={'calibration':calibration,'context':context,'uncertainty':uncertainty,'source':str(source),
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'method':f"每次响应的第5至14帧纵向加速度均值与第2至4帧左前轮角均值；{config['repetitions']}次重复取均值，轮角有相邻帧时序不确定性；零转向锚点来自零转向工况实测。",
        'limitations':'速度分档按名义初速线性插值，实际初速及其波动见原始记录；尚需闭环复核。桥接复核地图、天气及轮胎物理参数，但不能识别局部摩擦触发器等未知附着变化；不外推分档及表范围。'}
    out=folder/'候选执行器标定.json'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(f'候选标定已写入 {out}，尚需真实控制复核')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,default=None)
    args=parser.parse_args()
    candidates=[p for p in ROOT.glob('n13_*') if (p/'执行器原始采样.csv').is_file()]
    if args.run_dir is None and not candidates: raise FileNotFoundError('H盘没有 N13 采样目录')
    fit(args.run_dir or max(candidates,key=lambda p:p.stat().st_mtime))


if __name__=='__main__':main()
