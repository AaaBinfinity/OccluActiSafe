"""检查 C09 真实控制记录；任何缺项或断言失败返回非零状态。"""
from collections import defaultdict
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path('H:/OccluActiSafeDataset/artifacts')


def verify(folder):
    folder=Path(folder)
    config=json.loads((folder/'运行配置.json').read_text(encoding='utf-8'))
    if 'controller_checks' not in config: raise ValueError('不是控制复核运行')
    statuses=[json.loads(x) for x in (folder/'工况状态.jsonl').read_text(encoding='utf-8').splitlines()]
    completed={x['case_id']:x for x in statuses if x['status']=='completed'}
    expected=config['repetitions']*len(config['controller_checks'])
    checks=[]
    def record(name,ok,detail): checks.append({'name':name,'passed':bool(ok),'detail':detail})
    record('全部工况完成',set(completed)==set(range(expected)) and not any(x['status']=='failed' for x in statuses),
           {'completed':len(completed),'expected':expected})
    by_case=defaultdict(list)
    raw=folder/'执行器原始采样.csv'
    with raw.open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            if row['phase']=='response': by_case[int(row['case_id'])].append(row)
    metrics=[]
    for case_id,status in sorted(completed.items()):
        rows=by_case[case_id]; spec=status['check']; name=spec['name']
        wanted=spec.get('response_ticks',config['response_ticks'])
        record(f'{case_id} 响应帧完整',len(rows)==wanted,{'actual':len(rows),'expected':wanted})
        if not rows: continue
        mutual=all(not(float(r['actual_throttle'])>0 and float(r['actual_brake'])>0) for r in rows)
        bounds=all(0<=float(r['actual_throttle'])<=1 and 0<=float(r['actual_brake'])<=1 and abs(float(r['actual_steer']))<=1 for r in rows)
        limit=config['controller_parameters']['steering_rate_per_s']*config['fixed_delta_s']
        steer=[float(r['requested_steer']) for r in rows]
        rate=all(abs(b-a)<=limit+1e-6 for a,b in zip([0.]+steer,steer))
        record(f'{case_id} 互斥及范围',mutual and bounds,{'mutual':mutual,'bounds':bounds})
        record(f'{case_id} 请求转向变化',rate,{'limit_per_tick':limit})
        saturated=[r for r in rows if r['controller_saturated']=='True']
        if name=='饱和抗积分':
            ints=[float(r['controller_integral']) for r in saturated]
            record(f'{case_id} 饱和抗积分',bool(ints) and max(abs(x) for x in ints)<1e-9,
                   {'saturated_ticks':len(ints),'integrals':ints})
        wheels=[float(r['wheel_angle_deg']) for r in rows]
        if spec['angle_rad']:
            record(f'{case_id} 车轮方向',statistics.mean(wheels[-5:])*spec['angle_rad']>0,
                   {'requested_rad':spec['angle_rad'],'final_wheel_mean_deg':statistics.mean(wheels[-5:])})
        metrics.append({'case_id':case_id,'name':name,'repeat':status['repeat'],'samples':len(rows),
            'mean_speed_error_mps':statistics.mean(abs(float(r['speed_mps'])-spec['target_speed_mps']) for r in rows),
            'mean_actual_acceleration_mps2':statistics.mean(float(r['longitudinal_acceleration_mps2']) for r in rows),
            'final_wheel_mean_deg':statistics.mean(wheels[-5:]),'requested_angle_rad':spec['angle_rad'],
            'saturated_ticks':len(saturated)})
    pos=[m['mean_actual_acceleration_mps2'] for m in metrics if m['name']=='正加速度']
    neg=[m['mean_actual_acceleration_mps2'] for m in metrics if m['name']=='负加速度']
    record('正负加速度响应可区分',len(pos)==config['repetitions'] and len(neg)==config['repetitions'] and min(pos)>max(neg),
           {'positive_mps2':pos,'negative_mps2':neg})
    result={'passed':sum(x['passed'] for x in checks),'failed':sum(not x['passed'] for x in checks),
        'checks':checks,'metrics':metrics,'raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),
        'limitations':'仅检查配置所列工况；轮角对应相邻帧区间，数值不能代表整条路或其他附着条件。'}
    (folder/'控制复核结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(f"C09 子检查：通过 {result['passed']}，失败 {result['failed']}，结果 {folder}")
    if result['failed']: raise RuntimeError('真实控制复核有失败项')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,default=None);args=parser.parse_args()
    candidates=[]
    for p in ROOT.glob('n13_*'):
        f=p/'运行配置.json'
        if f.is_file() and 'controller_checks' in json.loads(f.read_text(encoding='utf-8')): candidates.append(p)
    if args.run_dir is None and not candidates: raise FileNotFoundError('H盘没有 N13 控制复核运行')
    verify(args.run_dir or max(candidates,key=lambda p:p.stat().st_mtime))


if __name__=='__main__':main()
