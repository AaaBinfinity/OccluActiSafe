"""复核真实扫值的覆盖、物理记录和基本控制断言。"""
from collections import defaultdict
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path('H:/OccluActiSafeDataset/artifacts')


def verify(folder):
    folder=Path(folder);config=json.loads((folder/'运行配置.json').read_text(encoding='utf-8'))
    statuses=[json.loads(s) for s in (folder/'工况状态.jsonl').read_text(encoding='utf-8').splitlines()]
    completed={s['case_id']:s for s in statuses if s['status']=='completed'}
    expected=config['repetitions']*len(config['speeds_mps'])*(2*len(config['pedals'])+2*len(config.get('steer_values',[-1.,-.3,.3,1.])))
    checks=[]
    def add(name,passed,detail):checks.append({'name':name,'passed':bool(passed),'detail':detail})
    add('全部工况完成',set(completed)==set(range(expected)) and not any(s['status']=='failed' for s in statuses),
        {'completed':len(completed),'expected':expected})
    grouped=defaultdict(list); all_rows=[];raw=folder/'执行器原始采样.csv'
    with raw.open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            all_rows.append(row)
            if row['phase']=='response':grouped[int(row['case_id'])].append(row)
    add('每个响应帧数',len(grouped)==expected and all(len(v)==config['response_ticks'] for v in grouped.values()),
        {'cases_with_response':len(grouped),'expected_ticks':config['response_ticks']})
    fields=('speed_mps','longitudinal_acceleration_mps2','lateral_acceleration_mps2','yaw_deg','wheel_angle_deg',
            'requested_throttle','requested_brake','requested_steer','actual_throttle','actual_brake','actual_steer')
    add('物理及控制记录有限',bool(all_rows) and all(math.isfinite(float(r[k])) for r in all_rows for k in fields),{'rows':len(all_rows)})
    mutual=all(not(float(r['actual_throttle'])>0 and float(r['actual_brake'])>0) for r in all_rows)
    bounds=all(0<=float(r['actual_throttle'])<=1 and 0<=float(r['actual_brake'])<=1 and abs(float(r['actual_steer']))<=1 for r in all_rows)
    add('真实控制互斥与限幅',mutual and bounds,{'mutual':mutual,'bounds':bounds})
    zero=sum(float(r['requested_throttle'])==0 and float(r['requested_brake'])==0 and float(r['requested_steer'])==0 for r in all_rows)
    add('零输入记录',zero>0,{'rows':zero})
    steers=[(case_id,rows) for case_id,rows in grouped.items() if rows[0]['kind']=='steer_step']
    signed=sum(any(float(row['wheel_angle_deg'])*float(rows[0]['value'])>0 for row in rows) for _,rows in steers)
    add('正负转向车轮角符号',len(steers)>0 and signed==len(steers),{'matching_cases':signed,'cases':len(steers)})
    brakes=[rows for rows in grouped.values() if rows[0]['kind']=='brake' and float(rows[0]['value'])==1]
    stops=sum(any(float(r['speed_mps'])<.1 for r in rows) for rows in brakes)
    add('全制动停车',len(brakes)==config['repetitions']*len(config['speeds_mps']) and stops==len(brakes),
        {'stopped_cases':stops,'braking_cases':len(brakes),'threshold_mps':.1})
    identities=defaultdict(set)
    for state in completed.values():identities[(state['speed'],state['kind'],state['value'])].add(state['repeat'])
    add('每个速度工况重复次数',len(identities)==expected//config['repetitions'] and
        all(v==set(range(config['repetitions'])) for v in identities.values()),
        {'conditions':len(identities),'repetitions':config['repetitions']})
    result={'passed':sum(x['passed'] for x in checks),'failed':sum(not x['passed'] for x in checks),
            'checks':checks,'response_cases':len(grouped),'raw_rows':len(all_rows),
            'raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest()}
    (folder/'扫值复核结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(f"扫值检查通过 {result['passed']}，失败 {result['failed']}；响应工况 {len(grouped)}/{expected}")
    if result['failed']:raise RuntimeError('扫值复核有失败项')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,default=None);args=parser.parse_args()
    candidates=[p for p in ROOT.glob('n13_*') if (p/'逐工况指标.json').is_file()]
    if args.run_dir is None and not candidates:raise FileNotFoundError('H盘没有已汇总的 N13 扫值')
    verify(args.run_dir or max(candidates,key=lambda p:p.stat().st_mtime))


if __name__=='__main__':main()
