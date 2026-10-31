"""汇总 N13 原始采样，不补造缺失工况；默认分析最新采样目录。"""
from collections import defaultdict
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path('H:/OccluActiSafeDataset/artifacts')


def analyze(folder, *, steering_only=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    folder=Path(folder)
    raw=folder/'执行器原始采样.csv'
    with raw.open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    by_case=defaultdict(list)
    for row in rows:
        if row['phase']=='response': by_case[int(row['case_id'])].append(row)
    results=[]
    statuses=folder/'工况状态.jsonl'
    case_names={}
    if statuses.is_file():
        for line in statuses.read_text(encoding='utf-8').splitlines():
            state=json.loads(line)
            if 'check' in state:case_names[state['case_id']]=state['check']['name']
    for case_id, items in sorted(by_case.items()):
        first=items[0]; initial=float(first['speed_mps'])
        distance=0.
        acceleration=[float(x['longitudinal_acceleration_mps2']) for x in items]
        wheel=[float(x['wheel_angle_deg']) for x in items]
        for left,right in zip(items,items[1:]):
            distance+=math.hypot(float(right['x_m'])-float(left['x_m']),float(right['y_m'])-float(left['y_m']))
        stop=next((i for i,x in enumerate(items) if float(x['speed_mps'])<.1),None)
        stop_distance=None
        if stop is not None:
            stop_distance=sum(math.hypot(float(b['x_m'])-float(a['x_m']),float(b['y_m'])-float(a['y_m']))
                              for a,b in zip(items[:stop],items[1:stop+1]))
        dt=.05
        results.append({'case_id':case_id,'repeat':int(first['repeat']),'kind':first['kind'],'value':float(first['value']),
            'target_initial_speed_mps':float(first['target_initial_speed_mps']),'actual_initial_speed_mps':initial,
            'end_speed_mps':float(items[-1]['speed_mps']),'samples':len(items),'distance_m':distance,
            'stop_distance_m':stop_distance,'constant_deceleration_stop_m':initial**2/(2*abs(statistics.mean(acceleration[:min(20,len(acceleration))]))) if first['kind']=='brake' and statistics.mean(acceleration[:min(20,len(acceleration))])<0 else None,
            'mean_acceleration_mps2':statistics.mean(acceleration),'min_acceleration_mps2':min(acceleration),
            'max_abs_jerk_mps3':max((abs(b-a)/dt for a,b in zip(acceleration,acceleration[1:])),default=0),
            'max_abs_wheel_angle_deg':max(map(abs,wheel)),
            'max_abs_wheel_rate_deg_s':max((abs(b-a)/dt for a,b in zip(wheel,wheel[1:])),default=0),
            'response_delay_s':next((i*dt for i,a in enumerate(acceleration) if abs(a-acceleration[0])>.1),None)})
    summary=folder/'逐工况指标.json'
    summary.write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    groups=defaultdict(list)
    for r in results: groups[(r['kind'],r['value'],r['target_initial_speed_mps'])].append(r)
    repeat=[]
    for (kind,value,speed),rs in sorted(groups.items()):
        for metric in ('actual_initial_speed_mps','mean_acceleration_mps2','stop_distance_m','max_abs_wheel_angle_deg'):
            vals=[x[metric] for x in rs if x[metric] is not None]
            repeat.append({'kind':kind,'value':value,'speed_mps':speed,'metric':metric,'count':len(vals),
                           'mean':statistics.mean(vals) if vals else None,
                           'range':max(vals)-min(vals) if vals else None,
                           'sample_sd':statistics.stdev(vals) if len(vals)>1 else None})
    (folder/'重复误差.json').write_text(json.dumps(repeat,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    selected=[]
    for speed in sorted({float(v[0]['target_initial_speed_mps']) for v in by_case.values()}):
        for kind,value in (('throttle',1.),('brake',1.),('steer_step',-1.),('steer_step',1.)):
            matches=[case_id for case_id,v in by_case.items() if float(v[0]['target_initial_speed_mps'])==speed
                     and v[0]['kind']==kind and float(v[0]['value'])==value]
            if matches: selected.append(min(matches))
    if not selected: selected=sorted(by_case)[:12]
    if selected and not steering_only:
        fig,axes=plt.subplots(3,1,figsize=(11,10),sharex=True,constrained_layout=True)
        for case_id in selected:
            items=by_case[case_id]; x=[float(r['sim_time_s'])-float(items[0]['sim_time_s']) for r in items]
            label=(f"{case_id}: {case_names[case_id]}" if case_id in case_names else
                   f"{items[0]['target_initial_speed_mps']} m/s {items[0]['kind']} {items[0]['value']}")
            for ax,key in zip(axes,('speed_mps','longitudinal_acceleration_mps2','wheel_angle_deg')):
                ax.plot(x,[float(r[key]) for r in items],label=label,linewidth=.8)
        for ax,label in zip(axes,('速度 (m/s)','纵向加速度 (m/s²)','左前轮角度 (°)')): ax.set_ylabel(label);ax.grid(alpha=.25)
        axes[-1].set_xlabel('响应时间 (s)');axes[0].legend(fontsize=7,ncol=3)
        for ext in ('png','svg'): fig.savefig(folder/f'执行器响应曲线.{ext}',dpi=180)
        plt.close(fig)
    pedal_cases=[items for items in by_case.values() if items[0]['kind'] in ('throttle','brake') and len(items)>=15]
    if pedal_cases and not steering_only:
        fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
        speeds=sorted({float(items[0]['target_initial_speed_mps']) for items in pedal_cases})
        for speed in speeds:
            group=[items for items in pedal_cases if float(items[0]['target_initial_speed_mps'])==speed]
            x=[float(items[0]['value'])*(1 if items[0]['kind']=='throttle' else -1) for items in group]
            y=[statistics.mean(float(r['longitudinal_acceleration_mps2']) for r in items[5:15]) for items in group]
            ax.scatter(x,y,s=20,label=f'{speed:g} m/s，各次重复')
        ax.set_xlabel('踏板输入（负为制动，正为油门）');ax.set_ylabel('响应第5–14帧平均纵向加速度 (m/s²)')
        ax.grid(alpha=.25);ax.legend()
        for ext in ('png','svg'):fig.savefig(folder/f'踏板与加速度重复散点.{ext}',dpi=180)
        plt.close(fig)
    steering_cases=[]
    for speed in sorted({float(v[0]['target_initial_speed_mps']) for v in by_case.values()}):
        for kind,value in (('steer_step',-1.),('steer_step',1.),('steer_ramp',-.3),('steer_ramp',.3)):
            matches=[(case_id,v) for case_id,v in by_case.items() if float(v[0]['target_initial_speed_mps'])==speed
                     and v[0]['kind']==kind and float(v[0]['value'])==value]
            if matches:steering_cases.append(min(matches,key=lambda pair:pair[0])[1])
    if steering_cases:
        fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
        for items in steering_cases:
            label=f"{items[0]['target_initial_speed_mps']} m/s {items[0]['kind']} {items[0]['value']}"
            xs=[float(x['x_m'])-float(items[0]['x_m']) for x in items]
            ys=[float(x['y_m'])-float(items[0]['y_m']) for x in items]
            times=[float(x['sim_time_s'])-float(items[0]['sim_time_s']) for x in items]
            axes[0].plot(xs,ys,label=label,linewidth=.9)
            axes[1].plot(times,[float(x['lateral_acceleration_mps2']) for x in items],linewidth=.9)
        axes[0].set_xlabel('世界坐标 x 相对位移 (m)');axes[0].set_ylabel('世界坐标 y 相对位移 (m)')
        axes[0].axis('equal');axes[0].legend(fontsize=6)
        axes[1].set_xlabel('响应时间 (s)');axes[1].set_ylabel('车体横向加速度 (m/s²)')
        for ax in axes:ax.grid(alpha=.25)
        for ext in ('png','svg'):fig.savefig(folder/f'转向轨迹与横向加速度.{ext}',dpi=180)
        plt.close(fig)
    files=[raw,summary,folder/'重复误差.json']
    for ext in ('png','svg'):
        for stem in ('执行器响应曲线','踏板与加速度重复散点','转向轨迹与横向加速度'):
            p=folder/f'{stem}.{ext}'
            if p.exists(): files.append(p)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (folder/'图表来源与哈希.json').write_text(json.dumps({'raw':raw.name,'files':hashes,
        'limitations':'每个轮角RPC只知位于相邻帧区间；统计含未完成工况的部分响应，不代表通过。'},ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'响应工况 {len(results)}，明细与图表：{folder}')
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,default=None)
    parser.add_argument('--steering-only',action='store_true',help='只新增转向图，保留既有图像原件')
    args=parser.parse_args()
    candidates=[p for p in ROOT.glob('n13_*') if (p/'执行器原始采样.csv').is_file()]
    if args.run_dir is None and not candidates: raise FileNotFoundError('H盘没有 N13 采样目录')
    analyze(args.run_dir or max(candidates,key=lambda p:p.stat().st_mtime),steering_only=args.steering_only)


if __name__=='__main__': main()
