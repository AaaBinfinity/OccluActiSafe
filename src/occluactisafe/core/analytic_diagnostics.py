"""N11固定解析实验矩阵与原始记录/中文开发图；无训练或测试集输入。"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timezone
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
from uuid import uuid4
from occluactisafe.common.config import PROJECT_ROOT,_yaml_file
from occluactisafe.common.run_manifest import _source_identity
from occluactisafe.planning.braking import braking_profile
from .runtime import run_episode


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path=None,*,overrides=None):
    p=Path(path or PROJECT_ROOT/'configs/analytic/n11.yaml')
    config=_yaml_file(p if p.is_absolute() else PROJECT_ROOT/p)
    if overrides:
        if set(overrides)-{'output_root'}:raise ValueError('仅允许覆盖output_root')
        config.update(overrides)
    return validate_config(config)


def validate_config(config):
    c=deepcopy(config)
    if set(c)!={'schema_version','output_root','duration_s','dt_s','propagation_s','scenarios'}:raise ValueError('N11配置缺失/未知字段')
    if c['schema_version']!='n11-analytic-1' or type(c['duration_s']) not in (int,float) or c['duration_s']!=6 or c['dt_s']!=.05 or c['propagation_s']!=.2:raise ValueError('当前N11冻结夹具6s/.05s/.2s')
    if c['scenarios']!=[{'scenario_id':'sufficient_35m','target_x_m':35.},{'scenario_id':'unavoidable_8m','target_x_m':8.}]:raise ValueError('N11冻结场景不可按结果替换')
    if not isinstance(c['output_root'],str) or not c['output_root'].strip() or '://' in c['output_root']:raise ValueError('output_root须为本地路径')
    p=Path(c['output_root']);c['output_root']=str((p if p.is_absolute() else PROJECT_ROOT/p).resolve());return c


def render_plots(directory,*,output_dir=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=Path(directory);out=Path(output_dir) if output_dir else directory;out.mkdir(parents=True,exist_ok=True)
    with (directory/'braking_phases.csv').open(encoding='utf-8',newline='') as f:braking=list(csv.DictReader(f))
    summary=json.loads((directory/'paired_episode_metrics.json').read_text(encoding='utf-8'))
    figures={};sources={'braking_phases.csv':sha(directory/'braking_phases.csv'),'paired_episode_metrics.json':sha(directory/'paired_episode_metrics.json')}
    episodes=[]
    for metric in summary:
        p=directory/metric['trajectory_file'];sources[p.name]=sha(p)
        episodes.append((metric,[json.loads(line) for line in p.read_text(encoding='utf-8').splitlines()]))
    def save(fig,name):
        try:
            fig.tight_layout()
            for ext in ('png','svg'):
                filename=name+'.'+ext;fig.savefig(out/filename,dpi=140,metadata={'Date':None} if ext=='svg' else None);figures[filename]=sha(out/filename)
        finally:plt.close(fig)
    with plt.rc_context({'svg.hashsalt':'n11-1','font.family':'DejaVu Sans','path.simplify':False}):
        fig,axes=plt.subplots(2,2,figsize=(10,7))
        for ax,field,unit in zip(axes.flat,('s','v','a','j'),('m','m/s','m/s2','m/s3')):
            ax.plot([float(r['time']) for r in braking],[float(r[field]) for r in braking]);ax.set(xlabel='time (s)',ylabel=f'{field} ({unit})')
        fig.suptitle('Exact phase integration, stop distance 21.175 m incl. queued delay');save(fig,'制动分段与停车')
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for metric,rows in episodes:
            index=0 if metric['target_x_m']==35 else 1;label=metric['strategy']+'/'+metric['information']
            axes[index,0].plot([r['time'] for r in rows],[r['state']['s'] for r in rows],label=label)
            axes[index,1].plot([r['time'] for r in rows],[r['state']['v'] for r in rows],label=label)
        for index,x in enumerate((35,8)):
            axes[index,0].axhline(x-4,linestyle=':',color='black',label='first-contact body boundary')
            for col in range(2):
                axes[index,col].set(xlabel='time (s)',ylabel='x (m)' if col==0 else 'speed (m/s)',title=f'target center {x} m; free-flight after contact')
                axes[index,col].legend(fontsize=7)
        save(fig,'解析分支配对轨迹')
        fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
        for metric,rows in episodes:
            if metric['target_x_m']!=35 or metric['strategy']!='risk':continue
            values=[r['audit']['risk']['candidates'][0]['R_max'] for r in rows]
            axes[0].plot([r['time'] for r in rows],[float('nan') if v is None else v for v in values],label=metric['information'])
            axes[1].step([r['time'] for r in rows],[{'NORMAL':0,'SLOW':1,'DEGRADED':2,'AEB':3}[r['audit']['decision']['action']] for r in rows],where='post',label=metric['information'])
        axes[0].set(ylabel='uncalibrated score; gaps=unavailable');axes[0].legend()
        axes[1].set(xlabel='time (s)',yticks=(0,1,2,3),yticklabels=('Normal','Caution','Degraded','AEB'));axes[1].legend();save(fig,'风险与制动状态时序')
    manifest={'sources':sources,'figures':figures,'smoothing':False,'downsampling':False,'kind':'analytic_world_not_carla'}
    write(out/'plot_manifest.json',manifest);return manifest


def run(config):
    c=validate_config(config);root=Path(c['output_root'])
    if not root.parent.is_dir():raise FileNotFoundError(f'产物父目录未挂载：{root.parent}')
    root.mkdir(exist_ok=True);directory=root/uuid4().hex;directory.mkdir();checks=[];metrics=[];started=time.perf_counter()
    write(directory/'config.json',c)
    write(directory/'manifest.json',{**_source_identity(PROJECT_ROOT),'started_utc':datetime.now(timezone.utc).isoformat(),
          'config_sha256':sha(directory/'config.json'),'input_mode':'analytic_fixture','data':None,'weights':None,'seed':None,
          'environment':{'os':platform.platform(),'python':sys.version,'executable':sys.executable,
                         **{k:importlib.metadata.version(k) for k in ('numpy','scipy','pyyaml','matplotlib')}},'gpu_used':False})
    def check(name,passed):
        checks.append({'id':name,'passed':bool(passed)});write(directory/'checks.json',checks)
        if not passed:raise AssertionError('N11断言失败：'+name)
    try:
        p=braking_profile(12,delay=.15);write(directory/'braking_segments.json',asdict(p))
        times=sorted({*(i/100 for i in range(401)),p.duration})
        with (directory/'braking_phases.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=('time','s','v','a','j'));w.writeheader();w.writerows({'time':t,**asdict(p.at(t))} for t in times)
        check('C02_21_175m',abs(p.at(10).s-21.175)<=1e-6)
        all_episodes={}
        for scene in c['scenarios']:
            for strategy in ('none','ttc','risk'):
                for information in ('local','cooperative'):
                    key=scene['scenario_id']+'_'+strategy+'_'+information
                    result=run_episode(scene['target_x_m'],strategy,information,duration=c['duration_s'],dt=c['dt_s'],propagation=c['propagation_s'])
                    all_episodes[key]=result
                    trajectory=key+'.jsonl'
                    with (directory/trajectory).open('w',encoding='utf-8') as f:
                        for row in result['rows']:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                    write(directory/(key+'_events.json'),{'events':result['events'],'requests':result['requests'],'channel_events':result['channel_events']})
                    metric={**{k:result[k] for k in ('initial_sha256','exogenous_schedule_sha256','episode_replay_sha256','strategy','information','analytic_contact','actual_collision','final')},
                            **scene,'trajectory_file':trajectory,'steps':len(result['rows']),'first_contact':result['events'][0] if result['events'] else None,
                            'requests':result['requests'],'calibrated_probability':None}
                    metrics.append(metric);write(directory/'paired_episode_metrics.json',metrics)
                    check(key+'_finite_mutual_control',all(r['control']['throttle']*r['control']['brake']==0 for r in result['rows']))
            group=[m for m in metrics if m['scenario_id']==scene['scenario_id']]
            check(scene['scenario_id']+'_paired_initial',len({m['initial_sha256'] for m in group})==1 and len({m['exogenous_schedule_sha256'] for m in group})==1)
            check(scene['scenario_id']+'_reference_contact',all(m['analytic_contact'] for m in group if m['strategy']=='none'))
            controls=[m for m in group if m['strategy']!='none' and m['information']=='cooperative']
            check(scene['scenario_id']+'_prespecified_outcome',all((not m['analytic_contact'] and m['final']['v']==0) if scene['target_x_m']==35 else m['analytic_contact'] for m in controls))
            a=all_episodes[scene['scenario_id']+'_risk_local']['rows'];b=all_episodes[scene['scenario_id']+'_risk_cooperative']['rows']
            check(scene['scenario_id']+'_common_prefix',all(x['state']==y['state'] and x['control']==y['control'] for x,y in zip(a,b) if x['time']<.2))
        render_plots(directory)
        result={'status':'passed','run_id':directory.name,'run_dir':str(directory),'checks_passed':len(checks),'checks_failed':0,'checks_skipped':0,'checks_not_run':0,
                'episodes':len(metrics),'world_scenarios':2,'state_rows':sum(m['steps'] for m in metrics),'wall_seconds':time.perf_counter()-started}
        write(directory/'result.json',result);write(directory/'status.json',{'status':'passed','completed_utc':datetime.now(timezone.utc).isoformat()})
    except BaseException as exc:
        write(directory/'status.json',{'status':'failed','type':type(exc).__name__,'reason':str(exc),'checks_completed':len(checks)})
        raise
    finally:write(directory/'evidence_index.json',{p.name:sha(p) for p in sorted(directory.iterdir()) if p.is_file() and p.name!='evidence_index.json'})
    return result
