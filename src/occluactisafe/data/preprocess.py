"""N04：冻结分组后逐文件生成输入/监督分离的轨迹窗口。"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import math
import numpy as np
import yaml
from occluactisafe.common.config import PROJECT_ROOT, _UniqueLoader, load_config
from occluactisafe.common.coordinates import Pose2D
from occluactisafe.common.run_manifest import create_run_manifest
from occluactisafe.data.adapters.tfd import _location_info, TRAJECTORY_LOCATIONS
from .split import build_group_split, canonical_hash, audit_group_split
from .windows import build_causal_window, number
from .labels import build_future_labels
from occluactisafe.prediction.dataset import MaskedScaler, save_sample

DEFAULTS={'audit_dir':'H:/OccluActiSafeDataset/artifacts/n03_20261029/full_audit/80e4390cb71442498769c622eebf06e9',
          'output_root':'H:/OccluActiSafeDataset/artifacts/n04','seed':20261030,
          'ratios':[0.65,0.15,0.10,0.10],'history_steps':50,'future_steps':50,'dt':0.1,
          'workers':6,'max_files':0,
          'active_views':['single-vehicle/trajectories','cooperative-vehicle-infrastructure/vehicle-trajectories']}


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def load_preprocess_config(path, *, overrides=None, cli_overrides=None, project_root=PROJECT_ROOT):
    root=Path(project_root).resolve();file=Path(path)
    file=file if file.is_absolute() else root/file
    with file.open(encoding='utf-8') as stream:layer=yaml.load(stream,Loader=_UniqueLoader)
    config={**DEFAULTS}
    for layer in (layer,overrides,cli_overrides):
        if layer is None:continue
        if not isinstance(layer,dict) or set(layer)-set(DEFAULTS):raise ValueError('N04配置包含未知字段')
        config.update(layer)
    for key in ('history_steps','future_steps','workers','seed','max_files'):
        if type(config[key]) is not int or config[key]< (1 if key in ('history_steps','future_steps','workers') else 0):raise ValueError(f'{key}非法')
    if config['workers']>8 or number(config['dt']) is None or config['dt']<=0:raise ValueError('workers或dt非法')
    if not isinstance(config['active_views'],list) or not config['active_views'] or set(config['active_views'])-set('/'.join(k) for k in TRAJECTORY_LOCATIONS):raise ValueError('输入视角非法')
    # 未证实因果性的融合结果只登记，不能作为此历史输入协议。
    if 'cooperative-vehicle-infrastructure/cooperative-trajectories' in config['active_views']:raise ValueError('未核验融合轨迹不能作为历史输入')
    for key in ('audit_dir','output_root'):
        p=Path(config[key]).expanduser();config[key]=str((p if p.is_absolute() else root/p).resolve())
    return config


def _index_file(args):
    data, span, identity = args;file=(data/span['source_file']).resolve()
    if not file.is_relative_to(data) or not file.is_file():raise ValueError(f'输入文件不存在或越界: {file}')
    if file.stat().st_size!=identity['size_bytes']:raise ValueError(f'文件大小与N03不符: {file}')
    with file.open(encoding='utf-8-sig',newline='') as stream:
        row=next(csv.DictReader(stream),None)
    if row is None or not row.get('city') or not row.get('intersect_id'):raise ValueError(f'缺少原始地理分组字段: {file}')
    collection,source,split,episode=_location_info(data,file,TRAJECTORY_LOCATIONS)
    return {'source_file':span['source_file'],'sha256':identity['sha256'],'size_bytes':identity['size_bytes'],
            'base_episode_id':episode,'location_id':row['city']+':'+row['intersect_id'],
            'recording_id':None,'original_split':split,'source':source,'collection':collection,
            'start_time':span['min_timestamp'],'end_time':span['max_timestamp']}


def _space(path):
    if shutil.disk_usage(path).free<10*1024**3:raise OSError('N04输出盘不足10 GiB，保留部分产物后停止')


def _make_sample(data, entry, config):
    path=(Path(data)/entry['source_file']).resolve();blob=path.read_bytes()
    if hashlib.sha256(blob).hexdigest()!=entry['sha256']:raise ValueError(f'原始文件哈希与N03不符: {path}')
    start=Decimal(entry['start_time']);domain='offline_tfd_reference:'+entry['source_file']
    reader=csv.DictReader(io.StringIO(blob.decode('utf-8-sig'),newline=''));records=[];cache={};raw_count=0
    for row in reader:
        raw_count+=1
        if row.get('city','')+':'+row.get('intersect_id','')!=entry['location_id']:
            raise ValueError(f'文件含多个地理标识，须先重建分组，不能继续写窗: {path}')
        stamp=row.get('timestamp','')
        if stamp not in cache:cache[stamp]=float(Decimal(stamp)-start)
        t=cache[stamp]
        # 输入显式为离线标注历史参考，不宣称真实处理/到达时间已由数据提供。
        records.append({'track_id':row.get('id',''),'observation_time':t,'source_available_time':t,
                        'arrival_time':t,'usable_time':t,'state_time':t,'clock_domain':domain,
                        'x':row.get('x'),'y':row.get('y'),'vx':row.get('v_x'),'vy':row.get('v_y'),
                        'velocity_reliable':number(row.get('v_x')) is not None and number(row.get('v_y')) is not None,
                        'heading':row.get('theta'),'length':row.get('length'),'width':row.get('width'),
                        'category':row.get('type',''),'tag':row.get('tag',''),'source_row':reader.line_num})
    cutoff=(config['history_steps']-1)*config['dt']
    egos=[r for r in records if r['tag']=='AV' and r['observation_time']<=cutoff and number(r['x']) is not None and number(r['y']) is not None]
    if not egos:return None,None,{'reason':'no_legal_ego_pose','raw_rows':raw_count}
    ego=max(egos,key=lambda r:r['observation_time']);yaw=number(ego['heading']);basis='observed_heading'
    if yaw is None:
        prior=sorted((r for r in egos if r['track_id']==ego['track_id'] and r['observation_time']<ego['observation_time']),key=lambda r:r['observation_time'])
        yaw=0.0;basis='world_axes_reference'
        if prior:
            dx,dy=float(ego['x'])-float(prior[-1]['x']),float(ego['y'])-float(prior[-1]['y'])
            gap=ego['observation_time']-prior[-1]['observation_time']
            if math.hypot(dx,dy)/gap>=0.2:yaw=math.atan2(dy,dx);basis='causal_motion_reference_not_body_heading'
    window=build_causal_window(records,cutoff,Pose2D(float(ego['x']),float(ego['y']),yaw),
                               clock_domain=domain,history_steps=config['history_steps'],dt=config['dt'])
    labels=build_future_labels(records,window,future_steps=config['future_steps'],dt=config['dt'])
    labels['target_eligible']=window['prediction_target_mask'] & window['observed_mask'].all(axis=1) & labels['future_mask'].all(axis=1)
    return window,labels,{'raw_rows':raw_count,'anchor_yaw_basis':basis,'reference_input_mode':'offline_annotation_history_reference'}


def _process_partition(args):
    data,entries,config,directory=args;directory=Path(directory);directory.mkdir()
    scaler=MaskedScaler();counts=Counter();coverage=Counter();eligible_coverage=Counter()
    with (directory/'index.jsonl').open('w',encoding='utf-8',newline='\n') as stream:
        for i,entry in enumerate(entries):
            if i%100==0:_space(directory)
            window,labels,extra=_make_sample(data,entry,config)
            index={k:entry[k] for k in ('source_file','group_id','project_role','location_id','original_split')}
            index.update(extra)
            if window is None:index.update(status='unavailable',sample_path=None);counts['unavailable']+=1
            else:
                name=hashlib.sha256(entry['source_file'].encode()).hexdigest()+'.npz'
                save_sample(directory/name,window,labels)
                if entry['project_role']=='train':scaler.update(window['features'],window['feature_mask'],role='train')
                index.update(status='written',sample_path=f'samples/{directory.name}/{name}',input_hash=window['input_hash'],
                             historical_targets=len(window['track_ids']),eligible_targets=int(labels['target_eligible'].sum()),
                             observed_slots=int(window['observed_mask'].sum()),total_slots=window['observed_mask'].size)
                for category in range(4):
                    n=int((window['categories']==category).sum());coverage[f"{entry['project_role']}:{category}"]+=n
                    n=int(((window['categories']==category)&labels['target_eligible']).sum());eligible_coverage[f"{entry['project_role']}:{category}"]+=n
                counts['written']+=1;counts['history_targets']+=len(window['track_ids']);counts['eligible_targets']+=int(labels['target_eligible'].sum())
            counts['raw_rows']+=extra['raw_rows']
            counts['files']+=1;stream.write(json.dumps(index,ensure_ascii=False,separators=(',',':'))+'\n')
            if (i+1)%100==0:write_json(directory/'progress.json',dict(counts))
    result={'counts':dict(counts),'history_category_counts':dict(coverage),'eligible_category_counts':dict(eligible_coverage),'scaler_statistics':scaler.statistics()}
    write_json(directory/'summary.json',result);write_json(directory/'progress.json',{**dict(counts),'complete':True})
    return result


def run_preprocess(config, *, project_root=PROJECT_ROOT):
    audit=Path(config['audit_dir']);out=Path(config['output_root']);root=Path(project_root).resolve()
    status=json.loads((audit/'audit_status.json').read_text(encoding='utf-8'))
    if status['status']!='audit_completed':raise ValueError('N03审计未完成')
    manifest=json.loads((audit/'data_manifest.json').read_text(encoding='utf-8'));data=Path(manifest['data_root']).resolve()
    if not data.is_dir():raise FileNotFoundError(f'原始数据未接入: {data}')
    if out==data or data in out.parents or out==audit or audit in out.parents:raise ValueError('输出不得覆盖N03或原始数据目录')
    review=json.loads((audit.parent.parent/'readiness_review.json').read_text(encoding='utf-8'))
    if not review.get('formal_data_ready_for_registered_local_scope') or review['dataset_sha256']!=manifest['dataset_sha256']:raise ValueError('N03本地范围就绪复核不匹配')
    quality=json.loads((audit/'data_quality.json').read_text(encoding='utf-8'))
    identities={f['path']:f for f in manifest['files']}
    print('N04：冻结全量文件分组；窗口协议',config,flush=True)
    spans=quality['raw']['time_spans'];entries=[]
    with ThreadPoolExecutor(max_workers=config['workers']) as pool:
        # 分批提交，避免为二十余万文件同时保留Future与打开文件任务。
        for start in range(0,len(spans),1000):
            entries.extend(pool.map(_index_file,((data,s,identities[s['source_file']]) for s in spans[start:start+1000])))
            if start%20000==0:print(f'分组元数据：{min(start+1000,len(spans))}/{len(spans)}',flush=True)
    del spans,quality,identities
    split=build_group_split(entries,seed=config['seed'],ratios=config['ratios'])
    selected=[r for r in split['files'] if r['project_input_allowed'] and '/'.join(r['source_file'].split('/')[:2]) in config['active_views']]
    if config['max_files']:
        # 小规模入口仍按角色取样，明确标limited，不冒充全量。
        limit=config['max_files'];chosen=[]
        for role in ('train','model_val','risk_calibration','test'):
            chosen.extend([r for r in selected if r['project_role']==role][:limit])
        selected=chosen
    if not selected:raise ValueError('没有符合所声明输入协议的文件')
    base=load_config(project_root=root,overrides={'runtime':{'seed':config['seed']}})
    run=create_run_manifest(config=base,input_mode='offline_annotation_history_reference',
                            data={'name':'TFD_N03_registered','version':manifest['dataset_version'],'sha256':manifest['dataset_sha256']},
                            output_root=out,project_root=root)
    directory=out/run['run_id'];write_json(directory/'n04_status.json',{'status':'running','run_id':run['run_id']})
    write_json(directory/'task_config.json',config);write_json(directory/'split_manifest.json',split)
    write_json(directory/'split_audit.json',audit_group_split(split))
    write_json(directory/'selection.json',{'scope':'limited' if config['max_files'] else 'full_declared_vehicle_history_protocol',
                                         'files':[r['source_file'] for r in selected],
                                         'active_views':config['active_views'],'original_split':'train',
                                         'excluded_original_val_test':'quarantined','excluded_other_views':'catalogued and grouped; not history input in this protocol',
                                         'supervision':'same-source future annotation; fused trajectories not inputs'})
    print(f"run_id={run['run_id']}；分组{split['group_count']}；将处理{len(selected)}个文件",flush=True)
    try:
        samples=directory/'samples';samples.mkdir();workers=min(config['workers'],len(selected))
        jobs=[(str(data),selected[len(selected)*i//workers:len(selected)*(i+1)//workers],config,str(samples/str(i))) for i in range(workers)]
        with ProcessPoolExecutor(max_workers=workers) as pool:results=list(pool.map(_process_partition,jobs))
        scaler=MaskedScaler();counts=Counter();coverage=Counter();eligible=Counter()
        for result in results:
            scaler.merge(result['scaler_statistics'],role='train');counts.update(result['counts']);coverage.update(result['history_category_counts']);eligible.update(result['eligible_category_counts'])
        if not scaler.count.any():raise ValueError('训练集合无有效历史，不能保存伪归一化')
        frozen=scaler.freeze();write_json(directory/'scaler.json',frozen)
        final={'status':'completed','run_id':run['run_id'],'scope':'limited' if config['max_files'] else 'full_declared_vehicle_history_protocol',
               'counts':dict(counts),'history_category_counts':dict(coverage),'eligible_category_counts':dict(eligible),
               'split_sha256':split['sha256'],'scaler_sha256':frozen['sha256'],
               'task_config_sha256':canonical_hash(config),'completed_utc':datetime.now(timezone.utc).isoformat(),
               'mode':'offline_annotation_history_reference','label_separation':'separate .labels.npz files',
               'official_protocol_note':'50/50 sampling retained; project geography holdout is not an official test score'}
        write_json(directory/'n04_status.json',final);return {**final,'run_dir':str(directory)}
    except BaseException as exc:
        write_json(directory/'n04_status.json',{'status':'failed','run_id':run['run_id'],'error_type':type(exc).__name__,'error':str(exc)})
        raise
