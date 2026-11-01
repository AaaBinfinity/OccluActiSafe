"""N14高画质开发场景复核：逐对象因果、全参与者配对、原生像素及独立视频。"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess

from PIL import Image
from analyze_occluded_junction import analyze,read,rows,STRATEGIES,speed
from export_n14_camera_videos import export,decode_check
from occluactisafe.common.schemas import ObservationPacket,ReceivedMessage
from occluactisafe.simulation.n14_participants import participant_specs

PARENT=Path('H:/OccluActiSafeDataset/artifacts/节点资料/N14_盲区路口/02_运行产物')
ROOT=Path(__file__).resolve().parents[1]


def additional_checks(folder):
    config=read(folder/'冻结配置.json');specs=participant_specs(config)
    classes={s['track_id']:s['category'] for s in specs};checks=[];raw=[];branches={}
    def check(name,passed,detail):checks.append(dict(name=name,passed=bool(passed),detail=detail))
    for branch in STRATEGIES:
        location=folder/branch;truth=rows(location/'评测真值.jsonl');observed=rows(location/'合法观测.jsonl')
        audit=rows(location/'在线决策.jsonl');identities=read(location/'评测对象身份.json');geometry=read(location/'车辆静态几何.json')
        branches[branch]=truth;by_time={round(t['time_s']/.05):t for t in truth}
        native={t['category']:[] for t in specs}
        for identity,value in identities['participants'].items():
            native[value['category']].append(geometry['boxes'][value['role']]['type_id'])
        check(branch+'真实参与者类别',set(identities['participants'])==set(classes)
              and all(v.startswith('walker.pedestrian.') for v in native.get('pedestrian',[]))
              and all(v.startswith('vehicle.') for v in native.get('vehicle',[])),native)
        causal=True;road_causal=True;controls=True;first_usable={}
        commands=rows(location/'批量RPC请求.jsonl')[-len(truth):]
        for record,observation,decision,batch in zip(truth,observed,audit,commands):
            local=ObservationPacket.from_dict(observation['local']).tracks if observation['local'] else ()
            expected={identity for identity,v in record['participant_visibility'].items() if v['local']}
            causal=causal and {t.source_track_id for t in local}==expected
            for track in local:
                current=record['participants'][track.source_track_id]
                causal=causal and track.category==classes[track.source_track_id]
                causal=causal and record['participant_pixels']['车端'][track.source_track_id]>0
                causal=causal and all(abs(getattr(track,k)-current[k])<1e-6 for k in ('x','y','vx','vy','length','width'))
            for message in observation['received']:
                received=ReceivedMessage.from_dict(message)
                for target in received.payload.targets:
                    track=target.track;source=by_time.get(round(track.state_time/.05))
                    road_causal=road_causal and source is not None and track.state_time<=observation['time_s']
                    if source is None:continue
                    current=source['participants'].get(track.source_track_id)
                    road_causal=road_causal and current is not None and source['participant_pixels']['路侧'].get(track.source_track_id,0)>0
                    if current is not None:
                        road_causal=road_causal and all(abs(getattr(track,k)-current[k])<1e-6 for k in ('x','y','vx','vy','length','width'))
                    first_usable.setdefault(track.source_track_id,received.usable_time)
            applied={c['actor_id']:c for c in batch['commands']}
            for identity,value in identities['participants'].items():
                command=applied.get(value['actor_id']);role=value['role']
                controls=controls and command is not None and command['value']==record['participant_controls'][role]
                if command is not None:controls=controls and command['kind']==('walker_control' if value['category']=='pedestrian' else 'control')
        check(branch+'逐目标本地可见性白名单',causal,{'target_count':len(classes),'frames':len(truth)})
        check(branch+'逐目标路侧消息观测溯源',road_causal,{'first_usable_by_target_s':first_usable})
        check(branch+'全部外部参与者控制回执',controls and len(commands)==len(truth),{'formal_batches':len(commands)})
        if config.get('settle_static_occluder'):
            freeze=read(location/'静态遮挡物冻结回执.json')
            batches=rows(location/'批量RPC请求.jsonl')
            matched=next((r for r in batches if r['sequence']==freeze['sequence']),None)
            expected=[dict(kind='physics',actor_id=identities['occluder_actor_id'],value=False)]
            check(branch+'相机启动前确认实体物理冻结',freeze['acknowledged'] and not freeze['image_callbacks_registered']
                  and freeze['settled_without_camera_frames']==130 and freeze['worker_tick_calls']==0
                  and matched is not None and matched['commands']==expected and matched['response_count']==1,freeze)
        no_static_contact=all(not any(t['participant_occluder_overlaps'].values())
            and not any(t['static_occluder_overlaps'].values()) for t in truth)
        expected_contacts={v['actor_id'] for v in identities['participants'].values()}
        collisions=read(location/'碰撞传感器.json')
        check(branch+'实体遮挡未造成额外接触',no_static_contact and all(c['other_id'] in expected_contacts for c in collisions),collisions)
        camera_index=read(location/'相机帧索引.json');native_ok=True;samples=0
        for camera in ('车端','路侧','跟车','俯瞰','前侧'):
            frames=[r for r in camera_index if r['source']==camera]
            native_ok=native_ok and bool(frames) and all(r.get('native_size')==[1920,1080] and len(r.get('raw_bgra_sha256',''))==64 for r in frames)
            for r in (frames[:1]+frames[-1:]):
                with Image.open(location/r['image']) as image:
                    native_ok=native_ok and image.size==(1920,1080) and hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()==r['rgb_pixels_sha256']
                samples+=1
        visual=read(location/'示意帧索引.json')
        with Image.open(location/visual[0]['image']) as image:native_ok=native_ok and image.size==(1920,1080)
        check(branch+'原生1080p实景与示意',native_ok,{'rgb_endpoint_pixel_checks':samples,'visibility_size':identities['visibility_size']})
        for spec in specs:
            identity=spec['track_id'];get_first=lambda field:next((r['time_s'] for r in truth if r['participant_visibility'][identity][field]),None)
            motions=[speed(r['participants'][identity]) for r in truth if r['time_s']>=spec['start_s']+.5]
            average=sum(motions)/len(motions) if motions else None
            if spec['category']=='pedestrian':
                check(branch+'行人实际运动与等待',len(motions)>=5 and abs(average-spec['speed_mps'])<=.15
                      and all(r['participant_controls'][spec['role']]['speed']==(0. if r['time_s']<spec['start_s'] else spec['speed_mps']) for r in truth),
                      {'target':identity,'measured_mean_speed_mps':average,'samples':len(motions),'start_s':spec['start_s']})
            raw.append(dict(strategy=branch,track_id=identity,category=spec['category'],first_local_s=get_first('local'),
                first_roadside_s=get_first('roadside'),first_message_usable_s=first_usable.get(identity),
                observed_mean_speed_mps=average,speed_samples=len(motions),
                geometric_overlap_frames=sum(r['participant_overlaps'][identity] for r in truth)))
        if len(specs)>1:
            covered=set().union(*(set(t['track_id'] for t in a['decision']['known_targets']) for a in audit))
            eligible={identity for t in truth for identity,v in t['participant_visibility'].items() if v['local']}
            if branch=='cooperative_aeb':eligible.update(first_usable)
            check(branch+'多目标未被单对象覆盖',eligible<=covered and (branch!='cooperative_aeb' or set(classes)<=covered),
                  {'world_targets':sorted(classes),'eligible':sorted(eligible),'known':sorted(covered)})
    differences=[];tolerance=config['pairing_tolerance']
    for branch in STRATEGIES[1:]:
        for left,right in zip(branches['reference'],branches[branch]):
            if left['time_s']>=config['prefix_s']:break
            actor_pairs={**{k:(left[k],right[k]) for k in ('ego','occluder')},
                         **{k:(left['participants'][k],right['participants'][k]) for k in classes},
                         **{k:(left['static_occluders'][k],right['static_occluders'][k]) for k in left['static_occluders']}}
            for identity,(a,b) in actor_pairs.items():
                if identity=='ego':ca,cb=left['control'],right['control']
                elif identity=='occluder':ca,cb=left['occluder_control'] or {},right['occluder_control'] or {}
                elif identity in classes:
                    role=next(s['role'] for s in specs if s['track_id']==identity)
                    ca,cb=left['participant_controls'][role],right['participant_controls'][role]
                else:ca=cb={}
                vehicle_control=max((abs(ca[k]-cb[k]) for k in ('throttle','brake','steer') if k in ca),default=0.)
                # walker原生命令无归一化油门/制动；同一外生调度须完全一致。
                walker_equal=not ('speed' in ca) or ca==cb
                differences.append(dict(branch=branch,time_s=left['time_s'],actor=identity,
                    position_m=math.hypot(a['x']-b['x'],a['y']-b['y']),velocity_mps=math.hypot(a['vx']-b['vx'],a['vy']-b['vy']),
                    yaw_rad=abs((a['yaw']-b['yaw']+math.pi)%(2*math.pi)-math.pi),
                    normalized_control_difference=vehicle_control,walker_command_equal=walker_equal,
                    time_equal=left['time_s']==right['time_s']))
    expected_rows=2*round(config['prefix_s']/.05)*(2+len(classes)+len(branches['reference'][0]['static_occluders']))
    check('所有动态及静态对象公共前缀',len(differences)==expected_rows and all(r['position_m']<=tolerance['position_m']
          and r['velocity_mps']<=tolerance['velocity_mps'] and r['yaw_rad']<=tolerance['yaw_rad']
          and r['normalized_control_difference']<=tolerance['control'] and r['walker_command_equal']
          and r['time_equal'] for r in differences),
          {'rows':len(differences),'tolerance':tolerance,'maxima':{key:max(r[key] for r in differences) for key in ('position_m','velocity_mps','yaw_rad')}})
    for file,records in (('全参与者公共前缀差异.csv',differences),('逐参与者可见性与事件.csv',raw)):
        with (folder/file).open('w',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    report={'passed':sum(c['passed'] for c in checks),'failed':sum(not c['passed'] for c in checks),'checks':checks}
    (folder/'多参与者追加验收.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report


def comparison_decode(folder):
    records=read(folder/'视频来源与哈希.json');results=[]
    for name,record in records.items():
        path=folder/record['video'] if name=='follow_comparison' else folder/name/record['video']
        proof=decode_check(path,record['frames'],record['fps'],(record['width'],record['height']))
        results.append(dict(video=str(path.relative_to(folder)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**proof))
    report=dict(passed=sum(r['passed'] for r in results),failed=sum(not r['passed'] for r in results),videos=results)
    (folder/'对照视频完整解码.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if report['failed']:raise RuntimeError('对照视频完整解码失败')
    return report


def plot_summary(folder,metrics):
    """逐案例原值，不将四个人工开发场景当独立统计样本估计总体收益。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei']
    fig,axes=plt.subplots(2,1,figsize=(12,9),constrained_layout=True)
    positions=list(range(len(metrics)));labels=[r['case_id']+'\n'+r['strategy'] for r in metrics]
    axes[0].bar(positions,[r['collision_sensor_events'] for r in metrics],color='#c65645',label='实际碰撞传感器事件数')
    axes[0].set_ylabel('事件数 / 单分支');axes[0].legend();axes[0].set_xticks(positions,labels,fontsize=8)
    for key,label,marker in [('first_message_usable_s','首包可用','o'),('first_local_seen_s','本地首次可见','x'),('first_warning_s','首次预警','^'),('first_brake_s','请求紧急制动','s')]:
        selected=[(i,r[key]) for i,r in enumerate(metrics) if r.get(key) is not None]
        axes[1].scatter([p[0] for p in selected],[p[1] for p in selected],label=label,marker=marker)
    axes[1].set_ylabel('仿真时间 / s');axes[1].set_xticks(positions,labels,fontsize=8);axes[1].legend(ncols=4)
    fig.suptitle('N14高画质开发案例原值：每类1案例×1种子；非统计收益，未发生事件不绘点')
    for extension in ('png','svg'):fig.savefig(folder/f'四场景碰撞与信息时序原值.{extension}',dpi=180)
    plt.close(fig)
    sources={r['id']:hashlib.sha256((Path(r['run_dir'])/'配对事件指标.csv').read_bytes()).hexdigest() for r in read(folder/'多场景验收报告.json')['cases'] if r.get('run_dir') and (Path(r['run_dir'])/'配对事件指标.csv').is_file()}
    (folder/'图表来源与哈希.json').write_text(json.dumps(dict(raw_csv='多场景原始指标.csv',raw_sha256=hashlib.sha256((folder/'多场景原始指标.csv').read_bytes()).hexdigest(),case_metric_sha256=sources),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def analyze_suite(folder=None,*,reuse_comparisons=False):
    if folder is None:
        candidates=list(PARENT.glob('n14_suite_*'))
        if not candidates:raise FileNotFoundError('没有本机高画质多场景运行')
        folder=max(candidates,key=lambda p:(p/'多场景运行清单.json').stat().st_mtime)
    folder=Path(folder).resolve();manifest=read(folder/'多场景运行清单.json')
    if manifest['probe_only'] or manifest['selected_case'] is not None or {c['id'] for c in manifest['cases']}!={'S01','S02','S03','S04'} or len(manifest['cases'])!=4:
        raise ValueError('静态/单场景探针不能提交为完整多场景验收')
    report={'analysis_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
            'analysis_source_sha256':{name:hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest() for name in ('analyze_n14_scenarios.py','analyze_occluded_junction.py','export_n14_camera_videos.py')},
            'started_at':datetime.now().astimezone().isoformat(),'cases':[],'scope':'四个开发场景/一个种子，受限轨迹输入；非N27完整专项或统计收益'}
    all_metrics=[]
    for case in manifest['cases']:
        result=dict(id=case['id'],name=case['name'],status='failed')
        try:
            if case['status']!='completed':raise RuntimeError('物理分支未完成：'+str(case))
            if hashlib.sha256(Path(case['config']).read_bytes()).hexdigest()!=case['config_sha256']:
                raise ValueError('采集前冻结的场景YAML被修改')
            run=Path(case['run_dir']);result['run_dir']=str(run)
            result['producer_version']=read(run/'源码版本.json')
            if case.get('reuse_verification'):
                from run_n14_scenarios import verify_file_hashes
                proof_path=Path(case['reuse_verification'])
                if hashlib.sha256(proof_path.read_bytes()).hexdigest()!=case['reuse_sha256']:raise ValueError('复用核验清单被修改')
                proof=read(proof_path);verify_file_hashes(run,proof['source_files'])
                old=proof['result'];result.update({k:v for k,v in old.items() if k not in ('id','name','status')})
                result.update(status='passed',reused=True,reuse_verification=str(proof_path),source_analysis_commit=proof['source_analysis_commit'])
                for row in read(run/'配对验收报告.json')['metrics']:all_metrics.append(dict(case_id=case['id'],case_name=case['name'],**row))
                report['cases'].append(result)
                (folder/'多场景验收报告.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
                continue
            extra=additional_checks(run)
            result.update(additional_checks=extra['passed'],additional_failed=extra['failed'])
            if reuse_comparisons:
                from run_n14_scenarios import verify_file_hashes
                previous=read(run/'对照视频完整解码.json')
                if previous['failed'] or len(previous['videos'])!=4:raise ValueError('原4个对照视频未完整解码通过')
                verify_file_hashes(run,{v['video']:v['sha256'] for v in previous['videos']})
                result['comparison_encoding_reused']=True
            try:base=analyze(run,encode=not reuse_comparisons)
            except RuntimeError:
                base=read(run/'配对验收报告.json')
                result.update(base_checks=base['passed'],base_failed=base['failed'])
                for row in base['metrics']:all_metrics.append(dict(case_id=case['id'],case_name=case['name'],**row))
                result['comparison_decode']=comparison_decode(run)
                result['independent_videos']=str(export(run,diagnostic=True))
                raise
            result.update(run_dir=str(run),base_checks=base['passed'],additional_checks=extra['passed'],additional_failed=extra['failed'])
            for row in base['metrics']:all_metrics.append(dict(case_id=case['id'],case_name=case['name'],**row))
            if extra['failed']:
                result['independent_videos']=str(export(run,diagnostic=True))
                raise RuntimeError('多目标/原生画质追加断言失败')
            result['comparison_decode']=comparison_decode(run)
            result['independent_videos']=str(export(run));result['status']='passed'
        except Exception as error:result['error']=repr(error)
        report['cases'].append(result)
        (folder/'多场景验收报告.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report.update(passed_cases=sum(x['status']=='passed' for x in report['cases']),failed_cases=sum(x['status']!='passed' for x in report['cases']),finished_at=datetime.now().astimezone().isoformat())
    (folder/'多场景验收报告.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if all_metrics:
        with (folder/'多场景原始指标.csv').open('w',encoding='utf-8-sig',newline='') as handle:
            fields=list(dict.fromkeys(k for row in all_metrics for k in row));writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(all_metrics)
        plot_summary(folder,all_metrics)
    index={p.relative_to(folder).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='高画质多场景证据索引.json'}
    (folder/'高画质多场景证据索引.json').write_text(json.dumps(dict(files=index,source_run_id=manifest['run_id'],
        external_cases=[dict(id=c['id'],source_run=c['run_dir'],verification=c['reuse_verification'],sha256=c['reuse_sha256']) for c in manifest['cases'] if c.get('reuse_verification')],
        scope=report['scope'],passed_cases=report['passed_cases'],failed_cases=report['failed_cases']),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if report['failed_cases']:raise RuntimeError('多场景验收存在失败，证据保留：'+str(folder))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--suite-dir',type=Path)
    parser.add_argument('--reuse-comparisons',action='store_true',help='仅重评估同一冻结原始数据；核对既有对照视频哈希并重新解码，避免重编码')
    args=parser.parse_args();analyze_suite(args.suite_dir,reuse_comparisons=args.reuse_comparisons)
