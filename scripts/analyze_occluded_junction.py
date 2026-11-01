"""复核N14配对、因果与事件，保存原始差异、图表和双视角视频。"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from occluactisafe.evaluation.episode_metrics import digest

ARTIFACTS=Path('H:/OccluActiSafeDataset/artifacts')
STRATEGIES=('reference','local_aeb','cooperative_aeb')
NAMES={'reference':'参考行驶','local_aeb':'仅车端AEB','cooperative_aeb':'车路协同AEB'}


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def rows(path):return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
def speed(s):return math.hypot(s['vx'],s['vy'])


def collision_requirement(config,strategy,collisions,overlaps):
    """落实原C07/F01：真实基线事件与几何代理分列，旧车辆演示筛选不扩展到行人。"""
    pedestrian=config.get('target_category')=='pedestrian'
    if strategy=='reference':return bool(collisions) and (pedestrian or bool(overlaps))
    if strategy=='local_aeb' and config.get('require_local_collision') and not pedestrian:
        return bool(collisions) and bool(overlaps)
    return None


def verify_budget_overview(folder,name,config,truth,audit,observations,check):
    from occluactisafe.common.schemas import ObservationPacket,ReceivedMessage
    b=config['braking'];cache={};causal=True;rule=True;latched=False
    for t,o,a in zip(truth,observations,audit):
        now=t['time_s'];allowed={}
        def remember(track,until,table):table[(track.source_id,track.source_track_id,track.state_time)]=(track,until)
        if o['local'] is not None:
            for track in ObservationPacket.from_dict(o['local']).tracks:remember(track,now,allowed)
        if name=='cooperative_aeb':
            for raw in o['received']:
                record=ReceivedMessage.from_dict(raw)
                if record.usable_time>now or record.arrival_time>now:causal=False
                for target in record.payload.targets:remember(target.track,record.payload.valid_until,cache)
            for key,(track,until) in cache.items():
                if until>=now and now-track.state_time<=b['max_track_age_s']:allowed[key]=(track,until)
        d=a['decision'];v=speed(t['ego'])
        expected_budget=v*(b['response_delay_s']+b['control_period_s'])+v*v/(2*b['deceleration_mps2'])+b['margin_m']
        rule=rule and abs(d['stopping_budget_m']-expected_budget)<1e-8
        for known in d['known_targets']:
            key=(known['source_id'],known['track_id'],known['state_time'])
            if key not in allowed:causal=False;continue
            track,_=allowed[key];age=now-track.state_time
            causal=causal and known.get('category',track.category)==track.category
            causal=causal and age>=0 and abs(known['age_s']-age)<1e-8
            causal=causal and abs(known['x']-(track.x+track.vx*age))<1e-6 and abs(known['y']-(track.y+track.vy*age))<1e-6
            expected=[[track.x+track.vx*(age+i*.1),track.y+track.vy*(age+i*.1)] for i in range(31)]
            causal=causal and known['prediction_xy']==expected
        if name!='reference' and now>=config['prefix_s'] and d['distance_to_conflict_envelope_m'] is not None and d['distance_to_conflict_envelope_m']<=expected_budget:latched=True
        rule=rule and bool(d['brake'])==latched
    check(name+'合法已知目标及CV路径',causal,{'frames':len(audit),'max_cache_age_s':b['max_track_age_s']})
    check(name+'动态制动预算规则',rule,{'parameters':b,'frames':len(audit)})
    display=rows(folder/name/'示意在线状态.jsonl');visual=read(folder/name/'示意帧索引.json')
    same=len(display)==len(audit)==len(visual) and all(v['decision']==a['decision'] and v['ego']==o['ego'] and v['time_s']==o['time_s']
        and v['legal_observation_sha256']==hashlib.sha256(json.dumps(o,sort_keys=True).encode()).hexdigest()
        for v,a,o in zip(display,audit,observations))
    check(name+'在线示意来源一致',same,{'frames':len(display),'evaluation_truth_used':False})
    window=read(folder/name/'实时示意窗口复核.json')
    shown=window['displayed_frames']==len(audit) and not window['closed_by_user'] and bool(window['snapshots'])
    shown=shown and all(hashlib.sha256(Path(s['path']).read_bytes()).hexdigest()==s['sha256'] for s in window['snapshots'])
    check(name+'实时窗口逐帧显示',shown,window)
    frames=read(folder/name/'相机帧索引.json')
    identity=lambda source:[(r['index'],r['frame'],r['time_s']) for r in frames if r['source']==source]
    check(name+'新增实景角度完整',all(identity(source)==identity('车端') for source in config['extra_cameras']),
          {'camera_names':list(config['extra_cameras']),'frames_each':len(identity('车端'))})
    geometry_path=folder/name/'车辆静态几何.json'
    if geometry_path.is_file():
        geometry=read(geometry_path);initial=read(folder/name/'初始状态.json');objects=read(folder/name/'评测对象身份.json')
        pose_errors={}
        for actor in ('ego','target','occluder'):
            initial_actor=initial[actor];current=truth[0][actor];yaw=initial_actor['yaw']
            initial_speed=config['speed_mps']
            if actor=='occluder' and not config.get('occluder_motion'):initial_speed=0.
            if actor=='target' and config.get('target_category')=='pedestrian':
                initial_speed=config.get('target_speed_mps',1.5) if config.get('target_start_s',0.)==0 else 0.
            expected=(initial_actor['x']+initial_speed*.05*math.cos(yaw),initial_actor['y']+initial_speed*.05*math.sin(yaw))
            pose_errors[actor]=math.hypot(current['x']-expected[0],current['y']-expected[1])
        valid_geometry=(geometry['tick_calls']==0 and geometry['physics_context']==config['calibration_context']
            and all(geometry['boxes'][a]['actor_id']==objects[a+'_actor_id']
                and abs(truth[0][a]['length']-2*geometry['boxes'][a]['extent']['x'])<1e-7
                and abs(truth[0][a]['width']-2*geometry['boxes'][a]['extent']['y'])<1e-7 for a in ('ego','target','occluder'))
            and max(pose_errors.values())<=.1)
        check(name+'活体几何缓存与初态位置',valid_geometry,{'first_step_position_errors_m':pose_errors,'sampler_tick_calls':geometry['tick_calls']})
    source_files=read(folder/'源码版本.json')['source_sha256']
    if 'src/occluactisafe/simulation/scenario_rpc.py' in source_files:
        proof=read(folder/name/'批量RPC复核.json');batches=rows(folder/name/'批量RPC请求.jsonl')
        acknowledged=(proof['worker_exit_code']==0 and proof['worker_tick_calls']==0
            and proof['acknowledged_batches']==len(batches) and len(batches)>=len(truth)
            and [b['sequence'] for b in batches]==list(range(len(batches)))
            and all(b['response_count']==len(b['commands']) for b in batches))
        formal=batches[-len(truth):]
        for batch,record in zip(formal,truth):
            controls={c['actor_id']:c['value'] for c in batch['commands'] if c['kind']=='control'}
            acknowledged=acknowledged and controls.get(objects['ego_actor_id'])==record['control']
        check(name+'控制回执与唯一父时钟',acknowledged,proof)
    if name=='cooperative_aeb':
        first=next((x for x in audit if x['decision']['brake']),None)
        warnings=[x for x in audit if x['decision']['warning'] and not x['decision']['brake']]
        check('协同先预警再按预算紧急制动',bool(warnings) and first is not None and warnings[0]['time_s']<first['time_s']
              and all(x['decision']['stopping_margin_m']>0 for x in warnings if x['time_s']>=config['prefix_s']),
              {'first_warning_s':warnings[0]['time_s'] if warnings else None,'first_emergency_s':first['time_s'] if first else None})


def video(folder):
    import imageio_ffmpeg
    from PIL import Image,ImageDraw,ImageFont
    index=read(folder/'相机帧索引.json');pairs={}
    for row in index:pairs.setdefault(row['index'],{})[row['source']]=row
    config=read(folder/'运行配置.json');fps=round(1/(.05*config['image_interval_ticks']))
    camera_width,height=config.get('rgb_size',(1280,720));width=2*camera_width
    filename=folder/'双视角仿真视频.mp4'
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
             '-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(filename)]
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',round(24*height/720))
    with (folder/'视频编码日志.txt').open('w',encoding='utf-8') as log:
        proc=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
        try:
            for _,pair in sorted(pairs.items()):
                if not {'车端','路侧'}<=set(pair):raise ValueError('视频双视角帧缺失')
                canvas=Image.new('RGB',(width,height))
                for i,name in enumerate(('车端','路侧')):
                    with Image.open(folder/pair[name]['image']) as image:
                        if image.size!=(camera_width,height):raise ValueError('原生相机尺寸不符')
                        canvas.paste(image.convert('RGB'),(camera_width*i,0))
                draw=ImageDraw.Draw(canvas);draw.rectangle((0,0,width,round(45*height/720)),fill='#101827')
                draw.text((15,7),f"{NAMES[folder.name]}  |  车端  |  t={pair['车端']['time_s']:.2f}s",font=font,fill='white')
                draw.text((camera_width+15,7),'路侧  |  trajectory_input · 真实CARLA RGB展示',font=font,fill='white')
                proc.stdin.write(canvas.tobytes())
        finally:proc.stdin.close();proc.wait(timeout=60)
        if proc.returncode:raise RuntimeError('视频编码失败')
    return {'video':filename.name,'frames':len(pairs),'fps':fps,'width':width,'height':height,
            'source_index_sha256':hashlib.sha256((folder/'相机帧索引.json').read_bytes()).hexdigest(),
            'sha256':hashlib.sha256(filename.read_bytes()).hexdigest()}


def follow_comparison(folder,data,config):
    import imageio_ffmpeg
    from PIL import Image,ImageDraw,ImageFont
    names=('local_aeb','cooperative_aeb');source={}
    for name in names:
        source[name]={r['index']:r for r in read(folder/name/'相机帧索引.json') if r['source']=='跟车'}
        if not source[name]:raise ValueError('跟车画面缺失')
    indexes=sorted(set().union(*(s.keys() for s in source.values())));fps=round(1/(.05*config['image_interval_ticks']))
    camera_width,height=config.get('rgb_size',(1280,720));width=2*camera_width
    file=folder/'仅车端与协同跟车同步对照.mp4';font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',round(23*height/720))
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
        '-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(file)]
    with (folder/'跟车视频编码日志.txt').open('w',encoding='utf-8') as log:
        proc=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
        try:
            for index in indexes:
                canvas=Image.new('RGB',(width,height));draw=ImageDraw.Draw(canvas)
                for i,name in enumerate(names):
                    selected=max(k for k in source[name] if k<=index);record=source[name][selected]
                    with Image.open(folder/name/record['image']) as image:
                        if image.size!=(camera_width,height):raise ValueError('跟车原生相机尺寸不符')
                        canvas.paste(image.convert('RGB'),(camera_width*i,0))
                    truth=data[name][0];sample=truth[min(selected,len(truth)-1)]
                    frozen=index>max(source[name]);collision_frame=selected>=len(truth)
                    draw=ImageDraw.Draw(canvas);draw.rectangle((camera_width*i,0,camera_width*(i+1),round(86*height/720)),fill='#101827')
                    suffix=' 已碰撞，末帧停留' if frozen else (' 碰撞触发帧' if collision_frame else '')
                    draw.text((camera_width*i+15,5),f'{NAMES[name]}   t={index*.05:.2f}s{suffix}',font=font,fill='white')
                    control=sample.get('actual_control',sample['control']);a=sample.get('longitudinal_acceleration_mps2',0.)
                    draw.text((camera_width*i+15,round(43*height/720)),f"{'碰撞前末次采样' if frozen or collision_frame else '实际'}速度 {speed(sample['ego']):.2f} m/s   制动 {control['brake']:.0%}   加速度 {a:.2f} m/s²",font=font,fill='#c4e6ff')
                proc.stdin.write(canvas.tobytes())
        finally:proc.stdin.close();proc.wait(timeout=60)
        if proc.returncode:raise RuntimeError('跟车对照编码失败')
    return {'video':file.name,'frames':len(indexes),'fps':fps,'width':width,'height':height,
            'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'padding':'碰撞结束分支仅冻结末帧并明确标注，无伪造未来画面'}


def analyze(folder,encode=True):
    folder=Path(folder);config=read(folder/'冻结配置.json');checks=[];data={};metrics=[]
    def check(name,condition,detail):checks.append({'name':name,'passed':bool(condition),'detail':detail})
    initial=[];prefix=[]
    for name in STRATEGIES:
        d=folder/name;truth=rows(d/'评测真值.jsonl');audit=rows(d/'在线决策.jsonl');observations=rows(d/'合法观测.jsonl')
        data[name]=(truth,audit,observations);collisions=read(d/'碰撞传感器.json');pref=read(d/'公共前缀.json')
        initial.append(pref['initial_sha256']);prefix.append(pref['sha256'])
        check(name+'清理恢复',read(d/'清理报告.json')['status']=='completed',read(d/'清理报告.json'))
        check(name+'连续合法帧',len(truth)==len(audit)==len(observations) and all(b['frame']==a['frame']+1 for a,b in zip(truth,truth[1:])),{'frames':len(truth)})
        check(name+'前缀无干预',all(not x['decision']['brake'] for x in audit if x['time_s']<config['prefix_s']),{'prefix_s':config['prefix_s']})
        legal=True;local_gated=True;first_usable=None;usable_history={}
        for t,o,a in zip(truth,observations,audit):
            if config.get('suite_case'):
                from occluactisafe.common.schemas import ObservationPacket
                tracks=ObservationPacket.from_dict(o['local']).tracks if o['local'] else ()
                if any(not t['participant_visibility'].get(track.source_track_id,{}).get('local',False)
                       or t['participant_pixels']['车端'].get(track.source_track_id,0)==0 for track in tracks):local_gated=False
                if 'vehicle-A' in a['decision']['used_sources'] and not tracks:legal=False
            else:
                if not t['local_visible'] and o['local'] is not None:local_gated=False
                if t['local_visible'] and t['target_pixels']['车端']==0:local_gated=False
                if 'vehicle-A' in a['decision']['used_sources'] and not t['local_visible']:legal=False
            if name!='cooperative_aeb' and 'roadside-1' in a['decision']['used_sources']:legal=False
            for received in o['received']:
                if received['usable_time']>o['time_s'] or received['arrival_time']>o['time_s']:legal=False
                if first_usable is None:first_usable=received['usable_time']
                if received['usable_time']<=o['time_s'] and received['arrival_time']<=o['time_s']:
                    from occluactisafe.common.schemas import ReceivedMessage
                    record=ReceivedMessage.from_dict(received)
                    for target in record.payload.targets:
                        track=target.track
                        usable_history[(track.source_id,track.source_track_id,track.state_time)]=record.payload.valid_until
            if 'roadside-1' in a['decision']['used_sources']:
                if 'braking' not in config:
                    if not o['received']:legal=False
                else:
                    known=[k for k in a['decision']['known_targets'] if k['source_id']=='roadside-1']
                    if not known:legal=False
                    for k in known:
                        key=(k['source_id'],k['track_id'],k['state_time'])
                        if usable_history.get(key,-1)<o['time_s'] or not 0<=o['time_s']-k['state_time']<=config['braking']['max_track_age_s']:legal=False
        check(name+'不可见目标隔离',local_gated,{'legal_observations':len(observations)})
        check(name+'消息可用时刻',legal,{'first_usable_time_s':first_usable})
        if config.get('occluder_motion') or config.get('suite_case'):
            objects=read(d/'评测对象身份.json');environment=read(d/'环境碰撞传感器.json')
            no_truck_contact=not any(r['role']=='occluder' or r['other_id']==objects['occluder_actor_id'] for r in environment)
            no_truck_overlap=all(not r['occluder_target_overlap'] and not r['occluder_ego_overlap'] for r in truth)
            check(name+'大车未参与碰撞',no_truck_contact and no_truck_overlap,environment)
        if config.get('occluder_motion'):
            lanes=read(d/'同向车道核验.json')
            same_direction=all(abs((r['occluder']['yaw']-r['ego']['yaw']+math.pi)%(2*math.pi)-math.pi)<.03 for r in truth)
            check(name+'同向合法车道',same_direction and all(x['yaw_error_deg']<5 for x in lanes.values()),lanes)
            tyaw=-math.radians(config['occluder']['yaw']);spawn=config['occluder'];cx,cy=config['conflict_xy']
            body_edge=max(abs(-math.sin(tyaw)*(r['occluder']['x']-spawn['x'])+math.cos(tyaw)*(r['occluder']['y']+spawn['y']))+r['occluder']['width']/2 for r in truth)
            front_min=min(-((r['occluder']['x']-cx)*math.cos(tyaw)+(r['occluder']['y']+cy)*math.sin(tyaw))-r['occluder']['length']/2 for r in truth)
            check(name+'大车车道及停车线余量',body_edge<=lanes['occluder']['lane_width']/2 and front_min>=config.get('occluder_min_front_clearance_m',1.5),
                  {'body_edge_m':body_edge,'minimum_front_distance_m':front_min})
        first_brake=next((a['time_s'] for a in audit if a['decision']['brake']),None)
        local_first=next((t['time_s'] for t in truth if t['local_visible']),None)
        road_first=next((t['time_s'] for t in truth if t['roadside_visible']),None)
        geo=[t['time_s'] for t in truth if t['geometry_overlap']]
        yaw=-math.radians(config['ego_spawn']['yaw']);p=config['ego_spawn'];cx,cy=config['conflict_xy']
        projection=lambda s:(s['x']-cx)*math.cos(yaw)+(s['y']+cy)*math.sin(yaw)
        cross=lambda s:-math.sin(yaw)*(s['x']-p['x'])+math.cos(yaw)*(s['y']+p['y'])
        lane_half=min(r['lane_width'] for r in config['ego_route'])/2
        max_lateral=max(abs(cross(t['ego']))+t['ego']['width']/2 for t in truth)
        result={'strategy':name,'frames':len(truth),'collision_sensor_events':len(collisions),'geometric_overlap_frames':len(geo),
            'first_geometric_overlap_s':geo[0] if geo else None,'first_brake_s':first_brake,'first_local_seen_s':local_first,
            'first_roadside_seen_s':road_first,'first_message_usable_s':first_usable,'end_speed_mps':speed(truth[-1]['ego']),
            'end_distance_before_conflict_m':-projection(truth[-1]['ego']),'max_lateral_body_edge_m':max_lateral,
            'lane_half_width_m':lane_half,'raw_prefix_sha256':pref['sha256']}
        if first_brake is not None:
            triggered=next(t for t in truth if t['time_s']==first_brake);v=speed(triggered['ego']);B=config['calibration']['capabilities']['effective_deceleration_mps2']
            budget=v*v/(2*B)+v*(config['calibration']['capabilities']['observed_response_delay_s']+config.get('braking',{}).get('control_period_s',0.))+5.
            result.update(brake_start_speed_mps=v,available_distance_m=-projection(triggered['ego']),stopping_budget_with_margin_m=budget)
            if config.get('occluder_motion') or config.get('suite_case'):
                physical=[r for r in truth if r['time_s']>=first_brake and r['actual_control']['brake']>0]
                no_jump=all(math.hypot(b['ego']['x']-a['ego']['x'],b['ego']['y']-a['ego']['y'])<=max(speed(a['ego']),speed(b['ego']))*.05+.1 for a,b in zip(truth,truth[1:]))
                check(name+'真实连续制动响应',len(physical)>=2 and speed(truth[-1]['ego'])<v-.3 and no_jump,
                      {'actual_brake_samples':len(physical),'brake_start_speed_mps':v,'last_speed_mps':speed(truth[-1]['ego']),'no_position_jump':no_jump})
        if 'braking' in config and config.get('overview',{}).get('enabled'):
            result.update(first_warning_s=audit[-1]['decision']['first_warning_s'],
                first_target_available_s=audit[-1]['decision']['first_target_available_s'])
            if first_brake is not None:
                first_decision=next(x['decision'] for x in audit if x['decision']['brake'])
                result.update(emergency_envelope_distance_m=first_decision['distance_to_conflict_envelope_m'],
                              emergency_stopping_budget_m=first_decision['stopping_budget_m'])
            verify_budget_overview(folder,name,config,truth,audit,observations,check)
        metrics.append(result)
        required=collision_requirement(config,name,collisions,geo)
        if required is not None:check('参考实际危险事件' if name=='reference' else '仅车端实际碰撞案例',required,result)
        if name=='cooperative_aeb':
            check('协同停车避撞及道路边界',not collisions and not geo and speed(truth[-1]['ego'])<.1 and max_lateral<=lane_half and -projection(truth[-1]['ego'])>3.,result)
            check('实测标定停车空间',first_brake is not None and result['available_distance_m']>result['stopping_budget_with_margin_m'],result)
            check('路侧先见且到达后干预',road_first is not None and first_usable is not None and first_brake>=first_usable and (local_first is None or first_brake<local_first),result)
            check('协同触发时确有相机物理遮挡',first_brake is not None and triggered['target_pixels']['车端']==0 and triggered['target_pixels']['路侧']>0,
                  {'time_s':first_brake,'pixels':triggered['target_pixels'] if first_brake is not None else None})
            if config.get('occluder_motion') or config.get('suite_case'):
                check('起始至触发持续完全遮挡',all(r['target_pixels']['车端']==0 for r in truth if first_brake is not None and r['time_s']<=first_brake),{'first_local_seen_s':local_first,'first_brake_s':first_brake})
    check('初始哈希一致',len(set(initial))==1,initial)
    differences=[];tolerance=config['pairing_tolerance'];base=data['reference']
    for name in STRATEGIES[1:]:
        branch=data[name]
        for i,(a,b) in enumerate(zip(base[0],branch[0])):
            if a['time_s']>=config['prefix_s']:break
            row={'branch':name,'index':i,'time_s':a['time_s']}
            for actor in (('ego','target','occluder') if config.get('occluder_motion') else ('ego','target')):
                left,right=a[actor],b[actor]
                row[actor+'_position_m']=math.hypot(left['x']-right['x'],left['y']-right['y'])
                row[actor+'_velocity_mps']=math.hypot(left['vx']-right['vx'],left['vy']-right['vy'])
                row[actor+'_yaw_rad']=abs((left['yaw']-right['yaw']+math.pi)%(2*math.pi)-math.pi)
            row['control_difference']=max(abs(a['control'][k]-b['control'][k]) for k in ('throttle','brake','steer'))
            if config.get('occluder_motion'):row['control_difference']=max(row['control_difference'],*(abs(a['occluder_control'][k]-b['occluder_control'][k]) for k in ('throttle','brake','steer')))
            # 观测坐标允许已批准的物理误差；消息身份、可用时刻和队列长度必须相同。
            identity=lambda o:[(r['payload']['sequence'],r['arrival_time'],r['usable_time']) for r in o['received']]
            row['message_identity_equal']=identity(base[2][i])==identity(branch[2][i]) and a['pending_count']==b['pending_count']
            differences.append(row)
    expected=2*round(config['prefix_s']/.05)
    check('运动前缀容差配对',len(differences)==expected and all(
        all(r[a+'_position_m']<=tolerance['position_m'] and r[a+'_velocity_mps']<=tolerance['velocity_mps'] and r[a+'_yaw_rad']<=tolerance['yaw_rad'] for a in (('ego','target','occluder') if config.get('occluder_motion') else ('ego','target')))
        and r['control_difference']<=tolerance['control'] and r['message_identity_equal'] for r in differences),
        {'rows':len(differences),'tolerance':tolerance,'raw_hashes':prefix,'maxima':{k:max(r[k] for r in differences) for k in differences[0] if k.endswith(('_m','_mps','_rad','difference'))}})
    identities=[digest({k:v for k,v in read(folder/n/'运行配置.json').items() if k!='strategy'}) for n in STRATEGIES]
    check('冻结身份及扰动一致',len(set(identities))==1,identities)
    with (folder/'公共前缀逐帧差异.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(differences[0]));w.writeheader();w.writerows(differences)
    report={'passed':sum(c['passed'] for c in checks),'failed':sum(not c['passed'] for c in checks),'checks':checks,'metrics':metrics,
            'validation_contract':'原C07/F01要求参考分支真实发生声明冲突；原生碰撞事件与截至末次决策帧的OBB代理分列。车辆演示保留两者同时出现及仅车端碰撞附加检查；该演示筛选不扩展到新增行人，不预设车端必然失败。协同仍须传感器/几何均零、停车及边界安全，因果与配对要求不变。',
            'requirement_scope':{'category':config.get('target_category','vehicle'),
                'original_local_collision_demo_flag':bool(config.get('require_local_collision')),
                'local_collision_assertion_applicable':bool(config.get('require_local_collision')) and config.get('target_category')!='pedestrian'},
            'input_mode':'trajectory_input','scope':'一个开发场景，实际碰撞传感器与几何事件分列；非图像检测/完整视频控制闭环'}
    (folder/'配对验收报告.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with (folder/'配对事件指标.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=list(dict.fromkeys(k for m in metrics for k in m));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(metrics)
    plot(folder,data,config)
    if encode:
        videos={n:video(folder/n) for n in STRATEGIES}
        if config.get('occluder_motion') or config.get('suite_case'):videos['follow_comparison']=follow_comparison(folder,data,config)
        (folder/'视频来源与哈希.json').write_text(json.dumps(videos,ensure_ascii=False,indent=2),encoding='utf-8')
    files={str(p.relative_to(folder)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob('*') if p.is_file() and p.name!='证据索引.json'}
    (folder/'证据索引.json').write_text(json.dumps({'files':files,'raw_truth_for_evaluation_only':True},ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"通过{report['passed']}，失败{report['failed']}，证据：{folder}")
    if report['failed']:raise RuntimeError('N14配对必需检查未通过')
    return report


def plot(folder,data,config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from occluactisafe.risk.obb import OBB,obb_corners
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(13,6),constrained_layout=True)
    for name,(truth,audit,_) in data.items():
        axes[0].plot([r['ego']['x'] for r in truth],[r['ego']['y'] for r in truth],label=NAMES[name])
        axes[1].plot([r['time_s'] for r in truth],[speed(r['ego']) for r in truth],label=NAMES[name])
    reference=data['reference'][0];axes[0].plot([r['target']['x'] for r in reference],[r['target']['y'] for r in reference],'--',color='gray',label='参考分支横向车')
    o=read(folder/'reference'/'初始状态.json')['occluder'];box=OBB((o['x'],o['y']),o['yaw'],o['length'],o['width'])
    axes[0].add_patch(Polygon(obb_corners(box),color='#e89b39',alpha=.7,label='遮挡大车初始车体'))
    axes[0].set_aspect('equal');axes[0].set_xlabel('项目X (m)');axes[0].set_ylabel('项目Y (m)')
    axes[1].set_xlabel('场景时间 (s)');axes[1].set_ylabel('自车速度 (m/s)')
    for ax in axes:ax.grid(alpha=.25);ax.legend()
    for ext in ('png','svg'):fig.savefig(folder/f'三策略轨迹与速度对照.{ext}',dpi=180)
    plt.close(fig)
    if config.get('occluder_motion') or config.get('suite_case'):
        fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True,constrained_layout=True)
        raw=[]
        for name,(truth,audit,_) in data.items():
            ts=[r['time_s'] for r in truth]
            axes[0].plot(ts,[speed(r['ego']) for r in truth],label=NAMES[name])
            axes[1].plot(ts,[r['longitudinal_acceleration_mps2'] for r in truth],label=NAMES[name])
            axes[2].step(ts,[r['actual_control']['brake'] for r in truth],where='post',label=NAMES[name])
            for r in truth:raw.append({'strategy':name,'time_s':r['time_s'],'speed_mps':speed(r['ego']),
                'acceleration_mps2':r['longitudinal_acceleration_mps2'],'requested_brake':r['control']['brake'],
                'actual_brake':r['actual_control']['brake'],'target_pixels_local':r['target_pixels']['车端']})
        for ax,title in zip(axes,('实际速度 (m/s)','实际纵向加速度 (m/s²)','实际制动输入')):
            ax.set_ylabel(title);ax.grid(alpha=.25);ax.legend()
        axes[-1].set_xlabel('场景时间 (s)；原始值，无平滑')
        for ext in ('png','svg'):fig.savefig(folder/f'制动时序与实际响应.{ext}',dpi=180)
        plt.close(fig)
        with (folder/'制动响应原始.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(raw[0]));writer.writeheader();writer.writerows(raw)
    if 'braking' in config:
        fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True,constrained_layout=True);raw=[]
        for name,(_,audit,_) in data.items():
            if name=='reference':continue
            ts=[r['time_s'] for r in audit];ds=[r['decision'] for r in audit]
            axes[0].plot(ts,[d['distance_to_conflict_envelope_m'] for d in ds],label=NAMES[name]+' 预测包络剩余距离')
            axes[0].plot(ts,[d['stopping_budget_m'] for d in ds],'--',label=NAMES[name]+' 停车预算')
            phase=lambda d:3 if d['mode']=='停车保持' else 2 if d['brake'] else 1 if d['warning'] else 0
            axes[1].step(ts,[phase(d) for d in ds],where='post',label=NAMES[name])
            for r in audit:raw.append({'strategy':name,'time_s':r['time_s'],'mode':r['decision']['mode'],
                'warning':r['decision']['warning'],'emergency':r['decision']['brake'],
                'envelope_distance_m':r['decision']['distance_to_conflict_envelope_m'],
                'stopping_budget_m':r['decision']['stopping_budget_m'],'margin_m':r['decision']['stopping_margin_m']})
        axes[0].set_ylabel('距离 (m)');axes[1].set_ylabel('决策阶段');axes[1].set_yticks([0,1,2,3],['正常','预警','紧急制动','停车保持'])
        axes[1].set_xlabel('仿真时间 (s)；仅当时合法输入，原值不平滑')
        for ax in axes:ax.grid(alpha=.25);ax.legend()
        for ext in ('png','svg'):fig.savefig(folder/f'预警与制动距离预算.{ext}',dpi=180)
        plt.close(fig)
        with (folder/'预警与制动预算原始.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(raw[0]));writer.writeheader();writer.writerows(raw)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path);parser.add_argument('--no-video',action='store_true');args=parser.parse_args()
    candidates=[p for p in ARTIFACTS.glob('n14_*') if (p/'cooperative_aeb'/'公共前缀.json').is_file()]
    if args.run_dir is None and not candidates:raise FileNotFoundError('没有完整三分支N14运行')
    analyze(args.run_dir or max(candidates,key=lambda p:p.stat().st_mtime),encode=not args.no_video)
