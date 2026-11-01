"""N14固定开发场景配置展开；只定义已授权的车辆/行人演示，不读运行结果选场景。"""
from copy import deepcopy
import json
import os
import subprocess


def epic_server():
    if os.name!='nt':raise RuntimeError('本次实际采集仅验证Windows，需另行核实服务配置')
    command="Get-NetTCPConnection -State Listen -LocalPort 2000 -ErrorAction Stop | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { Get-CimInstance Win32_Process -Filter ('ProcessId=' + $_) } | Select-Object ProcessId,ExecutablePath,CommandLine | ConvertTo-Json -Compress"
    process=subprocess.run(['pwsh','-NoProfile','-NonInteractive','-Command',command],capture_output=True,text=True,encoding='utf-8',timeout=15)
    if process.returncode or not process.stdout.strip():raise RuntimeError('未找到本机2000端口CARLA服务，先启动匹配Epic服务')
    proof=json.loads(process.stdout);items=proof if isinstance(proof,list) else [proof]
    if not all('carla' in x['ExecutablePath'].lower() and '-quality-level=epic' in x['CommandLine'].lower() for x in items):
        raise RuntimeError('当前2000端口进程未明确使用Epic，拒绝把低画质标为高画质')
    return proof


def expand_cases(base, suite):
    if suite['rgb_size'] != [1920,1080] or suite['visibility_size'] != [1280,720]:
        raise ValueError('本轮冻结展示1080p、可见性720p，不允许静默降级')
    if suite['render_quality']!='Epic' or suite['fps']!=20:raise ValueError('画质/仿真帧率与冻结规格不符')
    result=[];ids=set()
    for case in suite['cases']:
        if case['id'] in ids:raise ValueError('场景ID重复')
        ids.add(case['id']);config=deepcopy(base)
        config.update(rgb_size=suite['rgb_size'],visibility_size=suite['visibility_size'],render_quality='Epic',
                      suite_case=case,scenario_id=base['scenario_id']+'-'+case['id'])
        config['overview']=dict(config.get('overview',{}),enabled=True,live_window=True,output_scale=1.5)
        if case['kind']=='multiple_vehicles':
            pose=deepcopy(config['target_spawn']);pose['y']+=case['second_vehicle_headway_m']
            config['additional_targets']=[dict(role='target_2',track_id='crossing-2',category='vehicle',
                blueprint='vehicle.audi.a2',spawn=pose,speed_mps=config['speed_mps'],start_s=0.)]
        elif case['kind']=='pedestrian':
            cx,cy=config['conflict_xy'];config.pop('occluder_motion',None)
            config.update(target_category='pedestrian',target_blueprint='walker.pedestrian.0001',target_track_id='pedestrian-1',
                          speed_mps=case['ego_speed_mps'],target_speed_mps=case['pedestrian_speed_mps'],
                          target_start_s=case['pedestrian_start_s'],settle_static_occluder=True)
            config['target_spawn']=dict(x=cx+.14,y=cy+case['pedestrian_distance_m'],z=.5,yaw=-90.)
            dx,dy=case['occluder_offset_xy'];config['occluder']=dict(x=cx+dx,y=cy+dy,z=.5,yaw=case.get('occluder_yaw_deg',-90.))
            config['occluder_blueprint']=case.get('occluder_blueprint','vehicle.mercedes.sprinter')
            config['static_occluders']=[dict(role=f'static_{i}',blueprint='static.prop.clothcontainer',
                spawn=dict(x=cx+1.4,y=cy+offset,z=0.,yaw=0.)) for i,offset in enumerate((2.35,4.2,6.05))]
        elif case['kind']!='original':raise ValueError('未授权的场景类别')
        if not 4.5<=config['speed_mps']<=11.5:raise ValueError('开发初速须保留N13标定速度范围余量')
        result.append((case,config))
    return result
