"""N14 CARLA桥接：拓扑发现和受限轨迹场景。CARLA仅在桥接进程导入。"""
import json
import math
import os
from pathlib import Path
import sys
import time
import subprocess
from dataclasses import replace
from contextlib import ExitStack
from occluactisafe.simulation.runner import SynchronousRunner
from occluactisafe.simulation.sensors import FrameBuffer
from occluactisafe.risk.obb import OBB,segment_obb_interval
from occluactisafe.evaluation.episode_metrics import overlap,digest


def point(w):
    t=w.transform;return {'x':t.location.x,'y':t.location.y,'z':t.location.z,'yaw':t.rotation.yaw,
                         'road_id':w.road_id,'lane_id':w.lane_id,'lane_width':w.lane_width}


def intersection(a,b,c,d):
    ux,uy=b['x']-a['x'],b['y']-a['y'];vx,vy=d['x']-c['x'],d['y']-c['y']
    cross=ux*vy-uy*vx
    if abs(cross)<1e-6:return None
    dx,dy=c['x']-a['x'],c['y']-a['y'];t=(dx*vy-dy*vx)/cross;s=(dx*uy-dy*ux)/cross
    if 0<t<1 and 0<s<1:return [a['x']+t*ux,a['y']+t*uy]


def discover(client, output):
    import carla
    sys.path.insert(0,os.environ['OAS_CARLA_AGENTS'])
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    world=client.get_world();m=world.get_map();planner=GlobalRoutePlanner(m,2.)
    junctions={w.get_junction().id:w.get_junction() for pair in m.get_topology() for w in pair if w.is_junction}
    found=[]
    for identity,junction in sorted(junctions.items()):
        lines=[]
        for entry,exit in sorted(junction.get_waypoints(carla.LaneType.Driving),key=lambda p:(p[0].road_id,p[0].lane_id,p[1].road_id,p[1].lane_id)):
            delta=(exit.transform.rotation.yaw-entry.transform.rotation.yaw+180)%360-180
            if abs(delta)>5:continue
            before=entry.previous(70.);after=exit.next(30.)
            if not before or not after:continue
            a=min(before,key=lambda w:abs((w.transform.rotation.yaw-entry.transform.rotation.yaw+180)%360-180))
            b=min(after,key=lambda w:abs((w.transform.rotation.yaw-exit.transform.rotation.yaw+180)%360-180))
            route=planner.trace_route(a.transform.location,b.transform.location)
            if not route or any(abs((w.transform.rotation.yaw-entry.transform.rotation.yaw+180)%360-180)>5 for w,_ in route):continue
            lines.append({'entry':point(entry),'exit':point(exit),'route':[point(w) for w,_ in route]})
        for i,left in enumerate(lines):
            for right in lines[i+1:]:
                delta=abs((left['entry']['yaw']-right['entry']['yaw']+180)%360-180)
                if not 60<=delta<=120:continue
                conflict=intersection(left['entry'],left['exit'],right['entry'],right['exit'])
                if conflict:found.append({'junction_id':identity,'conflict_xy':conflict,'ego':left,'target':right})
    result={'map':m.name,'opendrive_sha256':__import__('hashlib').sha256(m.to_opendrive().encode()).hexdigest(),
            'carla_version':client.get_server_version(),'candidates':found}
    (output/'地图发现候选.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if not found:raise RuntimeError('没有上下游70/30m、近似直线且60–120度相交的路线对')
    print('candidates',len(found),flush=True)


def state(actor,bounding_box=None):
    t=actor.get_transform();v=actor.get_velocity();bb=bounding_box if bounding_box is not None else actor.bounding_box
    # CARLA Transform.transform原位修改Location，必须复制，不能污染缓存的局部偏移。
    offset=bb.location;center=t.transform(type(offset)(x=offset.x,y=offset.y,z=offset.z))
    return {'x':center.x,'y':-center.y,'vx':v.x,'vy':-v.y,'yaw':-math.radians(t.rotation.yaw),
            'length':2*bb.extent.x,'width':2*bb.extent.y}


def restricted_track(actor,source,now,bounding_box=None,*,track_id='crossing-1',category='vehicle'):
    from occluactisafe.common.schemas import SourceTrack
    s=state(actor,bounding_box)
    return SourceTrack.from_dict(dict(source_track_id=track_id,source_id=source,session_id='session-1',sensor_id=source,
        category=category,role='target',clock_domain='carla-n14',coordinate_frame_id='map-right-handed',reference_point='geometric_center',
        observation_time=now,last_seen_time=now,source_available_time=now,state_time=now,**s,
        size_source='annotation',size_uncertainty=0.,covariance_xy=[[0.,0.],[0.,0.]],observed_mask=True,
        state_valid_mask=True,size_valid=True,velocity_valid=True,yaw_valid=True,quality=1.))


def camera_visible(carla,camera,target,obstacle,range_m):
    """夹具几何适配层；仅当前帧。输出可见性，隐藏坐标不出桥接。"""
    import numpy as np
    transform=camera.get_transform();origin=transform.location;bb=target.bounding_box;tt=target.get_transform()
    center=tt.transform(bb.location);inv=np.array(transform.get_inverse_matrix())
    q=inv@np.array([center.x,center.y,center.z,1.])
    # CARLA相机x前、y右、z上；1280x720，水平FOV90度。
    if q[0]<=0 or abs(q[1])>q[0] or abs(q[2])>q[0]*720/1280 or origin.distance(center)>range_m:return False
    ot=obstacle.get_transform();ob=obstacle.bounding_box;oc=ot.transform(ob.location)
    box=OBB((oc.x,-oc.y),-math.radians(ot.rotation.yaw),2*ob.extent.x,2*ob.extent.y)
    targets=[center]+[tt.transform(carla.Location(x=bb.location.x+sx*bb.extent.x,y=bb.location.y+sy*bb.extent.y,z=bb.location.z)) for sx in (-1,1) for sy in (-1,1)]
    for p in targets:
        interval=segment_obb_interval((origin.x,-origin.y),(p.x,-p.y),box)
        if interval is None:return True
        lo,hi=interval;z0=origin.z+(p.z-origin.z)*lo;z1=origin.z+(p.z-origin.z)*hi
        if min(z0,z1)>oc.z+ob.extent.z or max(z0,z1)<oc.z-ob.extent.z:return True
    return False


def measure_geometry(client,output,config):
    """独立桥接子进程只读查询，绝不tick；由父进程既有runner推进预热帧。"""
    world=client.get_world();boxes={}
    for role,identity in config['actor_ids'].items():
        actor=world.get_actor(identity);box=actor.bounding_box
        if min(box.extent.x,box.extent.y,box.extent.z)<=0:raise RuntimeError('活体包围盒尺寸无效')
        boxes[role]={'actor_id':identity,'type_id':actor.type_id,
            'location':dict(x=box.location.x,y=box.location.y,z=box.location.z),
            'extent':dict(x=box.extent.x,y=box.extent.y,z=box.extent.z)}
    physics=world.get_actor(config['actor_ids']['ego']).get_physics_control()
    context={'map':world.get_map().name,'weather':str(world.get_weather()),'mass_kg':physics.mass,
        'drag_coefficient':physics.drag_coefficient,'wheels':[{'tire_friction':w.tire_friction,'max_steer_angle_deg':w.max_steer_angle,'radius_cm':w.radius} for w in physics.wheels]}
    (output/'车辆静态几何.json').write_text(json.dumps({'boxes':boxes,'physics_context':context,
        'tick_calls':0,'scope':'本轮活体车辆的一次只读几何/物理采样，正式循环缓存静态尺寸'},ensure_ascii=False,indent=2),encoding='utf-8')


def run_episode(client,output,config):
    import carla
    from occluactisafe.common.schemas import ObservationPacket
    from occluactisafe.common.clock import Timestamp
    from occluactisafe.core.roadside_edge import make_prediction
    from occluactisafe.communication.channel import SimChannel
    from occluactisafe.communication.protocol import ChannelConfig
    from occluactisafe.control.actuator import CalibratedActuator
    from occluactisafe.simulation.occluder_driver import OccluderDriver
    from occluactisafe.simulation.n14_participants import participant_specs,walker_action,batch_control
    from occluactisafe.simulation.n14_images import save_rgb_frame
    specs=participant_specs(config);spec_by_role={s['role']:s for s in specs};extra_actions={}
    rgb_width,rgb_height=config.get('rgb_size',(1280,720))
    mask_width,mask_height=config.get('visibility_size',(1280,720))
    # 每分支完整重载，世界帧由一个runner推进。无背景交通，无TM。
    client.set_timeout(20)
    moving=bool(config.get('occluder_motion')) and not config.get('probe_static',False)
    extra_cameras=config.get('extra_cameras',{})
    run_id=output.parent.name+'-'+output.name;frames={n:FrameBuffer(capacity=4) for n in ('车端','路侧','跟车',*extra_cameras)};images=[]
    masks={'车端':FrameBuffer(capacity=4),'路侧':FrameBuffer(capacity=4)}
    channel=SimChannel(ChannelConfig('I2V','carla-n14',propagation_delay_s=config['delay_s'],processing_delay_s=config['processing_s'],seed=config['seed']))
    collisions=[];environment_collisions=[];prefix=[];pending=[];sequence=0;observations=[];truth=[]
    def destroy(actor):
        response=client.apply_batch_sync([carla.command.DestroyActor(actor.id)],False)[0]
        if response.error:raise RuntimeError(response.error)
        return True
    def reconnect(timeout):client.set_timeout(timeout);return client
    with ExitStack() as resources, SynchronousRunner.reload_synchronously(client,run_id=run_id,episode_id=config['scenario_id'],generation=1,clock_domain='carla-n14',
            destroy_owned=destroy,cleanup_path=output/'清理报告.json',reconnect=reconnect,
            destroy_command=carla.command.DestroyActor,prepare_cleanup=client.set_timeout) as runner:
        world=runner._world
        if world.get_map().name!=config['map']:raise ValueError('地图不匹配')
        if __import__('hashlib').sha256(world.get_map().to_opendrive().encode()).hexdigest()!=config['opendrive_sha256']:raise ValueError('地图哈希不匹配')
        world.set_pedestrians_seed(config['seed'])
        def transform(p):return carla.Transform(carla.Location(x=p['x'],y=p['y'],z=p['z']),carla.Rotation(yaw=p['yaw'],pitch=p.get('pitch',0)))
        def spawn(blueprint,pose):
            bp=world.get_blueprint_library().find(blueprint);bp.set_attribute('role_name','oas:'+run_id)
            if bp.has_attribute('color'):bp.set_attribute('color','168,0,27')
            if bp.has_attribute('is_invincible'):bp.set_attribute('is_invincible','false')
            actor=world.spawn_actor(bp,transform(pose));runner.register_owned(actor);return actor
        ego=spawn('vehicle.audi.a2',config['ego_spawn'])
        participants={s['role']:spawn(s['blueprint'],s['spawn']) for s in specs};target=participants['target']
        settle_probe=bool(config.get('probe_static') and config.get('probe_settle_occluder')) or config.get('settle_static_occluder',False)
        obstacle=spawn(config['occluder_blueprint'],config['occluder']);obstacle.set_simulate_physics(moving or settle_probe)
        static_occluders={s['role']:spawn(s['blueprint'],s['spawn']) for s in config.get('static_occluders',[])}
        for actor in static_occluders.values():actor.set_simulate_physics(False)
        truck_driver=OccluderDriver(config['occluder'],config['conflict_xy'],**config['occluder_motion']) if moving else None
        sensor_bp=world.get_blueprint_library().find('sensor.other.collision')
        sensor_bp.set_attribute('role_name','oas:'+run_id)
        sensor=world.spawn_actor(sensor_bp,carla.Transform(),attach_to=ego);runner.register_owned(sensor)
        sensor.listen(lambda event:collisions.append({'frame':event.frame,'other_id':event.other_actor.id,'other_type':event.other_actor.type_id,
            'impulse':math.sqrt(event.normal_impulse.x**2+event.normal_impulse.y**2+event.normal_impulse.z**2)}))
        if moving or config.get('suite_case'):
            for role,actor in [('occluder',obstacle),*participants.items()]:
                if not actor.type_id.startswith('vehicle.'):continue
                bp=world.get_blueprint_library().find('sensor.other.collision');bp.set_attribute('role_name','oas:'+run_id)
                extra=world.spawn_actor(bp,carla.Transform(),attach_to=actor);runner.register_owned(extra)
                extra.listen(lambda event,r=role:environment_collisions.append({'role':r,'frame':event.frame,'other_id':event.other_actor.id,'other_type':event.other_actor.type_id}))
        cameras={}
        def spawn_cameras():
            camera_specs=[('车端',{'x':1.5,'y':0.,'z':1.6,'yaw':config.get('ego_camera_yaw_deg',0.)},True),
                          ('路侧',config['roadside_camera'],False),('跟车',{'x':-7.,'y':0.,'z':3.5,'yaw':0.,'pitch':-14.},True)]
            camera_specs.extend((name,spec['pose'],spec['attach_to_ego']) for name,spec in extra_cameras.items())
            for name,pose,attached in camera_specs:
                bp=world.get_blueprint_library().find('sensor.camera.rgb')
                bp.set_attribute('role_name','oas:'+run_id)
                for key,value in {'image_size_x':str(rgb_width),'image_size_y':str(rgb_height),'fov':'90','sensor_tick':'0.05'}.items():bp.set_attribute(key,value)
                camera=world.spawn_actor(bp,transform(pose),attach_to=ego if attached else None);runner.register_owned(camera)
                camera.listen(lambda image,n=name:frames[n].put(image.frame,image));cameras[name]=camera
                if name not in masks:continue
                label_bp=world.get_blueprint_library().find('sensor.camera.instance_segmentation')
                for key,value in {'role_name':'oas:'+run_id,'image_size_x':str(mask_width),'image_size_y':str(mask_height),'fov':'90','sensor_tick':'0.05'}.items():label_bp.set_attribute(key,value)
                labels=world.spawn_actor(label_bp,transform(pose),attach_to=ego if name=='车端' else None);runner.register_owned(labels)
                labels.listen(lambda image,n=name:masks[n].put(image.frame,image))
        rpc=None;rpc_ready=False
        def advance(action,target_action,occluder_action=None):
            now=runner.sim_time
            control={'request_id':str(runner.frame+1),'plan_id':'n14','solver_status':'success','safety_rechecked':True,
                'clock_domain':'carla-n14','plan_start_time':now,'valid_until':now+1,**action,'fallback_reason':None}
            request={'schema_version':'1.0','run_id':run_id,'episode_id':config['scenario_id'],'generation':1,
                'target_frame_id':runner.frame+1,'clock_domain':'carla-n14','request_time':now,'valid_until':now+1,'control':control}
            def apply(c):
                payload=[{'kind':'control','actor_id':ego.id,'value':{'throttle':c.throttle,'brake':c.brake,'steer':c.steer}}]
                for role,actor in participants.items():
                    value=target_action if role=='target' else extra_actions.get(role,{'throttle':0.,'brake':1.,'steer':0.})
                    payload.append(batch_control(role,actor.id,spec_by_role[role],value))
                if moving or settle_probe:payload.append({'kind':'control','actor_id':obstacle.id,'value':occluder_action or {'throttle':0.,'brake':1.,'steer':0.}})
                if not rpc_ready:
                    # 此时尚未注册图像回调；固定100帧确认保持控制并预热RPC子进程。
                    from occluactisafe.simulation.scenario_rpc import execute_batch
                    execute_batch(client,carla,payload,{ego.id,obstacle.id,*(a.id for a in participants.values())})
                else:
                    rpc.apply(payload)
            runner.advance(json.dumps(request),apply_control=apply)
        held={'throttle':0.,'brake':1.,'steer':0.}
        # 0.9.16 bounding_box会发同步RPC；独立进程采样避免与父进程tick的GIL/等待互锁。
        geometry_request=output/'几何采样请求.json'
        geometry_request.write_text(json.dumps({'world_id':world.id,'actor_ids':{'ego':ego.id,'occluder':obstacle.id,**{r:a.id for r,a in participants.items()},**{r:a.id for r,a in static_occluders.items()}}}),encoding='utf-8')
        from occluactisafe.simulation.scenario_rpc import ScenarioRPC
        rpc=ScenarioRPC(output,geometry_request);resources.callback(rpc.close)
        for _ in range(100):advance(held,held)
        rpc.ready(world.id);rpc_ready=True
        measured=json.loads((output/'车辆静态几何.json').read_text(encoding='utf-8'))
        boxes={}
        for box in measured['boxes'].values():
            boxes[box['actor_id']]=carla.BoundingBox(carla.Location(**box['location']),carla.Vector3D(**box['extent']))
        current=lambda actor:state(actor,boxes[actor.id])
        spawn_cameras()
        for _ in range(30):advance(held,held)
        if settle_probe:obstacle.set_simulate_physics(False)
        settled={'ego':ego.get_transform(),'occluder':obstacle.get_transform(),**{r:a.get_transform() for r,a in participants.items()}}
        if moving:
            # 三车均位于同向合法驾驶车道；只在初始化阶段预热大车动力。
            lane_metadata={}
            for name,actor in [('ego',ego),('occluder',obstacle)]:
                waypoint=world.get_map().get_waypoint(actor.get_location());delta=abs((actor.get_transform().rotation.yaw-waypoint.transform.rotation.yaw+180)%360-180)
                if str(waypoint.lane_type)!='Driving' or delta>5:raise ValueError('车辆不在同向Driving车道')
                lane_metadata[name]={'road_id':waypoint.road_id,'lane_id':waypoint.lane_id,'yaw_error_deg':delta,'lane_width':waypoint.lane_width}
            (output/'同向车道核验.json').write_text(json.dumps(lane_metadata,ensure_ascii=False,indent=2),encoding='utf-8')
        # 离线核实遮挡物不占用两条测试通道。
        oc=current(obstacle);route_clear=[]
        from occluactisafe.risk.obb import intersects
        for name in ('ego','target'):
            dimensions=(current(target)['length'],current(target)['width']) if name=='target' and specs[0]['category']=='pedestrian' else (4.,2.)
            for p in config[name+'_route']:
                if intersects(OBB((p['x'],-p['y']),-math.radians(p['yaw']),*dimensions),OBB((oc['x'],oc['y']),oc['yaw'],oc['length'],oc['width'])):
                    raise ValueError('遮挡物侵入测试路线')
                for prop in static_occluders.values():
                    state_=current(prop)
                    if intersects(OBB((p['x'],-p['y']),-math.radians(p['yaw']),*dimensions),OBB((state_['x'],state_['y']),state_['yaw'],state_['length'],state_['width'])):
                        raise ValueError('附加实体遮挡物侵入测试路线')
            route_clear.append(name)
        actual_context=measured['physics_context']
        if actual_context!=config['calibration_context']:raise ValueError('地图天气或物理上下文不符合N13标定')
        metadata={'ego':current(ego),'target':current(target),'occluder':current(obstacle),'mass_kg':actual_context['mass_kg'],
                  'tire_friction':[w['tire_friction'] for w in actual_context['wheels']],'weather':actual_context['weather'],'route_clear':route_clear}
        if config.get('suite_case'):
            metadata['participants']={s['track_id']:{'category':s['category'],'state':current(participants[s['role']])} for s in specs}
            metadata['static_occluders']={r:current(a) for r,a in static_occluders.items()}
        (output/'初始状态.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
        if moving:
            for _ in range(300):
                if math.hypot(obstacle.get_velocity().x,obstacle.get_velocity().y)>=config['speed_mps']:break
                advance(held,held,{'throttle':1.,'brake':0.,'steer':0.})
            else:raise RuntimeError('大车动力预热未完成')
        # N13表来自真实加速后的动力状态。只赋速度会保留低转速/挡位而失配。
        for warmup in range(200):
            if config.get('probe_static',False):break
            vehicles=[ego,*[a for role,a in participants.items() if spec_by_role[role]['category']=='vehicle']]
            if min(math.hypot(a.get_velocity().x,a.get_velocity().y) for a in vehicles)>=config['speed_mps']:break
            extra_actions={r:{'throttle':1.,'brake':0.,'steer':0.} for r in participants if r!='target'}
            full={'throttle':1.,'brake':0.,'steer':0.};advance(full,full,truck_driver.step(current(obstacle),initializing=True) if moving else None)
        else:raise RuntimeError('动力预热未达到初速')
        initial_commands=[]
        for name,actor in [('ego',ego),*participants.items()]:
            if not config.get('probe_static',False):
                pose=settled[name];initial_commands.append({'kind':'transform','actor_id':actor.id,'value':{'location':dict(x=pose.location.x,y=pose.location.y,z=pose.location.z),'rotation':dict(pitch=pose.rotation.pitch,yaw=pose.rotation.yaw,roll=pose.rotation.roll)}})
            spec=spec_by_role.get(name);pose=config['ego_spawn'] if name=='ego' else spec['spawn']
            if spec is None or spec['category']=='vehicle':
                yaw=math.radians(pose['yaw']);initial_speed=0. if config.get('probe_static',False) else config['speed_mps']
                initial_commands.append({'kind':'velocity','actor_id':actor.id,'value':dict(x=initial_speed*math.cos(yaw),y=initial_speed*math.sin(yaw),z=0.)})
                initial_commands.append({'kind':'angular_velocity','actor_id':actor.id,'value':dict(x=0.,y=0.,z=0.)})
        if moving:
            pose=settled['occluder'];initial_commands.append({'kind':'transform','actor_id':obstacle.id,'value':{'location':dict(x=pose.location.x,y=pose.location.y,z=pose.location.z),'rotation':dict(pitch=pose.rotation.pitch,yaw=pose.rotation.yaw,roll=pose.rotation.roll)}});yaw=math.radians(config['occluder']['yaw'])
            initial_commands.append({'kind':'velocity','actor_id':obstacle.id,'value':dict(x=config['speed_mps']*math.cos(yaw),y=config['speed_mps']*math.sin(yaw),z=0.)})
            initial_commands.append({'kind':'angular_velocity','actor_id':obstacle.id,'value':dict(x=0.,y=0.,z=0.)})
            truck_driver=OccluderDriver(config['occluder'],config['conflict_xy'],**config['occluder_motion'])
        rpc.apply(initial_commands)
        normal=held if config.get('probe_static',False) else {'throttle':.53,'brake':0.,'steer':0.}
        extra_actions={r:normal for r in participants if r!='target'}
        primary_initial=walker_action(specs[0],0.,held=config.get('probe_static',False)) if specs[0]['category']=='pedestrian' else normal
        advance(normal,primary_initial,truck_driver.step(current(obstacle),initializing=True) if moving else None)
        if any(a.id>=65536 for a in participants.values()):raise RuntimeError('实例相机16位ID范围不足，拒绝混淆实例')
        (output/'评测对象身份.json').write_text(json.dumps({'ego_actor_id':ego.id,'target_actor_id':target.id,
            'occluder_actor_id':obstacle.id,'origin_frame':runner.frame,'origin_sim_time_s':runner.sim_time,
            'instance_decoder':'官方bounding_boxes.py: G + (B << 8)',
            'participants':{s['track_id']:{'actor_id':participants[s['role']].id,'category':s['category'],'role':s['role']} for s in specs},
            'visibility_size':[mask_width,mask_height],
            'camera_calibration':{name:{'width':rgb_width,'height':rgb_height,'horizontal_fov_deg':90,
                'K':[[rgb_width/2,0.,rgb_width/2],[0.,rgb_width/2,rgb_height/2],[0.,0.,1.]],'camera_to_carla_world':camera.get_transform().get_matrix()}
                for name,camera in cameras.items()}},ensure_ascii=False,indent=2),encoding='utf-8')
        target_actuator=CalibratedActuator(config['calibration'],kp=.05,ki=.02,steering_rate_per_s=1.)
        extra_actuators={r:CalibratedActuator(config['calibration'],kp=.05,ki=.02,steering_rate_per_s=1.) for r in participants if r!='target'}
        def lane_angle(s,spawn):
            yaw=-math.radians(spawn['yaw']);dx=s['x']-spawn['x'];dy=s['y']+spawn['y']
            cross=-math.sin(yaw)*dx+math.cos(yaw)*dy;error=(yaw-s['yaw']+math.pi)%(2*math.pi)-math.pi
            return max(-.1,min(.1,error-.12*cross))
        for index in range(config['ticks']):
            if collisions:
                # 保留触发传感器后的真实当前帧；不再向在线策略发送隐藏/碰撞真值。
                for name,buffer in frames.items():
                    frame=buffer.get(runner.frame,3.)
                    if not frame['valid']:raise RuntimeError('碰撞关键帧缺失')
                    path=output/f'{name}碰撞画面_帧{index:04d}.png';proof=save_rgb_frame(frame['value'],path)
                    images.append({'image':path.name,'source':name,'frame':runner.frame,'time_s':index*.05,'index':index,'event':'collision_sensor',**proof})
                break
            now=index*.05;es=current(ego);ts=current(target)
            import numpy as np
            pixels={};mask_hashes={};all_pixels={}
            for name,buffer in masks.items():
                captured=buffer.get(runner.frame,3.)
                if not captured['valid']:raise RuntimeError('当前帧实例可见性核验缺失')
                image=captured['value'];raw=np.frombuffer(image.raw_data,dtype=np.uint8).reshape(image.height,image.width,4)
                identities=raw[:,:,1].astype(np.uint16)+(raw[:,:,0].astype(np.uint16)<<8)
                all_pixels[name]={s['track_id']:int(np.count_nonzero(identities==participants[s['role']].id)) for s in specs}
                pixels[name]=all_pixels[name][specs[0]['track_id']]
                mask_hashes[name]=__import__('hashlib').sha256(image.raw_data).hexdigest()
                if config.get('probe_static'):
                    label_path=output/f'{name}实例标签_帧{index:04d}.png';save_rgb_frame(image,label_path)
            # 部分车体在画面边缘出现也算可见，不能用中心点FOV误判为完全遮挡。
            visible={};local_tracks=[];road_tracks=[]
            for spec in specs:
                actor=participants[spec['role']];offset=boxes[actor.id].location
                center=actor.get_transform().transform(carla.Location(x=offset.x,y=offset.y,z=offset.z))
                local_ok=all_pixels['车端'][spec['track_id']]>0 and cameras['车端'].get_location().distance(center)<=config['local_range_m']
                road_ok=all_pixels['路侧'][spec['track_id']]>0 and cameras['路侧'].get_location().distance(center)<=config['roadside_range_m']
                visible[spec['track_id']]={'local':local_ok,'roadside':road_ok}
                for ok,source,destination in ((local_ok,'vehicle-A',local_tracks),(road_ok,'roadside-1',road_tracks)):
                    if ok:destination.append(restricted_track(actor,source,now,boxes[actor.id],track_id=spec['track_id'],category=spec['category']))
            local_visible=visible[specs[0]['track_id']]['local'];road_visible=visible[specs[0]['track_id']]['roadside']
            local=ObservationPacket('map-A','1','map-right-handed',index,'carla-n14',now,now,tuple(local_tracks)).to_dict() if local_tracks else None
            if index%2==0 and road_tracks:
                message=make_prediction(tuple(road_tracks),now,sequence,domain='carla-n14')
                message=replace(message,model_hash='n14-cv-current-observation-only');sequence+=1
                channel.send(message,Timestamp(now,'carla-n14'))
            pending.extend(channel.deliver(Timestamp(now,'carla-n14')))
            ready=[r for r in pending if r.usable_time<=now];pending=[r for r in pending if r.usable_time>now]
            observation={'time_s':now,'ego':es,'local':local,'received':[r.to_dict() for r in ready]}
            with (output/'合法观测.jsonl').open('a',encoding='utf-8') as evidence:evidence.write(json.dumps(observation,ensure_ascii=False)+'\n')
            # stdin/stdout仅传白名单观测及经核心侧决策的控制；truth另存。
            print('OAS:'+json.dumps({'kind':'step','observation':observation},allow_nan=False),flush=True)
            response=json.loads(sys.stdin.readline())
            if set(response)!={'throttle','brake','steer'}:raise ValueError('控制字段非法')
            action=response
            participant_states={s['track_id']:current(participants[s['role']]) for s in specs}
            participant_actions={}
            for spec in specs:
                actor=participants[spec['role']];s=participant_states[spec['track_id']]
                if spec['category']=='pedestrian':value=walker_action(spec,now,held=config.get('probe_static',False))
                else:
                    actuator=target_actuator if spec['role']=='target' else extra_actuators[spec['role']]
                    ta=held if config.get('probe_static',False) else actuator.step(speed_mps=math.hypot(s['vx'],s['vy']),target_speed_mps=spec['speed_mps'],
                        acceleration_mps2=0.,angle_rad=-lane_angle(s,spec['spawn']),dt_s=.05,
                        vehicle_blueprint='vehicle.audi.a2',surface_id=config['calibration']['surface_id'])
                    value={k:ta[k] for k in ('throttle','brake','steer')}
                participant_actions[spec['role']]=value
            target_action=participant_actions['target'];extra_actions={r:v for r,v in participant_actions.items() if r!='target'}
            truck_action=truck_driver.step(current(obstacle)) if moving else None
            actual=ego.get_control();acc=ego.get_acceleration();forward=ego.get_transform().get_forward_vector()
            row={'index':index,'time_s':now,'frame':runner.frame,'ego':es,'target':ts,'local_visible':local_visible,
                 'roadside_visible':road_visible,'geometry_overlap':overlap(es,ts),'control':action,
                 'pending_count':channel.pending_count+len(pending),'target_pixels':pixels,'visibility_mask_sha256':mask_hashes,
                 'actual_control':{'throttle':actual.throttle,'brake':actual.brake,'steer':actual.steer},
                 'longitudinal_acceleration_mps2':acc.x*forward.x+acc.y*forward.y+acc.z*forward.z,
                 'occluder':current(obstacle),'occluder_control':truck_action,
                 'occluder_target_overlap':overlap(current(obstacle),ts),'occluder_ego_overlap':overlap(current(obstacle),es)}
            if config.get('suite_case'):
                row.update(participants=participant_states,participant_visibility=visible,participant_pixels=all_pixels,
                    participant_controls=participant_actions,
                    static_occluders={r:current(a) for r,a in static_occluders.items()},
                    static_occluder_overlaps={r:any(overlap(current(a),s) for s in (es,*participant_states.values())) for r,a in static_occluders.items()},
                    participant_overlaps={identity:overlap(es,s) for identity,s in participant_states.items()},
                    participant_occluder_overlaps={identity:overlap(current(obstacle),s) for identity,s in participant_states.items()})
            truth.append(row);observations.append(observation)
            with (output/'评测真值.jsonl').open('a',encoding='utf-8') as evidence:evidence.write(json.dumps(row,ensure_ascii=False)+'\n')
            if now<config['prefix_s']:prefix.append({'observation':observation,'control':action,'target_control':target_action,'occluder_control':truck_action})
            if index%config['image_interval_ticks']==0:
                for name,buffer in frames.items():
                    frame=buffer.get(runner.frame,2.)
                    if not frame['valid']:raise RuntimeError('实际相机帧缺失')
                    image=output/f'{name}画面_帧{index:04d}.png';proof=save_rgb_frame(frame['value'],image)
                    images.append({'image':image.name,'source':name,'frame':runner.frame,'time_s':now,'index':index,**proof})
            advance(action,target_action,truck_action)
        time.sleep(.2)
        (output/'碰撞传感器.json').write_text(json.dumps(collisions,ensure_ascii=False,indent=2),encoding='utf-8')
        (output/'环境碰撞传感器.json').write_text(json.dumps(environment_collisions,ensure_ascii=False,indent=2),encoding='utf-8')
        (output/'评测真值.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in truth),encoding='utf-8')
        (output/'合法观测.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in observations),encoding='utf-8')
        (output/'相机帧索引.json').write_text(json.dumps(images,ensure_ascii=False,indent=2),encoding='utf-8')
        (output/'信道事件.json').write_text(json.dumps(channel.events,ensure_ascii=False,indent=2),encoding='utf-8')
        (output/'公共前缀.json').write_text(json.dumps({'sha256':digest(prefix),'frames':len(prefix),'initial_sha256':digest(metadata)},indent=2),encoding='utf-8')
    for buffer in frames.values():buffer.close()
    for buffer in masks.values():buffer.close()
    print('OAS:'+json.dumps({'kind':'complete'}),flush=True)


def main():
    import carla
    task,folder=sys.argv[1:3];output=Path(folder)
    client=carla.Client('127.0.0.1',2000);client.set_timeout(20)
    if client.get_client_version()!='0.9.16' or client.get_server_version()!='0.9.16':raise RuntimeError('CARLA版本不一致')
    if task=='discover':discover(client,output)
    elif task=='run':run_episode(client,output,json.loads(Path(sys.argv[3]).read_text(encoding='utf-8')))
    elif task=='geometry':measure_geometry(client,output,json.loads(Path(sys.argv[3]).read_text(encoding='utf-8')))
    else:raise ValueError('未知任务')


if __name__=='__main__':main()
