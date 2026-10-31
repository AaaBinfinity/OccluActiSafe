"""CARLA 桥接侧 N13 采样：只使用自车状态，不消费其他对象真值。"""
import csv
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import sys
import time
import traceback

from occluactisafe.simulation.runner import SynchronousRunner
from occluactisafe.simulation.sensors import FrameBuffer


def cases(config):
    if 'controller_checks' in config:
        for repeat in range(config['repetitions']):
            for check in config['controller_checks']:
                yield {'repeat':repeat,'speed':check['initial_speed_mps'],'kind':'controller',
                       'value':0.,'check':check}
        return
    for repeat in range(config['repetitions']):
        for speed in config['speeds_mps']:
            for pedal in config['pedals']:
                for kind in ('throttle', 'brake'):
                    yield {'repeat': repeat, 'speed': speed, 'kind': kind, 'value': pedal}
            for value in config.get('steer_values', (-1.0, -0.3, 0.3, 1.0)):
                for kind in ('steer_step', 'steer_ramp'):
                    yield {'repeat': repeat, 'speed': speed, 'kind': kind, 'value': value}


def run(output):
    import carla
    output = Path(output)
    config = json.loads((output / '运行配置.json').read_text(encoding='utf-8'))
    client = carla.Client(config['host'], config['port']); client.set_timeout(15)
    if client.get_client_version() != '0.9.16' or client.get_server_version() != '0.9.16':
        raise RuntimeError('客户端和服务端均须为 CARLA 0.9.16')
    frames = FrameBuffer(); previews = FrameBuffer()
    rows_path = output / '执行器原始采样.csv'
    fields = ['case_id','repeat','kind','value','target_initial_speed_mps','phase','tick','frame','sim_time_s',
              'x_m','y_m','yaw_deg','speed_mps','longitudinal_acceleration_mps2','lateral_acceleration_mps2',
              'wheel_angle_deg','wheel_query_frame_min','wheel_query_frame_max','requested_throttle','requested_brake','requested_steer',
              'actual_throttle','actual_brake','actual_steer','requested_acceleration_mps2','requested_angle_rad',
              'target_speed_mps','controller_integral','controller_saturated']
    run_id = output.name
    def destroy(actor):
        result = client.apply_batch_sync([carla.command.DestroyActor(actor.id)], False)[0]
        if result.error: raise RuntimeError(result.error)
        return True
    def reconnect(timeout):
        client.set_timeout(timeout); return client
    try:
        with rows_path.open('w', encoding='utf-8-sig', newline='') as stream, (output / '工况状态.jsonl').open('w', encoding='utf-8') as states, (output / 'saturation_events.jsonl').open('w', encoding='utf-8') as saturations:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); stream.flush()
            with ThreadPoolExecutor(max_workers=1) as wheel_executor, SynchronousRunner(client.get_world(),run_id=run_id,episode_id='calibration',generation=1,
                    clock_domain='carla',destroy_owned=destroy,cleanup_path=output/'清理报告.json',
                    reconnect=reconnect,destroy_command=carla.command.DestroyActor,prepare_cleanup=client.set_timeout) as runner:
                ego = runner.spawn_ego_and_camera(carla, frames)
                runner.spawn_preview_camera(carla, ego, previews)
                # 新 actor 在首帧前的客户端快照可能仍为零位姿；先推进一次再冻结生成点。
                now=runner.sim_time
                control={'request_id':'initial','plan_id':'n13','solver_status':'success',
                    'safety_rechecked':True,'clock_domain':'carla','plan_start_time':now,'valid_until':now+1,
                    'throttle':0.,'brake':1.,'steer':0.,'fallback_reason':None}
                request={'schema_version':'1.0','run_id':run_id,'episode_id':'calibration','generation':1,
                    'target_frame_id':runner.frame+1,'clock_domain':'carla','request_time':now,
                    'valid_until':now+1,'control':control}
                runner.advance(json.dumps(request),apply_control=lambda c: ego.apply_control(
                    carla.VehicleControl(throttle=c.throttle,brake=c.brake,steer=c.steer)))
                initial = ego.get_transform()
                physics = ego.get_physics_control()
                info = {'client_version':client.get_client_version(),'server_version':client.get_server_version(),
                        'vehicle_blueprint':ego.type_id,'vehicle_attributes':dict(ego.attributes),
                        'mass_kg':physics.mass,'drag_coefficient':physics.drag_coefficient,
                        'wheels':[{'tire_friction':w.tire_friction,'max_steer_angle_deg':w.max_steer_angle,
                                   'radius_cm':w.radius} for w in physics.wheels],
                        'map':runner._world.get_map().name,'weather':str(runner._world.get_weather()),
                        'spawn':str(initial),'note':'每个工况重置同一生成点和零速度；目标速度通过油门实际加速达到。车轮角RPC与唯一tick并发，采样帧仅能限定在记录的帧区间。'}
                (output/'车辆与环境.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
                def tick(throttle,brake,steer,case,phase,index,case_id,controller=None):
                    now=runner.sim_time
                    control={'request_id':str(runner.frame+1),'plan_id':'n13','solver_status':'success',
                        'safety_rechecked':True,'clock_domain':'carla','plan_start_time':now,'valid_until':now+1,
                        'throttle':float(throttle),'brake':float(brake),'steer':float(steer),'fallback_reason':None}
                    request={'schema_version':'1.0','run_id':run_id,'episode_id':'calibration','generation':1,
                        'target_frame_id':runner.frame+1,'clock_domain':'carla','request_time':now,
                        'valid_until':now+1,'control':control}
                    previous_frame=runner.frame
                    wheel_future=wheel_executor.submit(ego.get_wheel_steer_angle,carla.VehicleWheelLocation.FL_Wheel)
                    # 给同步 RPC 排队时间；车轮值时间仍只限定在相邻两帧间。
                    time.sleep(.03)
                    runner.advance(json.dumps(request),apply_control=lambda c: ego.apply_control(
                        carla.VehicleControl(throttle=c.throttle,brake=c.brake,steer=c.steer)))
                    wheel_angle=wheel_future.result(timeout=15)
                    transform=ego.get_transform(); velocity=ego.get_velocity(); acc=ego.get_acceleration()
                    forward=transform.get_forward_vector(); right=transform.get_right_vector(); actual=ego.get_control()
                    row={'case_id':case_id,'repeat':case['repeat'],'kind':case['kind'],'value':case['value'],
                         'target_initial_speed_mps':case['speed'],'phase':phase,'tick':index,'frame':runner.frame,
                         'sim_time_s':runner.sim_time,'x_m':transform.location.x,'y_m':transform.location.y,
                         'yaw_deg':transform.rotation.yaw,'speed_mps':math.sqrt(velocity.x**2+velocity.y**2+velocity.z**2),
                         'longitudinal_acceleration_mps2':acc.x*forward.x+acc.y*forward.y+acc.z*forward.z,
                         'lateral_acceleration_mps2':acc.x*right.x+acc.y*right.y+acc.z*right.z,
                         'wheel_angle_deg':wheel_angle,'wheel_query_frame_min':previous_frame,'wheel_query_frame_max':runner.frame,
                         'requested_throttle':throttle,'requested_brake':brake,'requested_steer':steer,
                         'actual_throttle':actual.throttle,'actual_brake':actual.brake,'actual_steer':actual.steer}
                    if controller is not None: row.update(controller)
                    writer.writerow(row)
                    if controller is not None and controller['controller_saturated']:
                        saturations.write(json.dumps({'case_id':case_id,'frame':runner.frame,'sim_time_s':runner.sim_time,
                            'throttle':throttle,'brake':brake,'steer':steer,'integral':controller['controller_integral']},
                            ensure_ascii=False,allow_nan=False)+'\n');saturations.flush()
                    return row
                for case_id,case in enumerate(cases(config)):
                    state={'case_id':case_id,**case,'status':'started'}
                    states.write(json.dumps(state,ensure_ascii=False)+'\n');states.flush()
                    try:
                        ego.set_transform(initial); ego.set_target_velocity(carla.Vector3D()); ego.set_target_angular_velocity(carla.Vector3D())
                        for i in range(30): tick(0,1,0,case,'reset',i,case_id)
                        reached=False
                        for i in range(config['warmup_max_ticks']):
                            row=tick(config.get('warmup_throttle',1.0),0,0,case,'warmup',i,case_id)
                            if row['speed_mps'] >= case['speed']:
                                reached=True; break
                        if not reached: raise RuntimeError('真实加速未达到指定初速')
                        if case['kind']=='controller':
                            from occluactisafe.control.actuator import CalibratedActuator
                            check=case['check']; actuator=CalibratedActuator(config['calibration'],**config['controller_parameters'])
                            if ego.type_id != config['calibration']['vehicle_blueprint']:
                                raise ValueError('实际蓝图与标定表不一致')
                        response_ticks=case['check'].get('response_ticks',config['response_ticks']) if case['kind']=='controller' else config['response_ticks']
                        for i in range(response_ticks):
                            value=case['value']; kind=case['kind']
                            controller=None
                            if kind=='controller':
                                speed=math.sqrt(sum(v*v for v in (ego.get_velocity().x,ego.get_velocity().y,ego.get_velocity().z)))
                                command=actuator.step(speed_mps=speed,target_speed_mps=check['target_speed_mps'],
                                    acceleration_mps2=check['acceleration_mps2'],angle_rad=check['angle_rad'],dt_s=.05,
                                    vehicle_blueprint=ego.type_id,surface_id=config['surface_id'])
                                throttle,brake,steer=(command[k] for k in ('throttle','brake','steer'))
                                controller={'requested_acceleration_mps2':check['acceleration_mps2'],
                                    'requested_angle_rad':check['angle_rad'],'target_speed_mps':check['target_speed_mps'],
                                    'controller_integral':actuator.integral,'controller_saturated':command['saturated']}
                            else:
                                throttle=value if kind=='throttle' else 0
                                brake=value if kind=='brake' else 0
                                steer=(value if kind=='steer_step' else value*min(1,i/40)) if kind.startswith('steer') else 0
                            row=tick(throttle,brake,steer,case,'response',i,case_id,controller)
                            if case_id in (0, 1, 8, 9) and i == 40:
                                preview=previews.get(runner.frame,2)
                                if preview['valid']:
                                    # CARLA Windows 原生接口不支持 Unicode 路径，先用 ASCII 文件名写入。
                                    temporary=output / f'preview_{case_id}.png'
                                    preview['value'].save_to_disk(str(temporary))
                                    temporary.replace(output / f'车辆画面_工况{case_id}.png')
                        state['status']='completed'
                    except Exception as exc:
                        state['status']='failed';state['error']=repr(exc)
                        traceback.print_exc()
                        states.write(json.dumps(state,ensure_ascii=False)+'\n');states.flush();stream.flush()
                        raise
                    states.write(json.dumps(state,ensure_ascii=False)+'\n');states.flush();stream.flush()
                    print(json.dumps(state,ensure_ascii=False),flush=True)
    finally:
        frames.close();previews.close()


if __name__ == '__main__':
    run(sys.argv[1])
