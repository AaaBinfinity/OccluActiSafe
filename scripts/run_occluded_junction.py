"""N14三分支轨迹输入闭环；IDE默认运行冻结场景，CLI可选。"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import uuid
import yaml
from discover_occluded_junction import ROOT,BRIDGE,ARTIFACTS
from occluactisafe.simulation.trajectory_input import TrajectoryAEB
from occluactisafe.control.actuator import CalibratedActuator

DEFAULT_CONFIG=ROOT/'configs/scenarios/occluded_junction.yaml'


def run(config_path=DEFAULT_CONFIG):
    config=yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    calibration_path=Path(config.pop('calibration_path'))
    if hashlib.sha256(calibration_path.read_bytes()).hexdigest()!=config['calibration_sha256']:raise ValueError('N13标定哈希不匹配')
    fitted=json.loads(calibration_path.read_text(encoding='utf-8'));config['calibration']=fitted['calibration'];config['calibration_context']=fitted['context']
    capabilities=config['calibration']['capabilities']
    config['braking']={'deceleration_mps2':capabilities['effective_deceleration_mps2'],
        'response_delay_s':capabilities['observed_response_delay_s'],'control_period_s':.05,
        'margin_m':config.get('aeb_margin_m',5.),'max_track_age_s':.5}
    output=ARTIFACTS/('n14_'+uuid.uuid4().hex);output.mkdir();print(output,flush=True)
    (output/'冻结配置.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    tracked=[ROOT/'src/occluactisafe/simulation'/n for n in ('scenarios.py','trajectory_input.py','runner.py')]
    tracked.extend([Path(__file__),ROOT/'src/occluactisafe/control/actuator.py'])
    if config.get('occluder_motion'):tracked.append(ROOT/'src/occluactisafe/simulation/occluder_driver.py')
    if config.get('overview',{}).get('enabled'):tracked.append(ROOT/'src/occluactisafe/simulation/aeb_overview.py')
    (output/'源码版本.json').write_text(json.dumps({'commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        'source_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}},ensure_ascii=False,indent=2),encoding='utf-8')
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src');env['PYTHONIOENCODING']='utf-8'
    for strategy in config.get('strategies',['reference','local_aeb','cooperative_aeb']):
        folder=output/strategy;folder.mkdir();child=dict(config,strategy=strategy)
        task=folder/'运行配置.json';task.write_text(json.dumps(child,ensure_ascii=False,indent=2),encoding='utf-8')
        policy=TrajectoryAEB(strategy,prefix_s=config['prefix_s'],horizon_s=config['horizon_s'],braking=config['braking'])
        actuator=CalibratedActuator(config['calibration'],kp=.05,ki=.02,steering_rate_per_s=1.)
        viewer=None
        if config.get('overview',{}).get('enabled'):
            from occluactisafe.simulation.aeb_overview import OverviewRecorder
            viewer=OverviewRecorder(folder,config,strategy)
        with (folder/'桥接日志.txt').open('w',encoding='utf-8') as log,(folder/'在线决策.jsonl').open('w',encoding='utf-8') as audit:
            process=subprocess.Popen([str(BRIDGE),'-m','occluactisafe.simulation.scenarios','run',str(folder),str(task)],
                env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',bufsize=1)
            complete=False
            try:
                for line in process.stdout:
                    if not line.startswith('OAS:'):log.write(line);continue
                    message=json.loads(line[4:])
                    if message['kind']=='complete':complete=True;break
                    if set(message)!={'kind','observation'} or message['kind']!='step':raise ValueError('桥接协议字段非法')
                    obs=message['observation'];decision=policy.step(obs);s=obs['ego'];spawn=config['ego_spawn']
                    yaw=-math.radians(spawn['yaw']);cross=-math.sin(yaw)*(s['x']-spawn['x'])+math.cos(yaw)*(s['y']+spawn['y'])
                    error=(yaw-s['yaw']+math.pi)%(2*math.pi)-math.pi
                    if decision['brake'] or config.get('probe_static',False):
                        # N13实测全刹停车/停车保持；低于映射最低速度时不外推PID表。
                        control={'throttle':0.,'brake':1.,'steer':0.}
                    else:
                        result=actuator.step(speed_mps=math.hypot(s['vx'],s['vy']),target_speed_mps=config['speed_mps'],
                            acceleration_mps2=0.,angle_rad=-max(-.1,min(.1,error-.12*cross)),dt_s=.05,
                            vehicle_blueprint='vehicle.audi.a2',surface_id=config['calibration']['surface_id'])
                        control={k:result[k] for k in ('throttle','brake','steer')}
                    audit.write(json.dumps({'time_s':obs['time_s'],'decision':decision,'control':control},ensure_ascii=False)+'\n');audit.flush()
                    if viewer is not None:viewer.update(obs,decision)
                    process.stdin.write(json.dumps(control)+'\n');process.stdin.flush()
                process.wait(timeout=30)
                if process.returncode or not complete:raise RuntimeError(f'{strategy}失败，保留证据：{folder}')
            finally:
                if process.poll() is None:
                    process.stdin.close()
                    try:process.wait(timeout=20)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                if viewer is not None:viewer.close()
        print(strategy,'completed',flush=True)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    run(parser.parse_args().config)
