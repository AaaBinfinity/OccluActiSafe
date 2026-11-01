"""N14无传感器RPC子进程：批量控制有确认，物理时钟始终由父runner持有。"""
import json
from pathlib import Path
import subprocess
import sys

PREFIX='OASRPC:'


def execute_batch(client, carla, commands, allowed):
    native=[]
    for item in commands:
        identity=item['actor_id']
        if identity not in allowed:raise ValueError('命令actor不属于本轮')
        kind=item['kind'];value=item['value']
        if kind=='control':command=carla.command.ApplyVehicleControl(identity,carla.VehicleControl(**value))
        elif kind=='walker_control':
            command=carla.command.ApplyWalkerControl(identity,carla.WalkerControl(
                direction=carla.Vector3D(**value['direction']),speed=value['speed'],jump=value['jump']))
        elif kind=='transform':command=carla.command.ApplyTransform(identity,carla.Transform(carla.Location(**value['location']),carla.Rotation(**value['rotation'])))
        elif kind=='velocity':command=carla.command.ApplyTargetVelocity(identity,carla.Vector3D(**value))
        elif kind=='angular_velocity':command=carla.command.ApplyTargetAngularVelocity(identity,carla.Vector3D(**value))
        elif kind=='physics':
            if not isinstance(value,bool):raise ValueError('物理开关必须为bool')
            command=carla.command.SetSimulatePhysics(identity,value)
        else:raise ValueError('未知批量命令类型')
        native.append(command)
    responses=client.apply_batch_sync(native,False)
    if len(responses)!=len(native):raise RuntimeError('批次响应数与命令数不一致')
    if any(r.error for r in responses):raise RuntimeError(str([r.error for r in responses]))
    return len(responses)


class ScenarioRPC:
    def __init__(self,folder,request):
        self.folder=Path(folder);self.count=0;self.closed=False
        self.log=(self.folder/'批量RPC进程日志.txt').open('w',encoding='utf-8')
        self.process=subprocess.Popen([sys.executable,'-m','occluactisafe.simulation.scenario_rpc',str(self.folder),str(request)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,encoding='utf-8',bufsize=1)

    def read(self):
        for line in self.process.stdout:
            if line.startswith(PREFIX):return json.loads(line[len(PREFIX):])
            self.log.write(line);self.log.flush()
        raise RuntimeError('批量RPC子进程退出，见日志')

    def ready(self,world_id):
        message=self.read()
        if message.get('kind')!='ready' or message.get('world_id')!=world_id:raise RuntimeError('RPC子进程世界身份不一致')

    def apply(self,commands):
        self.process.stdin.write(json.dumps({'sequence':self.count,'commands':commands})+'\n');self.process.stdin.flush()
        result=self.read()
        if not result.get('ok') or result.get('sequence')!=self.count:raise RuntimeError('批量RPC失败：'+str(result))
        self.count+=1

    def close(self):
        if self.closed:return
        self.closed=True
        self.process.stdin.close()
        try:self.process.wait(timeout=25)
        except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        self.log.close()
        (self.folder/'批量RPC复核.json').write_text(json.dumps({'acknowledged_batches':self.count,
            'worker_pid':self.process.pid,'worker_exit_code':self.process.returncode,'worker_tick_calls':0,'bootstrap_without_camera_frames':100,'camera_prime_frames':30,
            'method':'固定预热期后所有控制及复位批次由无传感器回调子进程同步确认；父runner独占tick'},ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    import carla
    from occluactisafe.simulation.scenarios import measure_geometry
    folder=Path(sys.argv[1]);config=json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
    client=carla.Client('127.0.0.1',2000);client.set_timeout(20)
    world=client.get_world()
    if world.id!=config['world_id']:raise RuntimeError('RPC采样世界已改变')
    measure_geometry(client,folder,config)
    print(PREFIX+json.dumps({'kind':'ready','world_id':world.id}),flush=True)
    sequence=0
    with (folder/'批量RPC请求.jsonl').open('x',encoding='utf-8') as log:
        for line in sys.stdin:
            try:
                request=json.loads(line)
                if request['sequence']!=sequence:raise ValueError('RPC批次乱序')
                count=execute_batch(client,carla,request['commands'],set(config['actor_ids'].values()))
                log.write(json.dumps(dict(request,response_count=count),ensure_ascii=False)+'\n');log.flush()
                print(PREFIX+json.dumps({'sequence':sequence,'ok':True}),flush=True);sequence+=1
            except Exception as error:
                print(PREFIX+json.dumps({'sequence':sequence,'ok':False,'error':repr(error)}),flush=True);raise


if __name__=='__main__':main()
