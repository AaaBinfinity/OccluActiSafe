"""独立桥接进程入口；仅此进程导入 CARLA 和评测旁路。"""
import base64
import json
import os
from multiprocessing.connection import Client
from pathlib import Path
import sys
import zlib

from occluactisafe.simulation.runner import SynchronousRunner
from occluactisafe.simulation.sensors import FrameBuffer
from occluactisafe.simulation.truth_logger import TruthLogger


def send(conn, data):
    conn.send_bytes(json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8'))


def main():
    import carla
    port = int(os.environ['OAS_IPC_PORT'])
    auth = bytes.fromhex(os.environ.pop('OAS_IPC_AUTH'))
    config = json.loads(os.environ.pop('OAS_BRIDGE_CONFIG'))
    with Client(('127.0.0.1', port), authkey=auth) as conn:
        client = carla.Client(config['host'], config['port']); client.set_timeout(15.0)
        client_version = client.get_client_version(); server_version = client.get_server_version()
        if client_version != '0.9.16' or server_version != client_version:
            raise RuntimeError(f'CARLA 版本不匹配：{client_version}/{server_version}')
        frames = FrameBuffer();display_frames=FrameBuffer(); truth = TruthLogger(Path(config['output']) / '评测真值.jsonl')
        try:
            def destroy_owned(actor):
                # 刚生成且尚未 tick 的 actor 可能不在客户端快照中；按本次持有的 ID 发不推进帧的销毁命令。
                response = client.apply_batch_sync([carla.command.DestroyActor(actor.id)], False)[0]
                if response.error: raise RuntimeError(f'actor {actor.id} 销毁失败：{response.error}')
                return True
            cleanup_connections=0
            def reconnect(timeout):
                nonlocal cleanup_connections
                cleanup_connections+=1
                if cleanup_connections==1:
                    client.set_timeout(timeout)
                    return client
                connected=carla.Client(config['host'],config['port']); connected.set_timeout(timeout)
                return connected
            with SynchronousRunner(client.get_world(), run_id=config['run_id'], episode_id='episode-1',
                                   generation=1, clock_domain='carla', destroy_owned=destroy_owned,
                                   cleanup_path=Path(config['output']) / '清理报告.json',
                                   reconnect=reconnect,destroy_command=carla.command.DestroyActor,
                                   prepare_cleanup=client.set_timeout,
                                   recovery_budget_s=config.get('recovery_budget_s',12.),cleanup_rpc_s=.5) as runner:
                ego = runner.spawn_ego_and_camera(carla, frames)
                if config.get('visualize',False):
                    runner.spawn_preview_camera(carla,ego,display_frames);runner.follow_ego(carla,ego)
                client.set_timeout(config.get('rpc_timeout_s', 5.0))
                send(conn, {'kind': 'ready', 'client_version': client_version, 'server_version': server_version,
                            'frame': runner.frame, 'sim_time': runner.sim_time})
                while True:
                    if not conn.poll(30): raise TimeoutError('核心请求超时')
                    try: message = json.loads(conn.recv_bytes(maxlength=2_000_000))
                    except EOFError: break
                    if message == {'kind': 'close'}: break
                    if message == {'kind': 'probe_server'}:
                        send(conn, {'kind': 'server_alive', 'version': client.get_server_version()})
                        continue
                    if not isinstance(message, dict) or set(message) != {'kind', 'request'} or message['kind'] != 'advance':
                        raise ValueError('未知 IPC 请求')
                    def apply_control(control):
                        ego.apply_control(carla.VehicleControl(throttle=control.throttle, brake=control.brake, steer=control.steer))
                    try:
                        event_json = runner.advance(json.dumps(message['request'], allow_nan=False), apply_control=apply_control)
                    except ValueError as exc:
                        send(conn, {'kind': 'rejected', 'frame': runner.frame, 'reason': str(exc)})
                        continue
                    event = json.loads(event_json); sensor = frames.get(event['frame_id'], config['sensor_timeout_s'])
                    if config.get('visualize',False):runner.follow_ego(carla,ego)
                    if sensor['valid']:
                        image = sensor['value']
                        camera = {'frame': image.frame, 'valid': True, 'width': image.width, 'height': image.height,
                                  'bgra': base64.b64encode(image.raw_data).decode('ascii'), 'reason': None}
                    else:
                        camera = {'frame': sensor['frame'], 'valid': False, 'width': None, 'height': None,
                                  'bgra': None, 'reason': sensor['reason']}
                    truth.log_ego(event['frame_id'], ego)
                    observation={'event':event,'camera':camera}
                    if config.get('visualize',False):
                        display=display_frames.get(event['frame_id'],config['sensor_timeout_s'])
                        if display['valid']:
                            image=display['value']
                            picture={'frame':image.frame,'valid':True,'width':image.width,'height':image.height,
                                     'encoding':'zlib_bgra','data':base64.b64encode(zlib.compress(image.raw_data,1)).decode('ascii'),'reason':None}
                        else:
                            picture={'frame':display['frame'],'valid':False,'width':None,'height':None,
                                     'encoding':'zlib_bgra','data':None,'reason':display['reason']}
                        send(conn,{'observation':observation,'display':picture})
                    else:send(conn,observation)
            send(conn, {'kind': 'closed'})
        finally:
            frames.close();display_frames.close(); truth.close()


if __name__ == '__main__':
    main()
