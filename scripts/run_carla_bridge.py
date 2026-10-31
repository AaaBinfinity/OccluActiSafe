"""N12 真服务同步入口；可在 IDE 无参数运行，产物写入移动硬盘。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
from datetime import datetime
from multiprocessing.connection import Listener

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from occluactisafe.common.schemas import AdvanceRequest, ControlResult
from occluactisafe.core.safety_monitor import check_control
from occluactisafe.simulation.observation_adapter import decode_tick


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bridge-python', type=Path, default=Path('H:/OccluActiSafeDataset/environments/carla_bridge_py311/Scripts/python.exe'))
    parser.add_argument('--output-root', type=Path, default=Path('H:/OccluActiSafeDataset/artifacts/n12'))
    parser.add_argument('--host', default='127.0.0.1'); parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--ticks', type=int, default=200); parser.add_argument('--sensor-timeout', type=float, default=0.5)
    args = parser.parse_args()
    if not args.bridge_python.is_file(): raise FileNotFoundError(f'桥接 Python 不存在：{args.bridge_python}')
    if args.ticks < 1 or not 0 <= args.sensor_timeout <= 10: raise ValueError('tick 数或传感器等待非法')
    run_id = datetime.now().strftime('n12_%Y%m%d_%H%M%S_%f')
    output = args.output_root.resolve() / run_id; output.mkdir(parents=True, exist_ok=False)
    config = {'host': args.host, 'port': args.port, 'run_id': run_id, 'output': str(output), 'sensor_timeout_s': args.sensor_timeout}
    (output / '运行配置.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    auth = secrets.token_bytes(32)
    listener = Listener(('127.0.0.1', 0), authkey=auth)
    listener._listener._socket.settimeout(45)
    env = os.environ.copy(); env.update({'OAS_IPC_PORT': str(listener.address[1]), 'OAS_IPC_AUTH': auth.hex(),
                                          'OAS_BRIDGE_CONFIG': json.dumps(config), 'PYTHONPATH': str(ROOT / 'src')})
    with (output / '桥接日志.txt').open('w', encoding='utf-8') as log:
        process = subprocess.Popen([str(args.bridge_python), '-m', 'occluactisafe.simulation.carla_client'],
                                   env=env, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        try:
            with listener.accept() as conn, (output / '时序.jsonl').open('x', encoding='utf-8') as records:
                if not conn.poll(45): raise TimeoutError('CARLA 握手超时')
                ready = json.loads(conn.recv_bytes());
                if set(ready) != {'kind', 'client_version', 'server_version', 'frame', 'sim_time'} or ready['kind'] != 'ready':
                    raise ValueError('桥接握手非法')
                (output / '握手.json').write_text(json.dumps(ready, ensure_ascii=False, indent=2), encoding='utf-8')
                frame, sim_time = ready['frame'], ready['sim_time']
                for index in range(args.ticks):
                    action = {'throttle': 0.0, 'brake': 1.0, 'steer': 0.0}; check_control(action)
                    control = ControlResult(str(index), 'stationary', 'success', True, 'carla', sim_time, sim_time + 1.0, **action, fallback_reason=None)
                    request = AdvanceRequest('1.0', run_id, 'episode-1', 1, frame + 1, 'carla', sim_time, sim_time + 1.0, control)
                    conn.send_bytes(json.dumps({'kind': 'advance', 'request': request.to_dict()}, allow_nan=False).encode())
                    if not conn.poll(30): raise TimeoutError('等待 tick 超时')
                    event, camera = decode_tick(json.loads(conn.recv_bytes(maxlength=2_000_000)))
                    if (event.run_id, event.episode_id, event.generation, event.clock_domain, event.frame_id) != (run_id, 'episode-1', 1, 'carla', frame + 1):
                        raise RuntimeError('tick 身份或帧号错误')
                    if abs((event.sim_time - sim_time) - .05) > 1e-4: raise RuntimeError('仿真步长异常')
                    raw = camera['bgra'].encode() if camera['valid'] else b''
                    records.write(json.dumps({'frame': event.frame_id, 'sim_time': event.sim_time, 'camera_valid': camera['valid'],
                                              'camera_reason': camera['reason'], 'camera_base64_sha256': hashlib.sha256(raw).hexdigest() if raw else None}) + '\n'); records.flush()
                    frame, sim_time = event.frame_id, event.sim_time
                    if index == 0:
                        rejected = []
                        for name, updates in [('duplicate', {}), ('reordered', {'target_frame_id': frame + 2}),
                                              ('old_generation', {'generation': 0})]:
                            candidate = request.to_dict(); candidate.update(updates)
                            conn.send_bytes(json.dumps({'kind': 'advance', 'request': candidate}, allow_nan=False).encode())
                            if not conn.poll(30): raise TimeoutError('异常请求测试超时')
                            result = json.loads(conn.recv_bytes())
                            if result.get('kind') != 'rejected' or result.get('frame') != frame:
                                raise RuntimeError('异常请求未被拒绝或改变了帧')
                            rejected.append({'case': name, 'frame': frame, 'reason': result['reason']})
                        (output / '异常请求.json').write_text(json.dumps(rejected, ensure_ascii=False, indent=2), encoding='utf-8')
                conn.send_bytes(b'{"kind":"close"}')
                if not conn.poll(30) or json.loads(conn.recv_bytes()) != {'kind': 'closed'}: raise RuntimeError('桥接未确认清理完成')
        finally:
            listener.close()
            try: process.wait(timeout=40)
            except subprocess.TimeoutExpired:
                process.terminate(); process.wait(timeout=10)
            if process.returncode: print(f'桥接进程退出码 {process.returncode}，详见 {output / "桥接日志.txt"}', file=sys.stderr)
    if process.returncode: raise RuntimeError('桥接进程异常退出')
    print(f'运行完成：{output}')


if __name__ == '__main__': main()
