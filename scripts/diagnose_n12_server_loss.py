"""真实 CARLA 服务端中断 A/B 诊断；默认运行不带 Traffic Manager 和调用关闭自动驾驶两组。"""
import argparse
import hashlib
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def plot_results(output, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=[f"{item['mode']}\nrepeat {item['repeat']}" for item in results]
    native=[int((item['exit_code'] & 0xffffffff) >= 0xC0000000) for item in results]
    fig,ax=plt.subplots(figsize=(10,4))
    ax.bar(range(len(results)),native)
    for index,item in enumerate(results): ax.text(index,native[index]+.05,item['exit_hex'],ha='center')
    ax.set(xticks=range(len(results)),xticklabels=labels,ylabel='native process abort (1=yes)',ylim=(0,1.3))
    fig.tight_layout(); images={}
    for suffix in ('png','svg'):
        path=output/('服务端中断与客户端退出对比.'+suffix); fig.savefig(path,dpi=150)
        images[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    plt.close(fig)
    write(output/'图表来源.json',{'source':'诊断结果.json',
          'source_sha256':hashlib.sha256((output/'诊断结果.json').read_bytes()).hexdigest(),'images':images})


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def worker(mode, output):
    import carla
    client = carla.Client('127.0.0.1', 2000); client.set_timeout(15)
    world = client.get_world()
    vehicle = None
    for point in world.get_map().get_spawn_points():
        vehicle = world.try_spawn_actor(world.get_blueprint_library().filter('vehicle.*')[0], point)
        if vehicle: break
    if vehicle is None: raise RuntimeError('没有可用生成点')
    if mode == 'disable_autopilot': vehicle.set_autopilot(False)
    client.set_timeout(3)
    write(output / '就绪.json', {'client': client.get_client_version(), 'actor_id': vehicle.id, 'mode': mode})
    # 等待足够长以暴露 TM 后台异常，且不引入 IPC 或请求超时逻辑。
    time.sleep(10)
    try: client.get_server_version()
    except RuntimeError as exc:
        write(output / 'Python捕获失联.json', {'error': str(exc), 'cleanup': 'server_unavailable'})
        return
    raise RuntimeError('未观察到服务端失联')


def listening():
    try:
        with socket.create_connection(('127.0.0.1', 2000), timeout=.3): return True
    except OSError: return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=('disable_autopilot', 'manual_spawn'))
    parser.add_argument('--worker-output', type=Path)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--bridge-only', action='store_true', help='只验证完整桥接的独立服务端中断路径')
    parser.add_argument('--server', type=Path, default=Path('H:/OccluActiSafeDataset/environments/carla_0.9.16_server/CarlaUE4.exe'))
    parser.add_argument('--bridge-python', type=Path, default=Path('H:/OccluActiSafeDataset/environments/carla_bridge_py311/Scripts/python.exe'))
    parser.add_argument('--output-root', type=Path, default=Path('H:/OccluActiSafeDataset/artifacts/n12_disconnect_diagnosis'))
    args = parser.parse_args()
    if args.worker: return worker(args.worker, args.worker_output)
    if os.name != 'nt': raise RuntimeError('本诊断入口只在本次 Windows 环境验证')
    if not args.server.is_file() or not args.bridge_python.is_file(): raise FileNotFoundError('服务端或桥接 Python 不存在')
    if not 1 <= args.repeats <= 5: raise ValueError('repeats 必须为 1～5')
    if listening(): raise RuntimeError('2000 端口已占用，拒绝影响现有服务')
    output = args.output_root.resolve() / datetime.now().strftime('ab_%Y%m%d_%H%M%S_%f'); output.mkdir(parents=True)
    sys.path.insert(0,str(ROOT/'src'))
    from occluactisafe.common.run_manifest import _source_identity
    write(output/'运行清单.json',{'source':_source_identity(ROOT),'repeats':args.repeats,
          'bridge_only':args.bridge_only,'server':str(args.server.resolve()),'bridge_python':str(args.bridge_python.resolve())})
    env = os.environ.copy(); env['PYTHONUTF8'] = '1'
    startup = subprocess.STARTUPINFO(); startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW; startup.wShowWindow = 0
    results = []
    for repeat in range(1 if args.bridge_only else args.repeats):
        for mode in (('bridge_probe',) if args.bridge_only else ('disable_autopilot', 'manual_spawn')):
            case = output / f'{mode}_{repeat}'; case.mkdir()
            with (case / '服务端日志.txt').open('wb') as server_log, (case / '客户端日志.txt').open('wb') as client_log:
                server = subprocess.Popen([str(args.server.resolve()), '-RenderOffScreen', '-nosound', '-quality-level=Low'],
                                          cwd=args.server.resolve().parent, stdout=server_log, stderr=subprocess.STDOUT,
                                          startupinfo=startup, creationflags=subprocess.CREATE_NO_WINDOW)
                process = None
                try:
                    deadline = time.monotonic() + 90
                    while not listening():
                        if server.poll() is not None or time.monotonic() > deadline: raise RuntimeError('服务端启动失败')
                        time.sleep(.5)
                    command = ([str(ROOT / '.venv/Scripts/python.exe'), str(ROOT / 'scripts/run_carla_bridge.py'),
                                '--bridge-python', str(args.bridge_python.resolve()), '--fault-test', 'server_loss', '--output-root', str(case)]
                               if mode == 'bridge_probe' else [str(args.bridge_python.resolve()), str(Path(__file__).resolve()),
                                                '--worker', mode, '--worker-output', str(case)])
                    process = subprocess.Popen(command, env=env,
                                               stdout=client_log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                    deadline = time.monotonic() + 40
                    while not ((case / '就绪.json').is_file() if mode != 'bridge_probe' else list(case.rglob('握手.json'))):
                        if process.poll() is not None or time.monotonic() > deadline: raise RuntimeError('客户端就绪失败')
                        time.sleep(.1)
                    # 仅结束本诊断新建的服务端进程树。
                    killed = subprocess.run(['taskkill', '/PID', str(server.pid), '/T', '/F'], capture_output=True)
                    (case / '中断服务记录.txt').write_bytes(killed.stdout + killed.stderr)
                    if killed.returncode: raise RuntimeError('未能中断本次服务端')
                    code = process.wait(timeout=65)
                    entry = {'mode': mode, 'repeat': repeat, 'exit_code': code,
                             'exit_hex': f'0x{code & 0xffffffff:08X}',
                             'python_caught_disconnect': (case / 'Python捕获失联.json').is_file()}
                    if mode == 'bridge_probe':
                        reports=list(case.rglob('清理报告.json')); exits=list(case.rglob('进程退出.json'))
                        entry['cleanup_report']=json.loads(reports[0].read_text(encoding='utf-8')) if reports else None
                        entry['bridge_exit']=json.loads(exits[0].read_text(encoding='utf-8')) if exits else None
                    results.append(entry); write(output / '诊断结果.json', results); print(entry, flush=True)
                finally:
                    if process is not None and process.poll() is None: process.terminate(); process.wait(timeout=10)
                    if server.poll() is None: subprocess.run(['taskkill', '/PID', str(server.pid), '/T', '/F'], capture_output=True)
                    server.wait(timeout=10)
            deadline = time.monotonic() + 10
            while listening() and time.monotonic() < deadline: time.sleep(.1)
            if listening(): raise RuntimeError('服务端端口未关闭，停止后续测试')
    plot_results(output,results)
    print(output)


if __name__ == '__main__': main()
