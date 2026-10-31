"""真实服务的旁路清理审计；需要在独立 CARLA Python 环境运行。"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def snapshot(client):
    world = client.get_world()
    settings = world.get_settings()
    return {'actor_ids': sorted(actor.id for actor in world.get_actors()),
            'synchronous_mode': settings.synchronous_mode,
            'fixed_delta_seconds': settings.fixed_delta_seconds,
            'substepping': settings.substepping,
            'max_substeps': settings.max_substeps,
            'max_substep_delta_time': settings.max_substep_delta_time}


def main():
    import carla
    parser = argparse.ArgumentParser()
    parser.add_argument('--core-python', type=Path, default=ROOT / '.venv/Scripts/python.exe')
    parser.add_argument('--bridge-python', type=Path, default=Path(sys.executable))
    parser.add_argument('--output-root', type=Path, default=Path('H:/OccluActiSafeDataset/artifacts/n12'))
    parser.add_argument('--host', default='127.0.0.1'); parser.add_argument('--port', type=int, default=2000)
    args = parser.parse_args()
    output = args.output_root.resolve() / datetime.now().strftime('n12_cleanup_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    client = carla.Client(args.host, args.port); client.set_timeout(15)
    versions = {'client': client.get_client_version(), 'server': client.get_server_version()}
    if versions != {'client': '0.9.16', 'server': '0.9.16'}: raise RuntimeError(f'版本不匹配：{versions}')
    results = []
    for fault in ('disconnect', 'timeout', 'exception'):
        before = snapshot(client)
        cmd = [str(args.core_python), str(ROOT / 'scripts/run_carla_bridge.py'), '--bridge-python', str(args.bridge_python),
               '--output-root', str(output / fault), '--host', args.host, '--port', str(args.port), '--fault-test', fault]
        env = os.environ.copy(); env['PYTHONUTF8'] = '1'
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=130,
                                env=env, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        # 异步世界恢复后，客户端快照可能在短时间内仍包含已销毁 actor。
        deadline = time.monotonic() + 5
        after = snapshot(client)
        while before != after and time.monotonic() < deadline:
            time.sleep(.1); after = snapshot(client)
        entry = {'fault': fault, 'exit_code': result.returncode, 'before': before, 'after': after,
                 'passed': result.returncode == 0 and before == after, 'stdout': result.stdout, 'stderr': result.stderr}
        results.append(entry)
        (output / '清理审计.json').write_text(json.dumps({'versions': versions, 'results': results}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'真实清理审计：{output}')
    if not all(x['passed'] for x in results): raise SystemExit(1)


if __name__ == '__main__': main()
