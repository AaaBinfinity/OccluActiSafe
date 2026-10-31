"""N13 实车仿真标定入口；先启动匹配的 CARLA 服务端，IDE 可直接运行。"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs/control/n13_sweep.json'


def run(config_path=DEFAULT_CONFIG):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    bridge = Path(config['bridge_python'])
    if not bridge.is_file():
        raise FileNotFoundError(f'CARLA 桥接解释器不存在：{bridge}')
    run_id = 'n13_' + uuid.uuid4().hex
    output = Path(config['output_root']) / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / '运行配置.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + env.get('PYTHONPATH', '')
    command = [str(bridge), '-m', 'occluactisafe.simulation.actuator', str(output)]
    print(f'运行标识：{run_id}\n证据目录：{output}', flush=True)
    with (output / '运行日志.txt').open('w', encoding='utf-8') as log:
        completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f'标定失败（退出码 {completed.returncode}），保留日志：{output / "运行日志.txt"}')
    print(f'原始采集完成：{output}', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    run(parser.parse_args().config)


if __name__ == '__main__':
    main()
