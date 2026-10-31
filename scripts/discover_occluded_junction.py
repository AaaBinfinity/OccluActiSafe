"""N14：读取实际CARLA拓扑，保存候选；IDE可直接运行。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import uuid

ROOT=Path(__file__).resolve().parents[1]
BRIDGE=Path('H:/OccluActiSafeDataset/environments/carla_bridge_py311/Scripts/python.exe')
ARTIFACTS=Path('H:/OccluActiSafeDataset/artifacts')


def launch(task, output, config=None):
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src');env['PYTHONIOENCODING']='utf-8'
    env['OAS_CARLA_AGENTS']='H:/OccluActiSafeDataset/environments/carla_0.9.16_server/PythonAPI/carla'
    command=[str(BRIDGE),'-m','occluactisafe.simulation.scenarios',task,str(output)]
    if config:command.append(str(Path(config).resolve()))
    with (output/'桥接日志.txt').open('w',encoding='utf-8') as log:
        result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'桥接失败，证据保留：{output}')


def run():
    output=ARTIFACTS/('n14_discovery_'+uuid.uuid4().hex);output.mkdir()
    print(output,flush=True);launch('discover',output);return output


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args();run()
