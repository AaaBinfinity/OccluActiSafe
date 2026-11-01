"""N14四场景高画质实际采集；IDE无参数，先启动匹配Epic CARLA。"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid

import yaml
from discover_occluded_junction import ROOT,ARTIFACTS
from run_occluded_junction import run
from occluactisafe.simulation.n14_suite import expand_cases,epic_server

DEFAULT_CONFIG=ROOT/'configs/scenarios/n14_suite.yaml'


def run_suite(config_path=DEFAULT_CONFIG,*,case_id=None,probe=False):
    config_path=Path(config_path).resolve();suite=yaml.safe_load(config_path.read_text('utf-8'))
    base_path=config_path.parent/suite['base_config'];base=yaml.safe_load(base_path.read_text('utf-8'))
    cases=expand_cases(base,suite)
    if case_id is not None:cases=[(c,v) for c,v in cases if c['id']==case_id]
    if not cases:raise ValueError('没有指定的已冻结场景')
    parent=ARTIFACTS/'节点资料/N14_盲区路口/02_运行产物'
    if not parent.is_dir():raise FileNotFoundError('硬盘N14节点目录未接入')
    render=epic_server();output=parent/(('n14_hd_probe_' if probe else 'n14_suite_')+uuid.uuid4().hex);output.mkdir()
    print(output,flush=True)
    report={'run_id':output.name,'started_at':datetime.now().astimezone().isoformat(),'status':'running',
            'probe_only':probe,'selected_case':case_id,'render_server':render,
            'suite_config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
            'base_config_sha256':hashlib.sha256(base_path.read_bytes()).hexdigest(),'cases':[]}
    save=lambda:(output/'多场景运行清单.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # 在任何物理分支执行前固定本轮全部展开配置及身份。
    for case,config in cases:
        folder=output/(case['id']+'_'+case['name']);folder.mkdir()
        if probe:config.update(probe_static=True,ticks=1,strategies=['reference'])
        path=folder/'冻结场景.yaml';path.write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
        report['cases'].append({'id':case['id'],'name':case['name'],'config':str(path),'config_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'status':'not_run'})
    save()
    for item in report['cases']:
        try:
            item['status']='running';save()
            item['run_dir']=str(run(Path(item['config']),output_root=Path(item['config']).parent));item['status']='completed'
        except Exception as error:item.update(status='failed',error=repr(error))
        finally:save()
    report['status']='completed' if all(c['status']=='completed' for c in report['cases']) else 'failed'
    report['finished_at']=datetime.now().astimezone().isoformat();save()
    if report['status']!='completed':raise RuntimeError('本次有失败场景，所有尝试/配置均已保留：'+str(output))
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--case');parser.add_argument('--probe',action='store_true')
    args=parser.parse_args();run_suite(args.config,case_id=args.case,probe=args.probe)
