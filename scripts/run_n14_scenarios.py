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


def verify_file_hashes(folder,files):
    folder=Path(folder).resolve()
    for name,expected in files.items():
        path=(folder/name).resolve()
        if not path.is_relative_to(folder) or not path.is_file():raise ValueError('复用证据路径缺失或越界：'+name)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('复用证据内容被修改：'+name)


def reusable_case(previous,item,destination):
    """仅复用本机同冻结配置、同物理生产源码且已完成复核的整组三策略。"""
    previous=Path(previous).resolve()
    if not previous.is_relative_to(ARTIFACTS.resolve()):raise ValueError('复用来源须在实际硬盘artifacts内')
    load=lambda p:json.loads(p.read_text('utf-8'))
    manifest=load(previous/'多场景运行清单.json')
    candidates=[c for c in manifest['cases'] if c['id']==item['id'] and c['config_sha256']==item['config_sha256'] and c['status']=='completed']
    if not candidates:return None
    case=candidates[0];analysis=load(previous/'多场景验收报告.json')
    result=next((c for c in analysis['cases'] if c['id']==item['id'] and c['status']=='passed'),None)
    if result is None:raise ValueError('同配置案例没有完整通过的分析，拒绝复用')
    run_dir=Path(case['run_dir']).resolve()
    if not run_dir.is_relative_to(previous):raise ValueError('首轮复用来源须是本机原批次内的实际运行')
    snapshot=load(run_dir/'源码版本.json')
    # 本入口只编排任务；原生采样、在线决策、执行器、runner与绘制源码必须逐字节相同。
    production={p:h for p,h in snapshot['source_sha256'].items() if p!='scripts/run_n14_scenarios.py'}
    verify_file_hashes(ROOT,production)
    verify_file_hashes(run_dir,load(run_dir/'证据索引.json')['files'])
    for record in result['comparison_decode']['videos']:
        if not record['passed']:raise ValueError('原对照视频解码未通过')
        verify_file_hashes(run_dir,{record['video']:record['sha256']})
    videos=Path(result['independent_videos']);video_report=load(videos/'独立视频来源与复核.json')
    if video_report['status']!='completed' or len(video_report['videos'])!=12:raise ValueError('原12视角视频未完整通过')
    for record in video_report['videos']:
        if not record['decode_check']['passed']:raise ValueError('原独立视频解码未通过')
        verify_file_hashes(videos,{record['video']:record['sha256']})
    extra=load(run_dir/'多参与者追加验收.json')
    if extra['failed']:raise ValueError('原追加断言未通过')
    source_files={p.relative_to(run_dir).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(run_dir.rglob('*')) if p.is_file()}
    proof=dict(source_suite=str(previous),source_run=str(run_dir),source_analysis_commit=analysis['analysis_commit'],
               source_analysis_report_sha256=hashlib.sha256((previous/'多场景验收报告.json').read_bytes()).hexdigest(),
               source_version=snapshot,verified_production_files=production,source_files=source_files,result=result,
               note='只复用本机本轮已完成且同配置的三策略；本入口已变但不重新产生该案例物理数据，实际生产版本保持原值')
    path=destination/'复用核验.json';path.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return dict(run_dir=str(run_dir),reuse_verification=str(path),reuse_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def run_suite(config_path=DEFAULT_CONFIG,*,case_id=None,probe=False,reuse_from=None):
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
            'base_config_sha256':hashlib.sha256(base_path.read_bytes()).hexdigest(),'cases':[],
            'reuse_from':str(Path(reuse_from).resolve()) if reuse_from else None}
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
            reused=reusable_case(reuse_from,item,Path(item['config']).parent) if reuse_from else None
            if reused:
                item.update(reused);print(item['id'],'verified reuse',flush=True)
            else:item['run_dir']=str(run(Path(item['config']),output_root=Path(item['config']).parent))
            item['status']='completed'
        except Exception as error:item.update(status='failed',error=repr(error))
        finally:save()
    report['status']='completed' if all(c['status']=='completed' for c in report['cases']) else 'failed'
    report['finished_at']=datetime.now().astimezone().isoformat();save()
    if report['status']!='completed':raise RuntimeError('本次有失败场景，所有尝试/配置均已保留：'+str(output))
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--case');parser.add_argument('--probe',action='store_true')
    parser.add_argument('--reuse-from',type=Path,help='只复用本机同配置且证据完整通过的案例，其余案例全三分支重跑')
    args=parser.parse_args();run_suite(args.config,case_id=args.case,probe=args.probe,reuse_from=args.reuse_from)
