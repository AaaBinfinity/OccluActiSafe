"""N17本机单模型/seed任务；代码参数、默认检查、完整记录和受控续训。"""
import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import runpy
import shutil
import uuid

import yaml

from occluactisafe.common.config import PROJECT_ROOT
from .detector_data import load_config
from .detector_models import expanded_models
from .detector_portable import stage_snapshot
from .detector_server import preflight,execution_lock,execute_worker,final_alias
from .detector_local_checkpoints import verify_checkpoint
from .detector_stopping import replay_stopping
from .run_store import read_json,write_json,sha256,now

DEFAULT_OPTIONS=PROJECT_ROOT/'configs/experiments/n17_local.py'
ALL_MODELS=expanded_models(['n','s','m','l','x'])


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def local_options(path=DEFAULT_OPTIONS,overrides=None):
    path=Path(path);path=(path if path.is_absolute() else PROJECT_ROOT/path).resolve()
    raw=runpy.run_path(str(path),run_name='n17_local_options')
    keys=('ACTION','MODEL','SEED','INPUT_ROOT','OUTPUT_ROOT','DEVICE','BATCH','WORKERS','AMP',
          'MAX_EPOCHS','MIN_EPOCHS','PATIENCE','RESUME_RUN','PAUSE_AFTER_EPOCH','PLOT_POINTS_PER_PAGE')
    opts={k:raw[k] for k in keys};opts.update(overrides or {})
    if set(opts)!=set(keys):raise ValueError('本机选项包含未知参数')
    candidates={m['family'].lower():m for m in ALL_MODELS}
    name=str(opts['MODEL']).lower().removesuffix('.pt')
    if name not in candidates:raise ValueError('MODEL须为三族n/s/m/l/x之一')
    opts['model']=copy.deepcopy(candidates[name]);opts['MODEL']=opts['model']['family']
    for key in ('SEED','DEVICE','BATCH','WORKERS','MAX_EPOCHS','MIN_EPOCHS','PATIENCE','PLOT_POINTS_PER_PAGE'):
        if type(opts[key]) is not int or opts[key]<(0 if key in ('SEED','DEVICE','WORKERS') else 1):raise ValueError('本机整数选项非法：'+key)
    if opts['SEED']>=2**32 or not 1<=opts['MIN_EPOCHS']<=opts['MAX_EPOCHS']<=300:raise ValueError('seed范围或300轮上限/最少轮数不合法')
    if type(opts['AMP']) is not bool or opts['ACTION'] not in ('check','plan','probe','smoke','train','resume'):raise ValueError('ACTION或AMP非法')
    if opts['PAUSE_AFTER_EPOCH'] is not None and (type(opts['PAUSE_AFTER_EPOCH']) is not int or not 1<=opts['PAUSE_AFTER_EPOCH']<=opts['MAX_EPOCHS']):raise ValueError('PAUSE_AFTER_EPOCH须为空或预算内正整数')
    if opts['ACTION']=='resume' and not opts['RESUME_RUN']:raise ValueError('resume须填写原中断任务根目录RESUME_RUN')
    def resolve(value):
        p=Path(value).expanduser();return (p if p.is_absolute() else PROJECT_ROOT/p).resolve()
    for key in ('INPUT_ROOT','OUTPUT_ROOT'):opts[key]=resolve(opts[key])
    if (opts['INPUT_ROOT'].is_relative_to(PROJECT_ROOT) or opts['OUTPUT_ROOT'].is_relative_to(PROJECT_ROOT)
            or opts['INPUT_ROOT'].is_relative_to(opts['OUTPUT_ROOT']) or opts['OUTPUT_ROOT'].is_relative_to(opts['INPUT_ROOT'])):
        raise ValueError('输入/输出须在仓库外且互不嵌套')
    if opts['RESUME_RUN']:opts['RESUME_RUN']=resolve(opts['RESUME_RUN'])
    opts['options_file']=str(path);opts['options_sha256']=sha256(path)
    return opts


def check_settings(opts):
    return {'action':'check','input_root':opts['INPUT_ROOT'],'output_root':opts['OUTPUT_ROOT'],
            'weights_root':opts['INPUT_ROOT']/'扩展权重','device':opts['DEVICE'],'font_path':'',
            'batch':opts['BATCH'],'workers':opts['WORKERS'],'amp':opts['AMP'],
            'minimum_free_gpu_mib':4096,'minimum_available_ram_mib':2048,'minimum_free_disk_gib':30,
            'expected_gpu_name':'5060 Ti','model_sizes':[opts['model']['scale']],
            '_path':opts['options_file'],'_sha256':opts['options_sha256']}


def development_rows(rows):
    selected=[]
    for role,n in (('train',16),('model_val',8)):
        for domain in ('SPD','CARLA'):
            part=[r for r in rows if r['role']==role and r['domain']==domain]
            if len(part)<n:raise ValueError('缺少真实开发联调输入，不构造图片代替')
            chosen={}
            # 预声明按GT类别覆盖抽样，不看任何模型输出，避免开发联调恰好没有行人。
            for cls in ('vehicle','pedestrian','cyclist'):
                matches=[r for r in part if any(g['class_id']==cls for g in r['gt'])]
                for row in matches[:max(1,n//4)]:chosen[row['key']]=row
            for row in part:
                if len(chosen)>=n:break
                chosen.setdefault(row['key'],row)
            selected.extend(list(chosen.values())[:n])
    return selected


def build_local_config(opts,environment,snapshot,job,development=False):
    cfg={k:v for k,v in copy.deepcopy(load_config()).items() if not k.startswith('_')}
    maximum,minimum,patience=(2,1,1) if development else (opts['MAX_EPOCHS'],opts['MIN_EPOCHS'],opts['PATIENCE'])
    if opts['ACTION']=='probe':maximum=minimum=patience=1
    cfg.update(output_root=str(job),weights_root=str(opts['INPUT_ROOT']/'扩展权重'),dataset_snapshot=snapshot,
               model_sizes=['n','s','m','l','x'],models=ALL_MODELS,device=0,font_path=environment['font_path'],
               class_name_aliases={'person':'pedestrian'},checkpoint_period=-1,record_epoch_timing=True)
    cfg['training'].update(epochs=maximum,batch=opts['BATCH'],nbs=opts['BATCH'],workers=opts['WORKERS'],amp=opts['AMP'],
                           patience=patience,cls_remap=True,save_json=True)
    cfg['budget_policy']={'criterion':'native_fitness','minimum_epochs':minimum,'patience':patience}
    recipe={'training':cfg['training'],'budget_policy':cfg['budget_policy'],'classes':cfg['classes'],
            'aliases':cfg['class_name_aliases'],'source_index_sha256':environment['input_verification']['source_index_sha256'],
            'actual_snapshot_index_sha256':snapshot['index_sha256'],'development_only':development,'weights_policy':'last_and_best_only'}
    # 路径重绑定后的索引含本机相同输入路径，不含新job路径；同配方可跨模型/seed核对。
    cfg['local_profile']={'model_id':opts['MODEL'],'seed':opts['SEED'],'recipe_sha256':digest(recipe),
                          'source_fingerprint_sha256':environment['source']['source_fingerprint_sha256'],
                          'plot_points_per_page':opts['PLOT_POINTS_PER_PAGE'],'pause_after_epoch':opts['PAUSE_AFTER_EPOCH'],
                          'development_only':development,'full_comparison_complete':False}
    return cfg


def prepare_resume(opts,cfg):
    source=Path(opts['RESUME_RUN']).resolve()
    if not source.is_relative_to(opts['OUTPUT_ROOT']):raise ValueError('恢复来源须在本节点实际输出目录内')
    task=read_json(source/'本机训练任务.json')
    if not task.get('frozen_config'):raise ValueError('原任务尚未准备完成，不能从不存在的检查点恢复')
    oldcfg=load_config(task['frozen_config']);local=cfg['local_profile']
    if task['status']=='completed':raise ValueError('任务已完成，不重复续训；新配方请新建任务')
    old=oldcfg['local_profile']
    if any(local[k]!=old[k] for k in ('model_id','seed','recipe_sha256','source_fingerprint_sha256')):raise ValueError('恢复模型/seed/配方/源码不同，拒绝混入旧任务')
    if task['gpu_uuid']!=cfg['_local_gpu_uuid']:raise ValueError('恢复GPU UUID不同')
    train=Path(task.get('training_directory',source/'运行/训练'));checkpoint=verify_checkpoint(train)
    if checkpoint['epoch']>=cfg['training']['epochs']:raise ValueError('检查点已到预算上限，无需增加训练')
    trace=[json.loads(line) for line in (train/'早停判定.jsonl').read_text(encoding='utf-8').splitlines()]
    committed=[r for r in trace if r['epoch']<=checkpoint['epoch']]
    if committed and committed[-1]['stop']:raise ValueError('检查点已触发早停，不应继续训练')
    state=read_json(train.parent/'训练状态.json')
    local.update(resume_checkpoint=str(train/'weights/last.pt'),resume_checkpoint_sha256=checkpoint['last_sha256'],
                 resume_best_checkpoint=str(train/'weights'/checkpoint['best_source_file']),resume_training_directory=str(train),
                 resume_epoch=checkpoint['epoch'],resume_batch_attempts=checkpoint['batch_attempts'],
                 resume_applied_updates=checkpoint['applied_updates'],original_class_transfer=state['class_transfer'])
    return {'source_run':str(source),'source_checkpoint_sha256':checkpoint['last_sha256'],'source_epoch':checkpoint['epoch'],
            'native_optimizer_epoch_resume':True,'source_directory_mutated':False}


def audit_local_batches(training,expected,actual_epochs,batch,amp):
    training=Path(training);epoch=1;seen=Counter();attempts=updates=skips=exposures=epoch_batches=0;per_epoch=[]
    if not expected:raise ValueError('不能用空训练输入记为完成')
    def verify_epoch():
        if seen!=expected or epoch_batches!=math.ceil(sum(expected.values())/batch):raise ValueError('某轮样本缺失/重复或实际batch预算不同')
    with (training/'逐批次优化尝试.csv').open(encoding='utf-8',newline='') as a,(training/'逐步优化指标.csv').open(encoding='utf-8',newline='') as b:
        applied=iter(csv.DictReader(b))
        for row in csv.DictReader(a):
            current=int(row['epoch'])
            if current!=epoch:
                if current!=epoch+1:raise ValueError('epoch不连续')
                verify_epoch();per_epoch.append({'epoch':epoch,'images':sum(seen.values())});epoch=current;seen=Counter();epoch_batches=0
            attempts+=1
            if int(row['attempt'])!=attempts:raise ValueError('批次尝试记录不连续')
            files=json.loads(row['image_files']);seen.update(str(Path(f).resolve()) for f in files)
            if len(files)!=int(row['image_count']) or not 0<len(files)<=batch:raise ValueError('批次曝光分母与原始文件清单或batch不同')
            epoch_batches+=1
            exposures+=len(files)
            if row['update_applied']=='True':
                updates+=1
                if int(row['step'])!=updates or next(applied,None)!=row:raise ValueError('实际更新与尝试记录不一致')
                for field in ('loss','gradient_norm','learning_rate','step_seconds','gpu_allocated_mib','host_rss_mib'):
                    if not math.isfinite(float(row[field])):raise ValueError('实际更新记录含非有限值')
            elif row['update_applied']=='False':
                skips+=1
                if not amp or float(row['amp_scale_after'])>=float(row['amp_scale_before']):raise ValueError('AMP跳步没有实际缩放证据')
            else:raise ValueError('更新状态非法')
        if next(applied,None) is not None:raise ValueError('存在无来源的优化更新')
    verify_epoch()
    if epoch!=actual_epochs:raise ValueError('末轮曝光与早停记录不一致')
    per_epoch.append({'epoch':epoch,'images':sum(seen.values())})
    return {'batch_attempts':attempts,'applied_updates':updates,'amp_skips':skips,'image_exposures':exposures,'per_epoch':per_epoch}


def audit_local_training(cfg,training):
    from .detector_data import load_index
    from .detector_local_visuals import read_json_local
    training=Path(training);rows,_=load_index(cfg)
    expected=Counter(str(Path(r['image']).resolve()) for r in rows if r['role']=='train')
    trace=[json.loads(s) for s in (training/'早停判定.jsonl').read_text(encoding='utf-8').splitlines()]
    termination=replay_stopping(trace,cfg['budget_policy'],cfg['training']['epochs'])
    batches=audit_local_batches(training,expected,termination['actual_epochs'],cfg['training']['batch'],cfg['training']['amp'])
    attempts,updates,skips,exposures=(batches[k] for k in ('batch_attempts','applied_updates','amp_skips','image_exposures'))
    epoch=termination['actual_epochs'];per_epoch=batches['per_epoch']
    checkpoint=verify_checkpoint(training)
    if checkpoint['status']!='completed' or checkpoint['applied_updates']!=updates or checkpoint['batch_attempts']!=attempts or checkpoint['epoch']!=epoch:raise ValueError('已完成检查点与实际更新/epoch不同')
    if {p.name for p in (training/'weights').iterdir()}!={'last.pt','best.pt'}:raise ValueError('weights必须且仅含last.pt/best.pt')
    graph=read_json_local(training/'训练图表来源_完整分项.json')
    if graph['points']!=updates or graph['validation_epochs']!=epoch or graph['step_csv_sha256']!=sha256(training/'逐步优化指标.csv'):raise ValueError('图表与实际记录不符')
    for item in graph['figures']:
        if sha256(training/item['file'])!=item['sha256']:raise ValueError('图表SHA改变')
    required=['args.yaml','超参数.yaml','运行选项.yaml','results.csv','训练结果总览_原生.png',
              '标签分布_原生.png','标签几何相关图.png','标签几何相关图.svg',
              'F1_置信度曲线.png','精确率_置信度曲线.png','召回率_置信度曲线.png','精确率_召回率曲线.png',
              '混淆矩阵_计数.png','混淆矩阵_归一化.png','早停依据与最佳指标.png',
              '逐轮分类精确率召回率与AP.png','逐轮实际耗时.png','逐轮显卡温度功耗与利用率.png',
              '逐轮验证原值/最佳权重验证.json','predictions.json','误检漏检与检测样例/诊断图来源与指标.json']
    missing=[name for name in required if not (training/name).is_file()]
    for pattern in ('训练批次示例_*.png','验证批次*_真实标签.png','验证批次*_模型预测.png'):
        if len(list(training.glob(pattern)))<3:missing.append(pattern+'至少3张')
    if missing:raise ValueError('参考图集/补充记录不完整：'+repr(missing))
    return {'status':'passed','scope':'单个模型/seed，不是完整矩阵或正式选型','actual_epochs':epoch,'termination':termination,
            'train_frames_per_epoch':sum(expected.values()),'image_exposures':exposures,'batch_attempts':attempts,
            'applied_updates':updates,'amp_skips':skips,'weights':checkpoint,'per_epoch':per_epoch,
            'full_comparison_complete':False,'test_media_opened':0,'required_output_files':required,
            'image_count':sum(p.suffix.lower() in ('.png','.svg','.jpg','.jpeg') for p in training.rglob('*'))}


def run_local(opts):
    settings=check_settings(opts)
    with execution_lock(opts['OUTPUT_ROOT']/f'.gpu_{opts["DEVICE"]}.lock'):
        rows,environment,checkdir=preflight(settings,models=[opts['model']])
        if opts['ACTION']=='check':return checkdir
        development=opts['ACTION'] in ('smoke','probe')
        if opts['ACTION']=='resume':development=bool(read_json(opts['RESUME_RUN']/'本机训练任务.json')['development_only'])
        if development:rows=development_rows(rows)
        folder='04_开发与复核/本机短程联调' if development else ('04_开发与复核/本机任务计划' if opts['ACTION']=='plan' else '02_运行产物')
        job=opts['OUTPUT_ROOT']/folder/('n17_local_'+opts['MODEL'].lower()+'_seed'+str(opts['SEED'])+'_'+uuid.uuid4().hex);job.mkdir(parents=True)
        snapshot=stage_snapshot(opts['INPUT_ROOT'],rows,job/'冻结数据元信息')
        cfg=build_local_config(opts,environment,snapshot,job,development);cfg['_local_gpu_uuid']=environment['gpu']['uuid']
        try:resume=prepare_resume(opts,cfg) if opts['ACTION']=='resume' else None
        except BaseException as exc:
            write_json(job/'本机训练任务.json',{'run_id':job.name,'status':'failed','phase':'prepare_resume',
                       'model':opts['MODEL'],'seed':opts['SEED'],'development_only':development,'error':repr(exc),
                       'source_run':str(opts['RESUME_RUN']),'training_started':False})
            raise
        frozen=job/'本次训练配置.yaml';frozen.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
        shutil.copy2(opts['options_file'],job/'本次代码选项.py')
        training_parent=job/opts['MODEL']/('seed_'+str(opts['SEED']))
        task={'run_id':job.name,'status':'planned' if opts['ACTION']=='plan' else 'running','started_at':now(),
              'model':opts['MODEL'],'seed':opts['SEED'],'development_only':development,'full_comparison_complete':False,
              'source_commit':environment['source']['source_commit'],'source_fingerprint_sha256':environment['source']['source_fingerprint_sha256'],
              'gpu_uuid':environment['gpu']['uuid'],'training_directory':str(training_parent/'训练'),
              'effective_options':{k:str(v) if isinstance(v,Path) else v for k,v in opts.items()},
              'seed_in_frozen_comparison':opts['SEED'] in (17,29,43),'frozen_config':str(frozen),'frozen_config_sha256':sha256(frozen),
              'environment_report':str(checkdir/'服务器预检.json'),'input':environment['input_verification'],'resume':resume}
        write_json(job/'本机训练任务.json',task)
        if opts['ACTION']=='plan':
            from ultralytics.cfg import get_cfg
            get_cfg(overrides=cfg['training']);print('仅计划，未训练：'+str(job),flush=True);return job
        params={'kind':'train','family':opts['MODEL'],'seed':opts['SEED'],'config':str(frozen),'attempt':str(job),
                'training_directory':str(training_parent),
                'expected_gpu_uuid':environment['gpu']['uuid'],'worker_cuda_visible_devices':environment['worker_cuda_visible_devices']}
        write_json(job/'任务参数.json',params)
        print(f'本次只训练 {opts["MODEL"]}，seed={opts["SEED"]}，上限{cfg["training"]["epochs"]}轮；产物：{job}',flush=True)
        try:
            code=execute_worker(job/'任务参数.json',job/'控制台.txt');task['worker_returncode']=code
            if code!=0:raise RuntimeError('训练中断/失败，原记录和已提交last/best保留，可选resume新目录续训')
            audit=audit_local_training(load_config(frozen),training_parent/'训练');write_json(job/'本次训练核验.json',audit)
            task.update(status='completed',finished_at=now(),audit=audit)
            write_json(job/'本机训练任务.json',task)
            if not development:final_alias(opts['OUTPUT_ROOT'],job)
            relative=(training_parent/'训练').relative_to(job).as_posix()
            (job/'结果导航.md').write_text(f'# 本次本机单模型训练\n\n模型：{opts["MODEL"]}；seed：{opts["SEED"]}；开发联调：{development}。\n\n权重：[last.pt]({relative}/weights/last.pt)、[best.pt]({relative}/weights/best.pt)。\n\n完整图表与原始记录在[训练目录]({relative}/)。实际完成{audit["actual_epochs"]}轮、{audit["applied_updates"]}次更新、{audit["amp_skips"]}次原生AMP跳步；单次任务不代表正式三seed/全模型选型。\n',encoding='utf-8')
            print('本次单模型任务完成：'+str(training_parent/'训练/weights'),flush=True)
            return job
        except BaseException as exc:
            task.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc),finished_at=now())
            write_json(job/'本机训练任务.json',task);print('未完成任务已保留：'+str(job),flush=True);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_OPTIONS)
    parser.add_argument('--action',choices=['check','plan','probe','smoke','train','resume'])
    parser.add_argument('--model');parser.add_argument('--seed',type=int);parser.add_argument('--run',type=Path)
    args=parser.parse_args();overrides={}
    for key,value in (('ACTION',args.action),('MODEL',args.model),('SEED',args.seed),('RESUME_RUN',args.run)):
        if value is not None:overrides[key]=value
    run_local(local_options(args.config,overrides))
