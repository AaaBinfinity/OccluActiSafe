"""N17服务器任务：预检、迁移对照、完整三seed训练评测、任务级续跑和回传。"""
import copy
import csv
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback
import uuid
import zipfile

import yaml

from occluactisafe.common.config import PROJECT_ROOT
from .detector_data import load_config
from .detector_portable import verify_input, stage_snapshot
from .run_store import read_json, write_json, sha256, now, manifest
from .source_version import source_info

DEFAULT_SERVER_CONFIG = PROJECT_ROOT/'configs/experiments/n17_server.yaml'
POLICIES = {'baseline': {}, 'person_alias': {'person': 'pedestrian'}}


def worker_cuda_selector(device,visible=None):
    """在子进程导入torch前只暴露指定卡，框架内部统一使用逻辑cuda:0。"""
    if visible is None:return str(device)
    tokens=[v.strip() for v in visible.split(',') if v.strip()]
    if device>=len(tokens):raise ValueError('GPU编号超出CUDA_VISIBLE_DEVICES映射')
    return tokens[device]


def server_settings(path=DEFAULT_SERVER_CONFIG):
    path = Path(path)
    path = (path if path.is_absolute() else PROJECT_ROOT/path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    allowed={'action','input_root','output_root','device','font_path','batch','workers','pilot_epochs',
             'formal_epochs','formal_policy','minimum_free_gpu_mib','minimum_available_ram_mib','minimum_free_disk_gib',
             'model_sizes','weights_root','amp','minimum_epochs','early_stopping_patience','checkpoint_period','expected_gpu_name'}
    if not isinstance(cfg,dict) or set(cfg)-allowed:raise ValueError('服务器配置含未知字段')
    cfg.setdefault('minimum_free_disk_gib',20)
    if cfg.get('action') not in ('check','pilot','formal','collect','prepare-weights','estimate'):raise ValueError('服务器action非法')
    def resolve(value):
        p = Path(value).expanduser()
        return (p if p.is_absolute() else PROJECT_ROOT/p).resolve()
    cfg['input_root'] = resolve(cfg['input_root'])
    cfg['output_root'] = resolve(cfg['output_root']) if cfg['output_root'] else cfg['input_root'].parent/'OccluActiSafe_N17_Results'
    cfg['weights_root'] = resolve(cfg['weights_root']) if cfg.get('weights_root') else cfg['input_root']/('扩展权重' if cfg.get('model_sizes') else 'weights')
    if cfg['weights_root'].is_relative_to(PROJECT_ROOT) or cfg['weights_root'].is_relative_to(cfg['output_root']):
        raise ValueError('权重目录须在仓库及结果目录外')
    cfg.setdefault('amp',False);cfg.setdefault('minimum_epochs',1)
    cfg.setdefault('early_stopping_patience',0);cfg.setdefault('checkpoint_period',1);cfg.setdefault('expected_gpu_name','')
    if type(cfg['amp']) is not bool or not isinstance(cfg['expected_gpu_name'],str):raise ValueError('amp或expected_gpu_name非法')
    if 'model_sizes' in cfg:
        from .detector_models import expanded_models
        expanded_models(cfg['model_sizes'])
    if cfg['output_root'].is_relative_to(PROJECT_ROOT) or cfg['input_root'].is_relative_to(PROJECT_ROOT):
        raise ValueError('输入/结果目录必须在Git仓库外')
    if cfg['input_root'].is_relative_to(cfg['output_root']) or cfg['output_root'].is_relative_to(cfg['input_root']):
        raise ValueError('输入和结果目录不能嵌套，避免把原图混入结果回传')
    for field in ('device','batch','workers','pilot_epochs','formal_epochs','minimum_free_gpu_mib','minimum_available_ram_mib','minimum_free_disk_gib','minimum_epochs','early_stopping_patience','checkpoint_period'):
        if type(cfg[field]) is not int or cfg[field] < (0 if field in ('device','workers','early_stopping_patience') else 1):
            raise ValueError(f'服务器参数非法：{field}')
    if cfg['formal_epochs'] < 3 or cfg['pilot_epochs'] > cfg['formal_epochs'] or cfg['formal_policy'] not in POLICIES:
        raise ValueError('正式训练至少3轮，pilot不能超过formal，配方须明确')
    if cfg['minimum_epochs'] > cfg['formal_epochs']:raise ValueError('最少轮数不能超过最大预算')
    if cfg.get('font_path'):
        cfg['font_path'] = str(resolve(cfg['font_path']))
    cfg['_path'],cfg['_sha256'] = str(path),sha256(path)
    return cfg


def server_base(settings):
    base=load_config()
    if settings.get('model_sizes'):
        from .detector_models import expanded_models
        base['model_sizes']=settings['model_sizes'];base['models']=expanded_models(settings['model_sizes'])
    return base


@contextmanager
def execution_lock(path):
    """进程退出由OS释放锁；不删除旧锁文件或结束未知进程。"""
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    f = path.open('a+b');f.seek(0)
    if path.stat().st_size == 0:
        f.write(b'0');f.flush();f.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError as exc:
        f.close();raise RuntimeError('该输出目录/GPU任务仍在运行，拒绝并发写入') from exc
    try:yield
    finally:
        f.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(f,fcntl.LOCK_UN)
        f.close()


def preflight(settings,models=None):
    if not settings['input_root'].is_dir():
        raise FileNotFoundError('先复制便携输入包，并填写configs/experiments/n17_server.yaml的input_root')
    directory = settings['output_root']/'03_验收证据'/('n17_server_check_'+uuid.uuid4().hex)
    report = {'status':'running','started_at':now(),'platform':platform.platform(),'python':sys.version,
              'configuration_sha256':settings['_sha256'],'input_root':str(settings['input_root']),
              'output_root':str(settings['output_root']),'training_started':False}
    write_json(directory/'服务器预检.json',report)
    try:
        from .visual_fonts import prepare_font_cache
        settings_dir=directory/'Ultralytics设置'
        prepare_font_cache(settings_dir,settings.get('font_path'))
        os.environ.update(YOLO_CONFIG_DIR=str(settings_dir),YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false')
        if sys.version_info[:2] != (3,11):raise RuntimeError('必须使用Python3.11')
        import torch,torchvision,ultralytics,psutil
        from ultralytics.utils import USER_CONFIG_DIR
        # IDE/kernel或续跑中框架可能已导入，实际缓存路径由首次导入固定。
        # 同时准备框架实际目录并留证，不能只改环境变量后假设缓存已切换。
        prepare_font_cache(Path(USER_CONFIG_DIR).parent,settings.get('font_path'))
        report['actual_ultralytics_cache']=str(USER_CONFIG_DIR)
        if not all((USER_CONFIG_DIR/name).is_file() for name in ('Arial.ttf','Arial.Unicode.ttf')):
            raise RuntimeError('框架实际字体缓存未就绪，拒绝运行中自动下载')
        from .visual_fonts import chinese_font
        if (torch.__version__.split('+')[0],torchvision.__version__.split('+')[0],ultralytics.__version__) != ('2.11.0','0.26.0','8.4.197'):
            raise RuntimeError('依赖版本不匹配；先按文档uv sync --locked，不能混用CPU torchvision')
        if not torch.cuda.is_available() or torch.version.cuda!='12.8':raise RuntimeError('需要可用CUDA12.8 PyTorch，不回退CPU')
        device=settings['device']
        if device >= torch.cuda.device_count():raise RuntimeError('GPU编号超出可见CUDA设备')
        torch.cuda.set_device(device)
        free,total=torch.cuda.mem_get_info(device)
        if free/2**20 < settings['minimum_free_gpu_mib']:raise RuntimeError('GPU空闲显存低于配置门槛，不自动改batch')
        if psutil.virtual_memory().available/2**20 < settings['minimum_available_ram_mib']:raise RuntimeError('可用系统内存不足')
        props=torch.cuda.get_device_properties(device)
        if settings.get('expected_gpu_name') and settings['expected_gpu_name'].lower() not in props.name.lower():
            raise RuntimeError('GPU型号不含配置expected_gpu_name，不用本机设备冒充服务器')
        report.update(torch=torch.__version__,torchvision=torchvision.__version__,ultralytics=ultralytics.__version__,
                      cuda=torch.version.cuda,gpu={'device':device,'name':props.name,'uuid':str(getattr(props,'uuid','unknown')),
                      'capability':list(torch.cuda.get_device_capability(device)),'total_mib':total/2**20,'free_mib':free/2**20},
                      available_ram_mib=psutil.virtual_memory().available/2**20,
                      host_memory_mib=psutil.virtual_memory().total/2**20,processor=platform.processor(),
                      cudnn_tf32=torch.backends.cudnn.allow_tf32,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                      CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),source=source_info())
        report['worker_cuda_visible_devices']=worker_cuda_selector(device,os.environ.get('CUDA_VISIBLE_DEVICES'))
        from importlib.metadata import distributions
        report['dependencies']={d.metadata['Name']:d.version for d in distributions() if d.metadata['Name']}
        if shutil.disk_usage(directory).free/2**30<settings['minimum_free_disk_gib']:raise RuntimeError('输出磁盘空闲空间不足，不自动清理')
        report['output_disk_free_bytes']=shutil.disk_usage(directory).free
        try:
            report['nvidia_smi']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total',
                                                        '--format=csv,noheader'],text=True,stderr=subprocess.STDOUT).strip()
        except (FileNotFoundError,subprocess.CalledProcessError) as exc:report['nvidia_smi_unavailable']=str(exc)
        gpu_device=torch.device('cuda',device)
        boxes=torch.tensor([[0.,0.,10.,10.],[0.,0.,9.,9.]],device=gpu_device)
        scores=torch.tensor([.9,.8],device=gpu_device)
        if torchvision.ops.nms(boxes,scores,.5).tolist()!=[0]:raise RuntimeError('CUDA NMS断言失败')
        with torch.inference_mode():
            conv=torch.nn.Conv2d(3,4,3).to(gpu_device)
            if not torch.isfinite(conv(torch.ones(1,3,16,16,device=gpu_device))).all():raise RuntimeError('GPU卷积非有限')
        torch.cuda.synchronize()
        if settings.get('amp'):
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                mixed=conv(torch.ones(1,3,16,16,device=gpu_device))
                if mixed.dtype!=torch.float16 or not torch.isfinite(mixed).all():raise RuntimeError('CUDA AMP卷积检查失败')
            torch.cuda.synchronize();report['gpu_amp_convolution']='passed'
        font=chinese_font(settings.get('font_path'))
        from fontTools.ttLib import TTFont
        with TTFont(str(font),fontNumber=0,lazy=True) as ft:
            if any(ord(c) not in ft.getBestCmap() for c in '训练车辆行人预警'):raise RuntimeError('配置字体缺少中文码位')
        report['font_path']=str(font);settings['font_path']=str(font)
        print('开始核验全部冻结RGB/标签/初始权重SHA，不读取test',flush=True)
        rows,data=verify_input(settings['input_root'],progress=True)
        report['input_verification']=data
        from ultralytics import YOLO
        from .detector_transfer import apply_class_aliases,verify_person_transfer
        base=server_base(settings);transfer=[]
        if models is not None:base['models']=models
        if settings.get('model_sizes'):
            from .detector_models import verify_weights
            report['expanded_weights']=verify_weights(settings['weights_root'],base['models'])
        for m in base['models']:
            model=YOLO(str(settings['weights_root']/m['weights']))
            apply_class_aliases(model,POLICIES['person_alias'])
            transfer.append({'family':m['family'],**verify_person_transfer(model,base['classes'])})
        report['class_transfer_checks']=transfer
        report.update(status='passed',finished_at=now(),gpu_nms='passed',gpu_convolution='passed')
        write_json(directory/'服务器预检.json',report)
        print(f'预检通过：{directory}',flush=True)
        return rows,report,directory
    except BaseException as exc:
        report.update(status='failed',finished_at=now(),error=repr(exc),traceback=traceback.format_exc())
        write_json(directory/'服务器预检.json',report)
        raise


def plan_tasks(mode,policies,models,seeds):
    tasks=[]
    for policy in policies:
        for model in models:
            for seed in (seeds[:1] if mode=='pilot' else seeds):
                tasks.append({'id':f'train_{policy}_{model["family"]}_{seed}','kind':'train','policy':policy,
                              'family':model['family'],'seed':seed,'status':'planned','attempts':[]})
        tasks.append({'id':'evaluate_'+policy,'kind':'evaluate','policy':policy,'status':'planned','attempts':[]})
    return tasks


def worker(task_path):
    task=read_json(task_path);cfg=load_config(task['config']);output=Path(task['attempt'])
    try:
        import torch
        if not torch.cuda.is_available():raise RuntimeError('worker没有CUDA，不回退CPU')
        actual_uuid=str(getattr(torch.cuda.get_device_properties(0),'uuid','unknown'))
        if task['expected_gpu_uuid']!='unknown' and actual_uuid!=task['expected_gpu_uuid']:
            raise RuntimeError('worker实际GPU与父进程预检UUID不同，拒绝误用GPU0')
        if task['kind']=='train':
            from .detector_matrix import train_one
            training_directory=Path(task.get('training_directory',output/'运行')).resolve()
            if not training_directory.is_relative_to(output.resolve()) or training_directory==output.resolve():raise ValueError('训练目录须在本次任务内')
            result=train_one(cfg,task['family'],task['seed'],training_directory)
            result.update(directory=str(training_directory),returncode=0)
        else:
            from .detector_evaluation import run_evaluation_matrix,evaluate_one,comparison_plots
            from .detector_audit import audit_evaluation_matrix,audit_training_matrix
            if task['mode']=='formal':
                root=run_evaluation_matrix(cfg,task['training_matrix'])
            else:
                root=output/'开发评测';root.mkdir()
                trained=read_json(task['training_matrix'])
                audit=audit_training_matrix(cfg,task['training_matrix'])
                if len(audit['runs'])!=len(cfg['models']) or any(r['status']!='passed' for r in audit['runs']):raise ValueError('开发训练预算复核失败')
                write_json(root/'开发预算复核.json',{'status':'passed','formal_matrix_complete':False,'runs':audit['runs']})
                results=[]
                for r in trained['runs']:
                    m=next(m for m in cfg['models'] if m['family']==r['family'])
                    try:
                        results.append(evaluate_one(cfg,m,r['checkpoint'],'finetuned',r['seed'],root/(r['family']+'_finetuned_'+str(r['seed']))))
                    except Exception as exc:
                        results.append({'candidate_id':r['family']+'_finetuned_'+str(r['seed']),'family':r['family'],
                                        'stage':'finetuned','seed':r['seed'],'status':'failed','error':repr(exc)})
                    write_json(root/'比较矩阵.json',{'run_id':root.name,'development_only':True,'runs':results})
                write_json(root/'矩阵公平性复核.json',audit_evaluation_matrix(cfg,results,root,len(cfg['models'])))
                comparison_plots(root,results)
            fairness=read_json(root/'矩阵公平性复核.json')
            if fairness['status']!='passed':raise ValueError('评测不完整，不能标成功')
            seal=root/'服务器评测产物清单.json';write_json(seal,manifest(root))
            result={'status':'completed','directory':str(root),'manifest':str(seal),'manifest_sha256':sha256(seal),
                    'formal_selection':task['mode']=='formal'}
        write_json(output/'worker_result.json',result)
    except BaseException as exc:
        write_json(output/'worker_failure.json',{'status':'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                   'error':repr(exc),'traceback':traceback.format_exc(),'time':now()})
        raise


def validate_completed(task,cfg_path,expected_source):
    attempt=task['attempts'][-1]
    if attempt.get('returncode')!=0:raise ValueError('已完成任务的worker退出码非0')
    result=read_json(Path(attempt['directory'])/'worker_result.json')
    if result['status']!='completed':raise ValueError('worker成功标记与原始状态不同')
    if task['kind']=='train':
        from .detector_audit import audit_training_matrix
        if result.get('source_fingerprint_sha256')!=expected_source:raise ValueError('训练源码指纹改变')
        p=Path(attempt['directory'])/'单次训练复核输入.json'
        if not p.exists():write_json(p,{'runs':[result]})
        elif read_json(p)!={'runs':[result]}:raise ValueError('已完成训练审计输入改变')
        audit=audit_training_matrix(load_config(cfg_path),p)
        if len(audit['runs'])!=1 or audit['runs'][0]['status']!='passed':raise ValueError('完整训练预算复核失败')
    else:
        if sha256(result['manifest'])!=result['manifest_sha256']:raise ValueError('评测清单已改变')
        for entry in read_json(result['manifest']):
            if sha256(Path(result['directory'])/entry['path'])!=entry['sha256']:raise ValueError('已完成评测文件SHA改变')
    return result


def execute_worker(task_file,log):
    task=read_json(task_file)
    command=[sys.executable,str(PROJECT_ROOT/'scripts/run_n17_server.py'),'--worker-task',str(task_file)]
    with Path(log).open('wb') as stream:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=0,
                                 env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONUTF8':'1',
                                      'CUDA_VISIBLE_DEVICES':task['worker_cuda_visible_devices']})
        try:
            for chunk in iter(lambda:process.stdout.read(8192),b''):
                stream.write(chunk);stream.flush()
                if getattr(sys.stdout,'buffer',None):sys.stdout.buffer.write(chunk);sys.stdout.buffer.flush()
                else:sys.stdout.write(chunk.decode('utf-8',errors='replace'));sys.stdout.flush()
            return process.wait()
        except BaseException:
            process.terminate()
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            raise


def training_matrix(job,state,policy):
    runs=[read_json(Path(t['attempts'][-1]['directory'])/'worker_result.json') for t in state['tasks']
          if t['policy']==policy and t['kind']=='train' and t['status']=='completed']
    expected=len(load_config(state['configs'][policy])['models'])*(1 if state['mode']=='pilot' else 3)
    if len(runs)!=expected:raise ValueError('训练任务不完整，不能开始评测')
    p=job/'配方'/policy/'训练矩阵.json'
    write_json(p,{'run_id':job.name+'_'+policy,'status':'completed','development_only':state['mode']=='pilot',
                  'runs':runs,'config_sha256':runs[0]['config_sha256'],'data_index_sha256':runs[0]['data_index_sha256']})
    return p


def final_alias(output,job):
    alias=output/'01_最终结果'/('last_'+job.name)
    alias.parent.mkdir(parents=True,exist_ok=True)
    if os.path.lexists(alias):
        if not os.path.samefile(alias,job):raise FileExistsError(alias)
        return alias
    if os.name=='nt':
        import base64
        quote=lambda p:"'"+str(p).replace("'","''")+"'"
        command="$ErrorActionPreference='Stop'; New-Item -ItemType Junction -Path "+quote(alias)+' -Target '+quote(job)+' | Out-Null'
        encoded=base64.b64encode(command.encode('utf-16-le')).decode()
        shell=shutil.which('pwsh') or shutil.which('powershell')
        if not shell:raise RuntimeError('缺少PowerShell，不能建立last目录联接')
        subprocess.run([shell,'-NoProfile','-NonInteractive','-EncodedCommand',encoded],check=True,capture_output=True)
    else:alias.symlink_to(job,target_is_directory=True)
    if not os.path.samefile(alias,job):raise ValueError('last入口目标不同')
    return alias


def finish_job(job,state,settings):
    results={t['policy']:read_json(Path(t['attempts'][-1]['directory'])/'worker_result.json')
             for t in state['tasks'] if t['kind']=='evaluate'}
    if state['mode']=='pilot':
        values={p:read_json(Path(r['directory'])/'比较矩阵.json')['runs'] for p,r in results.items()}
        paired=[]
        for family in (r['family'] for r in values['baseline']):
            before=next(r for r in values['baseline'] if r['family']==family)
            after=next(r for r in values['person_alias'] if r['family']==family)
            paired.append({'family':family,'baseline_pedestrian_recall':before['spd_recall']['pedestrian'],
                           'person_alias_pedestrian_recall':after['spd_recall']['pedestrian'],
                           'baseline_recall':before['spd_recall'],'person_alias_recall':after['spd_recall'],
                           'baseline_carla_fp':before['carla_empty_frame_fp'],'person_alias_carla_fp':after['carla_empty_frame_fp'],
                           'baseline_p95_ms':before['detector_p95_ms'],'person_alias_p95_ms':after['detector_p95_ms']})
        write_json(job/'开发配方对照.json',{'development_only':True,'formal_selection_allowed':False,'rows':paired})
        lines=['# N17服务器开发对照\n','单seed开发诊断，不替代正式三seed，不选部署模型。',
               '| 家族 | baseline行人召回 | person_alias行人召回 |','| --- | --- | --- |']
        lines += [f"| {r['family']} | {r['baseline_pedestrian_recall']:.2%} | {r['person_alias_pedestrian_recall']:.2%} |" for r in paired]
        lines.append('请同时核对全部类别、误检和延迟；本次只检验迁移因子，没有验证类别重采样或其他配方。')
    else:
        evaluation=Path(next(iter(results.values()))['directory']);selection=read_json(evaluation/'离线预选.json')
        lines=['# N17服务器完整比较\n',f"离线预选：{selection['status']}；选中：{selection['selected_candidate']}。",
               '完整声明尺寸矩阵、三seed与预训练/微调权重统一验证，数据/配置/权重/源码/设备均独立留证。',
               '这不是本机旧5060Ti结果，也不表示用户验收、定位或闭环已通过。',f'评测目录：`{evaluation}`']
    lines += ['', '所有原始CSV/JSON、逐步/逐轮指标、中文PNG/SVG、失败和权重都保留。',
              'N15遮挡代理/人工抽查/同源限制、两个零GT类、有限预算及负帧分母限制继续适用；N18未执行。']
    (job/'结果导航.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    state.update(status='completed',finished_at=now())
    write_json(job/'任务状态.json',state)
    alias=final_alias(settings['output_root'],job)
    write_json(job/'最终入口.json',{'physical':str(job),'last':str(alias),'summary_sha256':sha256(job/'结果导航.md'),
               'development_only':state['mode']=='pilot','user_acceptance':'pending'})
    if state['mode']=='pilot' and settings.get('model_sizes'):
        from .detector_timing import timing_budget
        timing_budget(settings,job,job/'正式训练耗时外推')


def run_job(settings,mode,resume=None,dry_run=False):
    if mode not in ('pilot','formal'):raise ValueError('未知训练动作')
    with execution_lock(settings['output_root']/f'.gpu_{settings["device"]}.lock'):
        rows,environment,checkdir=preflight(settings)
        source=environment['source']['source_fingerprint_sha256']
        if resume:
            job=Path(resume).resolve()
            if not job.is_relative_to(settings['output_root']/'02_运行产物'):raise ValueError('续跑目录不在配置输出根内')
            state=read_json(job/'任务状态.json')
            if (state['configuration_sha256']!=settings['_sha256'] or state['mode']!=mode
                    or state['source_fingerprint_sha256']!=source
                    or state['input_manifest_sha256']!=environment['input_verification']['manifest_sha256']
                    or state['gpu_uuid']!=environment['gpu']['uuid']):
                raise ValueError('配置/源码/输入/GPU身份改变，不能混入旧任务；请新建run')
            policies=list(POLICIES) if mode=='pilot' else [settings['formal_policy']]
            base=server_base(settings);expected=plan_tasks(mode,policies,base['models'],base['seeds'])
            fields=('id','kind','policy','family','seed')
            if [{k:t.get(k) for k in fields} for t in state['tasks']] != [{k:t.get(k) for k in fields} for t in expected]:
                raise ValueError('旧任务清单被删改')
            if state['status']=='completed':
                for task in state['tasks']:
                    p=state['configs'][task['policy']]
                    if sha256(p)!=state['config_hashes'][task['policy']]:raise ValueError('完成配置已改变')
                    validate_completed(task,p,source)
                final_alias(settings['output_root'],job)
                print('已完成任务仅复核，不改原结果或重跑：'+str(job),flush=True)
                return job
        else:
            job=settings['output_root']/'02_运行产物'/('n17_server_'+mode+'_'+uuid.uuid4().hex);job.mkdir(parents=True)
            snapshot=stage_snapshot(settings['input_root'],rows,job/'冻结数据元信息')
            policies=list(POLICIES) if mode=='pilot' else [settings['formal_policy']]
            configs={}
            base=server_base(settings)
            for policy in policies:
                cfg={k:v for k,v in copy.deepcopy(base).items() if not k.startswith('_')}
                cfg.update(output_root=str(job/'配方'/policy),weights_root=str(settings['weights_root']),
                           dataset_snapshot=snapshot,device=0,font_path=settings['font_path'],
                           class_name_aliases=POLICIES[policy])
                cfg['training'].update(epochs=settings['pilot_epochs'] if mode=='pilot' else settings['formal_epochs'],
                                       batch=settings['batch'],nbs=settings['batch'],workers=settings['workers'],cls_remap=True,
                                       amp=settings['amp'])
                cfg['checkpoint_period']=settings['checkpoint_period'];cfg['record_epoch_timing']=bool(settings.get('model_sizes'))
                if mode=='formal' and settings['early_stopping_patience']:
                    cfg['budget_policy']={'criterion':'native_fitness','minimum_epochs':settings['minimum_epochs'],
                                          'patience':settings['early_stopping_patience']}
                    cfg['training']['patience']=settings['early_stopping_patience']
                p=job/'配置'/(policy+'.yaml');p.parent.mkdir(exist_ok=True)
                p.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8');configs[policy]=str(p)
            state={'schema_version':'n17-server-job-1','mode':mode,'run_id':job.name,'status':'running','started_at':now(),
                   'configuration_sha256':settings['_sha256'],'source_fingerprint_sha256':source,
                   'source_commit':environment['source']['source_commit'],'gpu_uuid':environment['gpu']['uuid'],
                   'device':settings['device'],
                   'worker_cuda_visible_devices':environment['worker_cuda_visible_devices'],
                   'input_manifest_sha256':environment['input_verification']['manifest_sha256'],
                   'original_index_sha256':environment['input_verification']['source_index_sha256'],
                   'environment_report':str(checkdir/'服务器预检.json'),'configs':configs,
                   'config_hashes':{p:sha256(c) for p,c in configs.items()},
                   'tasks':plan_tasks(mode,policies,base['models'],base['seeds'])}
        if dry_run:
            from ultralytics.cfg import get_cfg
            from ultralytics.data.utils import check_det_dataset
            parsed=[]
            for policy,path in state['configs'].items():
                cfg=load_config(path);get_cfg(overrides=cfg['training'])
                data=check_det_dataset(str(Path(cfg['dataset_snapshot']['root'])/'dataset.yaml'),autodownload=False)
                if list(data['names'].values())!=cfg['classes']:raise ValueError('框架解析后的类别顺序改变')
                parsed.append(policy)
            state['status']='planned';write_json(job/'任务状态.json',state)
            write_json(job/'计划预检.json',{'status':'passed','training_started':False,'native_parameters_checked':parsed,
                       'training_tasks':sum(t['kind']=='train' for t in state['tasks']),
                       'mode':mode,'source_fingerprint_sha256':source})
            print(f'仅准备并核验任务计划，未训练：{job}',flush=True)
            return job
        state['status']='running';write_json(job/'任务状态.json',state)
        try:
            for task in state['tasks']:
                cfg_path=state['configs'][task['policy']]
                if sha256(cfg_path)!=state['config_hashes'][task['policy']]:raise ValueError('冻结的配方配置已修改')
                if task['status']=='completed':
                    validate_completed(task,cfg_path,source);print('已核验并跳过完成任务：'+task['id'],flush=True);continue
                attempt=job/'任务'/task['id']/('attempt_'+str(len(task['attempts'])+1));attempt.mkdir(parents=True)
                record={'directory':str(attempt),'status':'running','started_at':now()};task['attempts'].append(record);task['status']='running'
                params={**task,'mode':mode,'config':cfg_path,'attempt':str(attempt),
                        'worker_cuda_visible_devices':state['worker_cuda_visible_devices'],
                        'expected_gpu_uuid':state['gpu_uuid']}
                if task['kind']=='evaluate':params['training_matrix']=str(training_matrix(job,state,task['policy']))
                write_json(attempt/'任务参数.json',params);write_json(job/'任务状态.json',state)
                print(f'开始 {task["id"]}，控制台日志：{attempt/"控制台.txt"}',flush=True)
                record['returncode']=execute_worker(attempt/'任务参数.json',attempt/'控制台.txt')
                if record['returncode']!=0:raise RuntimeError(f'worker失败：{task["id"]}；保留日志后可任务级续跑')
                validate_completed(task,cfg_path,source)
                task['status']=record['status']='completed';record['finished_at']=now();write_json(job/'任务状态.json',state)
            _,verification=verify_input(settings['input_root'])
            write_json(job/'运行后输入完整性.json',verification)
            finish_job(job,state,settings)
            print(f'任务完成：{job/"结果导航.md"}',flush=True)
            return job
        except BaseException as exc:
            for task in state['tasks']:
                if task['status']=='running':
                    task['status']='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed'
                    task['attempts'][-1].update(status=task['status'],finished_at=now(),error=repr(exc))
            state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc),finished_at=now())
            write_json(job/'任务状态.json',state);print(f'未完成任务保留在：{job}',flush=True);raise


def collect_results(settings,run):
    job=Path(run).resolve()
    if not job.is_relative_to(settings['output_root']/'02_运行产物'):raise ValueError('只能回传配置输出目录内的任务')
    state=read_json(job/'任务状态.json')
    with execution_lock(settings['output_root']/f'.gpu_{state["device"]}.lock'):
        files=[]
        for current,dirs,names in os.walk(job,followlinks=False):
            for name in dirs:
                p=Path(current)/name
                if p.is_symlink() or getattr(p.lstat(),'st_reparse_tag',0):raise ValueError('结果中含目录链接，拒绝递归打包')
            files.extend(Path(current)/name for name in names if name!='回传文件清单.json')
        for p in files:
            if p.is_symlink() or not p.resolve().is_relative_to(job):raise ValueError('结果目录含越界文件链接')
        index={'status_at_collection':state['status'],'complete':state['status']=='completed','run_id':job.name,
               'source_commit':state['source_commit'],'files':[{'path':p.relative_to(job).as_posix(),'bytes':p.stat().st_size,
               'sha256':sha256(p)} for p in sorted(files)],'input_rgb_included':False,'created_at':now()}
        write_json(job/'回传文件清单.json',index)
        target=settings['output_root']/'回传包'/('N17服务器结果_'+uuid.uuid4().hex+'.zip');target.parent.mkdir(exist_ok=True)
        temporary=target.with_suffix('.zip.tmp')
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            for p in files+[job/'回传文件清单.json']:z.write(p,job.name+'/'+p.relative_to(job).as_posix())
        os.replace(temporary,target)
        with zipfile.ZipFile(target) as z:
            if z.testzip() is not None:raise ValueError('回传ZIP完整性检查失败')
        target.with_suffix('.zip.sha256').write_text(sha256(target)+'  '+target.name+'\n',encoding='utf-8')
        print(f'回传包：{target}',flush=True)
        return target
