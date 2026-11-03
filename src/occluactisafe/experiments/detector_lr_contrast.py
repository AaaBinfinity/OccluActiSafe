"""N17三族n/seed17/两学习率：安全排队、六任务上限及原协议统一评测。"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import yaml

from occluactisafe.common.config import PROJECT_ROOT
from .detector_controlled_trial import MODELS, LEARNING_RATES, STOP_EPOCH, validate_trial, audit_controlled_training
from .detector_data import load_config
from .detector_local import build_local_config, digest
from .detector_models import expanded_models
from .detector_portable import stage_snapshot
from .detector_server import execution_lock, preflight, final_alias
from .run_store import read_json, write_json, sha256, now, manifest
from .source_version import source_info

DEFAULT_OPTIONS = PROJECT_ROOT/'configs/experiments/n17_lr_contrast.py'
APPROVAL = 'n17-lr-6x10-20261103'
PROTOCOL = {'approval': APPROVAL, 'models': list(MODELS), 'seed': 17,
            'learning_rates': list(LEARNING_RATES), 'tasks': 6, 'schedule_epochs': 300,
            'minimum_epochs': 50, 'patience': 30, 'stop_after_epoch': 10,
            'train_frames': 17003, 'validation_frames': 3185, 'planned_image_exposures': 1020180,
            'imgsz': 960, 'batch': 2, 'nbs': 2, 'workers': 0, 'amp': True,
            'optimizer': 'SGD', 'warmup_epochs': 0.0, 'treatment_variable': 'training.lr0',
            'evaluation': {'imgsz': 960, 'batch': 1, 'confidence_keep': .01,
                           'confidence_operating': .25, 'half': False, 'warmup_frames': 20}}


def options(path=DEFAULT_OPTIONS, action=None, run_dir=None):
    path = Path(path); path = (path if path.is_absolute() else PROJECT_ROOT/path).resolve()
    raw = path.read_bytes(); scope = {'__name__': 'n17_lr_options', '__file__': str(path)}
    exec(compile(raw, str(path), 'exec'), scope)
    keys = ('ACTION', 'INPUT_ROOT', 'OUTPUT_ROOT', 'WAIT_FOR_RUN', 'RUN_DIR', 'WAIT_TIMEOUT_HOURS', 'DEVICE')
    cfg = {k: scope[k] for k in keys}
    if action is not None: cfg['ACTION'] = action
    if run_dir is not None: cfg['RUN_DIR'] = str(run_dir)
    if cfg['ACTION'] not in ('check', 'plan', 'run'):
        raise ValueError('对照入口仅支持check/plan/run')
    if type(cfg['DEVICE']) is not int or cfg['DEVICE'] < 0:
        raise ValueError('DEVICE须为非负整数')
    if type(cfg['WAIT_TIMEOUT_HOURS']) is not int or not 1 <= cfg['WAIT_TIMEOUT_HOURS'] <= 48:
        raise ValueError('等待上限须为1～48小时')
    for key in ('INPUT_ROOT', 'OUTPUT_ROOT', 'WAIT_FOR_RUN', 'RUN_DIR'):
        if cfg[key]:
            p = Path(cfg[key]).expanduser(); cfg[key] = (p if p.is_absolute() else PROJECT_ROOT/p).resolve()
    if not cfg['WAIT_FOR_RUN'] or not cfg['WAIT_FOR_RUN'].is_relative_to(cfg['OUTPUT_ROOT']):
        raise ValueError('必须明确指定本节点待正常结束的既有run')
    if (cfg['INPUT_ROOT'].is_relative_to(PROJECT_ROOT) or cfg['OUTPUT_ROOT'].is_relative_to(PROJECT_ROOT)
            or cfg['INPUT_ROOT'].is_relative_to(cfg['OUTPUT_ROOT']) or cfg['OUTPUT_ROOT'].is_relative_to(cfg['INPUT_ROOT'])):
        raise ValueError('输入/输出必须在仓库外且互不嵌套')
    if cfg['RUN_DIR'] and not cfg['RUN_DIR'].is_relative_to(cfg['OUTPUT_ROOT']):
        raise ValueError('只能接续本节点的既有对照计划')
    return {**cfg, '_path': str(path), '_sha256': __import__('hashlib').sha256(raw).hexdigest(), '_bytes': raw}


def prior_run_ready(path):
    record = read_json(Path(path)/'本机训练任务.json')
    if record['status'] in ('running', 'planned'):
        return False
    if (record['status'] != 'completed' or record.get('worker_returncode') != 0
            or record.get('audit', {}).get('status') != 'passed'):
        raise RuntimeError('待等待的用户任务未正常完成；保留现场，不启动新GPU任务')
    return True


def tasks():
    return [{'id': model+'_lr_'+str(lr).replace('.', 'p'), 'family': model, 'seed': 17,
             'learning_rate': lr, 'status': 'planned'} for model in MODELS for lr in LEARNING_RATES]


def pointer_path(cfg):
    return cfg['OUTPUT_ROOT']/'03_验收证据/n17_lr_contrast_delivery_20261103/已批准对照指针.json'


def public_options(cfg):
    return {k: str(v) if isinstance(v, Path) else v for k, v in cfg.items() if not k.startswith('_')}


def prepare_study(cfg):
    pointer = pointer_path(cfg)
    job = cfg['RUN_DIR'] or (Path(read_json(pointer)['directory']) if pointer.exists() else None)
    source = source_info()
    if job:
        if not job.resolve().is_relative_to(cfg['OUTPUT_ROOT']): raise ValueError('对照指针越界')
        state = read_json(job/'对照任务.json')
        if (state['protocol'] != PROTOCOL or state['source_fingerprint_sha256'] != source['source_fingerprint_sha256']
                or state['input_root'] != str(cfg['INPUT_ROOT']) or state['output_root'] != str(cfg['OUTPUT_ROOT'])
                or state['wait_for_run'] != str(cfg['WAIT_FOR_RUN']) or state['device'] != cfg['DEVICE']
                or [(t['family'], t['seed'], t['learning_rate']) for t in state['tasks']]
                   != [(t['family'], t['seed'], t['learning_rate']) for t in tasks()]):
            raise ValueError('既有六任务计划的源码/输入/设备/配方改变，拒绝重开或混用')
        return job, state
    job = cfg['OUTPUT_ROOT']/'04_开发与复核/学习率对照'/('n17_lr_contrast_'+uuid.uuid4().hex)
    job.mkdir(parents=True, exist_ok=False)
    state = {'run_id': job.name, 'status': 'planned', 'created_at': now(), 'protocol': PROTOCOL,
             'protocol_sha256': digest(PROTOCOL), 'source_commit': source['source_commit'],
             'source_fingerprint_sha256': source['source_fingerprint_sha256'],
             'input_root': str(cfg['INPUT_ROOT']), 'output_root': str(cfg['OUTPUT_ROOT']),
             'wait_for_run': str(cfg['WAIT_FOR_RUN']), 'device': cfg['DEVICE'], 'tasks': tasks(),
             'full_training_complete': False, 'formal_three_seed_comparison_complete': False,
             'selected_deployment_model': None}
    (job/'入口选项.py').write_bytes(cfg['_bytes'])
    write_json(job/'计划入口选项.json', public_options(cfg))
    write_json(job/'对照任务.json', state)
    write_json(pointer, {'approval': APPROVAL, 'directory': str(job), 'protocol_sha256': digest(PROTOCOL)})
    return job, state


def local_opts(cfg, family):
    model = next(m for m in expanded_models(['n']) if m['family'] == family)
    return {'ACTION': 'train', 'MODEL': family, 'SEED': 17, 'INPUT_ROOT': cfg['INPUT_ROOT'],
            'OUTPUT_ROOT': cfg['OUTPUT_ROOT'], 'DEVICE': cfg['DEVICE'], 'BATCH': 2, 'WORKERS': 0,
            'AMP': True, 'MAX_EPOCHS': 300, 'MIN_EPOCHS': 50, 'PATIENCE': 30,
            'PAUSE_AFTER_EPOCH': None, 'PLOT_POINTS_PER_PAGE': 20000, 'model': model}


def trial_config(cfg, task, environment, snapshot, directory):
    result = build_local_config(local_opts(cfg, task['family']), environment, snapshot, directory)
    result['models'] = expanded_models(['n']); result['model_sizes'] = ['n']; result['seeds'] = [17]
    result['training']['lr0'] = task['learning_rate']
    result['local_profile']['controlled_trial'] = {'protocol': 'n17-lr-6x10-v1',
        'stop_after_epoch': 10, 'learning_rate': task['learning_rate'], 'trial_id': task['id']}
    result['local_profile']['recipe_sha256'] = digest({'training': result['training'],
        'budget_policy': result['budget_policy'], 'classes': result['classes'],
        'aliases': result['class_name_aliases'], 'index_sha256': snapshot['index_sha256'],
        'controlled_protocol': APPROVAL})
    validate_trial(result, task['family'], 17)
    return result


def verify_fair_configs(configs):
    if len(configs) != 6: raise ValueError('对照必须完整包含六个配置')
    seen = set(); reference = None
    for cfg in configs:
        profile = cfg['local_profile']; model = profile['model_id']; lr = cfg['training']['lr0']
        validate_trial(cfg, model, profile['seed'])
        key = model, lr
        if key in seen: raise ValueError('对照配置重复')
        seen.add(key)
        frozen = load_config()
        if any(cfg[k] != frozen[k] for k in ('classes', 'evaluation', 'gates')):
            raise ValueError('对照改变了原类别、统一评测或准入门槛')
        shared = copy.deepcopy({k: cfg[k] for k in ('training', 'budget_policy', 'dataset_snapshot',
                               'classes', 'class_name_aliases', 'evaluation', 'gates')})
        shared['source_fingerprint_sha256'] = profile['source_fingerprint_sha256']
        shared['training'].pop('lr0')
        if reference is None: reference = shared
        if shared != reference: raise ValueError('六项配置除学习率外存在不公平差异')
    if seen != {(m, lr) for m in MODELS for lr in LEARNING_RATES}: raise ValueError('缺少已批准任务')
    return digest(reference)


def freeze_configs(cfg, job, state, rows, environment):
    snapshot = stage_snapshot(cfg['INPUT_ROOT'], rows, job/'冻结数据元信息')
    configs = []
    for task in state['tasks']:
        directory = job/'任务'/task['id']; directory.mkdir(parents=True, exist_ok=False)
        data = trial_config(cfg, task, environment, snapshot, directory)
        path = directory/'冻结训练配置.yaml'
        path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding='utf-8')
        task.update(directory=str(directory), config=str(path), config_sha256=sha256(path))
        configs.append(data)
    common = copy.deepcopy(configs[0]); common.pop('local_profile')
    common['output_root'] = str(job/'统一评测管理')
    path = job/'冻结统一评测配置.yaml'
    path.write_text(yaml.safe_dump(common, allow_unicode=True, sort_keys=False), encoding='utf-8')
    state.update(configs_frozen=True, shared_recipe_sha256=verify_fair_configs(configs),
                 evaluation_config=str(path), evaluation_config_sha256=sha256(path),
                 input_verification=environment['input_verification'], gpu_uuid=environment['gpu']['uuid'],
                 worker_cuda_visible_devices=environment['worker_cuda_visible_devices'])
    write_json(job/'配方公平性核验.json', {'status': 'passed', 'tasks': 6,
               'only_training_difference': 'lr0', 'shared_recipe_sha256': state['shared_recipe_sha256']})
    write_json(job/'对照任务.json', state)


def launch(task_path, log, visible):
    command = [sys.executable, str(PROJECT_ROOT/'scripts/run_n17_lr_contrast.py'), '--worker-task', str(task_path)]
    with Path(log).open('xb') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1', 'CUDA_VISIBLE_DEVICES': visible})
    return result.returncode


def verify_completed_study(job, state):
    if (not state.get('development_comparison_complete') or len(state['tasks']) != 6
            or any(t['status'] != 'completed' for t in state['tasks'])):
        raise ValueError('完成标记与六项实际任务状态不一致')
    for item in read_json(job/'完成产物清单.json'):
        file = (job/item['path']).resolve()
        if not file.is_relative_to(job.resolve()) or not file.is_file() or sha256(file) != item['sha256']:
            raise ValueError('已完成对照产物缺失、越界或SHA改变')


def worker(path):
    task = read_json(path)
    if task['kind'] not in ('train', 'evaluate'):
        raise ValueError('未知对照worker任务')
    if task['kind'] == 'train':
        config = load_config(task['config'])
        validate_trial(config, task['family'], task['seed'])
        from .detector_server import worker as training_worker
        training_worker(path)
        return
    directory = Path(task['attempt'])
    try:
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('统一评测必须CUDA，不回退CPU')
        if str(getattr(torch.cuda.get_device_properties(0), 'uuid', 'unknown')) != task['expected_gpu_uuid']:
            raise RuntimeError('统一评测GPU身份改变')
        if source_info()['source_fingerprint_sha256'] != task['source_fingerprint_sha256']:
            raise ValueError('对照源码已变化')
        if sha256(task['weights']) != task['weights_sha256']: raise ValueError('待评测best权重SHA改变')
        cfg = load_config(task['config'])
        for key, value in PROTOCOL['evaluation'].items():
            if cfg['evaluation'][key] != value: raise ValueError('统一评测协议改变：'+key)
        from .detector_evaluation import evaluate_one
        model = next(m for m in cfg['models'] if m['family'] == task['family'])
        result = evaluate_one(cfg, model, task['weights'], 'finetuned', 17, Path(task['evaluation_directory']))
        if result['status'] != 'completed': raise RuntimeError('统一评测未完整完成')
        write_json(directory/'worker_result.json', result)
    except BaseException as exc:
        write_json(directory/'worker_failure.json', {'status': 'failed', 'error': repr(exc), 'time': now()})
        raise


def run_tasks(cfg, job, state):
    for task in state['tasks']:
        directory = Path(task['directory']); training_parent = directory/task['family']/'seed_17'
        training = training_parent/'训练'
        if sha256(task['config']) != task['config_sha256']: raise ValueError('训练配置SHA改变')
        if task['status'] == 'completed':
            audit_controlled_training(load_config(task['config']), training)
            if sha256(task['evaluation_result_path']) != task['evaluation_result_sha256']:
                raise ValueError('已有评测摘要SHA改变')
            continue
        if task['status'] not in ('planned', 'trained'):
            raise RuntimeError('存在未完成/失败尝试，保留现场，不自动重跑增加预算')
        if task['status'] == 'planned':
            train_task = directory/'训练任务参数.json'
            write_json(train_task, {'kind': 'train', 'family': task['family'], 'seed': 17,
                'config': task['config'], 'attempt': str(directory), 'training_directory': str(training_parent),
                'expected_gpu_uuid': state['gpu_uuid'], 'worker_cuda_visible_devices': state['worker_cuda_visible_devices']})
            task.update(status='training', started_at=now()); write_json(job/'对照任务.json', state)
            print('开始10轮受控对照：'+task['id'], flush=True)
            code = launch(train_task, directory/'训练控制台.txt', state['worker_cuda_visible_devices'])
            task['training_returncode'] = code
            if code: raise RuntimeError('训练worker失败，保留日志/权重，不增加任务或伪造暂停')
            result = read_json(directory/'worker_result.json')
            if result['status'] != 'paused_at_declared_epoch': raise ValueError('训练没有按批准边界暂停')
            audit = audit_controlled_training(load_config(task['config']), training)
            write_json(directory/'受控训练核验.json', audit)
            task.update(status='trained', training_audit=str(directory/'受控训练核验.json'),
                        best=str(training/'weights/best.pt'), best_sha256=audit['weights']['best_sha256'])
            write_json(job/'对照任务.json', state)
        audit_controlled_training(load_config(task['config']), training)
        evaluation = directory/'统一评测'; attempt = directory/'评测执行'; attempt.mkdir(exist_ok=False)
        evaluation_task = directory/'评测任务参数.json'
        write_json(evaluation_task, {'kind': 'evaluate', 'family': task['family'], 'seed': 17,
            'config': state['evaluation_config'], 'weights': task['best'], 'weights_sha256': task['best_sha256'],
            'attempt': str(attempt), 'evaluation_directory': str(evaluation), 'expected_gpu_uuid': state['gpu_uuid'],
            'source_fingerprint_sha256': state['source_fingerprint_sha256']})
        task.update(status='evaluating'); write_json(job/'对照任务.json', state)
        print('统一FP32/ByteTrack评测：'+task['id'], flush=True)
        code = launch(evaluation_task, directory/'统一评测控制台.txt', state['worker_cuda_visible_devices'])
        task['evaluation_returncode'] = code
        if code: raise RuntimeError('统一评测失败，不把缺项计通过')
        result_path = attempt/'worker_result.json'; result = read_json(result_path)
        if (result['status'] != 'completed' or result['weight_sha256'] != task['best_sha256']
                or result['config_sha256'] != state['evaluation_config_sha256']
                or result['source_fingerprint_sha256'] != state['source_fingerprint_sha256']
                or result['actual_frames'] != 3185): raise ValueError('评测权重/配置/源码/帧分母不一致')
        task.update(status='completed', finished_at=now(), evaluation_result_path=str(result_path),
                    evaluation_result_sha256=sha256(result_path))
        write_json(job/'对照任务.json', state)


def summarize(job, state):
    import csv
    rows = []; full = []
    for task in state['tasks']:
        if task['status'] != 'completed': raise ValueError('对照任务不完整')
        result = read_json(task['evaluation_result_path'])
        full.append({'trial_id': task['id'], 'learning_rate': task['learning_rate'], 'result': result})
        for domain, metrics in result['domains'].items():
            rows.append({'family': task['family'], 'seed': 17, 'learning_rate': task['learning_rate'],
                         'domain': domain, 'frames': metrics['frames'], 'gt': metrics['ground_truth_count'],
                         'map50': metrics['map50'], 'map50_95': metrics['map50_95'],
                         **{c+'_recall': metrics['by_class'].get(c, {}).get('recall') for c in ('vehicle','pedestrian','cyclist')},
                         'negative_frames': metrics['negative_frames'],
                         'negative_frame_fp': metrics['false_positives_in_negative_frames'],
                         'negative_fp_per_frame': metrics['false_positives_per_empty_frame'],
                         'idf1': metrics['tracking']['idf1'],
                         'detector_p95_ms': metrics['latency']['detector_ms']['p95'],
                         'detection_tracking_p95_ms': metrics['latency']['detection_tracking_ms']['p95'],
                         'best_sha256': task['best_sha256']})
    with (job/'双域逐类与资源对照.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(job/'六任务统一评测原值.json', {'development_only': True, 'tasks': full})
    differences = []
    for model in MODELS:
        for domain in ('SPD', 'CARLA'):
            baseline = next(r for r in rows if r['family'] == model and r['domain'] == domain and r['learning_rate'] == .001)
            candidate = next(r for r in rows if r['family'] == model and r['domain'] == domain and r['learning_rate'] == .0003)
            differences.append({'family': model, 'domain': domain, 'seed': 17,
                **{key+'_difference_pp': 100*(candidate[key]-baseline[key])
                   for key in ('vehicle_recall','pedestrian_recall','cyclist_recall','map50_95')},
                'detector_p95_difference_ms': candidate['detector_p95_ms']-baseline['detector_p95_ms']})
    write_json(job/'配对差值与适用范围.json', {'direction':'lr0=.0003减去lr0=.001', 'differences':differences,
               'statistical_significance_claimed':False,'formal_three_seed_complete':False,
               'selected_deployment_model':None,'note':'单seed开发对照；负差值及门槛失败均保留，不默认低学习率获胜'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .visual_fonts import configure_plot_fonts
    configure_plot_fonts(); figures = []
    for domain in ('SPD', 'CARLA'):
        fig, axes = plt.subplots(3, 1, figsize=(11, 10))
        subset = [r for r in rows if r['domain'] == domain]
        labels = [r['family']+'\nlr='+str(r['learning_rate']) for r in subset]
        for ax, key, name in zip(axes, ('vehicle_recall','pedestrian_recall','cyclist_recall'), ('车辆','行人','骑行者')):
            ax.bar(labels, [100*r[key] for r in subset]); ax.set(ylim=(0,100), ylabel=name+'召回/%')
            ax.grid(axis='y', alpha=.2)
        fig.suptitle(domain+'学习率开发对照｜固定0.25/IoU0.5｜单seed，不代表正式选型')
        fig.tight_layout()
        for ext in ('png','svg'):
            file = job/(domain+'分类召回对照.'+ext); fig.savefig(file, dpi=180)
            figures.append({'file': file.name, 'sha256': sha256(file)})
        plt.close(fig)
    write_json(job/'对照图表来源.json', {'raw_csv_sha256': sha256(job/'双域逐类与资源对照.csv'),
               'source_result_sha256': [t['evaluation_result_sha256'] for t in state['tasks']],
               'smoothing': False, 'figures': figures})
    links = '\n'.join('- ['+t['id']+'](任务/'+t['id']+'/)' for t in state['tasks'])
    (job/'结果导航.md').write_text('# N17六项学习率开发对照\n\n六项各10轮受控暂停，保留300轮学习率计划；单seed开发对照，不是正式三seed或部署选型。\n\n'+links+'\n\n原始双域/分类/跟踪/资源见双域逐类与资源对照.csv，全部失败和过程图保留。\n', encoding='utf-8')


def progress(cfg):
    pointer = pointer_path(cfg)
    if not pointer.exists(): return None
    job = Path(read_json(pointer)['directory'])
    if not job.resolve().is_relative_to(cfg['OUTPUT_ROOT']): raise ValueError('对照进度指针越界')
    state = read_json(job/'对照任务.json')
    output = {'run_id':job.name,'directory':str(job),'status':state['status'],
              'completed_tasks':sum(t['status']=='completed' for t in state['tasks']),'total_tasks':6,'tasks':[]}
    for task in state['tasks']:
        item = {k:task[k] for k in ('id','status','learning_rate')}
        if task.get('directory'):
            result = Path(task['directory'])/task['family']/'seed_17/训练/results.csv'
            if result.is_file():
                text = result.read_text(encoding='utf-8'); lines = text.splitlines()
                if not text.endswith('\n'): lines = lines[:-1]
                if len(lines)>1: item['last_completed_epoch']=int(lines[-1].split(',')[0])
        output['tasks'].append(item)
    return output


def run(cfg):
    # 独立队列锁不占用当前GPU锁；同一授权指针只建立一组任务。
    with execution_lock(cfg['OUTPUT_ROOT']/'.n17_lr_contrast_queue.lock'):
        job, state = prepare_study(cfg)
        if cfg['ACTION'] == 'plan':
            print('仅计划六项，未启动GPU：'+str(job), flush=True); return job
        if state['status'] == 'completed':
            verify_completed_study(job, state)
            print('本次授权六项已完成，不重复运行：'+str(job), flush=True); return job
        if any(t['status'] not in ('planned', 'trained', 'completed') for t in state['tasks']):
            raise RuntimeError('既有任务仍运行或失败；不启动额外GPU预检或重跑')
        state['execution_options'] = public_options(cfg)
        state['execution_requested_at'] = now()
        write_json(job/'对照任务.json', state)
        started = time.monotonic(); lock = None; acquired = False
        try:
            while not prior_run_ready(cfg['WAIT_FOR_RUN']):
                state.update(status='waiting_for_existing_run', waiting_since=state.get('waiting_since', now()),
                             wait_observed_at=now())
                write_json(job/'对照任务.json', state)
                if time.monotonic()-started > cfg['WAIT_TIMEOUT_HOURS']*3600:
                    raise TimeoutError('等待用户任务超过显式上限，未启动GPU')
                print('等待用户seed29正常完成，不占用GPU：'+str(job), flush=True)
                time.sleep(30)
            # 已完成状态可能刚写完，待原父进程释放锁；不删除锁文件。
            while True:
                lock = execution_lock(cfg['OUTPUT_ROOT']/f'.gpu_{cfg["DEVICE"]}.lock')
                try: lock.__enter__(); acquired = True; break
                except RuntimeError:
                    if time.monotonic()-started > cfg['WAIT_TIMEOUT_HOURS']*3600: raise TimeoutError('GPU锁等待超时')
                    time.sleep(30)
            if source_info()['source_fingerprint_sha256'] != state['source_fingerprint_sha256']:
                raise ValueError('排队期间源码改变，拒绝沿用旧被测身份')
            if not state.get('configs_frozen'):
                settings = {'action': 'check', 'input_root': cfg['INPUT_ROOT'], 'output_root': job/'输入预检',
                    'weights_root': cfg['INPUT_ROOT']/'扩展权重', 'device': cfg['DEVICE'], 'font_path': '',
                    'batch': 2, 'workers': 0, 'amp': True, 'minimum_free_gpu_mib': 4096,
                    'minimum_available_ram_mib': 2048, 'minimum_free_disk_gib': 30,
                    'expected_gpu_name': '5060 Ti', 'model_sizes': ['n'], '_path': cfg['_path'], '_sha256': cfg['_sha256']}
                rows, environment, evidence = preflight(settings, models=expanded_models(['n']))
                state['environment_report'] = str(evidence/'服务器预检.json')
                freeze_configs(cfg, job, state, rows, environment)
            if sha256(state['evaluation_config']) != state['evaluation_config_sha256']:
                raise ValueError('统一评测配置SHA改变')
            state.update(status='running'); write_json(job/'对照任务.json', state)
            run_tasks(cfg, job, state); summarize(job, state)
            state.update(status='completed', finished_at=now(), development_comparison_complete=True)
            write_json(job/'对照任务.json', state)
            write_json(job/'完成产物清单.json', manifest(job))
            final_alias(cfg['OUTPUT_ROOT'], job)
            print('六项开发对照与统一GPU评测完成：'+str(job), flush=True)
        except BaseException as exc:
            for task in state['tasks']:
                if task['status'] in ('training', 'evaluating'):
                    task.update(failed_phase=task['status'], status='failed', error=repr(exc), failed_at=now())
            state.update(status='failed', failed_at=now(), error=repr(exc))
            write_json(job/'对照任务.json', state)
            raise
        finally:
            if acquired: lock.__exit__(None, None, None)
        return job


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--options', type=Path, default=DEFAULT_OPTIONS)
    parser.add_argument('--action', choices=('check','plan','run'))
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--worker-task', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_task: worker(args.worker_task); return 0
    cfg = options(args.options, args.action, args.run_dir)
    if cfg['ACTION'] == 'check':
        if not cfg['INPUT_ROOT'].is_dir() or not cfg['OUTPUT_ROOT'].is_dir(): raise FileNotFoundError('移动硬盘输入/输出缺失')
        ready = prior_run_ready(cfg['WAIT_FOR_RUN'])
        print(json.dumps({'status':'passed', 'mode':'read_only_configuration_check', 'protocol':PROTOCOL,
                          'prior_run_ready':ready, 'training_started':False, 'gpu_kernels_launched':False,
                          'existing_study':progress(cfg)}, ensure_ascii=False))
        return 0
    run(cfg); return 0


if __name__ == '__main__': main()
