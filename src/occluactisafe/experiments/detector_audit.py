"""N17真实矩阵输入、预算、完整帧与权重审计。"""
import csv
import json
import math
import ast
import hashlib
import subprocess
from collections import Counter
from pathlib import Path

from occluactisafe.common.config import PROJECT_ROOT
from .detector_data import load_index
from .detector_matrix import validate_matrix
from .run_store import read_json, sha256
from .source_version import historical_source


def audit_training_matrix(cfg, matrix_path):
    matrix = read_json(matrix_path)
    rows, data = load_index(cfg)
    train_frames = sum(r['role'] == 'train' for r in rows)
    expected_steps = math.ceil(train_frames/cfg['training']['batch'])*cfg['training']['epochs']
    expected_images = train_frames*cfg['training']['epochs']
    expected = set(validate_matrix(cfg))
    seen = set()
    initial_weights = {}
    algorithm_hashes = {}
    common_algorithm = None
    source_fingerprints = set()
    checks = []
    for run in matrix['runs']:
        identity = run['family'], run['seed']
        if identity in seen or identity not in expected:
            raise ValueError('训练矩阵重复/未知候选身份')
        seen.add(identity)
        if run['status'] != 'completed' or run.get('returncode',0) != 0:
            checks.append({'family':run['family'],'seed':run['seed'],'status':'failed',
                           'worker_status':run['status'],'returncode':run.get('returncode')})
            continue
        if run['config_sha256'] != cfg['_config_sha256'] or run['data_index_sha256'] != data['index_sha256']:
            raise ValueError('训练候选数据/配置哈希不公平')
        if run.get('class_name_aliases', {}) != cfg.get('class_name_aliases', {}):
            raise ValueError('训练分类行迁移配方与冻结配置不同')
        if run.get('source_fingerprint_sha256'):
            source_fingerprints.add(run['source_fingerprint_sha256'])
            if len(source_fingerprints) > 1:
                raise ValueError('训练矩阵的受保护源码内容不同')
        previous=initial_weights.setdefault(run['family'],run['initial_weight_sha256'])
        if previous!=run['initial_weight_sha256']:
            raise ValueError('同一模型族的训练种子使用不同初始权重')
        if sha256(run['checkpoint']) != run['checkpoint_sha256']:
            raise ValueError('完成权重哈希改变')
        path=Path(run['directory'])/'训练'/'逐步优化指标.csv'
        actual_epochs = cfg['training']['epochs']
        termination = None
        if cfg.get('budget_policy'):
            from .detector_stopping import replay_stopping
            trace = [json.loads(line) for line in (path.parent/'早停判定.jsonl').read_text(encoding='utf-8').splitlines()]
            termination = replay_stopping(trace, cfg['budget_policy'], cfg['training']['epochs'])
            if run.get('termination') != termination or run.get('checkpoint_rule') != 'native_model_val_best_fitness':
                raise ValueError('训练停止记录或best检查点策略不同')
            if Path(run['checkpoint']).name != 'best.pt' or sha256(run['last_checkpoint']) != run['last_checkpoint_sha256']:
                raise ValueError('早停检查点选择或last权重SHA改变')
            epochs = [json.loads(line) for line in (path.parent/'逐轮验证指标.jsonl').read_text(encoding='utf-8').splitlines()]
            if (len(epochs) != len(trace) or any(e['epoch'] != t['epoch'] or e['native_fitness'] != t['fitness'] for e,t in zip(epochs,trace))):
                raise ValueError('早停fitness与逐轮真实验证记录不同')
            if run.get('amp_requested') != cfg['training']['amp'] or run.get('amp_actual') != cfg['training']['amp']:
                raise ValueError('实际精度与冻结AMP配置不同')
            actual_epochs = termination['actual_epochs']
        run_steps = math.ceil(train_frames/cfg['training']['batch'])*actual_epochs
        run_images = train_frames*actual_epochs
        with path.open(encoding='utf-8',newline='') as f:
            steps=list(csv.DictReader(f))
        if [int(r['step']) for r in steps] != list(range(1,run_steps+1)):
            raise ValueError('实际优化步号/步数不符合完整冻结预算')
        exposures=sum(int(r['image_count']) for r in steps)
        if exposures != run_images:
            raise ValueError('实际样本曝光不等于完整训练预算')
        if termination:
            counts,images=Counter(),Counter()
            for row in steps:
                epoch=int(row['epoch']);counts[epoch]+=1;images[epoch]+=int(row['image_count'])
            if (dict(counts)!={e:math.ceil(train_frames/cfg['training']['batch']) for e in range(1,actual_epochs+1)}
                    or dict(images)!={e:train_frames for e in range(1,actual_epochs+1)}):
                raise ValueError('早停前某轮训练被截短或混入其他epoch')
        graph_path=path.parent/'训练图表来源_完整分项.json'
        graph=read_json(graph_path)
        if graph['points']!=run_steps or graph['validation_epochs']!=actual_epochs or graph['step_csv_sha256']!=sha256(path):
            raise ValueError('训练图表与原始逐步/逐轮记录不一致')
        if termination and graph.get('epoch_jsonl_sha256') != sha256(path.parent/'逐轮验证指标.jsonl'):
            raise ValueError('早停图表来源与逐轮验证SHA不同')
        for item in graph['figures']:
            if sha256(path.parent/item['file'])!=item['sha256']:
                raise ValueError('已保存训练图表的SHA不符')
        for row in steps:
            for field in ('loss','gradient_norm','learning_rate','step_seconds','gpu_allocated_mib','host_rss_mib'):
                if not math.isfinite(float(row[field])):
                    raise ValueError('训练日志含非有限原值')
        commit=run['source_commit']
        if commit not in algorithm_hashes:
            parts=[]
            for filename,function in (('detector_training.py','trainer_class'),('detector_matrix.py','train_one')):
                source=historical_source(commit,'src/occluactisafe/experiments/'+filename)
                node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==function)
                parts.append(ast.dump(node,include_attributes=False))
            algorithm_hashes[commit]=hashlib.sha256('\n'.join(parts).encode()).hexdigest()
        if common_algorithm is None:common_algorithm=algorithm_hashes[commit]
        if algorithm_hashes[commit]!=common_algorithm:
            raise ValueError('各训练run的optimizer/模型训练函数实现不同，拒绝混比')
        checks.append({'family':run['family'],'seed':run['seed'],'status':'passed',
                       'optimizer_steps':len(steps),'image_exposures':exposures,
                       'actual_epochs':actual_epochs, 'termination':termination,
                       'initial_weight_sha256':run['initial_weight_sha256'],
                       'source_commit':commit,'training_function_ast_sha256':algorithm_hashes[commit],
                       'figure_count':len(graph['figures']),'figure_source_sha256':sha256(graph_path),
                       'step_csv_sha256':sha256(path),'checkpoint_sha256':run['checkpoint_sha256']})
    return {'status':'passed' if seen==expected and all(r['status']=='passed' for r in checks) else 'failed',
            'expected_runs':len(expected),'actual_runs':len(matrix['runs']), 'train_frames':train_frames,
            'expected_optimizer_steps_per_run':expected_steps,'expected_image_exposures_per_run':expected_images,
            'total_optimizer_steps':sum(r.get('optimizer_steps',0) for r in checks),
            'total_image_exposures':sum(r.get('image_exposures',0) for r in checks),'runs':checks}


def audit_evaluation_matrix(cfg, results, root, expected_candidates):
    rows,data=load_index(cfg)
    expected={r['key'] for r in rows if r['role']=='model_val'}
    checks=[]
    for result in results:
        if result['status']!='completed':
            checks.append({'candidate':result['candidate_id'],'status':result['status']})
            continue
        if result['config_sha256']!=cfg['_config_sha256'] or result['data_index_sha256']!=data['index_sha256']:
            raise ValueError('评测候选使用不同数据/配置')
        directory=Path(root)/(result['family']+'_'+result['stage']+'_'+str(result['seed']))
        path=directory/'逐帧预测跟踪与计时.jsonl'
        records=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        if len(records)!=len(expected) or {r['key'] for r in records}!=expected:
            raise ValueError('验证矩阵漏帧、重复帧或测试帧混入')
        if result['actual_frames']!=len(expected):
            raise ValueError('结果分母与逐帧输出不一致')
        checks.append({'candidate':result['candidate_id'],'status':'passed','frames':len(records),
                       'data_index_sha256':data['index_sha256'],'prediction_sha256':sha256(path)})
    return {'status':'passed' if len(results)==expected_candidates and all(r['status']=='passed' for r in checks) else 'failed',
            'expected_candidates':expected_candidates,'actual_candidates':len(results),
            'frames_per_candidate':len(expected),'runs':checks,'test_media_opened':0}
