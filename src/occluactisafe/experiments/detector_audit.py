"""N17真实矩阵输入、预算、完整帧与权重审计。"""
import csv
import json
import math
from pathlib import Path

from .detector_data import load_index
from .detector_matrix import validate_matrix
from .run_store import read_json, sha256


def audit_training_matrix(cfg, matrix_path):
    matrix = read_json(matrix_path)
    rows, data = load_index(cfg)
    train_frames = sum(r['role'] == 'train' for r in rows)
    expected_steps = math.ceil(train_frames/cfg['training']['batch'])*cfg['training']['epochs']
    expected_images = train_frames*cfg['training']['epochs']
    expected = set(validate_matrix(cfg))
    seen = set()
    checks = []
    for run in matrix['runs']:
        identity = run['family'], run['seed']
        if identity in seen or identity not in expected:
            raise ValueError('训练矩阵重复/未知候选身份')
        seen.add(identity)
        if run['status'] != 'completed':
            checks.append({'family':run['family'],'seed':run['seed'],'status':run['status']})
            continue
        if run['config_sha256'] != cfg['_config_sha256'] or run['data_index_sha256'] != data['index_sha256']:
            raise ValueError('训练候选数据/配置哈希不公平')
        if sha256(run['checkpoint']) != run['checkpoint_sha256']:
            raise ValueError('完成权重哈希改变')
        path=Path(run['directory'])/'训练'/'逐步优化指标.csv'
        with path.open(encoding='utf-8',newline='') as f:
            steps=list(csv.DictReader(f))
        if [int(r['step']) for r in steps] != list(range(1,expected_steps+1)):
            raise ValueError('实际优化步号/步数不符合完整冻结预算')
        exposures=sum(int(r['image_count']) for r in steps)
        if exposures != expected_images:
            raise ValueError('实际样本曝光不等于完整训练预算')
        for row in steps:
            for field in ('loss','gradient_norm','learning_rate','step_seconds','gpu_allocated_mib','host_rss_mib'):
                if not math.isfinite(float(row[field])):
                    raise ValueError('训练日志含非有限原值')
        checks.append({'family':run['family'],'seed':run['seed'],'status':'passed',
                       'optimizer_steps':len(steps),'image_exposures':exposures,
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
