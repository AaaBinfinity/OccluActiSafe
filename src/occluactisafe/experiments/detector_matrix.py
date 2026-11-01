"""N17矩阵配置与串行真实GPU训练；每次失败仍在矩阵中。"""
from pathlib import Path
import os
import platform
import subprocess
import sys
import uuid

import yaml

from occluactisafe.common.config import PROJECT_ROOT
from .detector_data import load_index
from .run_store import now, sha256, read_json, write_json


def validate_matrix(cfg):
    if len(set(cfg['seeds'])) < 3:
        raise ValueError('N17至少三个不同训练种子')
    if {m['family'] for m in cfg['models']} != {'YOLO26', 'YOLOv8', 'YOLO11'}:
        raise ValueError('三族候选必须完整保留')
    if cfg['gates']['selection_split'] != 'model_val':
        raise ValueError('禁止用测试集选型')
    if cfg['training']['batch'] != cfg['training']['nbs']:
        raise ValueError('本次冻结有效batch必须等于物理batch')
    return [(m['family'], seed) for m in cfg['models'] for seed in cfg['seeds']]


def train_one(cfg, family, seed, directory):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('N17必需本机CUDA，不回退CPU')
    os.environ['YOLO_CONFIG_DIR'] = str(Path(cfg['output_root'])/'04_开发与复核'/'Ultralytics设置')
    from ultralytics import YOLO, __version__
    from ultralytics.utils.torch_utils import get_flops
    from .detector_training import trainer_class, training_plots
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    _, data = load_index(cfg)
    model_cfg = next(m for m in cfg['models'] if m['family'] == family)
    weight = Path(cfg['output_root'])/'02_运行产物'/'官方预训练权重'/model_cfg['weights']
    if not weight.exists():
        raise FileNotFoundError('先运行prepare_detector.py显式准备权重')
    state = dict(family=family, seed=seed, started_at=now(), status='running',
                 config_sha256=cfg['_config_sha256'], data_index_sha256=data['index_sha256'],
                 initial_weight_sha256=sha256(weight), python=sys.version, platform=platform.platform(),
                 ultralytics=__version__, torch=torch.__version__, cuda=torch.version.cuda,
                 source_commit=subprocess.check_output(['git', '-C', str(PROJECT_ROOT), 'rev-parse', 'HEAD'], text=True).strip())
    write_json(directory/'训练状态.json', state)
    try:
        model = YOLO(str(weight))
        state['pretrained_parameter_count'] = sum(p.numel() for p in model.model.parameters())
        state['pretrained_gflops_960'] = float(get_flops(model.model, imgsz=cfg['training']['imgsz']))
        write_json(directory/'训练状态.json', state)
        model.train(trainer=trainer_class(cfg['training']['batch']), data=str(Path(data['root'])/'dataset.yaml'),
                    device=0, seed=seed, project=str(directory), name='训练', exist_ok=False,
                    plots=False, save=True, save_period=1, val=True, pretrained=True,
                    **cfg['training'])
        checkpoint = directory/'训练'/'weights'/'last.pt'
        if not checkpoint.is_file():
            raise FileNotFoundError('训练未产生last.pt')
        state['parameter_count'] = sum(p.numel() for p in model.trainer.model.parameters())
        state['gflops_960'] = float(get_flops(model.trainer.model, imgsz=cfg['training']['imgsz']))
        state.update(status='completed', finished_at=now(), checkpoint=str(checkpoint),
                     checkpoint_sha256=sha256(checkpoint),
                     peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    except BaseException as exc:
        state.update(status='failed', finished_at=now(), error=repr(exc))
        raise
    finally:
        write_json(directory/'训练状态.json', state)
        training_plots(directory/'训练')
    return state


def run_training_matrix(cfg):
    pairs = validate_matrix(cfg)
    _, data = load_index(cfg)
    root = Path(cfg['output_root'])/'02_运行产物'/('n17_train_'+uuid.uuid4().hex)
    root.mkdir(parents=True)
    state = {'run_id': root.name, 'started_at': now(), 'status': 'running',
             'config_sha256': cfg['_config_sha256'], 'data_index_sha256': data['index_sha256'], 'runs': []}
    (root/'冻结矩阵配置.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    write_json(root/'训练矩阵.json', state)
    for family, seed in pairs:
        directory = root/family/('seed_'+str(seed))
        directory.parent.mkdir(parents=True, exist_ok=True)
        log = directory.parent/(directory.name+'_控制台.txt')
        command = [sys.executable, str(PROJECT_ROOT/'scripts/train_detector.py'), '--config', cfg['_config_path'],
                   '--family', family, '--seed', str(seed), '--output', str(directory)]
        print(f'N17开始 {family} seed={seed} epochs={cfg["training"]["epochs"]} batch={cfg["training"]["batch"]}', flush=True)
        with log.open('w', encoding='utf-8') as f:
            result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'})
        status_path = directory/'训练状态.json'
        row = read_json(status_path) if status_path.exists() else {'family': family, 'seed': seed,
                                                                 'status': 'failed', 'error': 'worker退出且缺状态'}
        row['directory'] = str(directory)
        row['returncode'] = result.returncode
        state['runs'].append(row)
        write_json(root/'训练矩阵.json', state)
        print(f'N17结束 {family} seed={seed} status={row["status"]} log={log}', flush=True)
    state.update(status='completed' if all(r['status'] == 'completed' for r in state['runs']) else 'failed', finished_at=now())
    write_json(root/'训练矩阵.json', state)
    return root
