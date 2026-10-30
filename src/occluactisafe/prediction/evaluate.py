"""N05流式CV验证：输入和监督分开读取，保留全目标及失败分母。"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import yaml
from occluactisafe.common.config import PROJECT_ROOT, _UniqueLoader, load_config
from occluactisafe.common.run_manifest import create_run_manifest
from occluactisafe.data.split import canonical_hash
from occluactisafe.prediction.dataset import PredictionDataset
from occluactisafe.prediction.baselines.cv import predict
from occluactisafe.evaluation.prediction_metrics import MetricAccumulator, score_target

DEFAULTS = {'windows_dir': '', 'output_root': '', 'role': 'model_val', 'top_k': 1,
            'miss_threshold_m': 2.0, 'max_windows': 0, 'allow_test': False,
            'frozen_config_sha256': None}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load_evaluation_config(path, *, overrides=None, cli_overrides=None):
    file = Path(path)
    if not file.is_absolute():
        file = PROJECT_ROOT / file
    with file.open(encoding='utf-8') as stream:
        layer = yaml.load(stream, Loader=_UniqueLoader)
    config = dict(DEFAULTS)
    for layer in (layer, overrides, cli_overrides):
        if layer is None:
            continue
        if not isinstance(layer, dict) or set(layer) - set(DEFAULTS):
            raise ValueError('N05配置含未知字段')
        config.update(layer)
    for key in ('windows_dir', 'output_root'):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f'{key}必须是明确的本地路径')
        p = Path(config[key]).expanduser()
        config[key] = str((p if p.is_absolute() else PROJECT_ROOT / p).resolve())
    if config['role'] not in ('train', 'model_val', 'test'):
        raise ValueError('N05不读取风险校准集')
    if type(config['top_k']) is not int or config['top_k'] != 1:
        raise ValueError('CV仅有K=1，不能复制为多模')
    if type(config['max_windows']) is not int or config['max_windows'] < 0 or type(config['allow_test']) is not bool:
        raise ValueError('窗口限量或测试开关非法')
    threshold = config['miss_threshold_m']
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('Miss阈值必须为有限正数')
    if config['role'] == 'test':
        expected = canonical_hash({k: v for k, v in config.items() if k != 'frozen_config_sha256'})
        if not config['allow_test'] or config['frozen_config_sha256'] != expected:
            raise ValueError('独立测试需显式开启并提供冻结配置哈希；N05默认不运行测试集')
    return config


def run(config):
    root, out = Path(config['windows_dir']), Path(config['output_root'])
    if root == out or root in out.parents or out in root.parents:
        raise ValueError('评测输出与N04输入必须分离')
    status = json.loads((root / 'n04_status.json').read_text(encoding='utf-8'))
    if status['status'] != 'completed':
        raise ValueError('N04窗口尚未完整完成')
    task = json.loads((root / 'task_config.json').read_text(encoding='utf-8'))
    n04_manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    split = json.loads((root / 'split_manifest.json').read_text(encoding='utf-8'))
    if canonical_hash({k: v for k, v in split.items() if k != 'sha256'}) != status['split_sha256']:
        raise ValueError('划分清单哈希不符')
    dataset = PredictionDataset(root, config['role'], normalized=False)
    if not len(dataset):
        raise ValueError('所选角色无窗口')
    entries = {r['source_file']: r for r in split['files']}
    for row in dataset.index:
        original = entries[row['source_file']]
        if original['project_role'] != config['role'] or original['original_split'] != 'train':
            raise ValueError('窗口索引与冻结分组不符')
    count = min(len(dataset), config['max_windows']) if config['max_windows'] else len(dataset)
    manifest = create_run_manifest(config=load_config(overrides={
                                       'runtime': {'seed': task['seed']},
                                       'prediction': {'history_steps': task['history_steps'],
                                                      'future_steps': task['future_steps'], 'future_dt': task['dt']},
                                       'planning': {'horizon_s': min(3.0, task['future_steps'] * task['dt'])}}),
                                   input_mode='offline_annotation_history_reference_cv',
                                   data=n04_manifest['data'], output_root=out)
    directory = out / manifest['run_id']
    write_json(directory / 'task_config.json', config)
    write_json(directory / 'status.json', {'status': 'running'})
    started = time.perf_counter()
    overall, groups = MetricAccumulator(), defaultdict(MetricAccumulator)
    coverage, exclusions = Counter(), Counter()
    input_digest = hashlib.sha256()
    predict_seconds = 0.0
    try:
        with (directory / 'per_agent_metrics.csv').open('w', encoding='utf-8', newline='') as metrics_file, (directory / 'predictions.csv').open('w', encoding='utf-8', newline='') as pred_file, (directory / 'input_inventory.jsonl').open('w', encoding='utf-8', newline='\n') as inventory:
            fields = ['source_file', 'group_id', 'track_id', 'category', 'view', 'input_hash',
                      'eligible', 'prediction_valid', 'reason', 'valid_steps', 'status',
                      'ADE_m', 'FDE_m', 'minADE_m', 'minFDE_m', 'miss']
            writer = csv.DictWriter(metrics_file, fieldnames=fields)
            writer.writeheader()
            pw = csv.writer(pred_file)
            pw.writerow(['source_file', 'track_id', 'mode', 'relative_time_s', 'local_x_m', 'local_y_m', 'prediction_valid'])
            for index in range(count):
                row = dataset.index[index]
                x = dataset.inputs(index)  # 不读取监督
                ft = np.arange(1, task['future_steps'] + 1) * task['dt']
                before = time.perf_counter()
                result = predict(x['features'][..., :2], x['history_times'],
                                 x['observed_mask'] & x['feature_mask'][..., :2].all(axis=-1), future_times=ft)
                predict_seconds += time.perf_counter() - before
                labels = dataset.labels(index)  # forward完成后才打开监督
                if not np.allclose(labels['future_times'], ft, rtol=0, atol=1e-12):
                    raise ValueError('未来时域不符')
                eligible = x['prediction_target_mask'] & x['observed_mask'].all(axis=1) & labels['future_mask'].all(axis=1)
                if not np.array_equal(eligible, labels['target_eligible']):
                    raise ValueError('监督对象定义不符')
                sample_path = root / row['sample_path']
                identity = {'source_file': row['source_file'], 'input_hash': x['input_hash'],
                            'sample_sha256': sha256(sample_path),
                            'labels_sha256': sha256(sample_path.with_suffix('.labels.npz'))}
                line = json.dumps(identity, sort_keys=True, separators=(',', ':')) + '\n'
                input_digest.update(line.encode()); inventory.write(line)
                coverage['windows'] += 1
                coverage['historical_targets'] += len(x['track_ids'])
                view = '/'.join(row['source_file'].split('/')[:2])
                for i in np.flatnonzero(x['prediction_target_mask']):
                    category = ('VEHICLE', 'PEDESTRIAN', 'BICYCLE', 'OTHER')[x['categories'][i]]
                    valid = bool(result['valid'][i])
                    record = {'source_file': row['source_file'], 'group_id': row['group_id'],
                              'track_id': x['track_ids'][i], 'category': category, 'view': view,
                              'input_hash': x['input_hash'], 'eligible': bool(eligible[i]),
                              'prediction_valid': valid, 'reason': '' if valid else 'fewer_than_two_positions'}
                    coverage['history_TARGET_AGENT'] += 1
                    if eligible[i]:
                        score = score_target(result['positions'][i], labels['future_positions'][i], labels['future_mask'][i],
                                             prediction_valid=valid, top_k=1, miss_threshold_m=config['miss_threshold_m'])
                        overall.add(score)
                        for key in ('category:' + category, 'view:' + view):
                            groups[key].add(score)
                    else:
                        score = {'status': 'ineligible', 'valid_steps': int(labels['future_mask'][i].sum())}
                        exclusions['incomplete_history_or_future'] += 1
                    writer.writerow({**record, **score})
                    for j, future in enumerate(ft):
                        p = result['positions'][i, 0, j]
                        pw.writerow([row['source_file'], x['track_ids'][i], 0, future,
                                     float(p[0]) if valid else '', float(p[1]) if valid else '', valid])
                if (index + 1) % 1000 == 0:
                    print(f'N05窗口 {index + 1}/{count}', flush=True)
        summary = {'protocol': {'model': 'CV', 'K': 1, 'top_k': 1, 'role': config['role'],
                               'history_steps': task['history_steps'], 'future_steps': task['future_steps'],
                               'future_horizon_s': task['future_steps'] * task['dt'],
                               'miss_threshold_m': config['miss_threshold_m'],
                               'coordinates': 'unnormalized rigid local frame, meters',
                               'target': 'history TARGET_AGENT with complete H/F; file-local identity',
                               'scope': 'limited' if count < len(dataset) else 'full_selected_role',
                               'official_test_score': False},
                   'overall': overall.summary(), 'groups': {k: v.summary() for k, v in sorted(groups.items())},
                   'coverage': dict(coverage), 'exclusions': dict(exclusions),
                   'input_inventory_sha256': input_digest.hexdigest(),
                   'timing': {'wall_seconds': time.perf_counter() - started, 'predict_seconds': predict_seconds,
                              'predict_window_denominator': count, 'device': 'CPU'},
                   'limitations': ['N04 source coordinate consistency unresolved; no filtering or clipping',
                                   'Annotation history reference only; no online perception or collision claim',
                                   'BICYCLE is not verified CYCLIST; rare classes have limited denominators']}
        write_json(directory / 'metrics.json', summary)
        evidence = {p.name: sha256(p) for p in directory.iterdir() if p.is_file() and p.name != 'status.json'}
        evidence['n04_inputs'] = {name: sha256(root / name) for name in
                                 ('n04_status.json', 'task_config.json', 'manifest.json', 'split_manifest.json', 'scaler.json')}
        write_json(directory / 'evidence_index.json', evidence)
        write_json(directory / 'status.json', {'status': 'completed', 'run_id': manifest['run_id'],
                   'completed_utc': datetime.now(timezone.utc).isoformat(), 'task_config_sha256': canonical_hash(config)})
        return {'run_id': manifest['run_id'], 'run_dir': str(directory), **summary}
    except BaseException as exc:
        write_json(directory / 'status.json', {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)})
        raise
