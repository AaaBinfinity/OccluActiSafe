"""N17已授权的10轮学习率对照：保留300轮计划，独立审计受控暂停。"""
from collections import Counter
from pathlib import Path
import json

from .detector_local_checkpoints import verify_checkpoint
from .detector_stopping import PlateauStopper
from .run_store import read_json, write_json, sha256

MODELS = ('YOLO26n', 'YOLOv8n', 'YOLO11n')
LEARNING_RATES = (.001, .0003)
STOP_EPOCH = 10


def validate_trial(cfg, family, seed):
    """只接受本次用户批准的变量；普通训练不走该分支。"""
    profile = cfg['local_profile']
    trial = profile.get('controlled_trial')
    if not isinstance(trial, dict) or trial.get('protocol') != 'n17-lr-6x10-v1':
        raise ValueError('缺少已冻结的N17六项学习率对照身份')
    train = cfg['training']
    expected = {'epochs': 300, 'batch': 2, 'nbs': 2, 'workers': 0, 'amp': True,
                'imgsz': 960, 'optimizer': 'SGD', 'warmup_epochs': 0.0}
    if any(train.get(k) != v for k, v in expected.items()):
        raise ValueError('受控对照的公共训练配方改变')
    if (family not in MODELS or seed != 17 or train['lr0'] not in LEARNING_RATES
            or trial.get('stop_after_epoch') != STOP_EPOCH
            or trial.get('learning_rate') != train['lr0']
            or profile.get('pause_after_epoch') is not None
            or profile.get('model_id') != family or profile.get('seed') != seed
            or cfg['budget_policy'] != {'criterion': 'native_fitness', 'minimum_epochs': 50, 'patience': 30}):
        raise ValueError('任务超出三族n/seed17/两学习率/第10轮暂停的批准预算')
    return trial


def controlled_termination(records, policy, maximum_epochs, stop_epoch):
    """部分计划不能用普通早停审计冒充完成；每轮原始stopper记录仍严格重放。"""
    if maximum_epochs != 300 or stop_epoch != STOP_EPOCH or len(records) != stop_epoch:
        raise ValueError('受控暂停未完成批准的10轮或改变了300轮学习率计划')
    if policy != {'criterion': 'native_fitness', 'minimum_epochs': 50, 'patience': 30}:
        raise ValueError('受控对照改变了最少轮数或早停规则')
    stopper = PlateauStopper(policy['minimum_epochs'], policy['patience'])
    for record in records:
        if stopper(record['epoch'], record['fitness']) or stopper.last_decision != record:
            raise ValueError('部分训练的早停记录不一致或被误标为原生早停')
    return {'actual_epochs': stop_epoch, 'maximum_epochs': maximum_epochs,
            'stop_reason': 'declared_development_pause', 'schedule_epochs': maximum_epochs,
            'best_epoch': stopper.best_epoch, 'best_fitness': stopper.best_fitness,
            'convergence_guaranteed': False, 'full_training_complete': False}


def pause_after_committed_epoch(trainer, stop_epoch=STOP_EPOCH):
    if getattr(trainer, '_n17_final_eval', False):
        return None
    epoch = trainer.epoch + 1
    if epoch < stop_epoch:
        return None
    if epoch != stop_epoch or stop_epoch != STOP_EPOCH:
        raise ValueError('受控训练越过批准暂停边界')
    checkpoint = verify_checkpoint(trainer.save_dir)
    if checkpoint['status'] != 'completed' or checkpoint['epoch'] != epoch:
        raise ValueError('暂停前last/best尚未完整提交')
    record = {'status': 'paused_at_declared_epoch', 'epoch': epoch,
              'schedule_epochs': 300, 'native_early_stop': False,
              'last_sha256': checkpoint['last_sha256'], 'best_sha256': checkpoint['best_sha256']}
    write_json(Path(trainer.save_dir)/'受控暂停.json', record)
    trainer.stop = True
    return record


def controlled_trainer_class(expected_batch, policy, options):
    from .detector_local_training import rich_trainer_class
    base = rich_trainer_class(expected_batch, policy, options)

    class ControlledTrainer(base):
        def _local_validation_finished(self, trainer):
            super()._local_validation_finished(trainer)
            pause_after_committed_epoch(self, options['controlled_trial']['stop_after_epoch'])

    return ControlledTrainer


def audit_controlled_training(cfg, training):
    from .detector_data import load_index
    from .detector_local import audit_local_batches
    training = Path(training)
    profile = cfg['local_profile']
    validate_trial(cfg, profile['model_id'], profile['seed'])
    trace = [json.loads(line) for line in (training/'早停判定.jsonl').read_text(encoding='utf-8').splitlines()]
    termination = controlled_termination(trace, cfg['budget_policy'], cfg['training']['epochs'], STOP_EPOCH)
    rows, _ = load_index(cfg)
    expected = Counter(str(Path(r['image']).resolve()) for r in rows if r['role'] == 'train')
    if sum(expected.values()) != 17003 or sum(r['role'] == 'model_val' for r in rows) != 3185:
        raise ValueError('受控对照没有使用完整冻结数据')
    batches = audit_local_batches(training, expected, STOP_EPOCH, 2, True)
    checkpoint = verify_checkpoint(training)
    pause = read_json(training/'受控暂停.json')
    if (checkpoint['status'] != 'completed' or checkpoint['epoch'] != STOP_EPOCH
            or checkpoint['applied_updates'] != batches['applied_updates']
            or checkpoint['batch_attempts'] != batches['batch_attempts']
            or pause['last_sha256'] != checkpoint['last_sha256']
            or pause['best_sha256'] != checkpoint['best_sha256']
            or pause['epoch'] != STOP_EPOCH or pause['native_early_stop']):
        raise ValueError('受控暂停/原始更新/权重提交身份不一致')
    if {p.name for p in (training/'weights').iterdir()} != {'last.pt', 'best.pt'}:
        raise ValueError('受控训练weights必须且仅有last.pt/best.pt')
    graph = read_json(training/'训练图表来源_完整分项.json')
    if (graph['points'] != batches['applied_updates'] or graph['validation_epochs'] != STOP_EPOCH
            or graph['step_csv_sha256'] != sha256(training/'逐步优化指标.csv')):
        raise ValueError('受控训练图表分母或原始CSV不一致')
    for figure in graph['figures']:
        if sha256(training/figure['file']) != figure['sha256']:
            raise ValueError('受控训练图表SHA不符')
    required = ['args.yaml', 'results.csv', 'predictions.json', '逐轮分类验证指标.jsonl',
                '逐轮验证原值/最佳权重验证.json', '误检漏检与检测样例/诊断图来源与指标.json',
                'F1_置信度曲线.png', '精确率_召回率曲线.png', '混淆矩阵_计数.png',
                '标签几何相关图.png', '逐轮实际耗时.png', '逐轮更新分母与AMP跳步.png']
    if any(not (training/name).is_file() for name in required):
        raise ValueError('受控训练缺少完整图集/预测/原始记录')
    return {'status': 'passed', 'scope': '已批准的全量数据10轮开发对照，不是完成300轮或三seed选型',
            'termination': termination, 'train_frames_per_epoch': 17003,
            **batches, 'weights': checkpoint, 'full_comparison_complete': False, 'test_media_opened': 0}
