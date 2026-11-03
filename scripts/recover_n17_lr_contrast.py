"""N17获准的一次第6轮恢复；IDE默认只读检查，显式--action run才续训。"""
import argparse
import copy
import json
from pathlib import Path
import sys

NODE = Path(r'H:\OccluActiSafeDataset\artifacts\节点资料\N17_检测器比较')
STUDY = NODE/'04_开发与复核/学习率对照/n17_lr_contrast_b158bcca95db42a88ecb60dc4f32f50b'
EVIDENCE = NODE/'03_验收证据/n17_lr_contrast_delivery_20261103/进程中断恢复_20261103'
RUNTIME = EVIDENCE/'冻结训练源码'
FINGERPRINT = '3ffcc6f81c317baa29bf61313ea3043686d2ce154edfc0a582890f8402351abb'
APPROVAL = 'n17-one-epoch-replay-approved-20261103'


def validate_budget(checkpoint, batches):
    if (checkpoint['status'] != 'completed' or checkpoint['epoch'] != 6
            or batches['image_exposures'] != 7*17003 or len(batches['per_epoch']) != 7
            or batches['batch_attempts'] != 7*8502):
        raise ValueError('现场超出第6轮检查点/一轮未保存训练的批准恢复预算')
    return {'approval': APPROVAL, 'original_planned_exposures': 1020180,
            'approved_extra_exposures': 17003, 'maximum_total_exposures': 1037183,
            'interrupted_attempt_exposures': batches['image_exposures'],
            'committed_prefix_exposures': 6*17003, 'replayed_uncommitted_epoch': 7,
            'remaining_training_epochs': 4, 'formal_three_seed_complete': False}


def recovery_configuration(config, training, checkpoint, old_state, destination):
    result = {k: copy.deepcopy(v) for k, v in config.items() if not k.startswith('_')}
    profile = result['local_profile']
    if (profile.get('resume_checkpoint') or profile['model_id'] != 'YOLO26n'
            or profile['seed'] != 17 or result['training']['epochs'] != 300
            or result['training']['lr0'] != .001 or checkpoint['epoch'] != 6):
        raise ValueError('只允许本次YOLO26n/seed17/原学习率从第6轮恢复一次')
    result['output_root'] = str(destination)
    profile.update(resume_checkpoint=str(training/'weights/last.pt'),
        resume_checkpoint_sha256=checkpoint['last_sha256'],
        resume_best_checkpoint=str(training/'weights/best.pt'), resume_training_directory=str(training),
        resume_epoch=6, resume_batch_attempts=checkpoint['batch_attempts'],
        resume_applied_updates=checkpoint['applied_updates'], original_class_transfer=old_state['class_transfer'])
    return result


def validate_state(state):
    if state.get('source_fingerprint_sha256') != FINGERPRINT or len(state['tasks']) != 6:
        raise ValueError('原六项源码身份或任务数不一致')
    first = state['tasks'][0]
    if (first['id'] != 'YOLO26n_lr_0p001' or first['family'] != 'YOLO26n'
            or first['seed'] != 17 or first['learning_rate'] != .001
            or first['status'] not in ('training', 'interrupted')
            or any(t['status'] != 'planned' for t in state['tasks'][1:])
            or state.get('recovery')):
        raise ValueError('不是本次唯一获准的中断现场，拒绝自动重试或增加任务')
    return first


def runtime():
    # 恢复编排与训练源码分离；原六项继续使用完全相同的167文件指纹。
    if not RUNTIME.is_dir():
        raise FileNotFoundError('缺少已核验的原冻结源码；不从当前源码冒充原版本')
    sys.path.insert(0, str(RUNTIME/'src'))
    from occluactisafe.common.config import PROJECT_ROOT
    from occluactisafe.experiments.source_version import source_info
    if PROJECT_ROOT.resolve() != RUNTIME.resolve():
        raise ValueError('Python已加载另一份项目代码，请用独立解释器运行本入口')
    info = source_info()
    if info['source_fingerprint_sha256'] != FINGERPRINT:
        raise ValueError('冻结训练源码被修改')
    return info


def inspect(state):
    from collections import Counter
    from occluactisafe.experiments.detector_data import load_config, load_index
    from occluactisafe.experiments.detector_local import audit_local_batches
    from occluactisafe.experiments.detector_local_checkpoints import verify_checkpoint
    from occluactisafe.experiments.run_store import read_json, sha256
    first = validate_state(state)
    for task in state['tasks']:
        if sha256(task['config']) != task['config_sha256']:
            raise ValueError('原六项配置SHA改变')
    if sha256(state['evaluation_config']) != state['evaluation_config_sha256']:
        raise ValueError('原统一评分配置SHA改变')
    directory = Path(first['directory']).resolve()
    if not directory.is_relative_to(STUDY.resolve()):
        raise ValueError('原任务目录越界')
    training = directory/'YOLO26n/seed_17/训练'
    config = load_config(first['config']); rows, _ = load_index(config)
    expected = Counter(str(Path(r['image']).resolve()) for r in rows if r['role'] == 'train')
    checkpoint = verify_checkpoint(training)
    batches = audit_local_batches(training, expected, 7, 2, True)
    # 此处仅核对7轮训练批次；第7轮验证未结束，不将其计为提交轮次。
    budget = validate_budget(checkpoint, batches)
    return first, directory, training, config, checkpoint, batches, budget


def run():
    import yaml
    from occluactisafe.experiments import detector_lr_contrast as queue
    from occluactisafe.experiments.detector_controlled_trial import audit_controlled_training
    from occluactisafe.experiments.detector_server import execution_lock
    from occluactisafe.experiments.run_store import read_json, write_json, sha256, now, manifest
    # 所有状态修改和恢复只在两把OS锁均取得后进行，未知活动任务不受干扰。
    with execution_lock(NODE/'.n17_lr_contrast_queue.lock'), execution_lock(NODE/'.gpu_0.lock'):
        state = read_json(STUDY/'对照任务.json')
        first, old, training, config, checkpoint, batches, budget = inspect(state)
        receipt = EVIDENCE/'中断现场与批准恢复.json'
        if receipt.exists():
            raise FileExistsError('已经申请过一次恢复，不自动重跑')
        original = manifest(old)
        # 保存完整原状态和原任务逐文件哈希，不改旧训练日志/图片/双权重。
        write_json(receipt, {'observed_at': now(), 'original_state': state,
            'original_state_sha256': sha256(STUDY/'对照任务.json'), 'original_task_files': original,
            'original_task_directory': str(old), 'checkpoint': checkpoint,
            'training_batches_only': batches, 'budget': budget,
            'reason': '训练与队列进程消失；第7轮验证中断，无Python异常栈；具体退出原因未确认',
            'controlled_pause_at_epoch10_reached': False})
        destination = STUDY/'恢复_20261103_第6轮/YOLO26n_lr_0p001'
        destination.mkdir(parents=True, exist_ok=False)
        resumed = recovery_configuration(config, training, checkpoint,
                                         read_json(training.parent/'训练状态.json'), destination)
        cfg_path = destination/'冻结恢复配置.yaml'
        cfg_path.write_text(yaml.safe_dump(resumed, allow_unicode=True, sort_keys=False), encoding='utf-8')
        first.update(directory=str(destination), config=str(cfg_path), config_sha256=sha256(cfg_path),
                     status='recovery_training', interrupted_attempt_directory=str(old))
        state.update(status='recovering', recovery={**budget, 'started_at': now(),
                     'receipt': str(receipt), 'receipt_sha256': sha256(receipt)})
        write_json(STUDY/'对照任务.json', state)
        try:
            task_path = destination/'恢复任务参数.json'
            write_json(task_path, {'kind': 'train', 'family': 'YOLO26n', 'seed': 17,
                'config': str(cfg_path), 'attempt': str(destination),
                'training_directory': str(destination/'YOLO26n/seed_17'),
                'expected_gpu_uuid': state['gpu_uuid'],
                'worker_cuda_visible_devices': state['worker_cuda_visible_devices']})
            print('从第6轮检查点恢复第7～10轮，额外曝光最多17003，原六项源码不变', flush=True)
            code = queue.launch(task_path, destination/'恢复训练控制台.txt', state['worker_cuda_visible_devices'])
            if code:
                raise RuntimeError('恢复worker退出码='+str(code)+'；保留现场，不再次自动重试')
            result = read_json(destination/'worker_result.json')
            if result['status'] != 'paused_at_declared_epoch':
                raise ValueError('恢复未到达原第10轮受控暂停边界')
            new_training = destination/'YOLO26n/seed_17/训练'
            audit = audit_controlled_training(resumed, new_training)
            restored = read_json(new_training/'检查点恢复核验.json')
            if (restored['start_epoch'] != 7 or restored['source_checkpoint_sha256'] != checkpoint['last_sha256']
                    or restored['applied_updates_restored'] != checkpoint['applied_updates']):
                raise ValueError('实际恢复来源或更新步数不一致')
            if manifest(old) != original:
                raise ValueError('原中断任务文件被改变')
            write_json(destination/'受控训练核验.json', audit)
            first.update(status='trained', training_audit=str(destination/'受控训练核验.json'),
                best=str(new_training/'weights/best.pt'), best_sha256=audit['weights']['best_sha256'],
                training_returncode=0)
            state['recovery'].update(restored_epoch=6, resumed_through_epoch=10,
                                     old_attempt_unchanged=True, restored_state=restored)
            state['status'] = 'recovery_trained'
            write_json(STUDY/'对照任务.json', state)
        except BaseException as exc:
            first.update(status='failed', failed_phase='approved_recovery', error=repr(exc))
            state.update(status='failed', error=repr(exc))
            write_json(STUDY/'对照任务.json', state)
            raise
    # 原队列自己重新获取锁，统一评分和后续5项均走冻结的原代码/配置。
    queue.run(queue.options(RUNTIME/'configs/experiments/n17_lr_contrast.py', action='run', run_dir=STUDY))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('check', 'run'), default='check')
    args = parser.parse_args()
    info = runtime()
    if args.action == 'run':
        run()
    else:
        from occluactisafe.experiments.run_store import read_json
        state = read_json(STUDY/'对照任务.json')
        if state.get('console_free_recovery'):
            print(json.dumps({'status':'passed','mode':'read_only_recovery_progress',
                'study_status':state['status'],'recovery':state['console_free_recovery'],
                'gpu_kernels_launched':False},ensure_ascii=False))
            return
        if state.get('recovery'):
            print(json.dumps({'status': 'passed', 'mode': 'read_only_recovery_progress',
                'study_status': state['status'], 'recovery': state['recovery'],
                'tasks': [{k: t[k] for k in ('id', 'status')} for t in state['tasks']],
                'gpu_kernels_launched': False}, ensure_ascii=False))
            return
        _, _, _, _, checkpoint, _, budget = inspect(state)
        print(json.dumps({'status': 'passed', 'mode': 'read_only_recovery_check',
            'source_fingerprint_sha256': info['source_fingerprint_sha256'],
            'checkpoint_epoch': checkpoint['epoch'], 'budget': budget,
            'gpu_kernels_launched': False, 'training_started': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
