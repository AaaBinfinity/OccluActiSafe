"""N17已批准的第二次恢复；默认只读，生产执行要求无控制台宿主。"""
import argparse
import csv
import json
from pathlib import Path
import sys

import recover_n17_lr_contrast as original

EVIDENCE = original.EVIDENCE/'无控制台恢复_20261103'
APPROVAL = EVIDENCE/'追加恢复批准.json'
DESTINATION = original.STUDY/'恢复_20261103_无控制台/YOLO26n_lr_0p001'


def approved_budget(record):
    expected = {'approved': True, 'additional_exposures': 9412,
                'cumulative_extra_exposures': 26415, 'maximum_total_exposures': 1046595}
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError('本次只批准额外9412曝光、累计26415，总上限1046595')
    return expected


def install_console_free_launch(queue):
    from n17_console_free import run_console_free
    python = str(Path(sys.executable).with_name('python.exe'))
    host = Path(__file__).with_name('n17_lr_headless_host.py')
    def launch(task_path, log, visible):
        return run_console_free([python, '-B', str(host), '--action', 'worker',
                                 '--worker-task', str(task_path)], log,
                                {'CUDA_VISIBLE_DEVICES': visible})
    queue.launch = launch


def inspect_current(state):
    from occluactisafe.experiments.run_store import read_json, manifest, sha256
    approval = approved_budget(read_json(APPROVAL))
    incident = read_json(original.EVIDENCE/'二次中断_控制台关闭_20261103/窗口关闭中断证据.json')
    first_receipt = read_json(original.EVIDENCE/'中断现场与批准恢复.json')
    expected = incident['previous_state']
    fields = ('id','family','seed','learning_rate','directory','config','config_sha256')
    if (state['source_fingerprint_sha256'] != original.FINGERPRINT
            or state['status'] != 'interrupted' or len(state['tasks']) != 6
            or state['tasks'][0]['status'] != 'interrupted'
            or any(t['status'] != 'planned' for t in state['tasks'][1:])
            or [{k:t[k] for k in fields} for t in state['tasks']]
               != [{k:t[k] for k in fields} for t in expected['tasks']]):
        raise ValueError('不是本次获准的控制台关闭现场，拒绝重复恢复')
    for record in (incident, first_receipt):
        directory = record.get('stopped_attempt_directory',record.get('original_task_directory'))
        entries = record.get('stopped_attempt_files',record.get('original_task_files'))
        if manifest(directory) != entries:
            raise ValueError('原中断目录内容改变')
    previous_training = Path(state['tasks'][0]['directory'])/'YOLO26n/seed_17/训练'
    with (previous_training/'逐批次优化尝试.csv').open(encoding='utf-8') as stream:
        lost = [r for r in csv.DictReader(stream) if int(r['epoch']) == 7]
    if len(lost) != 4706 or sum(int(r['image_count']) for r in lost) != 9412:
        raise ValueError('第二次未保存曝光不符合批准预算')
    # 从最初完整提交的epoch6恢复，原第一次恢复目录没有新的完整提交事务。
    details = original.inspect(first_receipt['original_state'])
    if sha256(details[2]/'weights/last.pt') != 'f15399a428d1ff2bbec5922b6c8e632a879a0207563adea9ed6f1ad06fae6269':
        raise ValueError('原第6轮检查点改变')
    return approval, incident, first_receipt, details


def run():
    import yaml
    from n17_lr_headless_host import console_handle
    if console_handle() != 0:
        raise RuntimeError('生产恢复只能由无控制台宿主启动，不接受普通终端run')
    from occluactisafe.experiments import detector_lr_contrast as queue
    from occluactisafe.experiments.detector_server import execution_lock
    from occluactisafe.experiments.detector_controlled_trial import audit_controlled_training
    from occluactisafe.experiments.run_store import read_json, write_json, sha256, manifest, now
    install_console_free_launch(queue)
    with execution_lock(original.NODE/'.n17_lr_contrast_queue.lock'), execution_lock(original.NODE/'.gpu_0.lock'):
        state = read_json(original.STUDY/'对照任务.json')
        approval, incident, prior, details = inspect_current(state)
        _, _, training, config, checkpoint, _, _ = details
        receipt = EVIDENCE/'第二次恢复执行声明.json'
        if receipt.exists(): raise FileExistsError('第二次恢复已申请过，不自动重试')
        write_json(receipt, {'time':now(),'approval':approval,'old_state':state,
                            'checkpoint':checkpoint,'extra_epoch7_exposures':9412})
        DESTINATION.mkdir(parents=True, exist_ok=False)
        resumed = original.recovery_configuration(config, training, checkpoint,
                        read_json(training.parent/'训练状态.json'), DESTINATION)
        cfg_path = DESTINATION/'冻结无控制台恢复配置.yaml'
        cfg_path.write_text(yaml.safe_dump(resumed,allow_unicode=True,sort_keys=False),encoding='utf-8')
        first = state['tasks'][0]
        first.update(directory=str(DESTINATION),config=str(cfg_path),config_sha256=sha256(cfg_path),
                     status='console_free_recovery_training',
                     interrupted_attempt_directories=[prior['original_task_directory'],incident['stopped_attempt_directory']])
        state.update(status='console_free_recovering',additional_budget_pending=False,
                     console_free_recovery={**approval,'started_at':now(),'receipt':str(receipt),
                                            'receipt_sha256':sha256(receipt),'console_handle':0})
        write_json(original.STUDY/'对照任务.json',state)
        try:
            task_path = DESTINATION/'无控制台恢复任务.json'
            write_json(task_path,{'kind':'train','family':'YOLO26n','seed':17,'config':str(cfg_path),
                'attempt':str(DESTINATION),'training_directory':str(DESTINATION/'YOLO26n/seed_17'),
                'expected_gpu_uuid':state['gpu_uuid'],'worker_cuda_visible_devices':state['worker_cuda_visible_devices']})
            print('无控制台恢复开始：epoch6 -> epoch7～10；累计额外曝光最多26415',flush=True)
            code = queue.launch(task_path,DESTINATION/'无控制台恢复训练日志.txt',state['worker_cuda_visible_devices'])
            if code: raise RuntimeError('无控制台恢复失败，退出码='+str(code)+'，不自动重试')
            result = read_json(DESTINATION/'worker_result.json')
            if result['status'] != 'paused_at_declared_epoch': raise ValueError('没有在第10轮受控暂停')
            new_training = DESTINATION/'YOLO26n/seed_17/训练'
            audit = audit_controlled_training(resumed,new_training)
            restored = read_json(new_training/'检查点恢复核验.json')
            if restored['start_epoch'] != 7 or restored['source_checkpoint_sha256'] != checkpoint['last_sha256']:
                raise ValueError('实际检查点恢复身份改变')
            for record in (incident,prior):
                directory=record.get('stopped_attempt_directory',record.get('original_task_directory'))
                if manifest(directory)!=record.get('stopped_attempt_files',record.get('original_task_files')):
                    raise ValueError('原两次中断目录发生改变')
            write_json(DESTINATION/'受控训练核验.json',audit)
            first.update(status='trained',training_returncode=0,training_audit=str(DESTINATION/'受控训练核验.json'),
                         best=str(new_training/'weights/best.pt'),best_sha256=audit['weights']['best_sha256'])
            state['console_free_recovery'].update(restored_state=restored,resumed_through_epoch=10)
            state['status']='console_free_recovery_trained'
            write_json(original.STUDY/'对照任务.json',state)
        except BaseException as exc:
            first.update(status='failed',failed_phase='console_free_recovery',error=repr(exc))
            state.update(status='failed',error=repr(exc));write_json(original.STUDY/'对照任务.json',state)
            raise
    # 含首项统一评分和其他5项，全部worker经同一无控制台启动包装，算法不变。
    queue.run(queue.options(original.RUNTIME/'configs/experiments/n17_lr_contrast.py',action='run',run_dir=original.STUDY))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('check','run'),default='check')
    args=parser.parse_args(argv);original.runtime()
    if args.action=='run':run();return
    from occluactisafe.experiments.run_store import read_json
    state=read_json(original.STUDY/'对照任务.json')
    if state.get('console_free_recovery'):
        print(json.dumps({'status':'passed','mode':'read_only_progress','study_status':state['status'],
                          'recovery':state['console_free_recovery'],'training_started_by_check':False},ensure_ascii=False))
    else:
        approval,_,_,_=inspect_current(state)
        print(json.dumps({'status':'passed','mode':'read_only_preflight','approval':approval,
                          'gpu_kernels_launched':False},ensure_ascii=False))


if __name__=='__main__':main()
