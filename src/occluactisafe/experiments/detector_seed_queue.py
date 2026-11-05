"""N17固定身份的串行seed队列；失败留证接下一项，绝不自动重试。"""
import argparse
from collections import Counter
import ctypes
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from occluactisafe.common.config import PROJECT_ROOT
from .detector_local import ALL_MODELS, local_options, run_local, selected_options
from .detector_models import ASSETS
from .detector_portable import INDEX_SHA
from .detector_server import execution_lock
from .detector_terminal import display_options, effective_options
from .run_store import append_json, now, read_json, sha256, write_json
from .source_version import source_info

DEFAULT_CONFIG = PROJECT_ROOT / 'configs/experiments/n17_seed_queue.py'
ENTRY = PROJECT_ROOT / 'scripts/train_n17_seeds.py'
SEEDS = (17, 29, 43)
OPTION_KEYS = ('MODEL', 'SEED', 'ACTION', 'INPUT_ROOT', 'OUTPUT_ROOT', 'DEVICE', 'BATCH',
               'WORKERS', 'AMP', 'MAX_EPOCHS', 'MIN_EPOCHS', 'PATIENCE', 'RESUME_RUN',
               'PAUSE_AFTER_EPOCH', 'PLOT_POINTS_PER_PAGE', 'CPU_THREADS', 'VALIDATION_BATCH',
               'LR0', 'VALIDATION_WORKERS', 'PREFETCH_FACTOR')


def resolve(path):
    p = Path(path).expanduser()
    return (p if p.is_absolute() else PROJECT_ROOT / p).resolve()


def queue_options(config=DEFAULT_CONFIG, overrides=None):
    path = resolve(config)
    raw = {'__name__': 'n17_queue_options', '__file__': str(path)}
    content = path.read_bytes()
    exec(compile(content, str(path), 'exec'), raw)
    values = {k: raw[k] for k in ('MODEL', 'SEEDS', 'ACTION', 'LOCAL_CONFIG', 'WAIT_HOURS')}
    changes = dict(overrides or {})
    values.update({k: changes.pop(k) for k in tuple(changes) if k in values})
    seeds = tuple(values['SEEDS'])
    if not seeds or any(type(s) is not int or s not in SEEDS for s in seeds) or seeds != tuple(s for s in SEEDS if s in seeds):
        raise ValueError('SEEDS须为17→29→43顺序的不重复非空子集')
    if values['ACTION'] not in ('check', 'plan', 'train'):
        raise ValueError('队列仅支持check/plan/train；不自动重训或恢复失败任务')
    if type(values['WAIT_HOURS']) not in (int, float) or not 0 < values['WAIT_HOURS'] <= 168:
        raise ValueError('WAIT_HOURS须大于0且不超过168小时')
    opts = local_options(resolve(values['LOCAL_CONFIG']),
                         {**changes, 'MODEL': values['MODEL'], 'SEED': seeds[0], 'ACTION': values['ACTION'],
                          'RESUME_RUN': '', 'PAUSE_AFTER_EPOCH': None})
    return {'local': opts, 'seeds': seeds, 'wait_hours': values['WAIT_HOURS'],
            'config_path': str(path), 'config_bytes': content}


def choose_queue(options, *, dry_run=False, input_fn=None, output=print):
    reader = input if input_fn is None else input_fn
    current = options['local']
    def ask(label, default, parse):
        while True:
            value = reader(f'{label} [{default}]：').strip()
            if value.lower() in ('q', 'quit', 'exit', '取消'):
                raise EOFError
            try:
                return parse(value or str(default))
            except (ValueError, KeyError, IndexError) as exc:
                output('输入无效，请重新输入：' + str(exc))
    def model(value):
        names = [m['family'] for m in ALL_MODELS]
        if value.isdigit():
            i = int(value) - 1
            if not 0 <= i < len(names):
                raise ValueError('模型编号须为1～15')
            return names[i]
        return {n.lower(): n for n in names}[value.lower().removesuffix('.pt')]
    def set_value(key, value):
        return selected_options(current, {key: value})[key]
    def action(value):
        v = {'1': 'check', '2': 'plan', '3': 'train'}.get(value, value.lower())
        if v not in ('check', 'plan', 'train'):
            raise ValueError('操作须为1/check、2/plan、3/train')
        return v
    try:
        output('N17单型号seed队列｜输入q取消；回车使用默认值')
        output('固定seed顺序：' + '→'.join(map(str, options['seeds'])))
        for i in range(0, len(ALL_MODELS), 3):
            output('   '.join(f'{j + 1:2d}. {ALL_MODELS[j]["family"]:<9}' for j in range(i, min(i + 3, len(ALL_MODELS)))))
        current = selected_options(current, {'MODEL': ask('模型编号或名称', current['MODEL'], model), 'ACTION': 'check'})
        lr = ask('SGD初始学习率lr0', current['LR0'], lambda v: set_value('LR0', float(v)))
        batch = ask('batch（当前n档32/worker2仍需实际容量核验）', current['BATCH'], lambda v: set_value('BATCH', int(v)))
        output('1. check：仅核输入身份/配置  2. plan：冻结队列  3. train：后台串行训练全部seed')
        chosen = ask('运行操作', 'check', action)
        opts = selected_options(current, {'LR0': lr, 'BATCH': batch, 'ACTION': chosen})
        display_options(opts, dry_run=dry_run, output=output)
        output(f'固定seeds={options["seeds"]}，共{len(options["seeds"])}项；完成一项自动接下一项。')
        output('全部按本次配方新建；旧seed17的50/30/worker1任务不会自动混入。失败保留后继续，不自动重试。')
        if reader('输入 y 确认整个队列，其他输入取消 [N]：').strip().lower() not in ('y', 'yes', '是', '确认'):
            raise EOFError
        return {**options, 'local': opts}
    except (EOFError, KeyboardInterrupt):
        output('已取消，没有启动队列或训练。')
        return None


def identity(opts):
    """只读小型索引/清单和初始权重；全量资产/CUDA预检仍由各真实worker执行。"""
    root = opts['INPUT_ROOT']
    paths = [root / '输入清单.json', root / 'data/逐帧数据索引.jsonl',
             root / '扩展权重' / opts['model']['weights']]
    if sha256(paths[-1]) != ASSETS[opts['model']['weights']]:
        raise ValueError('所选官方初始权重SHA不同，不下载或替换')
    if sha256(paths[1]) != INDEX_SHA:
        raise ValueError('输入索引不是约定的冻结版本')
    return {str(p): sha256(p) for p in paths}


def create_queue(options):
    opts = options['local']
    source = source_info()
    inputs = identity(opts)
    folder = '02_运行产物' if opts['ACTION'] == 'train' else '04_开发与复核/三seed队列计划'
    root = opts['OUTPUT_ROOT'] / folder / ('n17_seed_queue_' + uuid.uuid4().hex)
    root.mkdir(parents=True)
    (root / '冻结本机选项.py').write_bytes(opts['options_source_bytes'])
    (root / '冻结队列选项.py').write_bytes(options['config_bytes'])
    values = effective_options(opts)
    entries = [{'id': f'{opts["MODEL"]}_seed{s}', 'seed': s, 'status': 'pending'} for s in options['seeds']]
    plan = {'schema': 'n17-seed-queue-1', 'run_id': root.name, 'created_at': now(),
            'action': opts['ACTION'], 'model': opts['MODEL'], 'seeds': list(options['seeds']),
            'wait_hours': options['wait_hours'], 'source': source, 'inputs': inputs,
            'local_config_sha256': sha256(root / '冻结本机选项.py'),
            'queue_config_sha256': sha256(root / '冻结队列选项.py'), 'effective_options': values,
            'tasks': entries, 'automatic_retries': 0, 'automatic_resume': False,
            'training_exposure_upper_bound': len(entries) * 17003 * opts['MAX_EPOCHS']}
    write_json(root / '队列计划.json', plan)
    write_json(root / '队列状态.json', {'status': 'planned', 'tasks': entries,
               'plan_sha256': sha256(root / '队列计划.json')})
    (root / '结果导航.md').write_text(
        '# N17三seed队列\n\n[队列状态](队列状态.json)、[冻结计划](队列计划.json)、[事件原值](队列事件.jsonl)。'
        '\n\n每项独立目录保留全部图集/CSV/JSON及weights/last.pt、best.pt。失败不重试，完成一项自动接下一项；'
        '全队列结束后退出。旧实验不自动复用，三seed训练完成不等于统一评测、准入或N17用户验收。\n', encoding='utf-8')
    # 即使仅计划，也提供可访问的导航目标；后续只追加事件。
    append_json(root / '队列事件.jsonl', {'time': now(), 'event': 'planned', 'training_started': False})
    return root


def load_queue(root):
    root = resolve(root)
    state = read_json(root / '队列状态.json')
    plan = read_json(root / '队列计划.json')
    if plan['schema'] != 'n17-seed-queue-1' or sha256(root / '队列计划.json') != state['plan_sha256']:
        raise ValueError('冻结队列计划SHA不同')
    if [(t['id'], t['seed']) for t in state['tasks']] != [(t['id'], t['seed']) for t in plan['tasks']]:
        raise ValueError('队列状态中的任务身份/顺序不同')
    if sha256(root / '冻结本机选项.py') != plan['local_config_sha256'] or sha256(root / '冻结队列选项.py') != plan['queue_config_sha256']:
        raise ValueError('冻结选项SHA不同')
    output = resolve(plan['effective_options']['OUTPUT_ROOT'])
    if not root.is_relative_to(output) or root == output:
        raise ValueError('队列输出越界')
    return root, plan, state


def verify_identity(plan):
    if source_info()['source_fingerprint_sha256'] != plan['source']['source_fingerprint_sha256']:
        raise ValueError('队列源码身份已变，停止剩余任务，避免混配方')
    if any(sha256(p) != expected for p, expected in plan['inputs'].items()):
        raise ValueError('队列数据索引/输入清单/初始权重SHA已变')


def task_options(root, plan, seed):
    values = {k: plan['effective_options'][k] for k in OPTION_KEYS}
    values.update(SEED=seed)
    return local_options(root / '冻结本机选项.py', values)


def completed_result(job, opts, plan):
    job = Path(job).resolve()
    if not job.is_relative_to(opts['OUTPUT_ROOT']):
        raise ValueError('单项训练结果目录越界')
    task = read_json(job / '本机训练任务.json')
    if (task['status'] != 'completed' or task['development_only']
            or task['model'] != opts['MODEL'] or task['seed'] != opts['SEED']
            or task['source_fingerprint_sha256'] != plan['source']['source_fingerprint_sha256']):
        raise ValueError('单项训练没有完整成功或身份不符')
    frozen = Path(task['frozen_config']).resolve()
    if not frozen.is_relative_to(job) or sha256(frozen) != task['frozen_config_sha256']:
        raise ValueError('单项冻结训练配置SHA不同')
    from .detector_data import load_config
    cfg = load_config(frozen)
    audit_file = job / '本次训练核验.json'
    audit = read_json(audit_file)
    if audit['status'] != 'passed' or audit != task['audit']:
        raise ValueError('单项训练审计未通过或记录不同')
    return {'status': 'completed', 'job': str(job), 'task_sha256': sha256(job / '本机训练任务.json'),
            'audit_sha256': sha256(audit_file), 'recipe_sha256': cfg['local_profile']['recipe_sha256'],
            'actual_epochs': audit['actual_epochs'], 'image_exposures': audit['image_exposures'],
            'weights': audit['weights'], 'full_comparison_complete': False}


def worker(root, task_id):
    root, plan, _ = load_queue(root)
    entry = next(t for t in plan['tasks'] if t['id'] == task_id)
    directory = root / task_id
    directory.mkdir(exist_ok=True)
    # 独占创建票据，宿主崩溃或重复调用也不会重复训练同一项。
    with (directory / '单项执行票据.json').open('x', encoding='utf-8') as stream:
        json.dump({'pid': os.getpid(), 'time': now(), 'automatic_retry': False}, stream)
    try:
        verify_identity(plan)
        opts = task_options(root, plan, entry['seed'])
        # 仅此独立队列worker替换进程启动函数；既有单任务源码/当前活动训练不变。
        from . import detector_local
        original = detector_local.execute_worker
        detector_local.execute_worker = lambda task, log: execute_training_worker(root, entry, task, log)
        try:
            job = run_local(opts)
        finally:
            detector_local.execute_worker = original
        result = completed_result(job, opts, plan)
        write_json(directory / '单项结果.json', result)
    except BaseException as exc:
        write_json(directory / '单项结果.json', {'status': 'failed', 'error': repr(exc),
                   'time': now(), 'automatic_retry': False})
        raise


def console_handle():
    if os.name != 'nt':
        return None
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetConsoleWindow.restype = ctypes.c_void_p
    return int(kernel.GetConsoleWindow() or 0)


def execute_training_worker(root, entry, task_file, log):
    task = read_json(task_file)
    write_json(root / entry['id'] / '训练任务启动身份.json', {'file': str(task_file), 'sha256': sha256(task_file)})
    command = [sys.executable, '-u', str(ENTRY), '--worker-queue', str(root),
               '--task-id', entry['id'], '--model-task', str(task_file)]
    with Path(log).open('xb') as stream:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8',
                 'CUDA_VISIBLE_DEVICES': task['worker_cuda_visible_devices']})
        return process.wait()


def model_worker(root, task_id, task_file):
    root, plan, _ = load_queue(root)
    verify_identity(plan)
    if task_id not in [t['id'] for t in plan['tasks']]:
        raise ValueError('未知队列任务')
    record = read_json(root / task_id / '训练任务启动身份.json')
    task_file = resolve(task_file)
    if (task_file != resolve(record['file']) or sha256(task_file) != record['sha256']
            or not task_file.is_relative_to(resolve(plan['effective_options']['OUTPUT_ROOT']))):
        raise ValueError('训练worker任务路径或SHA不同')
    handle = console_handle()
    if handle:
        raise RuntimeError('训练worker仍附着控制台，拒绝启动')
    write_json(root / task_id / '训练worker身份.json', {'pid': os.getpid(), 'console_handle': handle, 'time': now()})
    from .detector_server import worker as server_worker
    return server_worker(task_file)


def launch_worker(root, entry):
    directory = root / entry['id']
    directory.mkdir(exist_ok=True)
    command = [sys.executable, '-u', str(ENTRY), '--worker-queue', str(root), '--task-id', entry['id']]
    with (directory / '队列控制台.txt').open('xb') as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'})
        return process.wait()


def wait_gpu(plan, deadline):
    opts = plan['effective_options']
    while True:
        verify_identity(plan)
        try:
            with execution_lock(resolve(opts['OUTPUT_ROOT']) / f'.gpu_{opts["DEVICE"]}.lock'):
                return
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise TimeoutError('等待既有GPU任务释放锁超时，未结束其他进程')
            time.sleep(min(15, max(0, deadline - time.monotonic())))


def result_summary(root, state):
    """只抄录实际成功回执；失败/未执行的指标留空，不能写成0。"""
    fields = ('seed', 'status', 'actual_epochs', 'image_exposures', 'job', 'error')
    with (root / '队列结果汇总.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for entry in state['tasks']:
            result = entry.get('result', {})
            writer.writerow({k: entry.get(k, result.get(k, '')) for k in fields})


def run_queue(root, *, runner=None, wait=None):
    root, plan, state = load_queue(root)
    if plan['action'] != 'train' or state['status'] != 'launched':
        raise ValueError('队列不是首次明确启动的train，拒绝重跑')
    runner = launch_worker if runner is None else runner
    wait = wait_gpu if wait is None else wait
    lock = resolve(plan['effective_options']['OUTPUT_ROOT']) / f'.n17_seed_queue_gpu_{plan["effective_options"]["DEVICE"]}.lock'
    with execution_lock(lock):
        state.update(status='running', host_pid=os.getpid(), started_at=now())
        write_json(root / '队列状态.json', state)
        try:
            recipe = None
            for entry in state['tasks']:
                verify_identity(plan)
                entry.update(status='waiting_for_gpu', started_at=now())
                write_json(root / '队列状态.json', state)
                wait(plan, time.monotonic() + plan['wait_hours'] * 3600)
                entry['status'] = 'running'
                write_json(root / '队列状态.json', state)
                append_json(root / '队列事件.jsonl', {'time': now(), 'event': 'started', 'task': entry['id']})
                try:
                    code = runner(root, entry)
                    entry['worker_returncode'] = code
                    verify_identity(plan)
                    result = read_json(root / entry['id'] / '单项结果.json')
                    if code != 0 or result['status'] != 'completed':
                        raise RuntimeError(f'单项退出码{code}，结果状态{result["status"]}')
                    if recipe is not None and recipe != result['recipe_sha256']:
                        raise ValueError('seed之间实际训练配方SHA不同')
                    recipe = result['recipe_sha256']
                    entry.update(status='completed', result=result, worker_returncode=code)
                except Exception as exc:
                    entry.update(status='failed', error=repr(exc))
                entry['finished_at'] = now()
                append_json(root / '队列事件.jsonl', {'time': now(), 'event': entry['status'], 'task': entry['id']})
                write_json(root / '队列状态.json', state)
            state['status'] = 'completed' if all(t['status'] == 'completed' for t in state['tasks']) else 'completed_with_failures'
        except BaseException as exc:
            state.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'blocked', error=repr(exc))
            raise
        finally:
            state.update(finished_at=now(), counts=dict(Counter(t['status'] for t in state['tasks'])),
                         full_comparison_complete=False)
            write_json(root / '队列状态.json', state)
            result_summary(root, state)
    return state


def host(root):
    root = resolve(root)
    handle = console_handle()
    if handle:
        raise RuntimeError('后台宿主仍附着控制台，拒绝启动训练')
    write_json(root / '后台宿主身份.json', {'pid': os.getpid(), 'time': now(),
               'console_handle': handle})
    try:
        run_queue(root)
    except BaseException as exc:
        state = read_json(root / '队列状态.json')
        if state['status'] == 'launched':
            state.update(status='blocked', error=repr(exc), finished_at=now())
            write_json(root / '队列状态.json', state)
        raise


def start_background(root):
    root, plan, state = load_queue(root)
    if plan['action'] != 'train' or state['status'] != 'planned':
        raise ValueError('只能首次启动已冻结的train队列')
    # 创建唯一启动票据，失败也不把本队列当作从未执行而悄悄重试。
    with (root / '队列启动票据.json').open('x', encoding='utf-8') as stream:
        json.dump({'time': now(), 'launcher_pid': os.getpid()}, stream)
    state['status'] = 'launched'
    write_json(root / '队列状态.json', state)
    try:
        with (root / '后台队列日志.txt').open('xb') as log:
            process = subprocess.Popen([sys.executable, '-u', str(ENTRY), '--host', str(root)],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, close_fds=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                start_new_session=os.name != 'nt',
                env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'})
        write_json(root / '后台启动回执.json', {'host_pid': process.pid, 'time': now(),
                   'scope': '启动回执不是训练完成证明'})
    except BaseException as exc:
        state.update(status='launch_failed', error=repr(exc))
        write_json(root / '队列状态.json', state)
        raise
    return process.pid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--model')
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--lr0', type=float)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--action', choices=('check', 'plan', 'train'))
    parser.add_argument('--dry-run', action='store_true', help='仅预览，不检查数据或启动进程')
    parser.add_argument('--no-interactive', action='store_true')
    parser.add_argument('--status', type=Path, help='只读指定队列状态')
    parser.add_argument('--host', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker-queue', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--task-id', help=argparse.SUPPRESS)
    parser.add_argument('--model-task', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.host:
        return host(args.host)
    if args.worker_queue:
        if args.model_task:
            return model_worker(args.worker_queue, args.task_id, args.model_task)
        return worker(args.worker_queue, args.task_id)
    if args.status:
        root, _, state = load_queue(args.status)
        import psutil
        runtime = {'host_process_present': psutil.pid_exists(read_json(root / '后台宿主身份.json')['pid'])} if (root / '后台宿主身份.json').is_file() else {}
        print(json.dumps({'queue': str(root), **state, **runtime}, ensure_ascii=False, indent=2))
        return state
    changes = {k: v for k, v in [('MODEL', args.model), ('SEEDS', args.seeds), ('ACTION', args.action),
                                ('LR0', args.lr0), ('BATCH', args.batch)] if v is not None}
    options = queue_options(args.config, changes)
    if not args.no_interactive and not changes:
        options = choose_queue(options, dry_run=args.dry_run)
        if options is None:
            return
    else:
        display_options(options['local'], dry_run=args.dry_run)
    print(f'队列：{options["local"]["MODEL"]}，seeds={options["seeds"]}；各项自动串行，失败留证、不重试。')
    if args.dry_run:
        print('仅预览：未启动检查、队列宿主或训练。')
        return options
    root = create_queue(options)
    print('冻结队列：' + str(root), flush=True)
    if options['local']['ACTION'] == 'train':
        pid = start_background(root)
        print(f'后台队列已启动，启动PID={pid}；GPU占用时自动等待。关闭本终端不影响后台接续。\n'
              f'状态查看：{sys.executable} "{ENTRY}" --status "{root}"\n'
              '每项进度：队列目录/<型号_seed>/队列控制台.txt。全部完成后后台退出。', flush=True)
    else:
        print('只核对索引/清单/初始权重并冻结计划；没有CUDA预检、训练或推理。')
    return root
