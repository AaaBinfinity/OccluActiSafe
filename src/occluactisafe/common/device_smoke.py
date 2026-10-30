"""明确构造的设备诊断；不包含GRU、预测损失或训练器。"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import yaml
from .config import PROJECT_ROOT, _UniqueLoader, load_config
from .device import import_torch, select_device, seed_everything, inventory, tensor_hash
from .run_manifest import create_run_manifest


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load_smoke_config(path, *, overrides=None, cli_overrides=None):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open(encoding='utf-8') as f:
        config = yaml.load(f, Loader=_UniqueLoader)
    required = {'device','seed','steps','matrix_size','batch_size','hidden_size',
                'allocator_limit_mib','minimum_free_mib','output_root','training_budget'}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError('N06配置字段不完整或未知')
    for layer in (overrides, cli_overrides):
        if layer is not None:
            if not isinstance(layer, dict) or set(layer) - required:
                raise ValueError('N06覆盖字段未知')
            config.update(layer)
    for key in ('seed','steps','matrix_size','batch_size','hidden_size','allocator_limit_mib','minimum_free_mib'):
        if type(config[key]) is not int or config[key] < (0 if key in ('seed','minimum_free_mib') else 1):
            raise ValueError(f'{key}必须为有效整数')
    if config['seed'] >= 2**32 or config['steps'] > 100 or config['matrix_size'] > 4096 or config['batch_size'] > 4096 or config['hidden_size'] > 4096:
        raise ValueError('设备smoke配置超出有界范围')
    budget = config['training_budget']
    fields = {'max_wall_seconds','max_optimizer_steps','max_epochs','max_gpu_allocator_mib',
              'max_host_working_set_mib','dataloader_workers','seeds'}
    if not isinstance(budget, dict) or set(budget) != fields:
        raise ValueError('训练预算字段非法')
    for key in fields - {'seeds'}:
        if type(budget[key]) is not int or budget[key] < (0 if key == 'dataloader_workers' else 1):
            raise ValueError('训练预算必须为有限整数上限')
    if not isinstance(budget['seeds'], list) or len(budget['seeds']) < 3 or len(set(budget['seeds'])) != len(budget['seeds']) or any(type(s) is not int or not 0 <= s < 2**32 for s in budget['seeds']):
        raise ValueError('学习模型比较种子至少3个不同有效整数')
    if not isinstance(config['output_root'], str) or not config['output_root'].strip():
        raise ValueError('输出路径不能为空')
    out = Path(config['output_root']).expanduser()
    config['output_root'] = str((out if out.is_absolute() else PROJECT_ROOT / out).resolve())
    return config


def _trial(torch, device, config):
    seed_everything(config['seed'], torch)
    sync = lambda: torch.cuda.synchronize(device) if device.type == 'cuda' else None
    size = config['matrix_size']
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    sync(); start = time.perf_counter()
    product = a @ b
    sync(); matrix_seconds = time.perf_counter() - start
    if not torch.isfinite(product).all().item() or product.device != device:
        raise RuntimeError('矩阵结果非有限或设备不符')
    # 对角与完整乘法在小尺寸子块上独立核查，而非只看没抛异常。
    expected = (a[0] * b[:, 0]).sum()
    torch.testing.assert_close(product[0, 0], expected, rtol=1e-4, atol=1e-4)
    model = torch.nn.Sequential(torch.nn.Linear(16, config['hidden_size']), torch.nn.Tanh(),
                                torch.nn.Linear(config['hidden_size'], 2)).to(device)
    x = torch.randn(config['batch_size'], 16, device=device)
    y = torch.randn(config['batch_size'], 2, device=device)
    initial = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    losses, grads = [], []
    sync(); start = time.perf_counter()
    for _ in range(config['steps']):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = (prediction - y).square().mean()
        loss.backward()
        if not torch.isfinite(loss).item() or any(p.grad is None or not torch.isfinite(p.grad).all().item() for p in model.parameters()):
            raise RuntimeError('loss或梯度非有限')
        grads.append(float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters())).item()))
        losses.append(float(loss.item()))
        optimizer.step()
    sync(); train_seconds = time.perf_counter() - start
    if not all(p.device == device and torch.isfinite(p).all().item() for p in model.parameters()) or x.device != device or y.device != device or prediction.device != device:
        raise RuntimeError('网络参数/张量非有限或未在指定设备')
    if not any(not torch.equal(a, b) for a, b in zip(initial, model.parameters())):
        raise RuntimeError('优化后参数没有改变')
    return {'losses': losses, 'gradient_norms': grads, 'parameters_changed': True,
            'parameter_hashes': [tensor_hash(p) for p in model.parameters()],
            'matrix_hash': tensor_hash(product), 'output_hash': tensor_hash(prediction),
            'tensor_device': str(prediction.device), 'matrix_seconds': matrix_seconds,
            'optimization_seconds': train_seconds, 'optimizer_steps': config['steps']}


def run(config):
    manifest = create_run_manifest(config=load_config(overrides={'runtime': {'seed': config['seed']}}),
                                   input_mode='device_mechanism_fixture', data=None,
                                   output_root=config['output_root'])
    directory = Path(config['output_root']) / manifest['run_id']
    write_json(directory / 'task_config.json', config)
    write_json(directory / 'status.json', {'status': 'running'})
    started = time.perf_counter()
    try:
        torch = import_torch()
        # 种子与cuBLAS配置在首次矩阵运算之前。
        seed_everything(config['seed'], torch)
        device = select_device(config['device'], torch)
        info = inventory(torch, device)
        write_json(directory / 'device_inventory.json', info)
        if device.type == 'cuda':
            if info['free_bytes'] < config['minimum_free_mib'] * 1024**2:
                raise RuntimeError('可用显存低于明确要求，不执行GPU任务')
            cap = config['allocator_limit_mib'] * 1024**2
            if cap > info['total_bytes']:
                raise ValueError('分配器预算大于设备总显存')
            torch.cuda.set_per_process_memory_fraction(cap / info['total_bytes'], device)
            torch.cuda.reset_peak_memory_stats(device)
        trials = [_trial(torch, device, config), _trial(torch, device, config)]
        deterministic_keys = ('losses','gradient_norms','parameter_hashes','matrix_hash','output_hash')
        if any(trials[0][k] != trials[1][k] for k in deterministic_keys):
            raise RuntimeError('同种子同构建同设备的诊断输出不确定')
        memory = None
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            free, total = torch.cuda.mem_get_info(device)
            memory = {'free_after_bytes': free, 'total_bytes': total,
                      'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                      'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
        try:
            driver = subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,driver_version,memory.total,memory.free','--format=csv'], text=True, timeout=15).strip() if device.type == 'cuda' else None
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError('无法记录GPU驱动清单') from exc
        result = {'run_id': manifest['run_id'], 'mode': info['mode'], 'device': info,
                  'driver_inventory': driver, 'memory': memory, 'trials': trials,
                  'deterministic': True, 'determinism_scope': 'same device/build, float32 matmul and Linear/Tanh/SGD only; no cross-version or other-operator guarantee',
                  'wall_seconds': time.perf_counter() - started,
                  'training_budget': config['training_budget'], 'data': None, 'weights': None}
        write_json(directory / 'device_smoke.json', result)
        write_json(directory / 'training_budget.json', config['training_budget'])
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file() and p.name != 'status.json'}
        hashes['uv.lock'] = hashlib.sha256((PROJECT_ROOT / 'uv.lock').read_bytes()).hexdigest()
        write_json(directory / 'evidence_index.json', hashes)
        write_json(directory / 'status.json', {'status': 'completed', 'run_id': manifest['run_id'],
                                               'completed_utc': datetime.now(timezone.utc).isoformat()})
        return {**result, 'run_dir': str(directory)}
    except BaseException as exc:
        write_json(directory / 'status.json', {'status': 'failed', 'run_id': manifest['run_id'],
                   'error_type': type(exc).__name__, 'error': str(exc),
                   'completed_utc': datetime.now(timezone.utc).isoformat()})
        raise
