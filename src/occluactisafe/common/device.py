"""N06训练设备与种子工具；模块导入不加载Torch或初始化CUDA。"""
import hashlib
import os
import random
import re
import numpy as np


def import_torch():
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise RuntimeError('PyTorch缺失或构建无法加载；先执行uv sync --locked --group training') from exc
    return torch


def select_device(name, torch):
    if name == 'cpu':
        return torch.device('cpu')
    if not isinstance(name, str) or not re.fullmatch(r'cuda:\d+', name):
        raise ValueError('设备必须显式为cuda:N或cpu（诊断）；禁止自动回退')
    if torch.version.cuda is None or not torch.cuda.is_available():
        raise RuntimeError('CUDA构建、驱动或GPU不可用；不回退CPU')
    index = int(name.split(':')[1])
    if index >= torch.cuda.device_count():
        raise ValueError(f'指定设备{name}不存在，可见GPU数={torch.cuda.device_count()}')
    device = torch.device(name)
    torch.cuda.set_device(device)
    return device


def seed_everything(seed, torch):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed须为[0,2**32)整数')
    # 在首次cuBLAS工作前设置；已由外部配置的不兼容值不能默默覆盖。
    workspace = os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if workspace not in (':4096:8', ':16:8'):
        raise ValueError('CUBLAS_WORKSPACE_CONFIG不符合确定性要求')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def inventory(torch, device):
    result = {'torch': torch.__version__, 'cuda_build': torch.version.cuda,
              'cudnn': torch.backends.cudnn.version(), 'device': str(device),
              'mode': 'gpu_acceptance' if device.type == 'cuda' else 'cpu_diagnostic',
              'build_config': torch.__config__.show()}
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        free, total = torch.cuda.mem_get_info(device)
        result.update(name=props.name, capability=list(torch.cuda.get_device_capability(device)),
                      arch_list=torch.cuda.get_arch_list(), total_bytes=total, free_bytes=free,
                      device_index=device.index, multiprocessors=props.multi_processor_count)
    return result


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
