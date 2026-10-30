"""N06直接运行真实GPU设备检查；CPU只可显式选择为诊断。"""
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / 'configs/runtime/n06.yaml'
OVERRIDES = {}


def run(config):
    from occluactisafe.common.device_smoke import run as smoke
    return smoke(config)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='N06设备机制夹具；不是课题模型训练')
    parser.add_argument('--config', type=Path, default=CONFIG_FILE)
    parser.add_argument('--device')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--allocator-limit-mib', type=int)
    parser.add_argument('--output-root')
    args = parser.parse_args(argv)
    try:
        from occluactisafe.common.device_smoke import load_smoke_config
        cli = {k: v for k, v in vars(args).items() if k != 'config' and v is not None}
        config = load_smoke_config(args.config, overrides=OVERRIDES, cli_overrides=cli)
        print(f'N06设备机制检查：{config}', flush=True)
        result = run(config)
    except (RuntimeError, ValueError, OSError, ImportError) as exc:
        print(f'N06检查失败：{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    print(f"N06完成：{result['mode']}；{result['run_dir']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
