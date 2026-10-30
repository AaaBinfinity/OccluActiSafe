"""N05 CV真实轨迹验证入口；IDE选择.venv后直接运行。"""
from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / 'configs/models/cv.yaml'
OVERRIDES = {}


def run(config):
    from occluactisafe.prediction.evaluate import run as evaluate
    return evaluate(config)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='CV单模式开放环验证，默认不读取独立test')
    parser.add_argument('--config', type=Path, default=CONFIG_FILE)
    parser.add_argument('--windows-dir')
    parser.add_argument('--output-root')
    parser.add_argument('--role')
    parser.add_argument('--max-windows', type=int)
    args = parser.parse_args(argv)
    try:
        from occluactisafe.prediction.evaluate import load_evaluation_config
        cli = {k: v for k, v in vars(args).items() if k != 'config' and v is not None}
        config = load_evaluation_config(args.config, overrides=OVERRIDES, cli_overrides=cli)
        result = run(config)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print(f'N05评测失败：{exc}', file=sys.stderr)
        return 1
    print(f"N05完成：{result['run_id']}；{result['run_dir']}；{result['overall']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
