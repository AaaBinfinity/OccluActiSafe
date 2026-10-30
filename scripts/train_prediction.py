"""N07真实GRU训练默认入口，IDE无必填CLI；导入不训练。"""
import argparse
from pathlib import Path
import sys

PROJECT_ROOT=Path(__file__).resolve().parents[1]
CONFIG_FILE=PROJECT_ROOT/'configs/experiments/tfd_gru.yaml'
OVERRIDES={}


def run(config):
    from occluactisafe.prediction.train import run as train
    return train(config)


def main(argv=None):
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description='N07 GRU训练、验证与检查点恢复')
    parser.add_argument('--config',type=Path,default=CONFIG_FILE)
    parser.add_argument('--resume');parser.add_argument('--seed',type=int)
    parser.add_argument('--output-root');parser.add_argument('--device')
    parser.add_argument('--preflight-only',action='store_true',default=None)
    args=parser.parse_args(argv)
    try:
        from occluactisafe.prediction.train import load_training_config
        cli={k:v for k,v in vars(args).items() if k not in ('config','seed') and v is not None}
        if args.seed is not None:cli['seeds']=[args.seed]
        config=load_training_config(args.config,overrides=OVERRIDES,cli_overrides=cli)
        print(f'N07明确任务配置：{config}',flush=True)
        result=run(config)
    except (OSError,RuntimeError,ValueError,ImportError) as exc:
        print(f'N07训练失败：{type(exc).__name__}: {exc}',file=sys.stderr);return 1
    print(f'N07完成：{result["status"]}；{result["run_dir"]}');return 0


if __name__=='__main__':raise SystemExit(main())
