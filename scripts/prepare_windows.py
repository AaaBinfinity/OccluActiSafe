"""N04直接运行入口：默认配置明确，CLI可选，不训练模型。"""
from pathlib import Path
import argparse
import sys

PROJECT_ROOT=Path(__file__).resolve().parents[1]
CONFIG_FILE=PROJECT_ROOT/'configs/data/n04.yaml'
OVERRIDES={}


def run(config):
    from occluactisafe.data.preprocess import run_preprocess
    return run_preprocess(config,project_root=PROJECT_ROOT)


def main(argv=None):
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description='N04分组、因果窗口与训练集归一化')
    parser.add_argument('--config',type=Path,default=CONFIG_FILE)
    parser.add_argument('--audit-dir');parser.add_argument('--output-root')
    parser.add_argument('--workers',type=int);parser.add_argument('--max-files',type=int)
    args=parser.parse_args(argv)
    try:
        from occluactisafe.data.preprocess import load_preprocess_config
        cli={k:v for k,v in {'audit_dir':args.audit_dir,'output_root':args.output_root,'workers':args.workers,'max_files':args.max_files}.items() if v is not None}
        config=load_preprocess_config(args.config,overrides=OVERRIDES,cli_overrides=cli,project_root=PROJECT_ROOT)
        result=run(config)
    except (ValueError,OSError) as exc:
        print(f'N04预处理失败：{exc}',file=sys.stderr);return 1
    print(f"N04完成：{result['run_id']}；{result['run_dir']}；范围={result['scope']}")
    return 0


if __name__=='__main__':raise SystemExit(main())
