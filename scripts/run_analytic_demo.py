"""N11解析闭环演示；IDE无必填CLI，不连接CARLA。"""
from pathlib import Path
import argparse
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / 'configs/analytic/n11.yaml'
OVERRIDES = {}


def main(argv=None):
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'): stream.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=CONFIG_FILE);parser.add_argument('--output-root')
    args=parser.parse_args(argv)
    try:
        from occluactisafe.core.analytic_diagnostics import load_config,run
        config=load_config(args.config,overrides={**OVERRIDES,**({'output_root':args.output_root} if args.output_root is not None else {})})
        result=run(config)
    except (ValueError,OSError,RuntimeError,ImportError,AssertionError) as exc:
        print(f'N11失败：{type(exc).__name__}: {exc}',file=sys.stderr);return 1
    print(f"N11完成（仅解析闭环）：{result['run_dir']}，{result['checks_passed']}条机制断言通过")
    return 0


if __name__=='__main__':raise SystemExit(main())
