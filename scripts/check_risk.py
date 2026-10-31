"""N10几何/原始风险机制诊断；IDE无必填参数直接运行。"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "configs/risk/n10.yaml"
OVERRIDES = {}


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"): stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="N10构造几何/覆盖/未知区机制；无训练、视频或避撞闭环")
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    parser.add_argument("--output-root")
    args = parser.parse_args(argv)
    try:
        from occluactisafe.risk.diagnostics import load_diagnostic_config, run_diagnostics
        cli = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
        config = load_diagnostic_config(args.config, overrides=OVERRIDES, cli_overrides=cli)
        print(f"N10构造机制：不读真实数据、不训练、不执行控制；配置={config}", flush=True)
        result = run_diagnostics(config)
    except (ValueError, RuntimeError, OSError, ImportError, AssertionError) as exc:
        print(f"N10失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"N10完成：{result['run_dir']}；{result['checks_passed']}条机制断言通过", flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
