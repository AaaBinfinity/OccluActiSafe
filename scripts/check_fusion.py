"""N09因果状态与关联机制诊断；IDE无必填参数直接运行。"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "configs/fusion/n09.yaml"
OVERRIDES = {}


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"): stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="N09构造观测机制；非真实数据/视频、训练或避撞实验")
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    parser.add_argument("--output-root")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    try:
        from occluactisafe.fusion.diagnostics import load_diagnostic_config, run_diagnostics
        cli = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
        config = load_diagnostic_config(args.config, overrides=OVERRIDES, cli_overrides=cli)
        print(f"N09构造观测诊断：无真实数据、训练或控制；配置={config}", flush=True)
        result = run_diagnostics(config)
    except (ValueError, RuntimeError, OSError, ImportError, AssertionError) as exc:
        print(f"N09失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"N09完成：{result['run_dir']}；{result['checks_passed']}条机制断言通过", flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
