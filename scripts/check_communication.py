"""N08默认进程内通信机制检查，支持IDE直接运行。"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "configs/communication/n08.yaml"
OVERRIDES = {}


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="N08进程内机制夹具；不是设备通信、训练或避撞实验")
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root")
    args = parser.parse_args(argv)
    try:
        from occluactisafe.communication.diagnostics import load_diagnostic_config, run_diagnostics
        cli = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
        config = load_diagnostic_config(args.config, overrides=OVERRIDES, cli_overrides=cli)
        print(f"N08机制检查：无真实数据/训练/网络设备；配置={config}", flush=True)
        result = run_diagnostics(config)
    except (ValueError, RuntimeError, OSError, ImportError, AssertionError) as exc:
        print(f"N08失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"N08完成：{result['run_dir']}；必需诊断{result['checks_passed']}项通过", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
