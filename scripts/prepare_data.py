"""N03：审计已获取的 TFD 数据；导入不触发运行或下载。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "configs" / "data" / "tfd.yaml"
OVERRIDES = {}


def run(config):
    """运行已获取数据的审计；参数语义见 N03 文档。"""
    from occluactisafe.data.tfd_audit import run_audit
    return run_audit(config, project_root=PROJECT_ROOT)


def main(argv=None):
    # Windows 重定向输出可能使用无法表示中文的系统编码；IDE 的替代流不一定支持 reconfigure。
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="审计已取得的 V2X-Seq TFD 数据")
    parser.add_argument("--config", type=Path, default=CONFIG_FILE,
                        help="N03 任务配置文件，默认 configs/data/tfd.yaml")
    parser.add_argument("--data-root", help="明确指定已准备的数据根目录")
    parser.add_argument("--output-root", help="输出证据目录")
    parser.add_argument("--scope", choices=("example", "full"), help="数据范围，不隐含改变数据")
    args = parser.parse_args(argv)
    from occluactisafe.data.tfd_audit import load_task_config
    cli = {key: value for key, value in {"data_root": args.data_root,
           "output_root": args.output_root, "dataset_scope": args.scope}.items() if value is not None}
    try:
        config = load_task_config(args.config, overrides=OVERRIDES, cli_overrides=cli,
                                  project_root=PROJECT_ROOT)
        result = run(config)
    except (ValueError, OSError, csv.Error) as exc:
        print(f"TFD 审计失败：{exc}", file=sys.stderr)
        return 1
    print(f"TFD 审计完成：run_id={result['run_id']}，目录={result['run_dir']}")
    print(f"范围={result['dataset_scope']}；完整数据就绪={result['formal_data_ready']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
