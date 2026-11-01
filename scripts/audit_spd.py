"""审计正式V2X-Seq SPD RGB双视角、标注、标定、配对和路口划分。IDE无必填参数。"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import uuid

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "data" / "spd.yaml"


def run(config: dict | None = None, *, dataset_root: str | Path | None = None,
        output_root: str | Path | None = None) -> dict:
    from occluactisafe.data.adapters.spd import audit_spd, sha256_file

    config = dict(config or yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8")))
    root = Path(dataset_root or os.environ.get("OAS_DATASET_ROOT") or config["dataset_root"]).expanduser().resolve()
    base = Path(output_root or config["output_root"]).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"SPD硬盘数据目录不可用：{root}")
    if not base.parent.exists():
        raise FileNotFoundError(f"N15 artifacts父目录未整理或硬盘未接入：{base.parent}")
    run_id = "n15_spd_" + uuid.uuid4().hex
    run_dir = base / "02_运行产物" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now().astimezone().isoformat()
    config_copy = {**config, "dataset_root": str(root), "output_root": str(base)}
    config_path = run_dir / "冻结审计配置.yaml"
    config_path.write_text(yaml.safe_dump(config_copy, allow_unicode=True, sort_keys=False), encoding="utf-8")
    try:
        report = audit_spd(root, seed=int(config.get("seed", 20261101)),
                           verify_image_decode=bool(config.get("verify_image_decode", True)))
        report["run_id"] = run_id
        report["started_at"] = started
        report["finished_at"] = datetime.now().astimezone().isoformat()
        report["config"] = {"path": str(config_path), "sha256": sha256_file(config_path)}
        report["source_url"] = config.get("source_url")
        report["dataset_version"] = config.get("dataset_version", "unknown")
        report["acquired_date"] = config.get("acquired_date", "unknown")
        report["usage_conditions"] = config.get("usage_conditions", "unverified")
        report["license_evidence"] = config.get("license_evidence", "unverified")
        report["archive_sha256"] = "unavailable: original downloaded archives are not present"
        report_path = run_dir / "SPD完整审计报告.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        split_path = run_dir / "SPD冻结分组清单.json"
        split_path.write_text(json.dumps(report["split"], ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        summary = {key: value for key, value in report.items() if key not in ("files", "split")}
        summary_path = run_dir / "SPD审计摘要.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        manifest = {"run_id": run_id, "node": "N15", "source_commit": os.environ.get("OAS_SOURCE_COMMIT", "uncommitted"),
                    "config_sha256": report["config"]["sha256"], "outputs": {}}
        for path in (report_path, split_path, summary_path):
            manifest["outputs"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        manifest_path = run_dir / "运行证据索引.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report["evidence_manifest"] = str(manifest_path)
        return {"run_id": run_id, "run_dir": str(run_dir), "status": "completed" if report["ready_for_rgb_audit_and_project_split"] else "failed",
                "report": report, "report_path": str(report_path), "split_path": str(split_path),
                "manifest_path": str(manifest_path)}
    except Exception as exc:
        failure = {"run_id": run_id, "status": "failed", "started_at": started,
                   "finished_at": datetime.now().astimezone().isoformat(),
                   "error": f"{type(exc).__name__}: {exc}", "config": str(config_path)}
        (run_dir / "失败记录.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root")
    parser.add_argument("--output-root")
    args = parser.parse_args(argv)
    try:
        config = yaml.safe_load(args.config.expanduser().resolve().read_text(encoding="utf-8"))
        result = run(config, dataset_root=args.dataset_root, output_root=args.output_root)
    except (OSError, ValueError, KeyError) as exc:
        print(f"SPD审计失败：{exc}", file=sys.stderr)
        return 1
    print(f"run_id={result['run_id']} status={result['status']} report={result['report_path']}")
    print(f"真实RGB帧={result['report']['source_indexes']}；输入清单SHA256={result['report']['input_file_manifest_sha256']}")
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
