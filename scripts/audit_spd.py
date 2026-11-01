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
    expected_artifacts = root.parent / "artifacts"
    if not base.is_relative_to(expected_artifacts.resolve()):
        raise ValueError(f"SPD审计产物必须保存在数据集硬盘artifacts内：{base}")
    base.mkdir(parents=True, exist_ok=True)
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
        report["visual_review_sheets"] = render_review_sheets(report, root, run_dir / "分层抽查图")
        from occluactisafe.data.adapters.spd import canonical_sha256
        report["report_sha256"] = canonical_sha256({k: v for k, v in report.items()
                                                     if k != "report_sha256"})
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


def render_review_sheets(report: dict, dataset_root: Path, output_dir: Path) -> list[dict]:
    """每个存在的类别×遮挡层生成≤30帧叠加联系表，图中文字和文件名中文。"""
    from PIL import Image, ImageDraw, ImageFont
    from occluactisafe.data.adapters.spd import sha256_file

    root = Path(dataset_root)
    output_dir.mkdir(parents=True, exist_ok=False)
    category_zh = {"Car": "小汽车", "Truck": "卡车", "Van": "厢式车", "Bus": "公交车",
                   "Pedestrian": "行人", "Cyclist": "骑行者", "Tricyclist": "三轮车骑行者",
                   "Motorcyclist": "摩托车骑行者", "Barrowlist": "手推车行人", "TrafficCone": "交通锥"}
    side_zh = {"vehicle-side": "车端", "infrastructure-side": "路侧"}
    sheets = []
    source_hashes = {entry["path"]: entry["sha256"] for entry in report["files"]}
    cell_w, cell_h, columns, rows = 400, 240, 4, 8
    font = ImageFont.load_default()
    for stratum in report["stratified_review_samples"]:
        samples = stratum["sample_frames"]
        if not samples:
            continue
        sheet = Image.new("RGB", (cell_w * columns, 35 + cell_h * rows), "#17212b")
        draw = ImageDraw.Draw(sheet)
        title = (f"{side_zh[stratum['side']]} | {category_zh.get(stratum['category'], stratum['category'])} | "
                 f"遮挡{stratum['occluded_state']} | 抽查{len(samples)}/{stratum['population_frames']}帧")
        draw.text((12, 10), title, fill="white", font=font)
        for index, sample in enumerate(samples):
            col, row = index % columns, index // columns
            x0, y0 = col * cell_w, 35 + row * cell_h
            image_path = root / sample["source_file"]
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((cell_w, cell_h - 25), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (cell_w, cell_h - 25), "#283746")
            tile.paste(image, ((cell_w - image.width)//2, (cell_h - 25 - image.height)//2))
            labels_path = (root / "V2X-Seq-SPD" / stratum["side"] / "label" / "camera" /
                           f"{sample['frame_id']}.json")
            labels = json.loads(labels_path.read_text(encoding="utf-8-sig"))
            scale_x, scale_y = image.width / 1920.0, image.height / 1080.0
            offset_x, offset_y = (cell_w - image.width)//2, (cell_h - 25 - image.height)//2
            painter = ImageDraw.Draw(tile)
            for obj in labels:
                if obj.get("type") != stratum["category"] or str(obj.get("occluded_state", "missing")) != str(stratum["occluded_state"]):
                    continue
                box = obj.get("2d_box", {})
                try:
                    coords = [float(box[k]) for k in ("xmin", "ymin", "xmax", "ymax")]
                except (KeyError, TypeError, ValueError):
                    continue
                x1, y1, x2, y2 = coords
                painter.rectangle((offset_x + x1*scale_x, offset_y + y1*scale_y,
                                   offset_x + x2*scale_x, offset_y + y2*scale_y),
                                  outline="#ffdf00", width=2)
            sheet.paste(tile, (x0, y0))
            draw.text((x0+6, y0+cell_h-21),
                      f"seq={sample['sequence_id']} frame={sample['frame_id']} 标签框=黄色",
                      fill="#ffffff", font=font)
        filename = (f"{side_zh[stratum['side']]}_{category_zh.get(stratum['category'], stratum['category'])}_"
                    f"遮挡{stratum['occluded_state']}_分层抽查.png")
        path = output_dir / filename
        sheet.save(path, format="PNG", optimize=False)
        sheets.append({"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size,
                       "side": stratum["side"], "category": stratum["category"],
                       "occluded_state": stratum["occluded_state"],
                       "sample_count": len(samples), "population_frames": stratum["population_frames"],
                       "source_files": [x["source_file"] for x in samples],
                       "source_file_hashes": [source_hashes[x["source_file"]] for x in samples],
                       "label_path_hashes": [source_hashes[
                           f"V2X-Seq-SPD/{stratum['side']}/label/camera/{x['frame_id']}.json"]
                                             for x in samples]})
    return sheets


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
