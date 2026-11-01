"""IDE直接运行的N16 SPD model_val图像/类别/时间适配审计入口。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import shutil
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from occluactisafe.data.spd_perception import (
    CLASS_MAP, iter_spd_perception_frames, load_frozen_spd_audit, load_frozen_spd_split,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "perception" / "n16.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _draw_ground_truth(frame, output: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont
    image = Image.fromarray(frame.image_rgb.copy())
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("msyh.ttc", 16) if Path("C:/Windows/Fonts/msyh.ttc").is_file() else ImageFont.load_default()
    colors = {"vehicle": "#00D084", "pedestrian": "#FF4D4F", "cyclist": "#FFD666",
              "barrowlist": "#40A9FF", "traffic_cone": "#9254DE"}
    legend = []
    labels_zh = {"vehicle": "车辆", "pedestrian": "行人", "cyclist": "骑行者",
                 "barrowlist": "手推车目标", "traffic_cone": "交通锥"}
    for index, gt in enumerate(frame.ground_truth, start=1):
        color = colors.get(gt.class_id, "#FFFFFF")
        x1, y1, x2, y2 = gt.xyxy
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        # 框内不逐个贴长标签，避免远处密集目标的文字互相覆盖。
        draw.ellipse((x1-3, y1-3, x1+3, y1+3), fill=color)
        legend.append((color, f"{index:02d} {labels_zh.get(gt.class_id, gt.class_id)} "
                               f"原类={gt.raw_category} ID={gt.track_id or '缺失'} "
                               f"官方遮挡={gt.official_occlusion_state} 截断={gt.truncation_state}"))
    row_height = 22
    panel_height = min(36 + row_height*len(legend), max(36, image.height-12))
    panel_width = min(image.width-24, 780)
    draw.rectangle((8, 8, 8+panel_width, 8+panel_height), fill=(0, 0, 0))
    draw.text((14, 10), "仅官方GT叠加；非检测器推理结果", fill="#FFFFFF", font=font)
    for index, (color, text) in enumerate(legend):
        y = 34+row_height*index
        if y+row_height > image.height-4:
            draw.text((14, y), "其余目标详见图像类别时间映射.json", fill="#FFFFFF", font=font)
            break
        draw.text((14, y), text, fill=color, font=font)
    image.save(output, format="PNG", optimize=False)


def run(config_path: Path | None = None) -> dict:
    config_path = (config_path or DEFAULT_CONFIG).resolve()
    import yaml
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_root = Path(config["dataset"]["root"]).expanduser().resolve()
    split_path = Path(config["dataset"]["frozen_split_manifest"]).expanduser().resolve()
    split = load_frozen_spd_split(split_path)
    audit_path = Path(config["dataset"]["frozen_audit_report"]).expanduser().resolve()
    expected_source_sha256, audit_internal_sha256 = load_frozen_spd_audit(audit_path, split)
    run_id = f"n16_spd_{uuid.uuid4().hex}"
    output_root = Path(config["visualization"]["output_root"]).expanduser().resolve()
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, run_dir / "冻结适配配置.yaml")
    started = datetime.now(timezone.utc).astimezone().isoformat()
    counts = Counter()
    categories = Counter()
    occlusions = Counter()
    sides = Counter()
    sizes = Counter()
    per_namespace_time = defaultdict(list)
    unsupported = []
    mapping = []
    visualized = defaultdict(int)
    errors = []
    for frame in iter_spd_perception_frames(dataset_root, split,
                                             project_role=config["dataset"]["project_role"],
                                             expected_source_sha256=expected_source_sha256):
        counts["frames"] += 1
        sides[frame.namespace.sensor_id] += 1
        counts["objects"] += len(frame.ground_truth)
        counts["negative_frames"] += int(not frame.ground_truth)
        counts["truncated_objects"] += frame.truncation_count
        per_namespace_time[frame.namespace.key].append((frame.timestamp_s, frame.frame_id))
        for label in frame.ground_truth:
            categories[label.raw_category] += 1
            sizes["small_le_32px"] += int(max(label.xyxy[2]-label.xyxy[0], label.xyxy[3]-label.xyxy[1]) <= 32)
            sizes["medium_33_96px"] += int(32 < max(label.xyxy[2]-label.xyxy[0], label.xyxy[3]-label.xyxy[1]) <= 96)
            sizes["large_over_96px"] += int(max(label.xyxy[2]-label.xyxy[0], label.xyxy[3]-label.xyxy[1]) > 96)
        for state, amount in frame.official_occlusion_counts.items():
            occlusions[state] += amount
        for item in frame.unsupported_labels:
            unsupported.append({"frame_id": frame.frame_id, "side": frame.namespace.sensor_id, **item})
        mapping.append({
            "frame_id": frame.frame_id, "timestamp_us": frame.timestamp_us,
            "timestamp_s": frame.timestamp_s, "clock_unit_source": "SPD image_timestamp microseconds",
            "camera_namespace": frame.namespace.key, "camera_side": frame.namespace.sensor_id,
            "episode_namespace": frame.namespace.episode_id,
            "intersection_loc": frame.intersection_loc, "project_role": frame.project_role,
            "image_sha256": frame.image_sha256,
            "annotation_sha256": frame.label_sha256,
            "annotation_path": frame.label_path.relative_to(dataset_root).as_posix(),
            "intrinsic_sha256_source_only_not_detector_input": frame.intrinsic_sha256,
            "objects": [{"xyxy_original_pixels": list(item.xyxy), "class_id": item.class_id,
                         "raw_category": item.raw_category, "track_id": item.track_id,
                         "pixel_scale": item.pixel_scale,
                         "official_occlusion_state": item.official_occlusion_state,
                         "truncation_state": item.truncation_state}
                        for item in frame.ground_truth],
        })
        # 每侧最多6张，轮转选择有标注与空帧，不把GT叠加图当模型检测结果。
        quota = int(config["visualization"]["maximum_examples_per_camera"])
        if visualized[frame.namespace.sensor_id] < quota:
            output = run_dir / (f"标注叠加_{'车端' if frame.namespace.sensor_id == 'vehicle-side' else '路侧'}_"
                                f"{frame.intersection_loc}_{frame.frame_id.replace(':', '_')}.png")
            _draw_ground_truth(frame, output)
            visualized[frame.namespace.sensor_id] += 1
    for namespace, values in per_namespace_time.items():
        values.sort(key=lambda item: item[1])
        for (previous, _), (current, _) in zip(values, values[1:]):
            counts["nonincreasing_timestamps"] += int(current <= previous)
    if not counts["frames"]:
        errors.append("model_val split为空")
    if counts["nonincreasing_timestamps"]:
        errors.append("同episode采集时间不严格递增")
    if unsupported:
        errors.append("存在未映射类别，逐项保留在unsupported_labels.json")
    counts["camera_episode_namespaces"] = len(per_namespace_time)
    counts["visualized_frames"] = sum(visualized.values())
    result = {
        "schema_version": "1.0", "node": "N16", "run_id": run_id,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": "passed" if not errors else "failed",
        "dataset": "V2X-Seq-SPD", "project_role": config["dataset"]["project_role"],
        "split_sha256": split.get("sha256"), "config_sha256": _sha256(config_path),
        "split_file_sha256": _sha256(split_path),
        "n15_audit_report_file_sha256": _sha256(audit_path),
        "n15_audit_report_internal_sha256": audit_internal_sha256,
        "n15_input_file_manifest_sha256": json.loads(audit_path.read_text(encoding="utf-8-sig")).get("input_file_manifest_sha256"),
        "class_map": CLASS_MAP, "counts": dict(counts),
        "raw_categories": dict(sorted(categories.items())),
        "official_occlusion_states": dict(sorted(occlusions.items())),
        "bbox_scale_bins": dict(sizes), "camera_sides": dict(sides),
        "detection_inference_performed": False,
        "ground_truth_is_scoring_only": True,
        "intrinsics_used_as_detector_input": False,
        "errors": errors,
        "limitations": ["此入口核验数据适配和GT叠加，不运行任何检测权重或报告模型精度",
                        "输入GT框不等价于独立人工复核；GT叠加图只用于抽查"],
    }
    (run_dir / "图像类别时间映射.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (run_dir / "不支持类别明细.json").write_text(
        json.dumps(unsupported, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "model_val适配审计.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    manifest = []
    for path in sorted(run_dir.iterdir()):
        if path.is_file():
            manifest.append({"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    (run_dir / "产物清单.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "status": result["status"], "counts": dict(counts),
                      "errors": errors, "run_dir": str(run_dir)}, ensure_ascii=False, indent=2))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    result = run(args.config)
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
