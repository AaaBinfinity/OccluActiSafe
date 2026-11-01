"""N16 SPD开发片段读取：RGB作输入、标注与标定只作离线评测/来源记录。"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator

import numpy as np

from occluactisafe.data.adapters.spd import (
    _read_json, normalize_visible_box, resolve_index_path, sha256_file,
)
from occluactisafe.evaluation.detection_metrics import GroundTruthObject
from occluactisafe.perception.types import TrackNamespace


CLASS_MAP = {
    "Car": "vehicle", "Truck": "vehicle", "Van": "vehicle", "Bus": "vehicle",
    "Pedestrian": "pedestrian", "Cyclist": "cyclist", "Tricyclist": "cyclist",
    "Motorcyclist": "cyclist", "Barrowlist": "barrowlist", "TrafficCone": "traffic_cone",
}


@dataclass(frozen=True)
class SPDPerceptionFrame:
    frame_id: str
    timestamp_us: int
    timestamp_s: float
    width: int
    height: int
    namespace: TrackNamespace
    intersection_loc: str
    project_role: str
    image_path: Path
    image_sha256: str
    label_path: Path
    label_sha256: str
    intrinsic_path: Path
    intrinsic_sha256: str
    image_rgb: np.ndarray
    ground_truth: tuple[GroundTruthObject, ...]
    raw_category_counts: dict[str, int]
    unsupported_labels: tuple[dict, ...]
    truncation_count: int
    official_occlusion_counts: dict[str, int]


def _index_rows(dataset_root: Path, side: str, cache: dict) -> dict[tuple[str, str], dict]:
    key = (str(dataset_root), side)
    if key not in cache:
        path = dataset_root / "V2X-Seq-SPD" / side / "data_info.json"
        rows = _read_json(path)
        if not isinstance(rows, list):
            raise ValueError(f"{path}顶层必须为数组")
        index = {}
        for row in rows:
            pair = (str(row.get("sequence_id", "")), str(row.get("frame_id", "")))
            if not all(pair) or pair in index:
                raise ValueError(f"SPD索引含空或重复帧主键：{side}:{pair}")
            index[pair] = row
        cache[key] = index
    return cache[key]


def load_spd_perception_frame(dataset_root: Path, split_row: dict,
                              *, index_cache: dict | None = None,
                              expected_source_sha256: dict[str, str] | None = None) -> SPDPerceptionFrame:
    """读取一帧真实SPD RGB与仅供评分的GT；不把标注或标定喂给检测器。"""
    root = Path(dataset_root).expanduser().resolve()
    side = split_row.get("side")
    if side not in ("vehicle-side", "infrastructure-side"):
        raise ValueError("SPD视角side无效")
    if split_row.get("project_role") not in ("train", "model_val", "test"):
        raise ValueError("隔离或未知split记录不能进入适配")
    cache = index_cache if index_cache is not None else {}
    indexed = _index_rows(root, side, cache)
    key = (str(split_row.get("sequence_id", "")), str(split_row.get("frame_id", "")))
    row = indexed.get(key)
    if row is None:
        raise ValueError(f"SPD帧索引缺失：{side}:{key}")
    image_path, _ = resolve_index_path(root, side, str(row.get("image_path", "")), kind="image")
    label_path, _ = resolve_index_path(root, side, str(row.get("label_camera_std_path", "")), kind="metadata")
    intrinsic_path, _ = resolve_index_path(root, side, str(row.get("calib_camera_intrinsic_path", "")), kind="metadata")
    if image_path is None or label_path is None or intrinsic_path is None:
        raise FileNotFoundError(f"SPD图像/相机标签/内参缺失：{side}:{key}")
    from PIL import Image
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    if (width, height) != (int(split_row["width"]), int(split_row["height"])):
        raise ValueError("实读RGB尺寸与冻结分组清单不一致")
    image_sha = sha256_file(image_path)
    expected_sha = split_row.get("image_sha256")
    if expected_sha and image_sha != expected_sha:
        raise ValueError(f"SPD RGB哈希与N15冻结清单不一致：{image_path}")
    image_rel = image_path.relative_to(root).as_posix()
    audit_image_sha = (expected_source_sha256 or {}).get(image_rel)
    if audit_image_sha is None or image_sha != audit_image_sha:
        raise ValueError(f"SPD RGB与N15不可变文件清单不一致：{image_rel}")
    for path in (label_path, intrinsic_path):
        rel = path.relative_to(root).as_posix()
        expected = (expected_source_sha256 or {}).get(rel)
        if expected is None:
            raise ValueError(f"N15不可变文件清单未登记SPD标注/标定源：{rel}")
        if sha256_file(path) != expected:
            raise ValueError(f"SPD标注/标定文件与N15不可变清单不一致：{rel}")
    timestamp_raw = row.get("image_timestamp")
    try:
        timestamp_us = int(timestamp_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SPD采集时间缺失或非法：{side}:{key}") from exc
    if timestamp_us < 0:
        raise ValueError("SPD图像采集时间不能为负")
    labels = _read_json(label_path)
    if not isinstance(labels, list):
        raise ValueError("SPD相机标注必须为数组")
    gt, unsupported, raw_counts, occ_counts = [], [], {}, {}
    truncation_count = 0
    for label_index, label in enumerate(labels):
        if not isinstance(label, dict):
            unsupported.append({"index": label_index, "reason": "label_not_object"})
            continue
        raw_category = str(label.get("type", ""))
        raw_counts[raw_category] = raw_counts.get(raw_category, 0) + 1
        if raw_category not in CLASS_MAP:
            unsupported.append({"index": label_index, "raw_category": raw_category,
                                "track_id": str(label.get("track_id", "")),
                                "reason": "unsupported_category"})
            continue
        normalized = normalize_visible_box(label, width, height)
        box = normalized["visible_box"]
        max_dimension = max(box["xmax"]-box["xmin"], box["ymax"]-box["ymin"])
        pixel_scale = ("small_le_32px" if max_dimension <= 32 else
                       "medium_33_96px" if max_dimension <= 96 else "large_over_96px")
        track_id = label.get("track_id")
        gt.append(GroundTruthObject(
            (box["xmin"], box["ymin"], box["xmax"], box["ymax"]),
            CLASS_MAP[raw_category], None if track_id in (None, "") else str(track_id), raw_category,
            pixel_scale, str(label.get("occluded_state", "missing")),
            str(label.get("truncated_state", "missing"))))
        occ = str(label.get("occluded_state", "missing"))
        occ_counts[occ] = occ_counts.get(occ, 0) + 1
        truncated = str(label.get("truncated_state", ""))
        if truncated.lower() not in ("0", "false", "none", "not_truncated", "not_truncated_by_image_edge", ""):
            truncation_count += 1
    if not any(key in json.loads(intrinsic_path.read_text(encoding="utf-8-sig"))
               for key in ("cam_K", "camera_matrix")):
        raise ValueError(f"SPD内参文件缺少相机矩阵：{intrinsic_path}")
    location = str(split_row.get("intersection_loc", ""))
    sequence = str(split_row.get("sequence_id", ""))
    namespace = TrackNamespace("spd", side, f"{location}-{sequence}")
    return SPDPerceptionFrame(
        frame_id=f"{side}:{sequence}:{key[1]}", timestamp_us=timestamp_us,
        timestamp_s=timestamp_us/1_000_000.0, width=width, height=height,
        namespace=namespace, intersection_loc=location,
        project_role=split_row["project_role"], image_path=image_path,
        image_sha256=image_sha, label_path=label_path,
        label_sha256=sha256_file(label_path), intrinsic_path=intrinsic_path,
        intrinsic_sha256=sha256_file(intrinsic_path), image_rgb=rgb,
        ground_truth=tuple(gt), raw_category_counts=raw_counts,
        unsupported_labels=tuple(unsupported), truncation_count=truncation_count,
        official_occlusion_counts=occ_counts)


def iter_spd_perception_frames(dataset_root: Path, split_manifest: dict, *,
                               project_role: str = "model_val",
                               expected_source_sha256: dict[str, str] | None = None) -> Iterator[SPDPerceptionFrame]:
    """稳定顺序遍历冻结split；缺行不会静默跳过。"""
    if project_role not in ("train", "model_val", "test"):
        raise ValueError("project_role必须是train/model_val/test")
    from occluactisafe.data.adapters.spd import audit_spd_split
    audit_spd_split(split_manifest)
    cache = {}
    rows = [row for row in split_manifest["files"] if row.get("project_role") == project_role]
    for row in sorted(rows, key=lambda item: (item["side"], item["sequence_id"], item["frame_id"])):
        yield load_spd_perception_frame(dataset_root, row, index_cache=cache,
                                        expected_source_sha256=expected_source_sha256)


def load_frozen_spd_split(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        value = json.load(stream)
    from occluactisafe.data.adapters.spd import canonical_sha256
    supplied_sha = value.get("sha256")
    calculated_sha = canonical_sha256({key: item for key, item in value.items() if key != "sha256"})
    if supplied_sha != calculated_sha:
        raise ValueError("SPD冻结分组清单自身sha256校验失败")
    from occluactisafe.data.adapters.spd import audit_spd_split
    audit_spd_split(value)
    return value


def load_frozen_spd_audit(path: Path, split_manifest: dict) -> tuple[dict[str, str], str]:
    """校验N15审计报告版本并返回原始文件SHA映射。"""
    from occluactisafe.data.adapters.spd import canonical_sha256
    audit_path = Path(path)
    with audit_path.open("r", encoding="utf-8-sig") as stream:
        report = json.load(stream)
    supplied_sha = report.get("report_sha256")
    calculated_sha = canonical_sha256({key: value for key, value in report.items()
                                       if key != "report_sha256"})
    if supplied_sha != calculated_sha:
        raise ValueError("N15 SPD审计报告自身sha256校验失败")
    if report.get("split_sha256") != split_manifest.get("sha256"):
        raise ValueError("N15审计报告与冻结分组清单不是同一版本")
    files = report.get("files")
    if not isinstance(files, list):
        raise ValueError("N15审计报告缺少不可变文件清单")
    hashes = {str(row["path"]): str(row["sha256"]) for row in files
              if isinstance(row, dict) and row.get("path") and row.get("sha256")}
    if len(hashes) != len(files):
        raise ValueError("N15不可变文件清单存在重复或无效记录")
    return hashes, supplied_sha
