"""V2X-Seq SPD 只读索引、逐帧完整性审计与路口分组。

不把 LiDAR 坐标标签解释为图像时刻世界坐标；图像包扁平布局只按索引路径
basename 作唯一回退，并将回退记入报告。原始文件不移动、不覆盖。
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

SIDES = ("vehicle-side", "infrastructure-side")
SIDE_IMAGE_PACKAGES = {
    "vehicle-side": "V2X-Seq-SPD-vehicle-side-image",
    "infrastructure-side": "V2X-Seq-SPD-infrastructure-side-image",
}
ROLES = ("train", "model_val", "test")
EXPECTED_LABEL_TYPES = {
    "Car", "Truck", "Van", "Bus", "Pedestrian", "Cyclist", "Tricyclist",
    "Motorcyclist", "Barrowlist", "TrafficCone",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> Any:
    def reject_constant(value: str):
        raise ValueError(f"非有限JSON常量：{value}")

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"重复JSON字段：{key}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream, parse_constant=reject_constant,
                         object_pairs_hook=unique_pairs)


def resolve_index_path(dataset_root: Path, side: str, relative: str,
                       *, kind: str) -> tuple[Path | None, str]:
    """先按官方相对路径查找；扁平 RGB 包只允许 basename 唯一映射。"""
    root = Path(dataset_root).resolve()
    side_root = (root / "V2X-Seq-SPD" / side).resolve()
    candidate = (side_root / relative).resolve()
    if not candidate.is_relative_to(side_root):
        raise ValueError(f"索引路径越界：{relative}")
    if candidate.is_file():
        return candidate, "indexed_relative_path"
    if kind == "image":
        image_root = (root / SIDE_IMAGE_PACKAGES[side]).resolve()
        flat = (image_root / Path(relative).name).resolve()
        if flat.parent == image_root and flat.is_file():
            return flat, "flat_package_basename"
    return None, "missing"


def _bbox_check(label: dict, width: int, height: int) -> str | None:
    box = label.get("2d_box")
    if not isinstance(box, dict):
        return "missing_2d_box"
    try:
        x1, y1, x2, y2 = (float(box[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
    except (KeyError, TypeError, ValueError):
        return "invalid_2d_box_fields"
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return "nonfinite_2d_box"
    if x2 <= x1 or y2 <= y1:
        return "nonpositive_2d_box"
    # Official boxes may meet image edges; outside coordinates remain an audit item.
    if x1 < -1e-3 or y1 < -1e-3 or x2 > width + 1e-3 or y2 > height + 1e-3:
        return "2d_box_outside_image"
    return None


def normalize_visible_box(label: dict, width: int, height: int) -> dict:
    """返回派生裁切框并保留raw_box；只修正≤0.001像素的浮点边界误差。"""
    error = _bbox_check(label, width, height)
    if error:
        raise ValueError(f"不可安全裁切的SPD可见框：{error}")
    raw = {key: float(label["2d_box"][key]) for key in ("xmin", "ymin", "xmax", "ymax")}
    clipped = {"xmin": max(0.0, raw["xmin"]), "ymin": max(0.0, raw["ymin"]),
               "xmax": min(float(width), raw["xmax"]), "ymax": min(float(height), raw["ymax"])}
    return {"raw_box": raw, "visible_box": clipped,
            "boundary_roundoff_clipped": clipped != raw}


def _group_assignment(group_ids: list[str], seed: int) -> dict[str, str]:
    unique = sorted(set(group_ids))
    if type(seed) is not int or seed < 0:
        raise ValueError("seed须为非负整数")
    if len(unique) < 3:
        raise ValueError(f"需要至少3个独立地理组才可冻结train/model_val/test，当前{len(unique)}")
    order = sorted(unique, key=lambda group: hashlib.sha256(
        f"spd-intersection-v1:{seed}:{group}".encode("utf-8")).hexdigest())
    # 三角色至少各一组；余数组进入训练角色。排序只依赖组身份与预先固定seed。
    assignment = {group: ROLES[index] if index < 3 else "train"
                  for index, group in enumerate(order)}
    return assignment


def build_spd_split(records: list[dict], *, seed: int = 20261101) -> dict:
    """按实际路口隔离双视角、配对帧及所有派生窗口。未知路口隔离不入集合。"""
    if not records:
        raise ValueError("SPD记录为空")
    locations = {str(row.get("intersection_loc", "")).strip() for row in records}
    locations.discard("")
    assignment = _group_assignment(sorted(locations), seed)
    files = []
    for row in sorted(records, key=lambda item: (item["side"], item["sequence_id"], item["frame_id"])):
        location = str(row.get("intersection_loc", "")).strip()
        role = assignment.get(location, "quarantined")
        files.append({**row, "group_id": f"intersection:{location}" if location else None,
                      "project_role": role, "project_input_allowed": role != "quarantined"})
    manifest = {
        "schema_version": "1.0", "protocol": "spd_intersection_holdout_v1",
        "seed": seed, "role_groups": {role: sorted(g for g, r in assignment.items() if r == role)
                                       for role in ROLES},
        "group_assignment": assignment, "files": files,
        "quarantined_unknown_location_records": sum(not str(x.get("intersection_loc", "")).strip()
                                                     for x in records),
        "note": "按原始intersection_loc整组隔离；原始序列与双侧配对帧继承该路口角色；不读取类别/框/未来标签分配。"
    }
    audit_spd_split(manifest)
    manifest["sha256"] = canonical_sha256(manifest)
    return manifest


def audit_spd_split(manifest: dict) -> dict:
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise ValueError("split files必须为列表")
    roles_by_group: dict[str, set[str]] = defaultdict(set)
    roles_by_source: dict[str, set[str]] = defaultdict(set)
    role_groups = manifest.get("role_groups", {})
    for role in ROLES:
        groups = role_groups.get(role, [])
        if not groups:
            raise ValueError(f"空split角色：{role}")
        for group in groups:
            roles_by_group[str(group)].add(role)
    for row in rows:
        role = row.get("project_role")
        if role not in (*ROLES, "quarantined"):
            raise ValueError(f"非法project_role：{role}")
        group = row.get("group_id")
        if group:
            roles_by_group[str(group)].add(role)
        source = str(row.get("source_file", ""))
        if source:
            roles_by_source[source].add(role)
    leaked_groups = {k: sorted(v) for k, v in roles_by_group.items() if len(v) > 1}
    leaked_sources = {k: sorted(v) for k, v in roles_by_source.items() if len(v) > 1}
    if leaked_groups or leaked_sources:
        raise ValueError(f"跨split泄漏：groups={leaked_groups}, sources={leaked_sources}")
    return {"cross_split_group_intersection": 0, "cross_split_source_intersection": 0,
            "role_group_counts": {role: len(role_groups[role]) for role in ROLES},
            "record_count": len(rows)}


def audit_spd(dataset_root: Path, *, seed: int = 20261101,
              verify_image_decode: bool = True) -> dict:
    """审计正式SPD必需RGB内容，返回可序列化报告、split及逐文件哈希清单。"""
    root = Path(dataset_root).expanduser().resolve()
    data_root = root / "V2X-Seq-SPD"
    if not data_root.is_dir():
        raise FileNotFoundError(f"SPD正式目录不存在：{data_root}")
    records: list[dict] = []
    files: dict[str, dict] = {}
    errors: list[dict] = []
    image_hash_to_paths: dict[str, list[str]] = defaultdict(list)
    label_counts: dict[str, Counter] = {s: Counter() for s in SIDES}
    label_occlusion_counts: dict[str, Counter] = {s: Counter() for s in SIDES}
    strata_frames: dict[tuple[str, str, str], set[tuple[str, str, str]]] = defaultdict(set)
    box_errors: Counter = Counter()
    duplicate_track_frames: list[dict] = []
    label_frame_empty_count = Counter()
    frame_counts = Counter()
    dimension_counts: Counter = Counter()
    timestamp_deltas: list[int] = []
    sequence_locations: dict[str, set[str]] = defaultdict(set)
    source_pair_records: dict[tuple[str, str, str], dict] = {}

    def register(path: Path, *, role: str) -> str:
        key = path.relative_to(root).as_posix()
        if key not in files:
            files[key] = {"path": key, "role": role, "bytes": path.stat().st_size,
                          "sha256": sha256_file(path)}
        image_hash_to_paths[files[key]["sha256"]].append(key)
        return files[key]["sha256"]

    for side in SIDES:
        side_root = data_root / side
        info_path = side_root / "data_info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"缺少{side} data_info.json")
        info_hash = register(info_path, role="frame_index")
        rows = _read_json(info_path)
        if not isinstance(rows, list):
            raise ValueError(f"{side}/data_info.json顶层须为列表")
        seen_frames: set[tuple[str, str]] = set()
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                errors.append({"side": side, "row": row_index, "error": "frame_index_not_object"})
                continue
            sequence = str(row.get("sequence_id", ""))
            frame = str(row.get("frame_id", ""))
            frame_key = (sequence, frame)
            if not sequence or not frame or frame_key in seen_frames:
                errors.append({"side": side, "row": row_index, "error": "empty_or_duplicate_sequence_frame",
                               "sequence_id": sequence, "frame_id": frame})
            seen_frames.add(frame_key)
            location = str(row.get("intersection_loc", "")).strip()
            if location:
                sequence_locations[f"{side}:{sequence}"].add(location)
            image_rel = row.get("image_path")
            image_path, resolution = resolve_index_path(root, side, str(image_rel or ""), kind="image")
            if image_path is None:
                errors.append({"side": side, "sequence_id": sequence, "frame_id": frame,
                               "kind": "rgb_image", "path": image_rel, "error": "missing"})
            else:
                image_sha = register(image_path, role="rgb_image")
                if verify_image_decode:
                    try:
                        from PIL import Image
                        with Image.open(image_path) as image:
                            image.verify()
                        with Image.open(image_path) as image:
                            width, height = image.size
                            dimension_counts[f"{width}x{height}"] += 1
                    except Exception as exc:
                        errors.append({"side": side, "sequence_id": sequence, "frame_id": frame,
                                       "kind": "rgb_decode", "path": str(image_path.relative_to(root)),
                                       "error": f"{type(exc).__name__}: {exc}"})
                        width = height = None
                else:
                    width = height = None
                frame_counts[(side, resolution)] += 1
                records.append({"side": side, "source_file": image_path.relative_to(root).as_posix(),
                                "index_file": info_path.relative_to(root).as_posix(),
                                "index_sha256": info_hash, "image_sha256": image_sha,
                                "sequence_id": sequence, "frame_id": frame,
                                "intersection_loc": location, "image_timestamp": row.get("image_timestamp"),
                                "pointcloud_timestamp": row.get("pointcloud_timestamp"),
                                "image_resolution_method": resolution, "width": width, "height": height})
            for field in ("calib_camera_intrinsic_path", "label_camera_std_path"):
                relative = row.get(field)
                if not isinstance(relative, str) or not relative:
                    errors.append({"side": side, "sequence_id": sequence, "frame_id": frame,
                                   "kind": field, "error": "missing_index_field"})
                    continue
                path, method = resolve_index_path(root, side, relative, kind="metadata")
                if path is None:
                    errors.append({"side": side, "sequence_id": sequence, "frame_id": frame,
                                   "kind": field, "path": relative, "error": "missing"})
                    continue
                register(path, role="camera_intrinsic" if field.startswith("calib") else "camera_label")
                try:
                    payload = _read_json(path)
                    if field.startswith("calib"):
                        if not isinstance(payload, dict) or not any(k in payload for k in ("cam_K", "camera_matrix")):
                            errors.append({"side": side, "sequence_id": sequence, "frame_id": frame,
                                           "kind": field, "error": "intrinsic_matrix_missing"})
                    else:
                        if not isinstance(payload, list):
                            raise ValueError("camera label top-level须为列表")
                        ids = set()
                        for object_index, label in enumerate(payload):
                            if not isinstance(label, dict):
                                box_errors["label_not_object"] += 1
                                continue
                            category = str(label.get("type", ""))
                            label_counts[side][category] += 1
                            occlusion = str(label.get("occluded_state", "missing"))
                            label_occlusion_counts[side][f"{category}|{occlusion}"] += 1
                            if image_path is not None:
                                strata_frames[(side, category, occlusion)].add(
                                    (sequence, frame, image_path.relative_to(root).as_posix()))
                            if category not in EXPECTED_LABEL_TYPES:
                                box_errors["unknown_category"] += 1
                            box_error = _bbox_check(label, width or 1920, height or 1080)
                            if box_error:
                                box_errors[box_error] += 1
                            track_id = label.get("track_id")
                            if category != "TrafficCone" and (track_id is None or str(track_id) == ""):
                                box_errors["missing_track_id"] += 1
                            if track_id is not None:
                                if str(track_id) in ids:
                                    box_errors["duplicate_track_id_in_frame"] += 1
                                    if len(duplicate_track_frames) < 500:
                                        duplicate_track_frames.append({"side": side, "sequence_id": sequence,
                                            "frame_id": frame, "track_id": str(track_id),
                                            "category": category, "token": label.get("token"),
                                            "box": label.get("2d_box")})
                                ids.add(str(track_id))
                            if _bbox_check(label, width or 1920, height or 1080) is None:
                                normalized = normalize_visible_box(label, width or 1920, height or 1080)
                                if normalized["boundary_roundoff_clipped"]:
                                    box_errors["boundary_roundoff_clipped"] += 1
                        if not payload:
                            label_frame_empty_count[side] += 1
                except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
                    errors.append({"side": side, "sequence_id": sequence, "frame_id": frame,
                                   "kind": field, "path": str(path.relative_to(root)),
                                   "error": f"{type(exc).__name__}: {exc}"})
            # LiDAR files are explicitly optional for this RGB experiment.
            for field in ("pointcloud_path", "label_lidar_std_path"):
                relative = row.get(field)
                path = (side_root / relative).resolve() if isinstance(relative, str) else None
                if path is None or not path.is_relative_to(side_root.resolve()) or not path.is_file():
                    frame_counts[(side, f"optional_missing:{field}")] += 1
            # Non-pointcloud extrinsics are required to audit camera mapping.
            for field in row:
                if not field.startswith("calib_") or field == "calib_camera_intrinsic_path":
                    continue
                relative = row.get(field)
                if not isinstance(relative, str) or not relative:
                    errors.append({"side": side, "sequence_id": sequence, "frame_id": frame,
                                   "kind": field, "error": "missing_index_field"})
                    continue
                path, _ = resolve_index_path(root, side, relative, kind="metadata")
                if path is None:
                    errors.append({"side": side, "sequence_id": sequence, "frame_id": frame,
                                   "kind": field, "path": relative, "error": "missing"})
                else:
                    register(path, role="extrinsic_calibration")
                    try:
                        value = _read_json(path)
                        if not isinstance(value, dict):
                            raise ValueError("外参JSON顶层须为对象")
                    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
                        errors.append({"side": side, "sequence_id": sequence, "frame_id": frame,
                                       "kind": field, "path": str(path.relative_to(root)),
                                       "error": f"{type(exc).__name__}: {exc}"})
            for ts_name in ("image_timestamp", "pointcloud_timestamp"):
                value = row.get(ts_name)
                if value is None or not re.fullmatch(r"\d+", str(value)):
                    errors.append({"side": side, "sequence_id": sequence, "frame_id": frame,
                                   "kind": "timestamp", "field": ts_name, "value": value,
                                   "error": "missing_or_noninteger"})
        source_pair_records.update({(side, str(x.get("sequence_id", "")), str(x.get("frame_id", ""))): x
                                    for x in rows if isinstance(x, dict)})

    cooperative_path = data_root / "cooperative" / "data_info.json"
    cooperative_hash = register(cooperative_path, role="pair_index") if cooperative_path.is_file() else None
    cooperative = _read_json(cooperative_path) if cooperative_path.is_file() else []
    pair_missing = Counter()
    pair_time_delta_us: list[int] = []
    paired_sequences = set()
    for pair in cooperative:
        vseq, vframe = str(pair.get("vehicle_sequence", "")), str(pair.get("vehicle_frame", ""))
        iseq, iframe = str(pair.get("infrastructure_sequence", "")), str(pair.get("infrastructure_frame", ""))
        vehicle = source_pair_records.get(("vehicle-side", vseq, vframe))
        infra = source_pair_records.get(("infrastructure-side", iseq, iframe))
        pair_missing["vehicle_frame_missing"] += vehicle is None
        pair_missing["infrastructure_frame_missing"] += infra is None
        paired_sequences.add((vseq, iseq))
        if vehicle and infra:
            try:
                pair_time_delta_us.append(int(infra["image_timestamp"]) - int(vehicle["image_timestamp"]))
            except (KeyError, TypeError, ValueError):
                pair_missing["timestamp_unavailable"] += 1
    # 部分vehicle data_info的intersection_loc为空；仅按官方cooperative序列链接，
    # 且目标序列唯一映射到一个有名路口时填派生分组，原始字段仍保留。
    vehicle_sequence_locations: dict[str, set[str]] = defaultdict(set)
    vehicle_sequence_infrastructure: dict[str, set[str]] = defaultdict(set)
    for pair in cooperative:
        vseq, iseq = str(pair.get("vehicle_sequence", "")), str(pair.get("infrastructure_sequence", ""))
        vehicle_sequence_infrastructure[vseq].add(iseq)
        vehicle_sequence_locations[vseq].update(sequence_locations.get(f"infrastructure-side:{iseq}", set()))
    raw_blank_vehicle_records = 0
    resolved_blank_vehicle_records = 0
    for row in records:
        if row["side"] != "vehicle-side" or row["intersection_loc"]:
            continue
        raw_blank_vehicle_records += 1
        locations = vehicle_sequence_locations.get(row["sequence_id"], set())
        if len(locations) == 1 and len(vehicle_sequence_infrastructure.get(row["sequence_id"], set())) == 1:
            row["intersection_loc_raw"] = ""
            row["intersection_loc"] = next(iter(locations))
            row["location_resolution_method"] = "official_cooperative_sequence_link"
            resolved_blank_vehicle_records += 1
        else:
            row["location_resolution_method"] = "unresolved_quarantined"
    split = build_spd_split(records, seed=seed)

    # Hash exact duplicate RGB content across all registered assets; package-only assets are indexed below.
    package_inventory = []
    for side, package in SIDE_IMAGE_PACKAGES.items():
        folder = root / package
        for path in sorted(folder.glob("*")) if folder.is_dir() else []:
            if not path.is_file():
                continue
            register(path, role=f"{side}_rgb_package")
            package_inventory.append(path.relative_to(root).as_posix())
    extra_root = root / "cooperative-vehicle-infrastructure-infrastructure-side-image"
    extra_files = sorted(p for p in extra_root.rglob("*") if p.is_file()) if extra_root.is_dir() else []
    for path in extra_files:
        register(path, role="unmapped_additional_image_package")
        package_inventory.append(path.relative_to(root).as_posix())
    by_hash: dict[str, list[str]] = defaultdict(list)
    for path, entry in files.items():
        if entry["role"].endswith("rgb_package") or entry["role"] == "unmapped_additional_image_package":
            by_hash[entry["sha256"]].append(path)
    exact_duplicates = [sorted(paths) for paths in by_hash.values() if len(paths) > 1]

    map_root = data_root / "maps"
    maps = []
    if map_root.is_dir():
        for path in sorted(x for x in map_root.rglob("*") if x.is_file()):
            maps.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                         "sha256": register(path, role="map_reference")})

    group_locations: dict[str, set[str]] = defaultdict(set)
    for key, locations in sequence_locations.items():
        if len(locations) > 1:
            errors.append({"kind": "sequence_location_conflict", "sequence": key,
                           "locations": sorted(locations)})
        for location in locations:
            group_locations[key].add(location)
    role_counts = Counter(x["project_role"] for x in split["files"])
    top = {"dataset_root": str(root), "official_layout": str(data_root),
           "scope": "formal_SPD_RGB_plus_metadata_no_raw_pointcloud_required",
           "seed": seed, "source_indexes": {
               "vehicle_side_frames": sum(1 for x in records if x["side"] == "vehicle-side"),
               "infrastructure_side_frames": sum(1 for x in records if x["side"] == "infrastructure-side"),
               "cooperative_pairs": len(cooperative), "paired_sequence_pairs": len(paired_sequences)},
           "image_resolution_methods": dict((f"{k[0]}:{k[1]}", v) for k, v in frame_counts.items()
                                             if isinstance(k, tuple) and str(k[1]).startswith("indexed" ) or
                                             isinstance(k, tuple) and str(k[1]).startswith("flat")),
           "decoded_image_dimensions": dict(dimension_counts),
           "label_object_counts": {side: dict(sorted(counts.items())) for side, counts in label_counts.items()},
           "label_category_occlusion_counts": {side: dict(sorted(counts.items()))
                                               for side, counts in label_occlusion_counts.items()},
           "stratified_review_samples": [
               {"side": side, "category": category, "occluded_state": occlusion,
                "population_frames": len(rows),
                "sample_frames": [{"sequence_id": seq, "frame_id": frame, "source_file": source}
                                  for seq, frame, source in sorted(rows,
                                      key=lambda x: hashlib.sha256(
                                          f"{seed}:{side}:{category}:{occlusion}:{x[0]}:{x[1]}".encode()).hexdigest())[:30]],
                "sample_count": min(30, len(rows)), "sample_rule": "按冻结seed和身份SHA排序，类别×遮挡层内固定样本；不足30则全查"}
               for (side, category, occlusion), rows in sorted(strata_frames.items())],
           "label_quality_findings": dict(box_errors),
           "duplicate_track_id_frames": duplicate_track_frames,
           "tracking_identity_note": "same-frame raw track_id duplicates are preserved and flagged; detector boxes remain scoreable, affected identity rows must be excluded from ID metrics unless independently disambiguated",
           "empty_camera_label_frames": dict(label_frame_empty_count),
           "optional_missing_pointcloud_files": {side: sum(v for (s, k), v in frame_counts.items()
               if s == side and str(k).startswith("optional_missing:pointcloud_path")) for side in SIDES},
           "optional_missing_lidar_labels": {side: sum(v for (s, k), v in frame_counts.items()
               if s == side and str(k).startswith("optional_missing:label_lidar_std_path")) for side in SIDES},
           "pair_reference_findings": dict(pair_missing),
           "pair_image_timestamp_delta_raw_integer": {
               "count": len(pair_time_delta_us), "min": min(pair_time_delta_us) if pair_time_delta_us else None,
               "median": sorted(pair_time_delta_us)[len(pair_time_delta_us)//2] if pair_time_delta_us else None,
               "max": max(pair_time_delta_us) if pair_time_delta_us else None,
               "unit_source": "integer field magnitudes and sequence cadence suggest microseconds, but local official schema does not declare units; values remain raw and are not converted for online use"},
           "sequences_by_location": {k: sorted(v) for k, v in sorted(sequence_locations.items())},
           "group_assignment": split["group_assignment"], "role_counts": dict(role_counts),
           "unknown_location_quarantined": split["quarantined_unknown_location_records"],
           "raw_blank_vehicle_location_records": raw_blank_vehicle_records,
           "blank_vehicle_location_resolved_by_official_pairs": resolved_blank_vehicle_records,
           "unresolved_blank_vehicle_location_records": raw_blank_vehicle_records-resolved_blank_vehicle_records,
           "map_references": maps, "additional_package": {
               "path": extra_root.relative_to(root).as_posix(), "file_count": len(extra_files),
               "bytes": sum(p.stat().st_size for p in extra_files),
               "inventory_paths": [p.relative_to(root).as_posix() for p in extra_files],
               "exact_rgb_hash_duplicates_with_selected_packages": sum(len(x) > 1 for x in exact_duplicates),
               "interpretation": "独立来源包；无data_info映射且不与正式RGB清单按内容哈希去重，保持隔离"},
           "errors": errors, "file_count": len(files),
           "total_bytes": sum(x["bytes"] for x in files.values()),
           "files": sorted(files.values(), key=lambda x: x["path"]),
           "input_file_manifest_sha256": canonical_sha256(sorted(files.values(), key=lambda x: x["path"])),
           "split": split, "split_sha256": split["sha256"],
           "ready_for_rgb_audit_and_project_split": not errors and all(role_counts[x] for x in ROLES)}
    top["report_sha256"] = canonical_sha256({k: v for k, v in top.items() if k != "report_sha256"})
    return top

