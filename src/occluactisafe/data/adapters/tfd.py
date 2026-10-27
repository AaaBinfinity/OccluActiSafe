"""DAIR-V2X-Seq TFD 文件只读适配，不推断未核实的坐标或时间单位。

目录和字段依据官方 dataset/v2x-seq-tfd/README.md；每条轨迹保留原始列
及文件行位置。官方协同轨迹的处理因果性尚未证明，只能作为离线数据。
"""

import csv
import json
from pathlib import Path

# README 示意拼写与官方 fusion/config.py 的实际大写枚举均保留；未知类别不当车辆。
OFFICIAL_TYPES = {
    "Vehicle": "vehicle", "Bicycle": "bicycle", "Pedestrain": "pedestrian",
    "VEHICLE": "vehicle", "BICYCLE": "bicycle", "PEDESTRIAN": "pedestrian",
    "UNKNOWN_UNMOVABLE": "unknown_unmovable", "UNKNOWN": "unknown",
}
TRAJECTORY_LOCATIONS = {
    ("single-infrastructure", "trajectories"): ("single", "infrastructure"),
    ("single-vehicle", "trajectories"): ("single", "vehicle"),
    ("cooperative-vehicle-infrastructure", "infrastructure-trajectories"): ("cooperative", "infrastructure"),
    ("cooperative-vehicle-infrastructure", "vehicle-trajectories"): ("cooperative", "vehicle"),
    ("cooperative-vehicle-infrastructure", "cooperative-trajectories"): ("cooperative", "cooperative"),
}
SIGNAL_LOCATIONS = {
    ("single-infrastructure", "traffic-light"): ("single", "infrastructure"),
    ("cooperative-vehicle-infrastructure", "traffic-light"): ("cooperative", "infrastructure"),
}


def _root(root):
    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"TFD 数据目录不存在：{path}")
    return path


def _locations(root, locations):
    """识别官方目录，包含 train/val/test_obs/data 嵌套布局。"""
    found = []
    for path in root.rglob("*.csv"):
        if path.is_file() and _location_info(root, path, locations) is not None:
            found.append(path)
    return sorted(found, key=lambda item: item.relative_to(root).as_posix())


def _location_info(root, path, locations):
    """提取已知视角和目录中的 split，不把目录名当作已复核的官方划分。"""
    parts = path.relative_to(root).parts
    matches = [(i, locations[tuple(parts[i:i + 2])]) for i in range(len(parts) - 2)
               if tuple(parts[i:i + 2]) in locations]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"源文件路径包含多个已知视角：{path}")
    index, (collection, source) = matches[0]
    others = parts[:index] + parts[index + 2:-1]
    splits = [part for part in others if part in {"train", "val", "test", "test_obs"}]
    if len(set(splits)) > 1:
        raise ValueError(f"源文件路径包含冲突 split：{path}")
    split = splits[0] if splits else "unspecified"
    # 前缀区分不同数据树；collection 区分单视角数据与协同数据。
    prefix = [part for part in parts[:index] if part not in {"train", "val", "test", "test_obs"}]
    episode_id = "/".join(prefix + [collection, split, path.stem])
    return collection, source, split, episode_id


def _map_files(root):
    return sorted((path for path in root.rglob("hdmap*.json")
                   if path.is_file() and path.parent.name == "maps"),
                  key=lambda item: item.relative_to(root).as_posix())


def _bad_constant(value):
    raise ValueError(f"地图包含非有限 JSON 值：{value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"地图包含重复 JSON 字段：{key}")
        result[key] = value
    return result


def iter_maps(root):
    """逐地图读取 LANE/STOPLINE/CROSSWALK；损坏文件显式报告，不构造车道。"""
    root = _root(root)
    for path in _map_files(root):
        item = {"source_file": path.relative_to(root).as_posix(),
                "map_id": path.stem[len("hdmap"):], "status": "present",
                "missing_sections": [], "original": None, "error": None}
        try:
            with path.open("r", encoding="utf-8-sig") as stream:
                data = json.load(stream, parse_constant=_bad_constant, object_pairs_hook=_unique_object)
            if not isinstance(data, dict):
                raise ValueError("地图 JSON 顶层必须是对象")
            item["original"] = data
            item["missing_sections"] = [key for key in ("LANE", "STOPLINE", "CROSSWALK") if key not in data]
            if "LANE" not in data:
                item["status"] = "incomplete"
            elif not isinstance(data["LANE"], dict):
                raise ValueError("地图 LANE 应为对象")
        except (OSError, UnicodeError, ValueError) as exc:
            item["status"] = "invalid"
            item["error"] = f"{type(exc).__name__}: {exc}"
        yield item


def _map_catalog(root):
    # 同名地图可在独立 split 目录出现；有损坏文件、同名冲突都不能冒充有效。
    result = {}
    for item in iter_maps(root):
        map_id = item["map_id"]
        existing = result.get(map_id)
        if existing is None:
            result[map_id] = item
        elif existing["status"] != item["status"] or existing.get("original") != item["original"]:
            result[map_id] = {"status": "ambiguous"}
    return result


def _map_status(raw, catalog):
    map_id = raw.get("intersect_id", "").strip()
    if not map_id:
        return "unknown"
    item = catalog.get(map_id)
    return "missing" if item is None else item["status"]


def discover_tfd(root):
    """返回发现的源文件索引；支持缺地图并明确标记，不做任何下载。"""
    root = _root(root)
    trajectories = _locations(root, TRAJECTORY_LOCATIONS)
    signals = _locations(root, SIGNAL_LOCATIONS)
    maps = _map_files(root)
    return {
        "root": str(root),
        "trajectory_files": [p.relative_to(root).as_posix() for p in trajectories],
        "traffic_light_files": [p.relative_to(root).as_posix() for p in signals],
        "map_files": [p.relative_to(root).as_posix() for p in maps],
        "map_status": "present" if maps else "missing",
        "time_unit": "unverified", "coordinate_system": "unverified",
        "rgb_images": "not_in_tfd_schema", "detection_2d_boxes": "not_in_tfd_schema",
    }


def _iter_csv(path):
    # csv.reader.line_num 是已消费的物理行数：取下一行首行位置，兼容多行字段。
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        try:
            headers = next(reader)
        except StopIteration:
            raise ValueError(f"空 CSV 文件：{path}")
        if not headers or any(not name for name in headers) or len(headers) != len(set(headers)):
            raise ValueError(f"CSV 表头为空或存在重复字段：{path}")
        while True:
            row_number = reader.line_num + 1
            try:
                row = next(reader)
            except StopIteration:
                break
            values = {name: row[index] if index < len(row) else "" for index, name in enumerate(headers)}
            yield row_number, values, row[len(headers):], [headers[i] for i in range(len(row), len(headers))]


def iter_trajectory_file(root, source_file, *, map_catalog=None):
    """逐行读取一个已发现的源文件，供大数据集分文件流式审计。"""
    root = _root(root)
    path = (root / source_file).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("轨迹路径须指向数据根目录内的现有文件")
    info = _location_info(root, path, TRAJECTORY_LOCATIONS)
    if info is None or path.suffix.lower() != ".csv":
        raise ValueError("轨迹文件不属于官方定义的目录结构")
    catalog = _map_catalog(root) if map_catalog is None else map_catalog
    collection, source, split, episode_id = info
    relative = path.relative_to(root).as_posix()
    for line, original, extra_values, missing_columns in _iter_csv(path):
        obj_type = original.get("type", "")
        normalized_type = obj_type.strip()
        yield {
            "source_file": relative, "source_row": line, "episode_id": episode_id,
            "split": split, "split_provenance": "directory_name_unverified",
            "source": source, "collection": collection,
            "track_id": original.get("id", ""), "timestamp": original.get("timestamp", ""),
            "x": original.get("x", ""), "y": original.get("y", ""),
            "z": original.get("z", ""), "vx": original.get("v_x", ""), "vy": original.get("v_y", ""),
            "heading": original.get("theta", ""), "length": original.get("length", ""),
            "width": original.get("width", ""), "height": original.get("height", ""),
            "city": original.get("city", ""), "intersection_id": original.get("intersect_id", ""),
            "object_subtype": original.get("sub_type", ""), "tag": original.get("tag", ""),
            "vic_tag": original.get("vic_tag", ""), "from_side": original.get("from_side", ""),
            "car_side_id": original.get("car_side_id", ""), "road_side_id": original.get("road_side_id", ""),
            "object_type": obj_type, "normalized_category": OFFICIAL_TYPES.get(normalized_type, "unknown"),
            "category_status": ("official_unknown" if normalized_type in ("UNKNOWN", "UNKNOWN_UNMOVABLE") else
                                "recognized" if normalized_type in OFFICIAL_TYPES else
                                "missing" if not normalized_type else "unrecognized"),
            "original": original, "extra_values": extra_values, "missing_columns": missing_columns,
            "map_status": _map_status(original, catalog),
            "time_unit": "unverified", "coordinate_system": "unverified",
            "usage_boundary": ("offline_only_future_leakage_unverified" if source == "cooperative"
                               else "offline_observation_availability_unknown"),
        }


def iter_trajectories(root):
    """逐行读取所有轨迹，不排序、去重、填补或推断在线可得性。"""
    root = _root(root)
    catalog = _map_catalog(root)
    for path in _locations(root, TRAJECTORY_LOCATIONS):
        yield from iter_trajectory_file(root, path, map_catalog=catalog)


def read_map_catalog(root):
    """加载一次地图目录，供多个 iter_trajectory_file 调用复用。"""
    return _map_catalog(_root(root))


def iter_traffic_lights(root):
    """逐行读取信号表，保留 lane_id/颜色/剩余值原文，不推断可用时刻。"""
    root = _root(root)
    catalog = _map_catalog(root)
    for path in _locations(root, SIGNAL_LOCATIONS):
        collection, source, split, episode_id = _location_info(root, path, SIGNAL_LOCATIONS)
        for line, original, extra_values, missing_columns in _iter_csv(path):
            yield {
                "source_file": path.relative_to(root).as_posix(), "source_row": line,
                "episode_id": episode_id, "split": split, "split_provenance": "directory_name_unverified",
                "collection": collection, "source": source,
                "timestamp": original.get("timestamp", ""), "lane_id": original.get("lane_id", ""),
                "original": original, "extra_values": extra_values, "missing_columns": missing_columns,
                "map_status": _map_status(original, catalog),
                "time_unit": "unverified", "availability": "unknown",
            }
