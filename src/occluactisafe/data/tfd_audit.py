"""N03 离线数据审计编排；输入必须是人工准备的数据，不自动下载。"""
from collections import Counter
from copy import deepcopy
from datetime import date, datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit

import yaml

from occluactisafe.common.config import PROJECT_ROOT, load_config
from occluactisafe.common.run_manifest import create_run_manifest
from occluactisafe.data.adapters.tfd import discover_tfd, read_map_catalog, iter_trajectories, iter_traffic_lights
from occluactisafe.data.quality import audit_trajectory_stream


_DEFAULTS = {
    "source": "v2x_seq_tfd", "data_root": "data/raw/v2x_seq_tfd/V2X-Seq-TFD-Example",
    "output_root": "artifacts/data_audit", "dataset_scope": "example",
    "dataset_version": "unknown", "acquired_date": "unknown",
    "source_url": "https://github.com/AIR-THU/DAIR-V2X-Seq#dataset-download",
    "usage_conditions": "unverified",
}


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for k, v in node.value:
        key = loader.construct_object(k, deep=deep)
        if not isinstance(key, str) or key in result:
            raise ValueError("任务 YAML 包含非字符串或重复键")
        result[key] = loader.construct_object(v, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def load_task_config(config_file, *, overrides=None, cli_overrides=None, project_root=PROJECT_ROOT):
    """包默认→任务 YAML→脚本覆盖→显式 CLI；路径基于项目根。"""
    root = Path(project_root).resolve()
    path = Path(config_file).expanduser()
    path = (path if path.is_absolute() else root / path).resolve()
    try:
        with path.open(encoding="utf-8") as stream:
            selected = yaml.load(stream, Loader=_UniqueLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"任务 YAML 无法解析：{path}") from exc
    config = deepcopy(_DEFAULTS)
    for layer in (selected, overrides, cli_overrides):
        if layer is None:
            continue
        if not isinstance(layer, dict) or set(layer) - set(_DEFAULTS):
            raise ValueError("TFD 任务配置包含未知字段或不是映射")
        config.update(layer)
    if config["source"] != "v2x_seq_tfd" or config["dataset_scope"] not in ("example", "full"):
        raise ValueError("数据来源或范围非法")
    if config["usage_conditions"] not in ("unverified", "verified"):
        raise ValueError("使用条件必须是 unverified 或 verified")
    for name in ("dataset_version", "source_url"):
        if not isinstance(config[name], str) or not config[name].strip():
            raise ValueError(f"{name} 必须是非空公开文字")
    acquired = config["acquired_date"]
    if not isinstance(acquired, str):
        raise ValueError("acquired_date 必须是 YYYY-MM-DD 或 unknown 字符串")
    if acquired != "unknown":
        try:
            parsed = date.fromisoformat(acquired)
        except ValueError as exc:
            raise ValueError("acquired_date 必须是 YYYY-MM-DD 或 unknown") from exc
        if parsed.isoformat() != acquired:
            raise ValueError("acquired_date 必须使用 YYYY-MM-DD")
    url = urlsplit(config["source_url"])
    if (url.scheme != "https" or not url.hostname or url.username or url.password or url.query
            or "=" in url.fragment or re.search(r"token|secret|password|credential|api[_-]?key", url.fragment, re.I)):
        raise ValueError("source_url 必须是无凭据和查询参数的公开 HTTPS 地址")
    for name in ("data_root", "output_root"):
        raw = config[name]
        if not isinstance(raw, str) or not raw.strip() or "://" in raw:
            raise ValueError(f"{name} 必须是本地路径")
        value = Path(raw).expanduser()
        config[name] = str((value if value.is_absolute() else root / value).resolve())
    if config["dataset_scope"] == "full" and Path(config["data_root"]) == (root / _DEFAULTS["data_root"]).resolve():
        raise ValueError("完整数据须明确指定不同于默认 Example 的数据路径")
    return config


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _line(stream, item):
    stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def _preview(path, points):
    """只绘制有效原始轨迹坐标；未核实单位或地图配准时不标米或叠图。"""
    if not points:
        return False
    xs, ys = zip(*points)
    x0, y0 = min(xs), min(ys)
    span = max(max(xs) - x0, max(ys) - y0, 1.0e-12)
    coords = " ".join(f"{30 + (x - x0) / span * 340:.3f},{370 - (y - y0) / span * 340:.3f}" for x, y in points)
    label = html.escape("TFD raw trajectory preview; units and map alignment unverified")
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400">\n'
                    f'<rect width="400" height="400" fill="white"/><polyline points="{coords}" '
                    f'fill="none" stroke="#13538b" stroke-width="2"/><text x="12" y="395" '
                    f'font-size="9">{label}</text></svg>\n', encoding="utf-8")
    return True


def run_audit(config, *, project_root=PROJECT_ROOT):
    """记录数据清单并逐文件审计；不构造缺失数据，也不进行训练/划分。"""
    root = Path(project_root).resolve()
    data = Path(config["data_root"]).resolve()
    output = Path(config["output_root"]).resolve()
    if not data.is_dir():
        raise FileNotFoundError(f"TFD 数据目录不存在：{data}；请先按官方入口准备数据")
    if output == data or data in output.parents:
        raise ValueError("输出目录不得位于原始数据目录内")
    discovery = discover_tfd(data)
    if not discovery["trajectory_files"]:
        raise ValueError(f"未找到官方 TFD 轨迹 CSV：{data}")
    filenames = sorted(set(discovery["trajectory_files"] + discovery["traffic_light_files"] + discovery["map_files"]))
    files = []
    for name in filenames:
        source_file = (data / name).resolve()
        if not source_file.is_relative_to(data):
            raise ValueError(f"数据文件越出数据根目录：{name}")
        files.append({"path": name, "size_bytes": source_file.stat().st_size,
                      "sha256": _sha(source_file)})
    source_digest = hashlib.sha256()
    for item in files:
        source_digest.update(item["path"].encode("utf-8") + b"\0" + bytes.fromhex(item["sha256"]))
    dataset_sha = source_digest.hexdigest()
    base = load_config(project_root=root)
    run = create_run_manifest(config=base, input_mode="offline_tfd_audit",
                              data={"name": "V2X-Seq-TFD", "version": config["dataset_version"], "sha256": dataset_sha},
                              output_root=output, project_root=root)
    run_dir = output / run["run_id"]
    task_config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False,
                                      separators=(",", ":")).encode("utf-8")).hexdigest()
    data_manifest = {"schema_version": "1.0", "source": config["source"],
                     "source_url": config["source_url"], "dataset_scope": config["dataset_scope"],
                     "dataset_version": config["dataset_version"], "usage_conditions_claim": config["usage_conditions"],
                     "usage_conditions_evidence": "not_provided_or_verified_by_audit",
                     "acquired_date_claim": config["acquired_date"],
                     "acquired_date_evidence": "user_supplied_unverified", "audit_utc": datetime.now(timezone.utc).isoformat(),
                     "data_root": str(data), "dataset_sha256": dataset_sha, "task_config_sha256": task_config_hash,
                     "provenance_verification": "user_provided_unverified",
                     "files": files, "discovery": discovery,
                     "note": "仅登记实际发现文件；下载时间和许可未知时明确记录，不以目录名冒充官方数据身份。"}
    _write_json(run_dir / "data_manifest.json", data_manifest)
    _write_json(run_dir / "task_config.json", config)
    status = {"status": "running", "run_id": run["run_id"]}
    _write_json(run_dir / "audit_status.json", status)
    try:
        # 一次逐文件解析地图后只保留轻量索引；轨迹和信号共享，避免将所有地图驻留内存。
        map_catalog = read_map_catalog(data)
        samples, points = [], []
        preview_key = None
        map_counts = Counter()
        with (run_dir / "trajectories.jsonl").open("w", encoding="utf-8") as output_stream, \
             (run_dir / "anomalies.jsonl").open("w", encoding="utf-8") as anomaly_stream:
            def accept(row):
                nonlocal preview_key
                _line(output_stream, row)
                map_counts[row.get("map_status", "unknown")] += 1
                if len(samples) < 20:
                    samples.append(row)
                key = (row.get("source_file"), row.get("track_id"))
                if preview_key is None:
                    preview_key = key
                if key == preview_key and len(points) < 500:
                    x, y = float(row["x"]), float(row["y"])
                    if math.isfinite(x) and math.isfinite(y):
                        points.append((x, y))
            def anomaly(row):
                _line(anomaly_stream, row)
            quality = audit_trajectory_stream(iter_trajectories(data, map_catalog=map_catalog), accept, anomaly)
        signals = 0
        with (run_dir / "traffic_lights.jsonl").open("w", encoding="utf-8") as stream:
            for signal in iter_traffic_lights(data, map_catalog=map_catalog):
                _line(stream, signal)
                signals += 1
        maps = map_catalog["maps"]
        _write_json(run_dir / "map_index.json", maps)
        _write_json(run_dir / "inspection_samples.json", samples)
        preview = _preview(run_dir / "trajectory_preview.svg", points)
        audited = {item["source_file"] for item in quality["raw"].get("file_field_summaries", [])}
        empty_files = sorted(set(discovery["trajectory_files"]) - audited)
        quality.update({"map_status_valid_row_counts": dict(sorted(map_counts.items())),
                        "traffic_light_row_count": signals, "map_file_count": len(maps),
                        "rgb_images": discovery["rgb_images"], "detection_2d_boxes": discovery["detection_2d_boxes"],
                        "preview_available": preview, "empty_or_header_only_trajectory_files": empty_files,
                        "field_dictionary": "data_dictionary.json"})
        _write_json(run_dir / "data_quality.json", quality)
        fields = {"identity": ["source_file", "source_row", "episode_id", "split", "source", "collection", "track_id"],
                  "raw_state": ["timestamp", "x", "y", "z", "vx", "vy", "heading", "length", "width", "height"],
                  "category": ["object_type", "object_subtype", "normalized_category", "original"],
                  "provenance": ["map_status", "time_unit", "coordinate_system", "usage_boundary"],
                  "units": "原始时间和坐标单位未确认，不强行转换；保留 original 原列", "real_data": "仅当实际输入是官方数据时才可作为真实数据审计"}
        _write_json(run_dir / "data_dictionary.json", fields)
        if quality["raw"]["row_count"] == 0:
            raise ValueError("轨迹文件均不含数据行，无法完成审计")
        # 即使标记 full，也不能凭目录名、用户提供的标签或一次审计自动宣布正式数据就绪。
        reasons = ["正式数据范围、许可、单位和官方划分须依据实际文件及来源证据人工复核"]
        if config["dataset_scope"] == "example":
            reasons.insert(0, "Example 仅用于接通，不能代表完整训练数据")
        status = {"status": "audit_completed", "run_id": run["run_id"],
                  "dataset_scope": config["dataset_scope"], "formal_data_ready": False,
                  "readiness_reasons": reasons, "raw_rows": quality["raw"]["row_count"],
                  "valid_rows": quality["valid"]["row_count"]}
        _write_json(run_dir / "audit_status.json", status)
        return {**status, "run_dir": str(run_dir)}
    except Exception as exc:
        _write_json(run_dir / "audit_status.json", {"status": "failed", "run_id": run["run_id"],
                    "error_type": type(exc).__name__, "error": str(exc), "formal_data_ready": False})
        raise
