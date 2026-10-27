"""TFD 原始记录的可追溯质量审计，不计算差分、窗口或划分。"""

from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from itertools import groupby


# 官方 README 与融合工具的类型写法分别保留；不把未知类当作车辆。
TYPE_TO_CATEGORY = {"Vehicle": "vehicle", "Bicycle": "bicycle", "Pedestrain": "pedestrian",
                    "VEHICLE": "vehicle", "BICYCLE": "bicycle", "PEDESTRIAN": "pedestrian",
                    "UNKNOWN": "unknown", "UNKNOWN_UNMOVABLE": "unknown_unmovable"}
DOCUMENTED_FIELDS = frozenset(("city", "timestamp", "id", "type", "sub_type", "tag", "x", "y", "z",
                               "length", "width", "height", "theta", "v_x", "v_y", "intersect_id"))
COOPERATIVE_FIELDS = frozenset(("vic_tag", "from_side", "car_side_id", "road_side_id"))


def _text(value):
    return "" if value is None else str(value).strip()


def _number(value):
    try:
        number = Decimal(_text(value))
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() else None


def _location(row):
    return {"source_file": row["source_file"], "source_row": row["source_row"]}


def audit_trajectories(records):
    """返回 (有效记录, JSON 可序列化报告)。每条记录带原文件和 CSV 行号。

    时间只按原值精确排序，报告跨度单位为原始时间单位，不假设秒或毫秒。
    分组键为原文件、轨迹 ID；相同键和时间的首条有效记录保留，
    其余有效重复记录剔除；所有原始行和异常位置均计入报告。
    """
    anomalies = []
    counts = Counter()
    raw_categories = Counter()
    raw_targets = defaultdict(set)
    raw_subtypes = Counter()
    raw_subtype_targets = defaultdict(set)
    valid_categories = Counter()
    valid_targets = defaultdict(set)
    valid_subtypes = Counter()
    valid_subtype_targets = defaultdict(set)
    raw_episodes, valid_episodes = set(), set()
    raw_tracks, valid_tracks = set(), set()
    seen_raw = {}
    seen_valid = set()
    previous = {}
    raw_times, valid_times = defaultdict(list), defaultdict(list)
    file_rows, file_nonempty = Counter(), defaultdict(Counter)
    file_fields, file_expected = defaultdict(set), defaultdict(set)
    valid = []
    raw_rows = 0

    def issue(code, row, reason, **extra):
        counts[code] += 1
        anomalies.append({"kind": code, **_location(row), "reason": reason, **extra})

    for row in records:
        if not isinstance(row, dict):
            raise TypeError("轨迹记录必须是字典")
        if not _text(row.get("source_file")) or not isinstance(row.get("source_row"), int) or row["source_row"] < 2:
            raise ValueError("每行必须包含原始文件路径和不小于2的CSV行号")
        if not _text(row.get("episode_id")) or not _text(row.get("source")):
            raise ValueError("每行必须包含 episode_id 和来源视角")
        raw_rows += 1
        episode = _text(row["episode_id"])
        source = _text(row["source"])
        identity = _text(row.get("track_id"))
        kind = _text(row.get("object_type"))
        original = row.get("original")
        if not isinstance(original, dict):
            raise ValueError("每行必须保留原始字段字典 original")
        file_id = row["source_file"]
        file_rows[file_id] += 1
        file_fields[file_id].update(original)
        file_expected[file_id].update(DOCUMENTED_FIELDS | (COOPERATIVE_FIELDS if source == "cooperative" else set()))
        file_nonempty[file_id].update(name for name, value in original.items() if _text(value))
        if row.get("extra_values"):
            issue("extra_fields", row, "CSV 行列数多于表头，保留额外值供核查", extra_values=row["extra_values"])
        if row.get("missing_columns"):
            issue("short_row", row, "CSV 行列数少于表头，缺列保留为空并记录", fields=row["missing_columns"])
        subtype = _text(original.get("sub_type")) or "<missing>"
        group = (row["source_file"], identity)
        span_group = (episode, source, row["source_file"])
        category = TYPE_TO_CATEGORY.get(kind, "unrecognized")
        raw_categories[kind or "<missing>"] += 1
        raw_subtypes[subtype] += 1
        raw_episodes.add(episode)
        if identity:
            raw_tracks.add(group)
            raw_targets[category].add(group)
            raw_subtype_targets[subtype].add(group)
        timestamp = _number(row.get("timestamp"))
        if timestamp is not None:
            raw_times[span_group].append(timestamp)

        invalid = False
        if not identity:
            issue("missing_track_id", row, "缺少轨迹 ID，不能确定目标身份，排除此行")
            invalid = True
        if timestamp is None:
            issue("invalid_timestamp", row, "时间缺失或非有限数值，不能排序，排除此行")
            invalid = True
        missing_coordinates = [name for name in ("x", "y") if not _text(row.get(name))]
        invalid_coordinates = [name for name in ("x", "y")
                               if _text(row.get(name)) and _number(row.get(name)) is None]
        if missing_coordinates:
            issue("missing_coordinates", row, "坐标缺失，排除此行", fields=missing_coordinates)
            invalid = True
        if invalid_coordinates:
            issue("invalid_coordinates", row, "坐标不是有限数值，排除此行", fields=invalid_coordinates)
            invalid = True
        if not kind:
            issue("missing_category", row, "类别缺失，不能映射类别，排除此行")
            invalid = True
        elif kind not in TYPE_TO_CATEGORY:
            issue("invalid_category", row, "类别不在官方声明枚举内，保留原值并排除此行", original_type=kind)
            invalid = True

        key = (group, timestamp)
        if identity and timestamp is not None:
            if key in seen_raw:
                issue("duplicate", row, "同文件、ID、时间重复；仅首条有效记录保留", first=seen_raw[key])
            else:
                seen_raw[key] = _location(row)
            if group in previous and timestamp < previous[group]:
                issue("out_of_order", row, "同目标原始输入时间倒序；有效记录排序后输出", previous_timestamp=str(previous[group]))
            previous[group] = timestamp
        if invalid:
            continue
        if key in seen_valid:
            continue
        seen_valid.add(key)
        valid.append((group, timestamp, row))
        valid_categories[kind] += 1
        valid_targets[category].add(group)
        valid_subtypes[subtype] += 1
        valid_subtype_targets[subtype].add(group)
        valid_episodes.add(episode)
        valid_tracks.add(group)
        valid_times[span_group].append(timestamp)

    valid.sort(key=lambda item: (item[0], item[1]))

    def spans(groups):
        return [{"episode_id": episode, "source": source, "source_file": source_file,
                 "min_timestamp": str(min(times)), "max_timestamp": str(max(times)),
                 "span_raw_time_units": str(max(times) - min(times))}
                for (episode, source, source_file), times in sorted(groups.items())]

    file_summaries = []
    for filename, count in sorted(file_rows.items()):
        fields = file_fields[filename] | file_expected[filename]
        file_summaries.append({"source_file": filename, "row_count": count,
                               "missing_documented_columns": sorted(file_expected[filename] - file_fields[filename]),
                               "field_missing_counts": {field: count - file_nonempty[filename][field] for field in sorted(fields)}})
    report = {
        "raw": {"row_count": raw_rows, "episode_count": len(raw_episodes),
                "episode_ids": sorted(raw_episodes),
                "track_count": len(raw_tracks), "type_row_counts": dict(sorted(raw_categories.items())),
                "category_track_counts": {k: len(v) for k, v in sorted(raw_targets.items())},
                "subtype_row_counts": dict(sorted(raw_subtypes.items())),
                "subtype_track_counts": {k: len(v) for k, v in sorted(raw_subtype_targets.items())},
                "file_field_summaries": file_summaries,
                "field_missing_counts": _global_field_missing(file_summaries),
                "time_spans": spans(raw_times)},
        "valid": {"row_count": len(valid), "episode_count": len(valid_episodes),
                  "episode_ids": sorted(valid_episodes),
                  "track_count": len(valid_tracks), "type_row_counts": dict(sorted(valid_categories.items())),
                  "category_track_counts": {k: len(v) for k, v in sorted(valid_targets.items())},
                  "subtype_row_counts": dict(sorted(valid_subtypes.items())),
                  "subtype_track_counts": {k: len(v) for k, v in sorted(valid_subtype_targets.items())},
                  "time_spans": spans(valid_times)},
        "excluded_row_count": raw_rows - len(valid),
        "anomaly_counts": {code: counts[code] for code in (
            "duplicate", "out_of_order", "missing_track_id", "invalid_timestamp",
            "missing_coordinates", "invalid_coordinates", "missing_category", "invalid_category", "extra_fields", "short_row")},
        "anomalies": anomalies,
        "group_identity": ["source_file", "track_id"],
        "time_unit": "unverified_raw_units",
        "category_mapping": TYPE_TO_CATEGORY.copy(),
        "note": "异常数量按类型计，单行可有多个异常；目标数按文件/ID 去重，不能相加为跨视角物理目标总数。一个目标的 type/sub_type 可能随行变化，分类目标数不能相加；Bicycle 不等于 Cyclist。",
    }
    return [item[2] for item in valid], report


def _global_field_missing(file_summaries):
    fields = {field for item in file_summaries for field in item["field_missing_counts"]}
    return {field: sum(item["field_missing_counts"].get(field, item["row_count"]) for item in file_summaries)
            for field in sorted(fields)}


def audit_trajectory_stream(records, on_valid, on_anomaly):
    """按来源文件连续分组审计，逐条交付结果，避免保留全量记录或异常。

    records 必须按 source_file 排序并保证每文件只出现一个连续组；回调由
    调用方负责持久化。内存上界为单个轨迹文件及全局 episode、文件索引，
    不假定大规模数据性能已验证。任何回调或输入错误向调用方传播。
    """
    if not callable(on_valid) or not callable(on_anomaly):
        raise TypeError("必须提供有效记录与异常记录的接收回调")
    combined = None
    episodes = {"raw": set(), "valid": set()}
    previous_file = None
    for source_file, batch in groupby(records, key=lambda row: row.get("source_file")):
        if not isinstance(source_file, str) or not source_file:
            raise ValueError("轨迹记录缺少原始文件路径")
        if previous_file is not None and source_file <= previous_file:
            raise ValueError("轨迹记录必须按 source_file 严格递增分组")
        previous_file = source_file
        valid, report = audit_trajectories(batch)
        for row in valid:
            on_valid(row)
        for anomaly in report["anomalies"]:
            on_anomaly(anomaly)
        if combined is None:
            combined = report
            combined["anomalies"] = None
            for name in ("raw", "valid"):
                episodes[name].update(report[name]["episode_ids"])
            continue
        combined["excluded_row_count"] += report["excluded_row_count"]
        for kind, count in report["anomaly_counts"].items():
            combined["anomaly_counts"][kind] += count
        for name in ("raw", "valid"):
            target, addition = combined[name], report[name]
            episodes[name].update(addition["episode_ids"])
            for field in ("row_count", "track_count"):
                target[field] += addition[field]
            for field in ("type_row_counts", "category_track_counts", "subtype_row_counts", "subtype_track_counts"):
                for key, count in addition[field].items():
                    target[field][key] = target[field].get(key, 0) + count
            target["time_spans"].extend(addition["time_spans"])
            if name == "raw":
                target["file_field_summaries"].extend(addition["file_field_summaries"])
    if combined is None:
        _, combined = audit_trajectories([])
        combined["anomalies"] = None
    for name in ("raw", "valid"):
        combined[name]["episode_ids"] = sorted(episodes[name])
        combined[name]["episode_count"] = len(episodes[name])
    combined["raw"]["field_missing_counts"] = _global_field_missing(combined["raw"]["file_field_summaries"])
    combined["anomaly_record_count"] = sum(combined["anomaly_counts"].values())
    combined["note"] += " 异常按回调逐条输出；汇总中 anomalies 为 null。"
    return combined
