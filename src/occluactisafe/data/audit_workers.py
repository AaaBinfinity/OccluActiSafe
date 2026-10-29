"""N03 有界多进程轨迹审计；每个文件仍使用同一套严格质量规则。"""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import shutil

from .adapters.tfd import iter_trajectory_file
from .quality import audit_trajectory_stream, _global_field_missing


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _partition(args):
    root, files, catalog, directory, min_free = args
    directory = Path(directory)
    directory.mkdir()
    samples, points, map_counts = [], [], Counter()
    preview_key = None
    counts = {"files": 0, "valid_rows": 0, "anomalies": 0}

    def space():
        if shutil.disk_usage(directory).free < min_free:
            raise OSError("审计输出磁盘可用空间不足 10 GiB")

    def records():
        for name in files:
            yield from iter_trajectory_file(root, name, map_catalog=catalog)
            counts["files"] += 1
            if counts["files"] % 100 == 0:
                _write(directory / "progress.json", counts)

    with gzip.open(directory / "trajectories.jsonl.gz", "wt", encoding="utf-8", newline="\n", compresslevel=1) as output, \
            (directory / "anomalies.jsonl").open("w", encoding="utf-8") as anomalies:
        def accept(row):
            nonlocal preview_key
            if counts["valid_rows"] % 10000 == 0:
                space()
            output.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
            counts["valid_rows"] += 1
            map_counts[row["map_status"]] += 1
            if len(samples) < 20:
                samples.append(row)
            key = row["source_file"], row["track_id"]
            if preview_key is None:
                preview_key = key
            if key == preview_key and len(points) < 500:
                points.append((float(row["x"]), float(row["y"])))

        def anomaly(row):
            if counts["anomalies"] % 10000 == 0:
                space()
            anomalies.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
            counts["anomalies"] += 1

        report = audit_trajectory_stream(records(), accept, anomaly)
    _write(directory / "quality.json", report)
    _write(directory / "extras.json", {"samples": samples, "points": points, "maps": map_counts})
    _write(directory / "progress.json", {**counts, "complete": True})
    return str(directory)


def parallel_trajectories(root, files, catalog, run_dir, workers, min_free):
    """按连续文件区间分片；拼接 gzip members，解压后与串行文件顺序一致。"""
    count = min(workers, len(files))
    partitions = [files[len(files) * i // count:len(files) * (i + 1) // count] for i in range(count)]
    shard_root = run_dir / "shards"
    shard_root.mkdir()
    args = [(str(root), batch, catalog, str(shard_root / str(i)), min_free) for i, batch in enumerate(partitions)]
    with ProcessPoolExecutor(max_workers=count) as pool:
        directories = list(pool.map(_partition, args))
    combined, samples, points, maps = None, [], [], Counter()
    episodes = {"raw": set(), "valid": set()}
    with (run_dir / "trajectories.jsonl.gz").open("wb") as trajectories, \
            (run_dir / "anomalies.jsonl").open("wb") as anomalies:
        for name in directories:
            directory = Path(name)
            report = json.loads((directory / "quality.json").read_text(encoding="utf-8"))
            extra = json.loads((directory / "extras.json").read_text(encoding="utf-8"))
            for section in episodes:
                episodes[section].update(report[section]["episode_ids"])
            if combined is None:
                combined = report
                points = extra["points"]
            else:
                combined["excluded_row_count"] += report["excluded_row_count"]
                combined["anomaly_record_count"] += report["anomaly_record_count"]
                for key, value in report["anomaly_counts"].items():
                    combined["anomaly_counts"][key] += value
                for section in ("raw", "valid"):
                    dst, src = combined[section], report[section]
                    for key in ("row_count", "track_count"):
                        dst[key] += src[key]
                    for key in ("type_row_counts", "category_track_counts", "subtype_row_counts", "subtype_track_counts"):
                        for label, value in src[key].items():
                            dst[key][label] = dst[key].get(label, 0) + value
                    dst["time_spans"].extend(src["time_spans"])
                combined["raw"]["file_field_summaries"].extend(report["raw"]["file_field_summaries"])
            samples.extend(extra["samples"][:20 - len(samples)])
            maps.update(extra["maps"])
            for source, output in ((directory / "trajectories.jsonl.gz", trajectories), (directory / "anomalies.jsonl", anomalies)):
                if shutil.disk_usage(run_dir).free < min_free:
                    raise OSError("合并审计产物时磁盘可用空间不足 10 GiB")
                with source.open("rb") as stream:
                    shutil.copyfileobj(stream, output, 1024 * 1024)
    for section in episodes:
        combined[section]["episode_ids"] = sorted(episodes[section])
        combined[section]["episode_count"] = len(episodes[section])
    combined["raw"]["field_missing_counts"] = _global_field_missing(combined["raw"]["file_field_summaries"])
    return combined, samples, points, maps
