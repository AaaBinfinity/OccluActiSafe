"""过程图片归档到移动硬盘原运行目录；中文命名，仓库只保存位置与哈希。"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import os
import tempfile
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = Path("H:/OccluActiSafeDataset/artifacts")
ARCHIVE_ROOT = PROJECT_ROOT / "docs/过程图"
NAMES = {
    "trajectory_preview": "轨迹预览", "epoch_validation": "逐轮验证指标",
    "step_optimization": "逐步优化指标", "step_resources": "逐步资源占用",
    "step_training": "逐步训练指标", "loss_validation_curves": "训练损失与验证曲线",
    "overfit_step_curves": "匀速拟合逐步诊断", "capacity_queue": "通信容量与排队",
    "duplex_timing": "双向通信时序", "association_gates": "跨源关联与门控",
    "source_filter_history": "来源滤波与历史状态", "geometry_intervals": "车体几何与区间细分",
    "raw_scores_coverage": "原始风险评分与时域覆盖", "visibility_constraints": "可见性与未知区域约束",
}
EXTENSIONS = {".png", ".svg", ".jpg", ".jpeg", ".webp", ".gif"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    """先写同目录临时文件再原子替换，避免直接截断现有清单。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".图片清单-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name); stream.write(raw)
        for attempt in range(8):
            try:
                os.replace(temporary, path)
                break
            except OSError as exc:
                if attempt == 7 or (not isinstance(exc, PermissionError) and exc.errno != 22): raise
                time.sleep(.1 * (attempt + 1))
    finally:
        if temporary is not None and temporary.exists(): temporary.unlink()


def contained(root, relative):
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()): raise ValueError("图片路径越界")
    return target


def verify(root, catalog):
    for entry in catalog["images"]:
        target = contained(root, entry["artifact_path"])
        if not target.is_file() or sha(target) != entry["sha256"]:
            raise ValueError(f"硬盘图片缺失或哈希不符：{target}")
    return len(catalog["images"])


def archive(artifacts_root=ARTIFACTS_ROOT, archive_root=ARCHIVE_ROOT, *, verify_only=False):
    artifacts_root, archive_root = Path(artifacts_root).resolve(), Path(archive_root).resolve()
    if not archive_root.is_relative_to(PROJECT_ROOT): raise ValueError("元数据目录必须位于项目内")
    if not artifacts_root.is_dir(): raise FileNotFoundError(f"硬盘产物根未接入：{artifacts_root}")
    if artifacts_root.is_relative_to(PROJECT_ROOT): raise ValueError("图片必须保存到仓库外的实际硬盘目录")
    catalog_path = archive_root / "图片来源与哈希.json"
    legacy = artifacts_root / "仓库归档时图片来源与哈希.json"
    raw = read(catalog_path) if catalog_path.is_file() else (read(legacy) if legacy.is_file() else {"images": []})
    indexed = {}
    for old in raw["images"]:
        relative = old.get("artifact_path", old.get("path"))
        if relative is None: raise ValueError("图片清单缺少路径")
        entry = {"original_artifact_path": old["original_artifact_path"], "artifact_path": relative,
                 "legacy_repository_path": old.get("legacy_repository_path", "docs/过程图/" + relative),
                 "sha256": old["sha256"], "bytes": old["bytes"]}
        for key in ("plot_manifest_original_path", "plot_manifest_sha256"):
            if key in old: entry[key] = old[key]
        indexed[entry["original_artifact_path"]] = entry
    catalog = {"schema_version": "process-images-2", "storage": "external_artifacts",
               "path_base": "本机实际移动硬盘项目artifacts目录", "images": list(indexed.values())}
    if verify_only:
        return {"verified": verify(artifacts_root, catalog), "moved": 0}
    archive_root.mkdir(parents=True, exist_ok=True)
    # 原始入库清单曾被不可变证据索引引用，整份移回硬盘留存，不改写它。
    if raw.get("schema_version") == "process-images-1" and catalog_path.exists():
        if legacy.exists():
            if legacy.read_bytes() != catalog_path.read_bytes(): raise ValueError("原始清单同名冲突")
        else: shutil.move(str(catalog_path), str(legacy))
    moved = 0

    def transfer(source, target, entry):
        nonlocal moved
        if not source.is_file() or sha(source) != entry["sha256"]: raise ValueError(f"源图片缺失或哈希不符：{source}")
        if source == target:
            # 已在硬盘原位且中文命名的新增帧不移动/删除；最后统一原子登记。
            # 中途清单写入失败时原图仍在原路径，可直接重扫续传，避免逐帧重写整个清单。
            indexed[entry["original_artifact_path"]] = entry
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha(target) != entry["sha256"]: raise ValueError(f"目标同名内容冲突：{target}")
        elif source != target: shutil.copy2(source, target)
        if sha(target) != entry["sha256"]: raise ValueError(f"目标图片哈希不符：{target}")
        indexed[entry["original_artifact_path"]] = entry
        catalog["images"] = sorted(indexed.values(), key=lambda e: e["original_artifact_path"])
        write(catalog_path, catalog)
        if source != target: source.unlink(); moved += 1

    for entry in list(indexed.values()):
        target = contained(artifacts_root, entry["artifact_path"])
        source = contained(PROJECT_ROOT, entry["legacy_repository_path"])
        if not source.is_relative_to(archive_root): raise ValueError("旧图片来源必须在过程图目录内")
        if source.exists(): transfer(source, target, entry)
        elif not target.is_file() or sha(target) != entry["sha256"]:
            raise ValueError(f"硬盘图片缺失或哈希不符：{target}")
    by_path = {e["artifact_path"]: e for e in indexed.values()}
    for task in sorted(artifacts_root.iterdir()):
        if not task.is_dir() or not re.fullmatch(r"n\d{2}(?:_.*)?", task.name): continue
        for source in sorted(task.rglob("*")):
            if not source.is_file() or source.suffix.lower() not in EXTENSIONS: continue
            source = contained(artifacts_root, source.relative_to(artifacts_root))
            relative = source.relative_to(artifacts_root)
            stem = NAMES.get(source.stem)
            if stem is None:
                if any('\u4e00' <= c <= '\u9fff' for c in source.stem): stem = source.stem
                else: raise ValueError(f"请先为新图类声明中文名称：{source.name}")
            target = contained(artifacts_root, relative.parent / (stem + source.suffix.lower()))
            canonical = target.relative_to(artifacts_root).as_posix()
            entry = by_path.get(canonical)
            if entry is None:
                entry = {"original_artifact_path": relative.as_posix(), "artifact_path": canonical,
                         "legacy_repository_path": "docs/过程图/" + canonical,
                         "sha256": sha(source), "bytes": source.stat().st_size}
                manifest = source.parent / "plot_manifest.json"
                if manifest.is_file():
                    entry.update(plot_manifest_original_path=manifest.relative_to(artifacts_root).as_posix(), plot_manifest_sha256=sha(manifest))
                by_path[canonical] = entry
            if source != target or entry["original_artifact_path"] not in indexed:
                transfer(source, target, entry)
            elif sha(source) != entry["sha256"]: raise ValueError(f"图片内容已改变：{source}")
    catalog["images"] = sorted(indexed.values(), key=lambda e: e["original_artifact_path"])
    write(catalog_path, catalog)
    return {"verified": verify(artifacts_root, catalog), "moved": moved}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, default=ARTIFACTS_ROOT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(archive(args.artifacts_root, verify_only=args.verify_only), ensure_ascii=False))
    return 0


if __name__ == "__main__": raise SystemExit(main())
