"""过程图片中文归档；IDE直接运行，原始日志/权重不迁移。"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify(root, catalog):
    for entry in catalog["images"]:
        target = (root / entry["path"]).resolve()
        if not target.is_relative_to(root.resolve()): raise ValueError("归档路径越界")
        if not target.is_file() or sha(target) != entry["sha256"]: raise ValueError(f"归档图片缺失或哈希不符：{target}")
    return len(catalog["images"])


def archive(artifacts_root=ARTIFACTS_ROOT, archive_root=ARCHIVE_ROOT, *, verify_only=False):
    artifacts_root, archive_root = Path(artifacts_root).resolve(), Path(archive_root).resolve()
    if not archive_root.is_relative_to(PROJECT_ROOT): raise ValueError("过程图必须位于项目目录内")
    catalog_path = archive_root / "图片来源与哈希.json"
    catalog = read(catalog_path) if catalog_path.is_file() else {"schema_version": "process-images-1", "images": []}
    verify(archive_root, catalog)
    if verify_only: return {"verified": len(catalog["images"]), "moved": 0}
    if not artifacts_root.is_dir(): raise FileNotFoundError(f"硬盘产物根未接入：{artifacts_root}")
    archive_root.mkdir(parents=True, exist_ok=True)
    indexed = {entry["original_artifact_path"]: entry for entry in catalog["images"]}
    moved = 0
    # 仅逐节点运行目录；不扫描/搬动数据、环境或依赖缓存中的图片。
    for task in sorted(artifacts_root.iterdir()):
        if not task.is_dir() or not re.fullmatch(r"n\d{2}(?:_.*)?", task.name): continue
        for source in sorted(task.rglob("*")):
            if not source.is_file() or source.suffix.lower() not in EXTENSIONS: continue
            source = source.resolve()
            if not source.is_relative_to(artifacts_root): raise ValueError("源图片路径越界")
            relative = source.relative_to(artifacts_root)
            stem = NAMES.get(source.stem)
            if stem is None:
                if any('\u4e00' <= c <= '\u9fff' for c in source.stem): stem = source.stem
                else: raise ValueError(f"请先为新图类声明中文名称：{source.name}")
            target = (archive_root / relative.parent / (stem + source.suffix.lower())).resolve()
            if not target.is_relative_to(archive_root): raise ValueError("目标图片路径越界")
            digest = sha(source); key = relative.as_posix()
            entry = {"original_artifact_path": key, "path": target.relative_to(archive_root).as_posix(),
                     "sha256": digest, "bytes": source.stat().st_size}
            plot_manifest = source.parent / "plot_manifest.json"
            if plot_manifest.is_file():
                entry["plot_manifest_original_path"] = plot_manifest.relative_to(artifacts_root).as_posix()
                entry["plot_manifest_sha256"] = sha(plot_manifest)
                entry["plot_source_record"] = read(plot_manifest)
            if key in indexed and indexed[key]["sha256"] != digest: raise ValueError(f"同一运行图片已变更，禁止覆盖：{source}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if sha(target) != digest: raise ValueError(f"目标同名内容冲突：{target}")
            else: shutil.copy2(source, target)
            if sha(target) != digest: raise ValueError(f"迁移后哈希不符：{target}")
            indexed[key] = entry
            catalog["images"] = sorted(indexed.values(), key=lambda e: e["original_artifact_path"])
            # 先保存可恢复映射并核对目标，再删除确切的单个源图片；不递归清理目录。
            write(catalog_path, catalog)
            source.unlink(); moved += 1
    catalog["images"] = sorted(indexed.values(), key=lambda e: e["original_artifact_path"])
    write(catalog_path, catalog)
    verify(archive_root, catalog)
    return {"verified": len(catalog["images"]), "moved": moved}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, default=ARTIFACTS_ROOT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(archive(args.artifacts_root, verify_only=args.verify_only), ensure_ascii=False))
    return 0


if __name__ == "__main__": raise SystemExit(main())
