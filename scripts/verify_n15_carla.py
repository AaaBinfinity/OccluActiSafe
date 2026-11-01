"""复核N15真实CARLA双视角样本并生成中文标注抽查图；IDE无必填参数。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_ROOT = Path("H:/OccluActiSafeDataset/artifacts/节点资料/N15_CARLA双视角标注与SPD审计/02_运行产物")
EXPECTED_SCENES = {"vehicle", "vehicle_occlusion", "cyclist", "pedestrian", "negative"}
SCENE_NAMES = {"vehicle": "单车", "vehicle_occlusion": "同车道遮挡",
               "cyclist": "骑行者", "pedestrian": "行人", "negative": "无目标负例"}
ROLE_NAMES = {"train": "训练", "model_val": "模型验证", "test": "测试"}
CATEGORY_COLORS = {"vehicle": (255, 80, 55), "cyclist": (45, 220, 255),
                   "pedestrian": (75, 255, 110)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def synchronized_frame(frame: dict) -> tuple[int, float]:
    views = frame["views"]
    if set(views) != {"车端", "路侧"}:
        raise ValueError("CARLA采集帧必须同时包含车端和路侧视角")
    vehicle, roadside = views["车端"], views["路侧"]
    if vehicle["frame_id"] != roadside["frame_id"] or vehicle["frame_id"] != frame["carla_frame_id"]:
        raise ValueError("车端与路侧CARLA frame不一致")
    return int(frame["carla_frame_id"]), float(frame["source_time_s"])


def validate_box(label: dict, width: int, height: int) -> None:
    box = label["bbox_xyxy_visible_pixels"]
    if (len(box) != 4 or not (0 <= box[0] < box[2] <= width)
            or not (0 <= box[1] < box[3] <= height)
            or int(label["visible_pixel_count"]) <= 0):
        raise ValueError(f"可见实例框越界或空框：{label.get('track_id')}")


def _fonts():
    from PIL import ImageFont
    from scripts.audit_spd import review_font_identity
    identity = review_font_identity()
    return (ImageFont.truetype(identity["path"], 40),
            ImageFont.truetype(identity["path"], 28), identity)


def _draw_review(run_dir: Path, group: dict, scene: dict, frame: dict,
                 output: Path, fonts, width: int, height: int) -> dict:
    from PIL import Image, ImageDraw
    title_font, body_font, font_identity = fonts
    panel_width, panel_height = width, height
    canvas = Image.new("RGB", (panel_width * 2, panel_height + 74), (24, 29, 37))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 14), f"{ROLE_NAMES[group['project_role']]} · {group['map']} · "
              f"{SCENE_NAMES[group['scene_kind']]} · CARLA帧 {frame['carla_frame_id']} · "
              f"{frame['source_time_s']:.3f} s", font=title_font, fill=(245, 247, 250))
    source_hashes = {}
    for index, view_name in enumerate(("车端", "路侧")):
        record = frame["views"][view_name]
        scene_dir = Path(group["manifest"]).parent
        image_path = scene_dir / record["rgb"]
        label_path = image_path.parent / record["label"]
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        labels = json.loads(label_path.read_text(encoding="utf-8"))
        source_hashes[view_name] = {"rgb": record["rgb_sha256"], "label": record["label_sha256"],
                                   "mask": record["mask_sha256"]}
        draw_panel = ImageDraw.Draw(image)
        for item in labels["labels"]:
            color = CATEGORY_COLORS[item["category"]]
            x1, y1, x2, y2 = item["bbox_xyxy_visible_pixels"]
            draw_panel.rectangle((x1, y1, x2 - 1, y2 - 1), outline=color, width=4)
            distance = item.get("camera_distance_m")
            detail = f"{item['category']} id={item['actor_id']} {item['visible_pixel_count']}px"
            if distance is not None:
                detail += f" {distance:.1f}m"
            draw_panel.text((x1, max(0, y1 - 38)), detail, font=body_font, fill=color,
                            stroke_width=3, stroke_fill=(15, 18, 22))
        x_offset = index * panel_width
        canvas.paste(image, (x_offset, 74))
        hidden = labels["hidden_actor_review"]
        caption = f"{view_name} | 可见 {len(labels['labels'])} | 隐藏/画外待复核 {len(hidden)}"
        draw.rectangle((x_offset, panel_height + 44, x_offset + panel_width - 1, panel_height + 73),
                       fill=(24, 29, 37))
        draw.text((x_offset + 14, panel_height + 47), caption, font=body_font, fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)
    return {"path": output.name, "sha256": sha256(output), "bytes": output.stat().st_size,
            "font_sha256": font_identity["sha256"], "source_sha256": source_hashes,
            "frame_id": int(frame["carla_frame_id"]), "source_time_s": float(frame["source_time_s"])}


def _draw_sequence_sheet(group: dict, scene: dict, output: Path, fonts) -> dict:
    from PIL import Image, ImageDraw
    from PIL import ImageFont
    title_font, body_font, font_identity = fonts
    small_font = ImageFont.truetype(font_identity["path"], 15)
    cell_w, cell_h, columns, rows = 384, 216, 5, 6
    panel_w, panel_h = cell_w * columns, cell_h * rows
    canvas = Image.new("RGB", (panel_w * 2, panel_h + 104), (24, 29, 37))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 14), f"{ROLE_NAMES[group['project_role']]} · {group['map']} · "
              f"{SCENE_NAMES[group['scene_kind']]} · 全部30帧 · 车端/路侧同步", font=title_font,
              fill=(245, 247, 250))
    for index, frame in enumerate(scene["frames"]):
        row, column = divmod(index, columns)
        for view_index, view_name in enumerate(("车端", "路侧")):
            record = frame["views"][view_name]
            scene_dir = Path(group["manifest"]).parent
            label = json.loads((scene_dir / record["label"]).read_text(encoding="utf-8"))
            image_path = scene_dir / record["rgb"]
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            image_draw = ImageDraw.Draw(image)
            for item in label["labels"]:
                x1, y1, x2, y2 = item["bbox_xyxy_visible_pixels"]
                color = CATEGORY_COLORS[item["category"]]
                image_draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=color, width=8)
            image = image.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            x = view_index * panel_w + column * cell_w
            y = 104 + row * cell_h
            canvas.paste(image, (x, y))
            tag = f"{view_name} #{index:02d} 可见{len(label['labels'])} 隐藏{len(label['hidden_actor_review'])}"
            draw.rectangle((x, y, x + cell_w - 1, y + 26), fill=(18, 23, 31))
            draw.text((x + 8, y + 4), tag, font=small_font, fill=(255, 255, 255))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)
    return {"path": output.name, "sha256": sha256(output), "bytes": output.stat().st_size,
            "font_sha256": font_identity["sha256"], "frames": len(scene["frames"]),
            "frame_ids": [int(frame["carla_frame_id"]) for frame in scene["frames"]],
            "source_group_manifest_sha256": group["manifest_sha256"]}


def audit_run(run_dir: Path, *, create_visuals: bool = True) -> dict:
    from PIL import Image
    from scripts.audit_spd import review_font_identity
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "CARLA双视角数据清单.json"
    config_path = run_dir / "冻结采集配置.yaml"
    if not manifest_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("运行目录缺少冻结配置或CARLA双视角数据清单")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "completed" or not manifest.get("world_restored"):
        raise ValueError("CARLA运行未完整结束或未恢复原世界")
    if manifest["failed_scenes"]:
        raise ValueError("CARLA运行存在失败场景")
    if manifest["carla_version"] != "0.9.16" or manifest["source_commit"] == "uncommitted":
        raise ValueError("CARLA版本或不可变源码提交缺失")
    if len(manifest["groups"]) != 15 or {g["scene_kind"] for g in manifest["groups"]} != EXPECTED_SCENES:
        raise ValueError("场景组数量/类别不符合N15冻结矩阵")
    role_by_map = {}
    scene_groups = {}
    image_count = 0
    referenced_files = set()
    counters = Counter()
    visual_records = []
    contact_sheets = []
    fonts = _fonts() if create_visuals else None
    for group in manifest["groups"]:
        if group["status"] != "completed" or group["frames"] != 30:
            raise ValueError(f"场景未完整采集30帧：{group['episode_id']}")
        old_role = role_by_map.setdefault(group["map"], group["project_role"])
        if old_role != group["project_role"]:
            raise ValueError("同一地图被分入多个split")
        scene_groups.setdefault(group["scene_kind"], set()).add(group["project_role"])
        scene_path = Path(group["manifest"])
        if not scene_path.is_file() or sha256(scene_path) != group["manifest_sha256"]:
            raise ValueError(f"场景清单SHA-256不符：{scene_path}")
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
        if scene["status"] != "completed" or scene["group_id"] != group["group_id"]:
            raise ValueError(f"场景清单状态/组标识不符：{scene_path}")
        if len(scene["frames"]) != 30 or len(scene["labels"]) != 60:
            raise ValueError(f"场景图像或双视角标注数量不符：{scene_path}")
        for frame_index, frame in enumerate(scene["frames"]):
            frame_id, timestamp = synchronized_frame(frame)
            for view_name in ("车端", "路侧"):
                record = frame["views"][view_name]
                if abs(float(record.get("source_time_s", timestamp)) - timestamp) > 1e-9:
                    raise ValueError("双视角采集时刻不一致")
                label_path = scene_path.parent / record["label"]
                rgb_path = scene_path.parent / record["rgb"]
                mask_path = scene_path.parent / record["mask"]
                label = json.loads(label_path.read_text(encoding="utf-8"))
                if label["frame_id"] != frame_id or label["view"] != view_name or label["source_time_s"] != timestamp:
                    raise ValueError(f"标签frame/视角/时刻不符：{label_path}")
                if label["rgb_sha256"] != record["rgb_sha256"] or label["instance_sha256"] != record["mask_sha256"]:
                    raise ValueError(f"标签来源哈希与场景索引不一致：{label_path}")
                if label["projected_3d_boxes_used_as_visible_gt"]:
                    raise ValueError("检测到3D投影框作为可见GT")
                for path, expected_hash in ((label_path, record["label_sha256"]),
                                            (rgb_path, record["rgb_sha256"]),
                                            (mask_path, record["mask_sha256"])):
                    if not path.is_file() or sha256(path) != expected_hash:
                        raise ValueError(f"数据文件缺失或SHA-256不符：{path}")
                    referenced_files.add(path.resolve())
                with Image.open(rgb_path) as rgb, Image.open(mask_path) as mask:
                    if rgb.size != (1920, 1080) or mask.size != (1920, 1080):
                        raise ValueError(f"RGB/实例掩码分辨率不是1920×1080：{rgb_path}")
                    if rgb.format != "PNG" or mask.format != "PNG":
                        raise ValueError(f"RGB/实例掩码不是PNG：{rgb_path}")
                    import numpy as np
                    pixels = np.asarray(mask.convert("RGB"), dtype=np.uint8)
                    instance_ids = pixels[:, :, 1].astype(np.uint32) + (pixels[:, :, 2].astype(np.uint32) << 8)
                catalog_by_id = {int(actor["actor_id"]): actor for actor in scene["actor_catalog"]}
                visible_ids = set()
                for item in label["labels"]:
                    validate_box(item, 1920, 1080)
                    actor_id = int(item["actor_id"])
                    if actor_id not in catalog_by_id or actor_id in visible_ids:
                        raise ValueError(f"可见目标身份不在场景目录或重复：{label_path}")
                    visible_ids.add(actor_id)
                    ys, xs = np.nonzero(instance_ids == actor_id)
                    if (xs.size != int(item["visible_pixel_count"])
                            or [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                            != item["bbox_xyxy_visible_pixels"]):
                        raise ValueError(f"可见框/像素数与CARLA实例掩码不符：{label_path}")
                    counters[f"visible_{item['category']}"] += 1
                    counters["visible_labels"] += 1
                    proxy = item.get("occlusion_fraction_proxy")
                    proxy_bin = item.get("occlusion_bin")
                    if item.get("occlusion_proxy_method") != "visible_instance_pixels_vs_projected_3d_bbox_area":
                        raise ValueError(f"遮挡代理方法标记缺失：{label_path}")
                    if proxy is None:
                        expected_bin = "unknown"
                    else:
                        proxy = float(proxy)
                        if not math.isfinite(proxy) or not 0.0 <= proxy <= 1.0:
                            raise ValueError(f"遮挡比例代理不在[0,1]：{label_path}")
                        expected_bin = ("low_0_10pct" if proxy < 0.10 else
                                        "medium_10_50pct" if proxy < 0.50 else "high_50_100pct")
                    if proxy_bin != expected_bin:
                        raise ValueError(f"遮挡代理分箱与比例不一致：{label_path}")
                    counters[f"occlusion_proxy_{proxy_bin}"] += 1
                hidden_ids = {int(item["actor_id"]) for item in label["hidden_actor_review"]}
                if (visible_ids & hidden_ids or visible_ids | hidden_ids != set(catalog_by_id)
                        or any(np.any(instance_ids == actor_id) for actor_id in hidden_ids)):
                    raise ValueError(f"可见/隐藏身份不能与实例掩码一一对应：{label_path}")
                counters["hidden_actor_review_records"] += len(hidden_ids)
                if label["negative_frame"]:
                    counters["negative_view_frames"] += 1
                if not label["labels"] and not label["negative_frame"]:
                    raise ValueError("空可见标签帧未标记为负例")
            image_count += 2
        if create_visuals:
            frame = scene["frames"][len(scene["frames"]) // 2]
            output_dir = run_dir / "双视角中文标注抽查图"
            output = output_dir / f"{ROLE_NAMES[group['project_role']]}_{group['map']}_{SCENE_NAMES[group['scene_kind']]}_双视角标注抽查.png"
            visual_records.append(_draw_review(run_dir, group, scene, frame, output, fonts, 1920, 1080))
            sequence_sheet = output_dir / f"{ROLE_NAMES[group['project_role']]}_{group['map']}_{SCENE_NAMES[group['scene_kind']]}_三十帧全量抽查.png"
            contact_sheets.append(_draw_sequence_sheet(group, scene, sequence_sheet, fonts))
    if set(role_by_map.values()) != {"train", "model_val", "test"}:
        raise ValueError("地图级train/model_val/test划分不完整")
    if any(roles != {"train", "model_val", "test"} for roles in scene_groups.values()):
        raise ValueError("各场景没有覆盖三个地图级split")
    for role, map_name in (("train", "Town03_Opt"), ("model_val", "Town05_Opt"), ("test", "Town10HD_Opt")):
        if role_by_map.get(map_name) != role:
            raise ValueError("地图与冻结split映射不符")
    config_hash = sha256(config_path)
    audit = {"schema_version": "n15-carla-integrity-1", "run_id": manifest["run_id"],
             "status": "passed", "checked_at": datetime.now().astimezone().isoformat(),
             "capture_source_commit": manifest["source_commit"], "carla_version": manifest["carla_version"],
             "verification_source_commit": os.environ.get("OAS_SOURCE_COMMIT", "uncommitted"),
             "verifier_source_sha256": sha256(Path(__file__).resolve()),
             "world_restored": manifest["world_restored"], "config_sha256": config_hash,
             "capture_manifest_sha256": sha256(manifest_path), "group_manifest_count": len(manifest["groups"]),
             "scenario_frames": sum(g["frames"] for g in manifest["groups"]),
             "synchronized_view_frames": image_count, "verified_rgb_mask_label_files": len(referenced_files),
             "counts": dict(sorted(counters.items())), "maps": role_by_map,
             "scene_groups": {key: sorted(value) for key, value in scene_groups.items()},
             "result_scope": "CARLA数据完整性及标注来源复核；不代表N15全部标准通过或模型效果通过",
             "limitations": [
                 "遮挡分层使用可见实例像素数与投影3D轴对齐包围框面积比值，仅是可见率代理；受框与目标真实轮廓差异影响，不能当作真实遮挡率或统一跨类别视觉遮挡真值",
                 "hidden_actor_review只能区分零实例像素，无法仅据此区分完全遮挡和画外目标",
                 "每个split仅一个地图且地图与天气绑定，不能分离地图/天气泛化效果"],
             "failures": [], "visuals": visual_records, "all_frame_contact_sheets": contact_sheets,
             "font_sha256": (review_font_identity()["sha256"] if create_visuals else None)}
    audit["verified_rgb_mask_label_files"] = len(referenced_files)
    audit_path = run_dir / "N15双视角完整性复核.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"audit": audit, "audit_path": str(audit_path)}


def latest_completed_run(root: Path) -> Path:
    candidates = []
    for path in Path(root).glob("n15_carla_*"):
        manifest_path = path / "CARLA双视角数据清单.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("status") == "completed" and len(manifest.get("groups", [])) == 15:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("未找到完整的N15 CARLA正式采集运行")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="显式复核指定run；默认选择最新完整15组运行")
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--no-visuals", action="store_true", help="仅执行数据完整性复核")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else latest_completed_run(args.artifacts_root)
    result = audit_run(run_dir, create_visuals=not args.no_visuals)
    print(f"run_id={result['audit']['run_id']} status=passed "
          f"files={result['audit']['verified_rgb_mask_label_files']} "
          f"visuals={len(result['audit']['visuals'])} audit={result['audit_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
