"""CARLA实例分割到可见语义框的离线适配；不将3D投影框当作可见GT。"""
from __future__ import annotations

from pathlib import Path
from typing import Any


CARLA_CLASS_MAP = {
    "vehicle": "vehicle", "car": "vehicle", "truck": "truck", "bus": "bus",
    "motorcycle": "motorcycle", "bicycle": "cyclist", "rider": "cyclist",
    "pedestrian": "pedestrian", "person": "pedestrian",
}


def decode_instance_id_rgb(rgb_pixel: tuple[int, int, int]) -> tuple[int, int]:
    """CARLA instance PNG RGB通道：绿色低8位、蓝色高8位；红色是语义tag。"""
    red, green, blue = rgb_pixel
    if any(type(channel) is not int or not 0 <= channel <= 255 for channel in rgb_pixel):
        raise ValueError("RGB像素通道必须为0～255整数")
    return green + (blue << 8), red


def extract_visible_instances(mask_rgb: Any, actor_catalog: list[dict], *,
                              frame_id: int, episode_id: str, source_time_s: float,
                              rgb_path: str | Path, instance_path: str | Path) -> dict:
    """由真实实例像素计算可见框；全遮挡对象仅留hidden清单，不生成检测框。

    actor_catalog每项含actor_id、blueprint_id、category、track_id；实例分割传感器
    原生坐标与RGB尺寸须一致。`occluded_state`对可见实例保守记为unknown，只有
    完全无像素可严格判定全遮挡；不从3D投影补框。
    """
    import numpy as np

    array = np.asarray(mask_rgb)
    if array.ndim != 3 or array.shape[2] < 3 or array.dtype.kind not in "ui":
        raise ValueError("实例分割输入须为H×W×RGB整数像素")
    if type(frame_id) is not int or frame_id < 0 or not episode_id:
        raise ValueError("frame_id/episode_id非法")
    if source_time_s < 0:
        raise ValueError("source_time_s不能为负")
    height, width = array.shape[:2]
    actors: dict[int, dict] = {}
    for actor in actor_catalog:
        identity = actor.get("actor_id")
        if type(identity) is not int or not 0 <= identity <= 65535 or identity in actors:
            raise ValueError("CARLA instance RGB编码要求唯一actor_id位于0～65535")
        category = CARLA_CLASS_MAP.get(str(actor.get("category", "")).lower())
        if category is None:
            raise ValueError(f"不支持CARLA目标类别：{actor.get('category')}")
        if not actor.get("track_id") or not actor.get("blueprint_id"):
            raise ValueError("目标身份须带track_id与blueprint_id")
        actors[identity] = {**actor, "normalized_category": category}

    ids = array[:, :, 1].astype(np.uint32) + (array[:, :, 2].astype(np.uint32) << 8)
    semantic = array[:, :, 0].astype(np.uint16)
    labels, hidden = [], []
    for actor_id, actor in sorted(actors.items()):
        ys, xs = np.nonzero(ids == actor_id)
        if xs.size == 0:
            hidden.append({"actor_id": actor_id, "track_id": actor["track_id"],
                           "category": actor["normalized_category"],
                           "visibility": "fully_occluded_or_out_of_view"})
            continue
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
        touches_edge = x1 == 0 or y1 == 0 or x2 == width or y2 == height
        labels.append({
            "category": actor["normalized_category"], "track_id": actor["track_id"],
            "actor_id": actor_id, "blueprint_id": actor["blueprint_id"],
            "bbox_xyxy_visible_pixels": [x1, y1, x2, y2],
            "visible_pixel_count": int(xs.size),
            "semantic_tags": sorted(int(x) for x in np.unique(semantic[ids == actor_id])),
            "truncated_state": "touches_image_edge" if touches_edge else "not_truncated_by_image_edge",
            "occluded_state": "visible_occlusion_degree_unknown",
        })
    return {"schema_version": "1.0", "episode_id": episode_id, "frame_id": frame_id,
            "source_time_s": float(source_time_s), "image_size": [width, height],
            "rgb_path": str(rgb_path), "instance_path": str(instance_path),
            "rgb_sha256": _optional_sha256(rgb_path), "instance_sha256": _optional_sha256(instance_path),
            "label_source": "actual_carla_instance_segmentation_visible_pixels",
            "projected_3d_boxes_used_as_visible_gt": False,
            "negative_frame": len(labels) == 0, "labels": labels, "hidden_actor_review": hidden}


def _optional_sha256(path: str | Path) -> str | None:
    from .spd import sha256_file
    candidate = Path(path)
    return sha256_file(candidate) if candidate.is_file() else None


def read_instance_png(path: str | Path):
    """读取已保存的RGB通道顺序PNG，拒绝调色板/灰度图与无效尺寸。"""
    from PIL import Image
    with Image.open(Path(path)) as image:
        if image.mode != "RGB":
            raise ValueError(f"实例PNG必须保留RGB通道，当前{image.mode}")
        return image.copy()

