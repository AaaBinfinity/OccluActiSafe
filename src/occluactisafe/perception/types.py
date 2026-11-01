"""检测与像素轨迹统一数据契约（N16）。"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class TrackNamespace:
    source_id: str
    sensor_id: str
    episode_id: str

    def __post_init__(self) -> None:
        if not all(isinstance(v, str) and v.strip() for v in
                   (self.source_id, self.sensor_id, self.episode_id)):
            raise ValueError("source/sensor/episode命名空间字段必须为非空字符串")

    @property
    def key(self) -> str:
        return f"{self.source_id}:{self.sensor_id}:{self.episode_id}"


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    score: float
    class_id: str

    def __post_init__(self) -> None:
        if len(self.xyxy) != 4 or not all(math.isfinite(float(v)) for v in self.xyxy):
            raise ValueError("检测框必须是4个有限原图xyxy坐标")
        x1, y1, x2, y2 = self.xyxy
        if x2 <= x1 or y2 <= y1:
            raise ValueError("检测框面积必须为正")
        if not math.isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("检测置信度必须在[0,1]")
        if not isinstance(self.class_id, str) or not self.class_id:
            raise ValueError("class_id必须为非空统一类别")


@dataclass(frozen=True)
class Detections:
    frame_id: str
    timestamp_s: float
    width: int
    height: int
    namespace: TrackNamespace
    items: tuple[Detection, ...]
    status: str = "ok"
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not self.frame_id or not math.isfinite(self.timestamp_s):
            raise ValueError("frame_id不能为空且时间戳必须有限")
        if type(self.width) is not int or type(self.height) is not int or min(self.width, self.height) <= 0:
            raise ValueError("原图尺寸必须为正整数")
        if self.status not in ("ok", "empty", "error", "missing"):
            raise ValueError("未知检测状态")
        if self.status == "error" and self.items:
            raise ValueError("错误帧不能包含伪造检测")
        if self.status == "error" and not self.error_message:
            raise ValueError("错误状态必须保留错误说明")
        if self.status != "error" and self.error_message is not None:
            raise ValueError("非错误状态不能携带错误说明")
        for item in self.items:
            x1, y1, x2, y2 = item.xyxy
            if x1 < 0 or y1 < 0 or x2 > self.width or y2 > self.height:
                raise ValueError("检测框必须处于原图像素范围")


@dataclass(frozen=True)
class PixelTrack:
    track_id: str
    xyxy: tuple[float, float, float, float]
    class_id: str
    score: float
    timestamp_s: float
    frame_id: str
    namespace: TrackNamespace
    confirmed: bool
    age_frames: int
    missed_frames: int


def validate_detections(items: Iterable[Detection], width: int, height: int) -> tuple[Detection, ...]:
    result = tuple(items)
    for item in result:
        x1, y1, x2, y2 = item.xyxy
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            raise ValueError("检测框超出原图范围")
    return result
