"""检测器最小统一协议。"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from occluactisafe.perception.types import Detections, TrackNamespace


@runtime_checkable
class Detector(Protocol):
    def detect(self, frame, *, frame_id: str, timestamp_s: float,
               namespace: TrackNamespace) -> Detections:
        """输入原始RGB帧；返回原图坐标检测，保留空帧或明确错误状态。"""
