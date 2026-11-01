"""Ultralytics结果适配；库在使用时懒加载，N16不下载权重。"""
from __future__ import annotations

from pathlib import Path
import math

from occluactisafe.perception.types import Detection, Detections, TrackNamespace


class UltralyticsDetector:
    def __init__(self, model=None, *, weights: str | Path | None = None,
                 class_map: dict[int, str] | None = None, confidence: float = 0.01,
                 imgsz: int = 960, device: str | None = None, predict_kwargs: dict | None = None):
        if model is None:
            if weights is None:
                raise ValueError("必须提供已存在权重路径或模型对象；不会自动下载权重")
            path = Path(weights).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"权重文件不存在：{path}")
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("需安装并锁定Ultralytics后才能加载权重") from exc
            model = YOLO(str(path))
        if not class_map:
            raise ValueError("必须显式提供框架类别到项目统一类别的映射")
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence必须在[0,1]")
        if type(imgsz) is not int or imgsz <= 0:
            raise ValueError("imgsz必须为正整数")
        self.model = model
        self.class_map = dict(class_map or {})
        self.confidence = confidence
        self.imgsz = imgsz
        self.device = device
        self.predict_kwargs = dict(predict_kwargs or {})

    def detect(self, frame, *, frame_id: str, timestamp_s: float,
               namespace: TrackNamespace) -> Detections:
        shape = getattr(frame, "shape", None)
        if shape is None or len(shape) != 3 or int(shape[2]) != 3:
            raise ValueError("frame必须是H×W×3 RGB原图数组")
        height, width = int(shape[0]), int(shape[1])
        if height <= 0 or width <= 0:
            raise ValueError("原图尺寸必须为正")
        kwargs = {"conf": self.confidence, "imgsz": self.imgsz, "verbose": False,
                  **self.predict_kwargs}
        if self.device is not None:
            kwargs["device"] = self.device
        # Ultralytics接收numpy数组时按BGR解释；本项目契约输入为RGB，先做显式通道变换。
        try:
            import numpy as np
            frame_bgr = np.ascontiguousarray(frame[..., ::-1])
            results = list(self.model.predict(source=frame_bgr, **kwargs))
        except Exception as exc:
            return Detections(frame_id, timestamp_s, width, height, namespace, (),
                              "error", f"{type(exc).__name__}: {exc}")
        if not results:
            return Detections(frame_id, timestamp_s, width, height, namespace, (), "empty")
        if len(results) != 1:
            return Detections(frame_id, timestamp_s, width, height, namespace, (),
                              "error", f"单帧推理返回{len(results)}个结果，期望1个")
        result = results[0]
        orig_shape = tuple(getattr(result, "orig_shape", ()))
        if orig_shape and orig_shape != (height, width):
            raise ValueError(f"模型结果原图尺寸{orig_shape}与输入{(height, width)}不一致")
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return Detections(frame_id, timestamp_s, width, height, namespace, (), "empty")
        xyxy = boxes.xyxy.detach().cpu().tolist()
        scores = boxes.conf.detach().cpu().tolist()
        classes = boxes.cls.detach().cpu().tolist()
        items = []
        for coords, score, class_index in zip(xyxy, scores, classes):
            index = int(class_index)
            if index not in self.class_map:
                return Detections(frame_id, timestamp_s, width, height, namespace, (),
                                  "error", f"框架类别{index}没有统一类别映射")
            label = self.class_map[index]
            # Ultralytics Results.boxes.xyxy已还原到orig_shape；禁止将网络内部缩放坐标传出。
            items.append(Detection(tuple(float(v) for v in coords), float(score), label))
        return Detections(frame_id, timestamp_s, width, height, namespace,
                          tuple(items), "ok" if items else "empty")
