"""统一检测与跟踪契约。"""
from occluactisafe.perception.registry import register_detector
from occluactisafe.perception.adapters.ultralytics import UltralyticsDetector

register_detector("ultralytics", UltralyticsDetector)
