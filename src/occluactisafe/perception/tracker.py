"""基于IoU高低分两阶段关联的ByteTrack风格像素跟踪器（N16）。"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np

from scipy.optimize import linear_sum_assignment

from occluactisafe.perception.types import Detections, PixelTrack


@dataclass(frozen=True)
class ByteTrackConfig:
    high_threshold: float = 0.5
    low_threshold: float = 0.1
    new_track_threshold: float = 0.5
    high_iou_threshold: float = 0.3
    low_iou_threshold: float = 0.2
    min_hits: int = 2
    max_lost_frames: int = 30
    max_lost_seconds: float = 1.0

    def __post_init__(self):
        for name in ("high_threshold", "low_threshold", "new_track_threshold",
                     "high_iou_threshold", "low_iou_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name}必须在[0,1]")
        if self.low_threshold >= self.high_threshold:
            raise ValueError("低分阈值必须小于高分阈值")
        if type(self.min_hits) is not int or self.min_hits < 1:
            raise ValueError("min_hits必须为正整数")
        if type(self.max_lost_frames) is not int or self.max_lost_frames < 0:
            raise ValueError("max_lost_frames必须为非负整数")
        if not math.isfinite(self.max_lost_seconds) or self.max_lost_seconds <= 0:
            raise ValueError("max_lost_seconds必须为正数")


@dataclass
class _TrackState:
    local_id: int
    box: tuple[float, float, float, float]
    class_id: str
    score: float
    timestamp_s: float
    frame_id: str
    hits: int = 1
    missed_frames: int = 0
    mean: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float64))
    covariance: np.ndarray = field(default_factory=lambda: np.eye(8, dtype=np.float64))


@dataclass
class _NamespaceState:
    next_id: int = 1
    last_timestamp_s: float | None = None
    last_numeric_frame: int | None = None
    tracks: dict[int, _TrackState] = field(default_factory=dict)


def _predict(track: _TrackState, timestamp_s: float):
    dt = max(0.0, timestamp_s - track.timestamp_s)
    mean, _ = _kalman_predict(track.mean, track.covariance, dt)
    return _xyah_to_box(mean[:4])


def _box_to_xyah(box):
    x1, y1, x2, y2 = box
    height = max(1e-3, y2-y1)
    return np.asarray([(x1+x2)/2, (y1+y2)/2, max(1e-3, x2-x1)/height, height], dtype=np.float64)


def _xyah_to_box(value):
    cx, cy, aspect, height = value
    height = max(1e-3, float(height))
    width = max(1e-3, float(aspect)*height)
    return (float(cx-width/2), float(cy-height/2),
            float(cx+width/2), float(cy+height/2))


def _kalman_predict(mean, covariance, dt):
    transition = np.eye(8, dtype=np.float64)
    transition[:4, 4:] = np.eye(4, dtype=np.float64)*dt
    height = max(1.0, float(mean[3]))
    position_std = np.asarray([height/20, height/20, 0.1, height/20])
    velocity_std = np.asarray([height/160, height/160, 0.01, height/160])
    process = np.diag(np.square(np.r_[position_std, velocity_std])*max(dt, 1e-3))
    return transition @ mean, transition @ covariance @ transition.T + process


def _kalman_update(track, detection, timestamp_s):
    dt = max(0.0, timestamp_s-track.timestamp_s)
    mean, covariance = _kalman_predict(track.mean, track.covariance, dt)
    observation = np.zeros((4, 8), dtype=np.float64)
    observation[:, :4] = np.eye(4, dtype=np.float64)
    height = max(1.0, float(mean[3]))
    measurement_std = np.asarray([height/20, height/20, 0.1, height/20])
    noise = np.diag(np.square(measurement_std))
    innovation = _box_to_xyah(detection.xyxy)-observation @ mean
    innovation_cov = observation @ covariance @ observation.T + noise
    gain = covariance @ observation.T @ np.linalg.inv(innovation_cov)
    identity = np.eye(8, dtype=np.float64)
    track.mean = mean + gain @ innovation
    # Joseph form preserves symmetry and positive-semidefinite covariance numerically.
    residual = identity-gain @ observation
    track.covariance = residual @ covariance @ residual.T + gain @ noise @ gain.T
    track.box = detection.xyxy
    track.class_id = detection.class_id
    track.score = detection.score
    track.timestamp_s = timestamp_s


def _iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right-left) * max(0.0, bottom-top)
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _assign(track_ids, predicted, detections, threshold):
    if not track_ids or not detections:
        return [], set(track_ids), set(range(len(detections)))
    cost = []
    overlaps = []
    for track_id in track_ids:
        row, ious = [], []
        for det in detections:
            overlap = _iou(predicted[track_id], det.xyxy) if det.class_id else 0.0
            ious.append(overlap)
            row.append(1.0 - overlap if overlap >= threshold else 1e6)
        cost.append(row); overlaps.append(ious)
    rows, cols = linear_sum_assignment(cost)
    matches = [(track_ids[r], c) for r, c in zip(rows, cols) if overlaps[r][c] >= threshold]
    return matches, set(track_ids) - {x for x, _ in matches}, set(range(len(detections))) - {y for _, y in matches}


class ByteTrack:
    """跟踪只按 source/sensor/episode 命名空间隔离，不读取GT。"""
    def __init__(self, config: ByteTrackConfig | None = None):
        self.config = config or ByteTrackConfig()
        self._states: dict[str, _NamespaceState] = {}

    def reset(self, *, source_id: str | None = None, sensor_id: str | None = None,
              episode_id: str | None = None) -> None:
        if source_id is None and sensor_id is None and episode_id is None:
            self._states.clear()
            return
        keys = [key for key in self._states
                if (source_id is None or key.split(":", 2)[0] == source_id)
                and (sensor_id is None or key.split(":", 2)[1] == sensor_id)
                and (episode_id is None or key.split(":", 2)[2] == episode_id)]
        for key in keys:
            del self._states[key]

    def track(self, detections: Detections, timestamp: float | None = None) -> tuple[PixelTrack, ...]:
        now = detections.timestamp_s if timestamp is None else float(timestamp)
        if not math.isfinite(now) or not math.isclose(now, detections.timestamp_s, abs_tol=1e-9):
            raise ValueError("track时间必须与检测采集时间一致")
        key = detections.namespace.key
        state = self._states.setdefault(key, _NamespaceState())
        if state.last_timestamp_s is not None and now <= state.last_timestamp_s:
            raise ValueError("同一episode时间戳必须严格递增")
        try:
            frame_num = int(detections.frame_id)
        except (TypeError, ValueError):
            frame_num = None
        frame_gap = (max(1, frame_num - state.last_numeric_frame)
                     if frame_num is not None and state.last_numeric_frame is not None else 1)
        state.last_timestamp_s = now
        if frame_num is not None:
            state.last_numeric_frame = frame_num
        for track in state.tracks.values():
            track.missed_frames += frame_gap

        high = [det for det in detections.items if det.score >= self.config.high_threshold]
        low = [det for det in detections.items
               if self.config.low_threshold <= det.score < self.config.high_threshold]
        candidates = sorted(state.tracks)
        predicted = {track_id: _predict(state.tracks[track_id], now) for track_id in candidates}
        high_matches, unmatched_ids, unmatched_high = _assign(
            candidates, predicted, high, self.config.high_iou_threshold)
        for track_id, det_index in high_matches:
            self._update(state.tracks[track_id], high[det_index], now, detections.frame_id,
                         frame_gap)
        low_matches, unmatched_ids, _ = _assign(
            sorted(unmatched_ids), predicted, low, self.config.low_iou_threshold)
        for track_id, det_index in low_matches:
            self._update(state.tracks[track_id], low[det_index], now, detections.frame_id,
                         frame_gap)
        for index in sorted(unmatched_high):
            det = high[index]
            if det.score >= self.config.new_track_threshold:
                local_id = state.next_id
                state.next_id += 1
                measurement = _box_to_xyah(det.xyxy)
                mean = np.r_[measurement, np.zeros(4, dtype=np.float64)]
                height = max(1.0, float(measurement[3]))
                std = np.asarray([height/10, height/10, 0.1, height/10,
                                  height/16, height/16, 0.01, height/16])
                covariance = np.diag(np.square(std))
                state.tracks[local_id] = _TrackState(local_id, det.xyxy, det.class_id,
                                                      det.score, now, detections.frame_id,
                                                      mean=mean, covariance=covariance)
        for track_id, track in list(state.tracks.items()):
            if (track.missed_frames > self.config.max_lost_frames
                    or now - track.timestamp_s > self.config.max_lost_seconds):
                del state.tracks[track_id]
        output = []
        for track_id, track in sorted(state.tracks.items()):
            if track.frame_id != detections.frame_id or track.hits < self.config.min_hits:
                continue
            output.append(PixelTrack(
                track_id=f"{key}:track-{track_id}", xyxy=track.box,
                class_id=track.class_id, score=track.score, timestamp_s=now,
                frame_id=detections.frame_id, namespace=detections.namespace,
                confirmed=track.hits >= self.config.min_hits, age_frames=track.hits,
                missed_frames=track.missed_frames))
        return tuple(output)

    @staticmethod
    def _update(track, detection, now, frame_id, frame_gap):
        _kalman_update(track, detection, now)
        track.frame_id = frame_id
        track.hits += 1
        track.missed_frames = 0
