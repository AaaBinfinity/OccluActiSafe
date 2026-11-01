"""检测/跟踪统一评分，缺失帧按漏检处理。"""
from __future__ import annotations

from dataclasses import dataclass
import math
from collections import defaultdict

from scipy.optimize import linear_sum_assignment

from occluactisafe.perception.types import Detection, PixelTrack


@dataclass(frozen=True)
class GroundTruthObject:
    xyxy: tuple[float, float, float, float]
    class_id: str
    track_id: str | None
    raw_category: str = ""
    pixel_scale: str = "unknown"
    official_occlusion_state: str = "unknown"
    truncation_state: str = "unknown"


@dataclass(frozen=True)
class MetricFrame:
    frame_id: str
    ground_truth: tuple[GroundTruthObject, ...]
    detections: tuple[Detection, ...] = ()
    tracks: tuple[PixelTrack, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()


def box_iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _match(gt, predictions, threshold):
    if not gt or not predictions:
        return [], set(range(len(gt))), set(range(len(predictions)))
    overlaps = [[box_iou(g.xyxy, p.xyxy) if g.class_id == p.class_id else 0.0
                 for p in predictions] for g in gt]
    costs = [[1.0-value if value >= threshold else 1e6 for value in row] for row in overlaps]
    rows, cols = linear_sum_assignment(costs)
    matches = [(r, c) for r, c in zip(rows, cols) if overlaps[r][c] >= threshold]
    return matches, set(range(len(gt))) - {r for r, _ in matches}, set(range(len(predictions))) - {c for _, c in matches}


def score_detections(frames: list[MetricFrame], *, iou_threshold: float = 0.5) -> dict:
    """返回micro TP/FP/FN、逐类召回与AP@阈值；空预测帧不会被丢弃。"""
    if not math.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
        raise ValueError("IoU阈值必须在(0,1]")
    tp = fp = fn = 0
    by_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0})
    ap_data = defaultdict(list)
    frame_map = {frame.frame_id: frame for frame in frames}
    if len(frame_map) != len(frames):
        raise ValueError("MetricFrame.frame_id必须唯一并覆盖完整评测帧")
    for frame in frames:
        matches, missing, extra = _match(frame.ground_truth, frame.detections, iou_threshold)
        tp += len(matches); fn += len(missing); fp += len(extra)
        matched_gt, matched_pred = {g for g, _ in matches}, {p for _, p in matches}
        for index, gt in enumerate(frame.ground_truth):
            row = by_class[gt.class_id]; row["gt"] += 1
            if index not in matched_gt: row["fn"] += 1
        for index, pred in enumerate(frame.detections):
            row = by_class[pred.class_id]; row["pred"] += 1
            is_tp = index in matched_pred
            row["tp" if is_tp else "fp"] += 1
        # AP必须按置信度贪心匹配；操作点的最大IoU分配不能决定AP的TP顺序。
        remaining = set(range(len(frame.ground_truth)))
        for pred in sorted(frame.detections, key=lambda p: -p.score):
            possible = [(box_iou(frame.ground_truth[i].xyxy, pred.xyxy), i) for i in remaining
                        if frame.ground_truth[i].class_id == pred.class_id]
            overlap, index = max(possible, default=(0.0, -1))
            is_tp = overlap >= iou_threshold
            if is_tp:
                remaining.remove(index)
            ap_data[pred.class_id].append((pred.score, is_tp))
    for row in by_class.values():
        denom = row["tp"] + row["fn"]
        row["recall"] = row["tp"] / denom if denom else None
        row["precision"] = row["tp"] / (row["tp"] + row["fp"]) if row["tp"] + row["fp"] else None
    aps = {}
    for class_id in sorted(set(by_class) | set(ap_data)):
        entries = ap_data[class_id]
        positives = by_class[class_id]["gt"]
        if positives == 0:
            aps[class_id] = None
            continue
        if not entries:
            aps[class_id] = 0.0
            continue
        tp_cum = fp_cum = 0
        points = []
        for _, is_tp in sorted(entries, key=lambda item: -item[0]):
            tp_cum += int(is_tp); fp_cum += int(not is_tp)
            points.append((tp_cum/positives, tp_cum/(tp_cum+fp_cum)))
        aps[class_id] = sum((recall-prev_recall)*max((p for r, p in points if r >= recall), default=0.0)
                            for prev_recall, recall in zip([0.0]+[r for r, _ in points], [r for r, _ in points]))
    eligible = [v for v in aps.values() if v is not None]
    return {"frames": len(frames), "tp": tp, "fp": fp, "fn": fn,
            "ground_truth_count": tp+fn, "prediction_count": tp+fp,
            "precision": tp/(tp+fp) if tp+fp else None,
            "recall": tp/(tp+fn) if tp+fn else None,
            "iou_threshold": iou_threshold, "ap_by_class": aps,
            "map_at_iou_threshold": sum(eligible)/len(eligible) if eligible else None,
            "by_class": dict(sorted(by_class.items()))}


def score_tracking(frames: list[MetricFrame], *, iou_threshold: float = 0.5) -> dict:
    """按帧IoU匹配构建全局ID配对，报告IDF1、ID switch及轨迹覆盖。"""
    total_gt = total_pred = idtp = switches = matched_instances = 0
    pair_counts = defaultdict(int)
    last_assignment = {}
    for frame in frames:
        gt = [item for item in frame.ground_truth if item.track_id is not None]
        pred = list(frame.tracks)
        total_gt += len(gt); total_pred += len(pred)
        matches, _, _ = _match(gt, pred, iou_threshold)
        for gi, pi in matches:
            gt_id = gt[gi].track_id
            pred_id = pred[pi].track_id
            pair_counts[(gt_id, pred_id)] += 1
            matched_instances += 1
            previous = last_assignment.get(gt_id)
            if previous is not None and previous != pred_id:
                switches += 1
            last_assignment[gt_id] = pred_id
    if pair_counts:
        gt_ids = sorted({pair[0] for pair in pair_counts})
        pred_ids = sorted({pair[1] for pair in pair_counts})
        matrix = [[-pair_counts[(g, p)] for p in pred_ids] for g in gt_ids]
        rows, cols = linear_sum_assignment(matrix)
        idtp = sum(pair_counts[(gt_ids[r], pred_ids[c])] for r, c in zip(rows, cols))
    idfn, idfp = total_gt-idtp, total_pred-idtp
    denom = 2*idtp + idfp + idfn
    return {"frames": len(frames), "gt_instances": total_gt, "predicted_tracks": total_pred,
            "idtp": idtp, "idfp": idfp, "idfn": idfn,
            "idf1": 2*idtp/denom if denom else None, "id_switches": switches,
            "matched_instances": matched_instances,
            "track_coverage": matched_instances/total_gt if total_gt else None,
            "iou_threshold": iou_threshold}


def score_detection_strata(frames: list[MetricFrame], *, iou_threshold: float = 0.5) -> dict:
    """输出按摄像头侧的完整检测指标及按目标尺度/官方遮挡状态的召回分母。"""
    by_side = {}
    sides = sorted({dict(frame.metadata).get("camera_side", "unknown") for frame in frames})
    for side in sides:
        subset = [frame for frame in frames if dict(frame.metadata).get("camera_side", "unknown") == side]
        by_side[side] = score_detections(subset, iou_threshold=iou_threshold)
    target_rows = defaultdict(lambda: {"gt": 0, "tp": 0, "fn": 0})
    for frame in frames:
        matches, _, _ = _match(frame.ground_truth, frame.detections, iou_threshold)
        matched = {gt_index for gt_index, _ in matches}
        for index, gt in enumerate(frame.ground_truth):
            row = target_rows[(gt.class_id, gt.pixel_scale, gt.official_occlusion_state)]
            row["gt"] += 1
            row["tp" if index in matched else "fn"] += 1
    serialized = {}
    for (class_id, scale, occlusion), values in sorted(target_rows.items()):
        values["recall"] = values["tp"]/values["gt"] if values["gt"] else None
        serialized[f"{class_id}|{scale}|{occlusion}"] = values
    return {"iou_threshold": iou_threshold, "by_camera_side": by_side,
            "target_recall_by_class_scale_official_occlusion": serialized,
            "false_positive_denominator": "camera-side frames; see by_camera_side.frames"}
