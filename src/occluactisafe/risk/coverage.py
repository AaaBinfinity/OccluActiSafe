"""绝对时间共同窗口、已知目标缺口与N02/N08预测入口。"""

from dataclasses import dataclass
import json
import math
from occluactisafe.common.coordinates import Pose2D
from occluactisafe.common.clock import Timestamp
from occluactisafe.common.schemas import PredictionMessage
from occluactisafe.communication.validation import validate_received, _points
from occluactisafe.perception.participants import ParticipantGeometry, geometry_from_track
from .obb import real
from .occupancy import PoseTrajectory


def _id(value):
    if not isinstance(value, str) or not value.strip(): raise ValueError("标识不能为空")


@dataclass(frozen=True)
class ModeForecast:
    mode_id: str
    weight: float
    path: PoseTrajectory

    def __post_init__(self):
        _id(self.mode_id); real(self.weight, "weight", nonnegative=True)
        if not isinstance(self.path, PoseTrajectory): raise ValueError("模式需要显式轨迹")


@dataclass(frozen=True)
class TargetForecast:
    target_id: str
    geometry: ParticipantGeometry
    modes: tuple
    last_seen: float
    state_time: float
    anchor_time: float
    available_time: float
    valid_until: float
    max_observation_age_s: float = 2.
    max_state_age_s: float = 2.
    max_anchor_age_s: float = 1.
    heading_assumption: str = "declared_pose_trajectory"

    def __post_init__(self):
        _id(self.target_id); _id(self.heading_assumption)
        if not isinstance(self.geometry, ParticipantGeometry): raise ValueError("缺少合法类别几何")
        modes = tuple(self.modes)
        if not modes or any(not isinstance(m, ModeForecast) for m in modes): raise ValueError("目标不得为空模式")
        if len({m.mode_id for m in modes}) != len(modes): raise ValueError("模式ID重复")
        if abs(math.fsum(m.weight for m in modes) - 1.) > 1e-8: raise ValueError("模式权重和须为1")
        domains = {(m.path.clock_domain, m.path.coordinate_frame_id) for m in modes}
        if len(domains) != 1: raise ValueError("模式时钟域/坐标不一致")
        for name in ("last_seen", "state_time", "anchor_time", "available_time", "valid_until"):
            real(getattr(self, name), name)
        if not self.last_seen <= self.state_time <= self.anchor_time <= self.available_time <= self.valid_until:
            raise ValueError("目标信息时间倒序")
        for name in ("max_observation_age_s", "max_state_age_s", "max_anchor_age_s"):
            real(getattr(self, name), name, nonnegative=True)
        object.__setattr__(self, "modes", modes)

    def adoption_gap(self, now):
        if now < self.available_time: return "not_yet_available"
        if now > self.valid_until: return "message_expired"
        for field, limit in (("last_seen", self.max_observation_age_s), ("state_time", self.max_state_age_s), ("anchor_time", self.max_anchor_age_s)):
            if now - getattr(self, field) > limit: return f"stale_{field}"
        return None


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    reference_policy: str
    path: PoseTrajectory
    geometry: ParticipantGeometry
    physically_feasible: bool

    def __post_init__(self):
        _id(self.candidate_id); _id(self.reference_policy)
        if not isinstance(self.path, PoseTrajectory) or not isinstance(self.geometry, ParticipantGeometry): raise ValueError("候选缺轨迹/几何")
        if type(self.physically_feasible) is not bool: raise ValueError("必须显式声明上游物理可行性，本节点不证明动力学")


def common_coverage(targets, *, expected_target_ids, now, plan_start, horizon_s=3.):
    real(now, "now"); real(plan_start, "plan_start"); real(horizon_s, "horizon_s", positive=True)
    if plan_start < now: raise ValueError("规划起点不能早于采用时刻")
    if expected_target_ids is None: raise ValueError("必须显式提供已知目标注册表，空表不等于环境无人")
    expected = tuple(expected_target_ids)
    for target_id in expected: _id(target_id)
    if len(set(expected)) != len(expected): raise ValueError("已知目标ID重复")
    targets = tuple(targets)
    if len({t.target_id for t in targets}) != len(targets): raise ValueError("预测目标ID重复；先完成关联而非重复计数")
    if any(t.target_id not in expected for t in targets): raise ValueError("预测目标未登记，需先显式登记合法可用目标")
    end = plan_start + horizon_s; rows = []; prefix_complete = True
    gaps = [{"target_id": identity, "reason": "known_target_prediction_missing"} for identity in expected if identity not in {t.target_id for t in targets}]
    for target in targets:
        gap = target.adoption_gap(now)
        if gap: gaps.append({"target_id": target.target_id, "reason": gap}); continue
        for mode in target.modes:
            if mode.weight == 0: continue
            stop = mode.path.contiguous_end(plan_start)
            prefix_stop = mode.path.contiguous_end(now)
            prefix_ok = prefix_stop is not None and prefix_stop >= plan_start
            prefix_complete &= prefix_ok
            rows.append({"target_id": target.target_id, "mode_id": mode.mode_id, "weight": mode.weight,
                         "contiguous_end": stop, "prefix_complete": prefix_ok})
            if stop is None: gaps.append({"target_id": target.target_id, "mode_id": mode.mode_id, "reason": "start_uncovered"})
            else: end = min(end, stop)
    if gaps: end = plan_start; prefix_complete = False
    return {"now": now, "plan_start": plan_start, "requested_end": plan_start + horizon_s, "common_end": end,
            "requested_horizon_s": horizon_s, "effective_horizon_s": max(0., end - plan_start),
            "complete": not gaps and end >= plan_start + horizon_s, "prefix_complete": prefix_complete,
            "expected_target_ids": list(expected), "targets": rows, "gaps": gaps}


def source_target_id(track):
    return json.dumps([track.source_id, track.session_id, track.sensor_id, track.source_track_id], separators=(",", ":"))


def targets_from_received(record, *, now, policy, state=None, pedestrian_radius=None, center_offsets=None, identity_map=None):
    """N08当前接收门控与N02严格模型；未声称完整预测覆盖或已完成跨源关联。"""
    stamp = Timestamp(real(now, "now"), policy.clock_domain)
    result = validate_received(record, now=stamp, policy=policy, state=state, plan_start=stamp, horizon_s=0.)
    if not result.accepted: raise ValueError(f"预测接收拒绝: {result.reason}")
    message = result.received.payload
    if not isinstance(message, PredictionMessage): raise ValueError("风险适配只接受预测消息")
    targets = []
    for target in message.targets:
        track = target.track; identity = source_target_id(track)
        geometry = geometry_from_track(track, pedestrian_radius=pedestrian_radius, center_offset_m=(center_offsets or {}).get(identity))
        modes = []
        for i, weight in enumerate(target.weights):
            times, points, mask = _points(message, target, i)
            yaw = track.yaw if geometry.category != "pedestrian" else 0.
            poses = tuple(Pose2D(x, y, yaw) for x, y in points)
            modes.append(ModeForecast(str(i), weight, PoseTrajectory(times, poses, message.clock_domain, message.coordinate_frame_id, mask)))
        targets.append(TargetForecast((identity_map or {}).get(identity, identity), geometry, tuple(modes), track.last_seen_time,
                                      track.state_time, message.anchor_time, record.usable_time, message.valid_until,
                                      policy.max_observation_age_s, policy.max_state_age_s, policy.max_anchor_age_s,
                                      "constant_source_heading" if geometry.category != "pedestrian" else "circle_heading_not_applicable"))
    return tuple(targets)
