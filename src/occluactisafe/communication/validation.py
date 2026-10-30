"""接收可用性与覆盖门控；不实现融合、危险记忆或控制动作。"""

from bisect import bisect_left
from dataclasses import dataclass
import time
from types import MappingProxyType

from occluactisafe.common.clock import Timestamp
from occluactisafe.common.schemas import ObservationMessage, PredictionMessage, ReceivedMessage
from .protocol import finite


@dataclass(frozen=True)
class ReceiverPolicy:
    recipient_id: str
    map_id: str
    map_version: str
    coordinate_frame_id: str
    clock_domain: str
    source_sessions: dict[str, str]
    max_observation_age_s: float = 2.0
    max_state_age_s: float = 2.0
    max_anchor_age_s: float = 1.0
    planning_horizon_s: float = 3.0

    def __post_init__(self):
        for name in ("recipient_id", "map_id", "map_version", "coordinate_frame_id", "clock_domain"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name}必须为非空字符串")
        if not isinstance(self.source_sessions, dict) or not self.source_sessions:
            raise ValueError("必须显式登记可信source/session")
        if any(not isinstance(v, str) or not v.strip() for pair in self.source_sessions.items() for v in pair):
            raise ValueError("source/session必须为非空字符串")
        object.__setattr__(self, "source_sessions", MappingProxyType(dict(self.source_sessions)))
        for name in ("max_observation_age_s", "max_state_age_s", "max_anchor_age_s", "planning_horizon_s"):
            finite(getattr(self, name), name)


class ReceiverState:
    """显式会话表及消息高水位；无目标状态、协方差或风险缓存。"""
    def __init__(self, policy: ReceiverPolicy):
        self.policy = policy
        self.source_sessions = dict(policy.source_sessions)
        self.sequences = {}
        self.generated = {}
        self.references = {}
        self.last_decision_time = None

    def activate_session(self, source_id, session_id):
        if source_id not in self.source_sessions or not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("只能由接收者为已登记来源显式激活合法会话")
        if self.source_sessions[source_id] == session_id:
            return
        # 新包不能调用此接口；调用者负责会话认证。旧会话不再接受。
        self.source_sessions[source_id] = session_id
        self.sequences = {k: v for k, v in self.sequences.items() if k[0] != source_id}
        self.generated = {k: v for k, v in self.generated.items() if k[0] != source_id}
        self.references = {k: v for k, v in self.references.items() if k[0] != source_id}


@dataclass(frozen=True)
class ReceiveResult:
    accepted: bool
    reason: str
    received: ReceivedMessage | None
    coverage: dict | None
    validation_wall_s: float


def _track_key(track):
    return (track.source_id, track.session_id, track.sensor_id, track.source_track_id)


def _points(message, target, mode):
    times = message.future_times
    values = target.trajectories[mode]
    valid = target.valid_mask[mode]
    track = target.track
    if track.state_valid_mask and track.state_time == message.anchor_time:
        times = (message.anchor_time, *times)
        values = ((track.x, track.y), *values)
        valid = (True, *valid)
    return times, values, valid


def _sample(times, values, valid, when):
    i = bisect_left(times, when)
    if i < len(times) and times[i] == when:
        if valid[i]:
            return values[i]
        raise ValueError("查询采样被掩码排除")
    if i == 0 or i == len(times) or not valid[i - 1] or not valid[i]:
        raise ValueError("不能跨缺失区间插值或超时域外推")
    ratio = (when - times[i - 1]) / (times[i] - times[i - 1])
    return tuple(a + (b - a) * ratio for a, b in zip(values[i - 1], values[i]))


def prediction_coverage(message: PredictionMessage, start: float, end: float) -> dict:
    """所有目标的正权重模式必须连续覆盖同一绝对时间窗口。"""
    finite(start, "coverage_start"); finite(end, "coverage_end")
    if end < start:
        raise ValueError("覆盖终点不得早于起点")
    report = {"required_start": start, "required_end": end, "complete": bool(message.targets), "targets": []}
    for target in message.targets:
        modes = []
        for mode, weight in enumerate(target.weights):
            if weight <= 0:
                continue
            times, points, valid = _points(message, target, mode)
            try:
                _sample(times, points, valid, start)
                _sample(times, points, valid, end)
                if any(not ok for t, ok in zip(times, valid) if start < t < end):
                    raise ValueError("请求窗口中存在掩码洞")
                reason = "complete"
            except ValueError as exc:
                reason = str(exc)
                report["complete"] = False
            modes.append({"mode": mode, "weight": weight, "complete": reason == "complete", "reason": reason})
        if not modes:
            report["complete"] = False
        report["targets"].append({"identity": list(_track_key(target.track)), "modes": modes})
    return report


def validate_received(received, now: Timestamp, policy: ReceiverPolicy, *, state: ReceiverState | None = None,
                      plan_start: Timestamp | None = None, horizon_s: float | None = None) -> ReceiveResult:
    """校验完成才提交高水位；拒绝结果不向核心暴露payload。"""
    started = time.perf_counter()
    coverage = None

    def result(reason):
        return ReceiveResult(reason == "accepted", reason, received if reason == "accepted" else None,
                             coverage, time.perf_counter() - started)

    if not isinstance(policy, ReceiverPolicy):
        raise ValueError("policy须为ReceiverPolicy")
    if state is not None and state.policy is not policy:
        raise ValueError("ReceiverState必须绑定同一个policy对象")
    try:
        if not isinstance(received, ReceivedMessage):
            return result("invalid_message")
        # 重验深层不可变契约；不能因之前构造成功就信任外部输入。
        ReceivedMessage.from_json(received.to_json())
    except (ValueError, TypeError):
        return result("invalid_message")
    p = received.payload
    if not isinstance(now, Timestamp) or now.clock_domain != policy.clock_domain or p.clock_domain != policy.clock_domain:
        return result("clock_domain_mismatch")
    if now.seconds < 0:
        return result("invalid_decision_time")
    if state is not None and state.last_decision_time is not None and now.seconds < state.last_decision_time:
        return result("decision_time_regression")
    for name in ("recipient_id", "map_id", "map_version", "coordinate_frame_id"):
        if getattr(p, name) != getattr(policy, name):
            return result(name + "_mismatch")
    sessions = state.source_sessions if state is not None else policy.source_sessions
    if p.source_id not in sessions:
        return result("unknown_source")
    if p.session_id != sessions[p.source_id]:
        return result("session_mismatch")
    if now.seconds < received.arrival_time:
        return result("not_arrived")
    if now.seconds < received.usable_time:
        return result("not_usable")
    if now.seconds > p.valid_until:
        return result("expired")
    try:
        received.require_available(now)
    except ValueError:
        return result("future_information")
    tracks = p.tracks if isinstance(p, ObservationMessage) else tuple(t.track for t in p.targets)
    if len({_track_key(t) for t in tracks}) != len(tracks):
        return result("duplicate_target_identity")
    for t in tracks:
        if now.seconds - t.last_seen_time > policy.max_observation_age_s:
            return result("observation_too_old")
        if now.seconds - t.state_time > policy.max_state_age_s:
            return result("state_too_old")
    if isinstance(p, ObservationMessage):
        if now.seconds - p.observation_time > policy.max_observation_age_s:
            return result("observation_too_old")
        reference = (p.observation_time, p.source_available_time)
    else:
        if now.seconds - p.anchor_time > policy.max_anchor_age_s:
            return result("anchor_too_old")
        reference = (p.anchor_time, p.input_cutoff)
    key = (p.source_id, p.session_id)
    reference_key = (*key, p.message_type)
    if state is not None:
        if p.sequence <= state.sequences.get(key, -1):
            return result("duplicate_or_out_of_order")
        if p.generated_time < state.generated.get(key, float("-inf")):
            return result("generation_regression")
        previous = state.references.get(reference_key)
        if previous is not None and any(v < old for v, old in zip(reference, previous)):
            return result("reference_time_regression")
    if isinstance(p, PredictionMessage):
        start = now if plan_start is None else plan_start
        horizon = policy.planning_horizon_s if horizon_s is None else finite(horizon_s, "horizon_s")
        if not isinstance(start, Timestamp) or start.clock_domain != policy.clock_domain:
            return result("plan_clock_mismatch")
        if start.seconds < now.seconds:
            return result("plan_in_past")
        coverage = prediction_coverage(p, start.seconds, start.seconds + horizon)
        if not coverage["complete"]:
            return result("coverage_insufficient")
    if state is not None:
        state.sequences[key] = p.sequence
        state.generated[key] = p.generated_time
        state.references[reference_key] = reference
        state.last_decision_time = now.seconds
    return result("accepted")


def query_prediction(received: ReceivedMessage, *, now: Timestamp, when: Timestamp,
                     policy: ReceiverPolicy, target_index=0, mode=0):
    """单时刻合法预测查询；不会替调用方批准完整规划或生成控制。"""
    if not isinstance(received.payload, PredictionMessage):
        raise ValueError("只能查询预测消息")
    checked = validate_received(received, now, policy, plan_start=when, horizon_s=0)
    if not checked.accepted:
        raise ValueError(checked.reason)
    if type(target_index) is not int or type(mode) is not int or target_index < 0 or mode < 0:
        raise ValueError("目标与模式索引须为非负整数")
    p = received.payload
    target = p.targets[target_index]
    if target.weights[mode] <= 0:
        raise ValueError("不能查询零权重模式")
    return _sample(*_points(p, target, mode), when.seconds)
