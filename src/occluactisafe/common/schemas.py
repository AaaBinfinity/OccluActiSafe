"""N02 数据契约：内部数组与严格的 JSON 传输对象分离。

坐标采用右手平面，位置 m，速度 m/s，时间 s，角度 rad；
时间值必须属于对象声明的 clock_domain。传输对象不包含接收端时间。
"""

from dataclasses import asdict, dataclass, fields, is_dataclass
import json
import math
from typing import Optional, Tuple, Union, get_args, get_origin, get_type_hints

from occluactisafe.common.errors import ContractError
from occluactisafe.common.clock import (
    Timestamp, validate_observation_times, validate_prediction_times,
    validate_receive_times, require_available,
)

SCHEMA_VERSION = "1.0"


def _convert(value, annotation, path):
    """将严格 JSON 值转为指定类型，同时拒绝未知字段和非有限数。"""
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Union:
        for option in args:
            try:
                return _convert(value, option, path)
            except (TypeError, ValueError):
                continue
        raise ContractError(f"{path} 类型或值不符合契约")
    if annotation is type(None):
        if value is not None:
            raise ContractError(f"{path} 应为 null")
        return None
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ContractError(f"{path} 应为数组")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(v, args[0], f"{path}[{i}]") for i, v in enumerate(value))
        if len(value) != len(args):
            raise ContractError(f"{path} 数组长度错误")
        return tuple(_convert(v, a, f"{path}[{i}]") for i, (v, a) in enumerate(zip(value, args)))
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ContractError(f"{path} 必须为有限数值")
        return float(value)
    if annotation is int:
        if type(value) is not int:
            raise ContractError(f"{path} 必须为整数")
        return value
    if annotation is bool:
        if type(value) is not bool:
            raise ContractError(f"{path} 必须为布尔值")
        return value
    if annotation is str:
        if not isinstance(value, str) or not value.strip():
            raise ContractError(f"{path} 必须为非空字符串")
        return value
    if isinstance(annotation, type) and is_dataclass(annotation):
        if isinstance(value, annotation):
            return value
        if not isinstance(value, dict):
            raise ContractError(f"{path} 必须为对象")
        return annotation.from_dict(value)
    raise ContractError(f"{path} 使用了不支持的类型 {annotation}")


def _duplicates(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ContractError(f"重复 JSON 字段: {key}")
        obj[key] = value
    return obj


def _bad_constant(value):
    raise ContractError(f"JSON 不允许非有限值: {value}")


class Record:
    """通过字段声明显式白名单；构造及反序列化均严格校验。"""

    def __post_init__(self):
        hints = get_type_hints(type(self))
        for field in fields(self):
            value = getattr(self, field.name)
            converted = _convert(value, hints[field.name], field.name)
            # 保证数组深层不可变，避免创建后绕过校验。
            object.__setattr__(self, field.name, converted)
        self._check()

    def _check(self):
        pass

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ContractError(f"{cls.__name__} 必须为对象")
        expected = {item.name for item in fields(cls)}
        extra, missing = set(value) - expected, expected - set(value)
        if extra or missing:
            raise ContractError(f"{cls.__name__} 未知字段 {sorted(extra)}；缺少字段 {sorted(missing)}")
        hints = get_type_hints(cls)
        return cls(**{name: _convert(val, hints[name], name) for name, val in value.items()})

    def to_dict(self):
        # JSON 对 tuple 自动编码为数组。二次校验防止绕过构造器修改状态。
        self.__post_init__()
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw):
        try:
            value = json.loads(raw, object_pairs_hook=_duplicates, parse_constant=_bad_constant)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"非法 JSON: {exc}") from exc
        return cls.from_dict(value)


def _time(domain, value):
    return Timestamp(value, domain)


def _version(value):
    if value != SCHEMA_VERSION:
        raise ContractError(f"不支持 schema_version: {value}")


def _frame(sequence, frame_id):
    if sequence < 0 or frame_id < 0:
        raise ContractError("sequence 和 frame_id 必须为非负整数")


def _positive(value, name):
    if value <= 0:
        raise ContractError(f"{name} 必须大于零")


@dataclass(frozen=True)
class EgoState(Record):
    ego_id: str
    clock_domain: str
    coordinate_frame_id: str
    reference_point: str
    observation_time: float
    source_available_time: float
    state_time: float
    x: float
    y: float
    vx: Optional[float]
    vy: Optional[float]
    yaw: Optional[float]
    length: float
    width: float
    velocity_valid: bool
    yaw_valid: bool

    def _check(self):
        validate_observation_times(_time(self.clock_domain, self.observation_time),
                                   _time(self.clock_domain, self.observation_time),
                                   _time(self.clock_domain, self.state_time),
                                   _time(self.clock_domain, self.source_available_time))
        _positive(self.length, "length"); _positive(self.width, "width")
        if self.velocity_valid != (self.vx is not None and self.vy is not None):
            raise ContractError("velocity_valid 与速度可空字段不一致")
        if self.yaw_valid != (self.yaw is not None):
            raise ContractError("yaw_valid 与航向可空字段不一致")
        if (self.vx is None) != (self.vy is None):
            raise ContractError("vx 和 vy 必须同时有效或同时缺失")
        if self.reference_point not in ("geometric_center", "rear_axle"):
            raise ContractError("未知状态参考点")


@dataclass(frozen=True)
class SourceTrack(Record):
    source_track_id: str
    source_id: str
    session_id: str
    sensor_id: str
    category: str
    role: str
    clock_domain: str
    coordinate_frame_id: str
    reference_point: str
    observation_time: float
    last_seen_time: float
    source_available_time: float
    state_time: float
    x: Optional[float]
    y: Optional[float]
    vx: Optional[float]
    vy: Optional[float]
    yaw: Optional[float]
    length: Optional[float]
    width: Optional[float]
    size_source: str
    size_uncertainty: Optional[float]
    covariance_xy: Optional[Tuple[Tuple[float, float], Tuple[float, float]]]
    observed_mask: bool
    state_valid_mask: bool
    size_valid: bool
    velocity_valid: bool
    yaw_valid: bool
    quality: Optional[float]

    def _check(self):
        validate_observation_times(_time(self.clock_domain, self.observation_time),
                                   _time(self.clock_domain, self.last_seen_time),
                                   _time(self.clock_domain, self.state_time),
                                   _time(self.clock_domain, self.source_available_time))
        if self.category not in ("vehicle", "pedestrian", "cyclist", "unknown"):
            raise ContractError("未知目标类别")
        if self.role not in ("ego", "target"):
            raise ContractError("role 必须区分 ego 与 target")
        if self.reference_point not in ("geometric_center", "rear_axle"):
            raise ContractError("未知状态参考点")
        if self.state_valid_mask != (self.x is not None and self.y is not None):
            raise ContractError("state_valid_mask 与位置可空字段不一致")
        if (self.x is None) != (self.y is None):
            raise ContractError("x 和 y 必须同时有效或同时缺失")
        if self.size_valid != (self.length is not None and self.width is not None):
            raise ContractError("size_valid 与尺寸可空字段不一致")
        if (self.length is None) != (self.width is None):
            raise ContractError("length 和 width 必须同时有效或同时缺失")
        if self.size_valid:
            _positive(self.length, "length"); _positive(self.width, "width")
        if self.velocity_valid != (self.vx is not None and self.vy is not None):
            raise ContractError("velocity_valid 与速度可空字段不一致")
        if (self.vx is None) != (self.vy is None):
            raise ContractError("vx 和 vy 必须同时有效或同时缺失")
        if self.yaw_valid != (self.yaw is not None):
            raise ContractError("yaw_valid 与航向可空字段不一致")
        if self.observed_mask and not self.state_valid_mask:
            raise ContractError("有效观测不能标为无效状态")
        if self.quality is not None and not 0 <= self.quality <= 1:
            raise ContractError("quality 须在 [0, 1]")
        if self.size_uncertainty is not None and self.size_uncertainty < 0:
            raise ContractError("尺寸不确定度不能为负")
        if self.covariance_xy is not None:
            (xx, xy), (yx, yy) = self.covariance_xy
            if (xx < 0 or yy < 0 or not math.isclose(xy, yx, abs_tol=1e-9, rel_tol=0)
                    or xx * yy - xy * yx < -1e-9):
                raise ContractError("位置协方差必须对称且半正定")


@dataclass(frozen=True)
class ObservationPacket(Record):
    map_id: str
    map_version: str
    coordinate_frame_id: str
    frame_id: int
    clock_domain: str
    observation_time: float
    source_available_time: float
    tracks: Tuple[SourceTrack, ...]

    def _check(self):
        if self.frame_id < 0 or self.source_available_time < self.observation_time:
            raise ContractError("帧号或观测可用时刻非法")
        for track in self.tracks:
            if track.clock_domain != self.clock_domain or track.coordinate_frame_id != self.coordinate_frame_id:
                raise ContractError("目标时钟域或坐标系与观测包不一致")
            if (track.observation_time > self.observation_time or
                    track.source_available_time > self.source_available_time or
                    track.state_time > self.source_available_time):
                raise ContractError("观测包包含未来或尚不可用的目标")


@dataclass(frozen=True)
class PredictionTarget(Record):
    track: SourceTrack
    trajectories: Tuple[Tuple[Tuple[float, float], ...], ...]
    weights: Tuple[float, ...]
    valid_mask: Tuple[Tuple[bool, ...], ...]

    def _check(self):
        count = len(self.trajectories)
        if not count or len(self.weights) != count or len(self.valid_mask) != count:
            raise ContractError("模式、权重和掩码数量不一致")
        lengths = {len(mode) for mode in self.trajectories}
        if len(lengths) != 1 or next(iter(lengths)) == 0:
            raise ContractError("各模式必须具有相同的非空时域")
        t = next(iter(lengths))
        if any(len(mask) != t for mask in self.valid_mask):
            raise ContractError("掩码时域不匹配")
        if any(w < 0 for w in self.weights):
            raise ContractError("模式权重必须非负")
        if any(w != 0 and not any(mask) for w, mask in zip(self.weights, self.valid_mask)):
            raise ContractError("无效模式的权重必须为零")
        if any(any(mask) for mask in self.valid_mask):
            if not math.isclose(sum(self.weights), 1.0, rel_tol=1e-6, abs_tol=1e-6):
                raise ContractError("有效模式的权重之和必须为1")
        elif sum(self.weights) != 0:
            raise ContractError("无效目标模式的权重之和必须为0")


@dataclass(frozen=True)
class ObservationMessage(Record):
    schema_version: str
    message_type: str
    source_id: str
    session_id: str
    sequence: int
    recipient_id: str
    map_id: str
    map_version: str
    coordinate_frame_id: str
    frame_id: int
    clock_domain: str
    observation_time: float
    source_available_time: float
    generated_time: float
    transmitted_time: float
    valid_until: float
    tracks: Tuple[SourceTrack, ...]

    def _check(self):
        _version(self.schema_version); _frame(self.sequence, self.frame_id)
        if self.message_type != "observation":
            raise ContractError("message_type 必须为 observation")
        if not self.observation_time <= self.source_available_time <= self.generated_time <= self.transmitted_time <= self.valid_until:
            raise ContractError("观测消息的采集、处理、发送和有效期顺序非法")
        for track in self.tracks:
            if track.clock_domain != self.clock_domain or track.coordinate_frame_id != self.coordinate_frame_id:
                raise ContractError("目标时钟域或坐标系与消息不一致")
            if (track.observation_time > self.observation_time or
                    track.source_available_time > self.generated_time or
                    track.state_time > self.generated_time):
                raise ContractError("消息含未来或尚不可用的目标")


@dataclass(frozen=True)
class PredictionMessage(Record):
    schema_version: str
    message_type: str
    source_id: str
    session_id: str
    sequence: int
    recipient_id: str
    map_id: str
    map_version: str
    coordinate_frame_id: str
    frame_id: int
    clock_domain: str
    model_hash: str
    anchor_time: float
    input_cutoff: float
    completed_time: float
    generated_time: float
    transmitted_time: float
    valid_until: float
    future_times: Tuple[float, ...]
    future_end: float
    targets: Tuple[PredictionTarget, ...]

    def _check(self):
        _version(self.schema_version); _frame(self.sequence, self.frame_id)
        if self.message_type != "prediction":
            raise ContractError("message_type 必须为 prediction")
        validate_prediction_times(*(_time(self.clock_domain, t) for t in (
            self.input_cutoff, self.anchor_time, self.completed_time,
            self.transmitted_time, self.future_end, self.valid_until)))
        if not self.completed_time <= self.generated_time <= self.transmitted_time:
            raise ContractError("预测完成、生成、发送时刻顺序非法")
        if self.valid_until < self.transmitted_time:
            raise ContractError("发送时消息已超过有效期")
        if not self.future_times or self.future_times[0] <= self.anchor_time:
            raise ContractError("未来采样必须晚于预测锚点")
        if any(a >= b for a, b in zip(self.future_times, self.future_times[1:])):
            raise ContractError("future_times 必须严格递增")
        if not math.isclose(self.future_end, self.future_times[-1], abs_tol=1e-9, rel_tol=0):
            raise ContractError("future_end 必须与末个未来时刻一致")
        for target in self.targets:
            track = target.track
            if track.clock_domain != self.clock_domain or track.coordinate_frame_id != self.coordinate_frame_id:
                raise ContractError("预测目标时钟域或坐标系与消息不一致")
            if track.observation_time > self.input_cutoff or track.source_available_time > self.input_cutoff:
                raise ContractError("预测使用了截止之后的采集或不可用观测")
            if track.state_time > self.anchor_time:
                raise ContractError("目标锚点状态不能来自未来")
            if any(len(mode) != len(self.future_times) for mode in target.trajectories):
                raise ContractError("目标轨迹时域与 future_times 不一致")


@dataclass(frozen=True)
class ReceivedMessage(Record):
    payload: Union[ObservationMessage, PredictionMessage]
    arrival_time: float
    usable_time: float
    clock_domain: str

    def _check(self):
        if self.clock_domain != self.payload.clock_domain:
            raise ContractError("接收时钟域与消息时钟域不一致")
        validate_receive_times(_time(self.clock_domain, self.payload.transmitted_time),
                               _time(self.clock_domain, self.arrival_time),
                               _time(self.clock_domain, self.usable_time))

    def require_available(self, decision_time: Timestamp):
        """校验决策时刻可消费；接收成功不等于尚在有效期或已可用。"""
        payload = self.payload
        require_available(decision_time,
                          usable_time=_time(self.clock_domain, self.usable_time),
                          valid_until=_time(self.clock_domain, payload.valid_until))
        if isinstance(payload, ObservationMessage):
            require_available(decision_time,
                              observation_time=_time(self.clock_domain, payload.observation_time),
                              source_available_time=_time(self.clock_domain, payload.source_available_time))
            tracks = payload.tracks
        else:
            require_available(decision_time, state_time=_time(self.clock_domain, payload.anchor_time))
            tracks = tuple(target.track for target in payload.targets)
        for track in tracks:
            require_available(decision_time,
                              observation_time=_time(self.clock_domain, track.observation_time),
                              source_available_time=_time(self.clock_domain, track.source_available_time),
                              state_time=_time(self.clock_domain, track.state_time))


@dataclass(frozen=True)
class DecisionBundle(Record):
    request_id: str
    mode: str
    reason: str
    clock_domain: str
    plan_start_time: float
    deadline: float
    valid_until: float
    reference_trajectory: Tuple[Tuple[float, float], ...]
    risk_kind: str
    risk_value: Optional[float]
    risk_horizon: float
    reference_policy: str
    constraint_context: Tuple[str, ...]

    def _check(self):
        if self.mode not in ("Normal", "Caution", "AEB", "AES", "Degraded", "Recovery"):
            raise ContractError("未知决策模式")
        if self.risk_kind not in ("uncalibrated_score", "calibrated_probability", "unknown_region_constraint"):
            raise ContractError("未知风险口径")
        if self.risk_kind == "calibrated_probability" and (self.risk_value is None or not 0 <= self.risk_value <= 1):
            raise ContractError("校准事件概率必须位于 [0, 1]")
        if self.risk_kind == "uncalibrated_score" and self.risk_value is None:
            raise ContractError("未校准评分必须提供实际数值")
        if self.risk_kind == "unknown_region_constraint" and self.risk_value is not None:
            raise ContractError("未知区约束不伪造风险数值")
        # 求解截止时间可能早于计划执行起点；二者不能被强制排序。
        if self.valid_until < self.plan_start_time:
            raise ContractError("有效期早于计划开始时刻")
        _positive(self.risk_horizon, "risk_horizon")


@dataclass(frozen=True)
class ControlResult(Record):
    request_id: str
    plan_id: str
    solver_status: str
    safety_rechecked: bool
    clock_domain: str
    plan_start_time: float
    valid_until: float
    throttle: float
    brake: float
    steer: float
    fallback_reason: Optional[str]

    def _check(self):
        if self.solver_status not in ("success", "fallback", "timeout", "infeasible", "error"):
            raise ContractError("未知求解状态")
        if self.valid_until < self.plan_start_time:
            raise ContractError("控制有效期早于计划起点")
        if not (0 <= self.throttle <= 1 and 0 <= self.brake <= 1 and -1 <= self.steer <= 1):
            raise ContractError("控制请求超出归一化范围")
        if self.throttle > 0 and self.brake > 0:
            raise ContractError("同一控制周期不能同时请求油门和制动")
        if self.solver_status == "success" and (not self.safety_rechecked or self.fallback_reason is not None):
            raise ContractError("成功控制必须经安全复核且不能带回退原因")
        if self.solver_status != "success" and self.fallback_reason is None:
            raise ContractError("失败或回退必须声明原因")


@dataclass(frozen=True)
class NoResult(Record):
    task_id: str
    clock_domain: str
    time: float
    reason: str


@dataclass(frozen=True)
class TickEvent(Record):
    schema_version: str
    run_id: str
    episode_id: str
    generation: int
    frame_id: int
    clock_domain: str
    sim_time: float
    observations: Tuple[ObservationPacket, ...]

    def _check(self):
        _version(self.schema_version); _frame(self.generation, self.frame_id)
        for packet in self.observations:
            if packet.clock_domain != self.clock_domain or packet.frame_id != self.frame_id:
                raise ContractError("TickEvent 观测帧或时钟域不一致")
            if packet.observation_time > self.sim_time or packet.source_available_time > self.sim_time:
                raise ContractError("TickEvent 不能包含未来采集或尚未处理完成的观测")


@dataclass(frozen=True)
class AdvanceRequest(Record):
    schema_version: str
    run_id: str
    episode_id: str
    generation: int
    target_frame_id: int
    clock_domain: str
    request_time: float
    valid_until: float
    control: ControlResult

    def _check(self):
        _version(self.schema_version); _frame(self.generation, self.target_frame_id)
        if self.control.clock_domain != self.clock_domain or self.valid_until < self.request_time:
            raise ContractError("AdvanceRequest 时钟域或有效期非法")
        if not self.control.safety_rechecked:
            raise ContractError("桥接请求必须携带经安全复核的控制，包括回退")
        if self.control.valid_until < self.request_time or self.valid_until > self.control.valid_until:
            raise ContractError("请求携带的控制已过期或请求期限超出控制期限")


class PredictionBundle:
    """内部 NumPy 预测结果，[B,N,K,T,2]；不会自动转换成网络消息。"""
    def __init__(self, trajectories, weights, valid_mask, future_times, anchor_time,
                 input_cutoff, model_hash, coordinate_frame_id, clock_domain):
        import numpy as np
        self.trajectories = np.array(trajectories, copy=True)
        self.weights = np.array(weights, copy=True)
        self.valid_mask = np.array(valid_mask, copy=True)
        self.future_times = np.array(future_times, copy=True)
        self.anchor_time = _convert(anchor_time, float, "anchor_time")
        self.input_cutoff = _convert(input_cutoff, float, "input_cutoff")
        self.model_hash = _convert(model_hash, str, "model_hash")
        self.coordinate_frame_id = _convert(coordinate_frame_id, str, "coordinate_frame_id")
        self.clock_domain = _convert(clock_domain, str, "clock_domain")
        p = self.trajectories
        if p.ndim != 5 or p.shape[-1] != 2 or not all(p.shape[:-1]):
            raise ContractError("轨迹形状须为非空 [B,N,K,T,2]")
        if self.weights.shape != p.shape[:3] or self.valid_mask.shape != p.shape[:4]:
            raise ContractError("预测权重或有效掩码形状不匹配")
        if self.valid_mask.dtype.kind != "b":
            raise ContractError("预测掩码必须为布尔数组")
        if self.future_times.shape != (p.shape[3],):
            raise ContractError("future_times 长度须等于 T")
        if any(a.dtype.kind not in "iuf" for a in (p, self.weights, self.future_times)):
            raise ContractError("轨迹、权重和时间必须为数值数组")
        if not all(np.isfinite(a).all() for a in (p, self.weights, self.future_times)):
            raise ContractError("内部预测不能含 NaN 或无穷大")
        if self.input_cutoff > self.anchor_time or self.future_times[0] <= self.anchor_time or not np.all(np.diff(self.future_times) > 0):
            raise ContractError("预测输入截止、锚点或未来时间顺序非法")
        if (self.weights < 0).any():
            raise ContractError("模式权重必须非负")
        active = self.valid_mask.any(axis=-1)
        if (self.weights[~active] != 0).any():
            raise ContractError("无效模式权重必须为0")
        sums = self.weights.sum(axis=-1)
        valid_targets = active.any(axis=-1)
        if not np.allclose(sums[valid_targets], 1.0, rtol=1e-6, atol=1e-6) or not np.all(sums[~valid_targets] == 0):
            raise ContractError("有效目标权重须归一，无效目标权重须为0")
        for array in (self.trajectories, self.weights, self.valid_mask, self.future_times):
            array.setflags(write=False)
