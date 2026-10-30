"""复用N02白名单协议，固定UTF-8计量与方向检查。"""

from dataclasses import dataclass
import hashlib
import json
import math

from occluactisafe.common.errors import ContractError
from occluactisafe.common.schemas import ObservationMessage, PredictionMessage

Message = ObservationMessage | PredictionMessage


def finite(value, name, *, minimum=0):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < minimum):
        raise ValueError(f"{name} 必须为不小于{minimum}的有限数")
    return float(value)


def integer(value, name, *, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} 必须为不小于{minimum}的整数")
    return value


def decode_message(raw: bytes | str) -> Message:
    """未知字段、重复键及非有限值最终全部由N02严格构造器拒绝。"""
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise ContractError("消息必须为UTF-8") from exc
    if not isinstance(raw, str):
        raise ContractError("消息必须为UTF-8字节或JSON文本")
    try:
        # 仅分派类型；随后原始文本交严格解析器，不能丢弃重复键。
        value = json.loads(raw)
        kind = value.get("message_type") if isinstance(value, dict) else None
    except (TypeError, ValueError) as exc:
        raise ContractError("非法JSON") from exc
    cls = {"observation": ObservationMessage, "prediction": PredictionMessage}.get(kind)
    if cls is None:
        raise ContractError("不支持的消息类型")
    return cls.from_json(raw)


def encode_message(payload: Message) -> bytes:
    if not isinstance(payload, (ObservationMessage, PredictionMessage)):
        raise ContractError("只接受N02观测或预测消息")
    return payload.to_json().encode("utf-8")


def target_count(payload: Message) -> int:
    return len(payload.tracks if isinstance(payload, ObservationMessage) else payload.targets)


def check_direction(payload: Message, direction: str):
    if direction not in ("V2I", "I2V"):
        raise ContractError("方向必须为V2I或I2V")
    if direction == "V2I" and not isinstance(payload, ObservationMessage):
        raise ContractError("预测仅允许沿I2V发送")


def service_seconds(byte_count: int, capacity_bps: float | None) -> float:
    integer(byte_count, "byte_count")
    if capacity_bps is None:
        return 0.0
    capacity = finite(capacity_bps, "capacity_bps", minimum=1e-12)
    result = 8 * byte_count / capacity
    if not math.isfinite(result):
        raise ValueError("服务时间溢出")
    return result


@dataclass(frozen=True)
class ChannelConfig:
    direction: str
    clock_domain: str
    propagation_delay_s: float = 0.2
    processing_delay_s: float = 0.0
    jitter_s: float = 0.0
    loss_rate: float = 0.0
    seed: int = 17
    capacity_bps: float | None = None
    max_queue_wait_s: float = 0.5
    max_pending_messages: int = 128
    max_message_bytes: int = 262144
    blackout_intervals: tuple[tuple[float, float], ...] = ()

    def __post_init__(self):
        if self.direction not in ("V2I", "I2V") or not isinstance(self.clock_domain, str) or not self.clock_domain.strip():
            raise ValueError("方向或时钟域非法")
        for name in ("propagation_delay_s", "processing_delay_s", "jitter_s", "loss_rate", "max_queue_wait_s"):
            finite(getattr(self, name), name)
        if self.loss_rate > 1:
            raise ValueError("loss_rate不得大于1")
        integer(self.seed, "seed", minimum=0)
        integer(self.max_pending_messages, "max_pending_messages")
        integer(self.max_message_bytes, "max_message_bytes")
        if self.capacity_bps is not None:
            finite(self.capacity_bps, "capacity_bps", minimum=1e-12)
        intervals = tuple(tuple(pair) for pair in self.blackout_intervals)
        for pair in intervals:
            if len(pair) != 2:
                raise ValueError("断联区间须有起止值")
            start, end = (finite(v, "blackout") for v in pair)
            if start >= end:
                raise ValueError("断联区间必须递增")
        object.__setattr__(self, "blackout_intervals", intervals)


@dataclass(frozen=True)
class Disturbance:
    propagation_delay_s: float
    lost: bool

    def __post_init__(self):
        finite(self.propagation_delay_s, "propagation_delay_s")
        if type(self.lost) is not bool:
            raise ValueError("lost必须为布尔值")


def message_key(payload: Message) -> tuple[str, str, int]:
    return payload.source_id, payload.session_id, payload.sequence


def disturbance_for(config: ChannelConfig, key: tuple[str, str, int]) -> Disturbance:
    """不依赖调用顺序、Python哈希随机化或全局随机数生成器。"""
    raw = json.dumps([config.seed, config.direction, *key], ensure_ascii=False,
                     separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    loss_draw = int.from_bytes(digest[:8], "big") / 2**64
    jitter_draw = int.from_bytes(digest[8:16], "big") / 2**64
    delay = max(0.0, config.propagation_delay_s + (2 * jitter_draw - 1) * config.jitter_s)
    return Disturbance(delay, loss_draw < config.loss_rate)
