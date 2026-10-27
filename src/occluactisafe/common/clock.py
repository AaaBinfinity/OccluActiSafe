"""带时钟域的秒值校验；不把到达或外推状态当作新采集。"""

from dataclasses import dataclass
import math
from typing import Optional


@dataclass(frozen=True)
class Timestamp:
    """秒及明确时钟域；不同会话/时钟应使用不同标识。"""

    seconds: float
    clock_domain: str

    def __post_init__(self) -> None:
        if isinstance(self.seconds, bool) or not isinstance(self.seconds, (int, float)) or not math.isfinite(self.seconds):
            raise ValueError("时间必须是有限的秒数")
        if not isinstance(self.clock_domain, str) or not self.clock_domain.strip():
            raise ValueError("时钟域不能为空")


def require_same_domain(*timestamps: Timestamp) -> None:
    """检查传入时间的域一致，禁止直接比较仿真钟与墙钟。"""
    if not timestamps:
        raise ValueError("至少需要一个时间戳")
    if any(not isinstance(value, Timestamp) for value in timestamps):
        raise ValueError("所有时间必须包含秒数与时钟域")
    if any(value.clock_domain != timestamps[0].clock_domain for value in timestamps[1:]):
        raise ValueError("时钟域不一致")


def _ordered(earlier: Timestamp, later: Timestamp, earlier_name: str, later_name: str) -> None:
    if earlier.seconds > later.seconds:
        raise ValueError(f"{earlier_name} 不得晚于 {later_name}")


def validate_observation_times(
    observation_time: Timestamp,
    last_seen_time: Timestamp,
    state_time: Timestamp,
    source_available_time: Timestamp,
) -> None:
    """验证原始观测、最近真实观测、外推状态和源处理完成时刻。"""
    require_same_domain(observation_time, last_seen_time, state_time, source_available_time)
    _ordered(observation_time, last_seen_time, "observation_time", "last_seen_time")
    _ordered(last_seen_time, state_time, "last_seen_time", "state_time")
    _ordered(last_seen_time, source_available_time, "last_seen_time", "source_available_time")


def validate_prediction_times(
    input_cutoff: Timestamp,
    anchor_time: Timestamp,
    completed_time: Timestamp,
    transmitted_time: Timestamp,
    future_end: Timestamp,
    valid_until: Timestamp,
) -> None:
    """检查预测输入/起点、计算/发送顺序与未来覆盖。

    valid_until 是采用期限，与 future_end 覆盖终点独立；不能互相代替。
    """
    require_same_domain(input_cutoff, anchor_time, completed_time, transmitted_time, future_end, valid_until)
    _ordered(input_cutoff, anchor_time, "input_cutoff", "anchor_time")
    _ordered(input_cutoff, completed_time, "input_cutoff", "completed_time")
    _ordered(completed_time, transmitted_time, "completed_time", "transmitted_time")
    if future_end.seconds <= anchor_time.seconds:
        raise ValueError("future_end 必须晚于 anchor_time")


def validate_receive_times(transmitted_time: Timestamp, arrival_time: Timestamp, usable_time: Timestamp) -> None:
    """验证发送、到达、处理完成；接收元数据只能由接收方记录。"""
    require_same_domain(transmitted_time, arrival_time, usable_time)
    _ordered(transmitted_time, arrival_time, "transmitted_time", "arrival_time")
    _ordered(arrival_time, usable_time, "arrival_time", "usable_time")


def require_available(
    decision_time: Timestamp,
    *,
    observation_time: Optional[Timestamp] = None,
    source_available_time: Optional[Timestamp] = None,
    state_time: Optional[Timestamp] = None,
    usable_time: Optional[Timestamp] = None,
    valid_until: Optional[Timestamp] = None,
) -> None:
    """消费时检查已采集、源处理完成、状态时刻和接收可用且未过期。

    仅对调用方提供的字段检查；需要哪类信息由各数据契约明确。
    同时提供采集、源完成或接收可用时，检查这些时间自身的先后次序；
    外推状态时刻不要求早于源处理完成。到期时刻本身仍有效。
    """
    fields = {
        "observation_time": observation_time,
        "source_available_time": source_available_time,
        "state_time": state_time,
        "usable_time": usable_time,
        "valid_until": valid_until,
    }
    present = [(name, value) for name, value in fields.items() if value is not None]
    require_same_domain(decision_time, *(value for _, value in present))
    pairs = (
        (observation_time, source_available_time, "observation_time", "source_available_time"),
        (source_available_time, usable_time, "source_available_time", "usable_time"),
        (observation_time, usable_time, "observation_time", "usable_time"),
    )
    for earlier, later, earlier_name, later_name in pairs:
        if earlier is not None and later is not None:
            _ordered(earlier, later, earlier_name, later_name)
    for name, value in present:
        if name == "valid_until":
            _ordered(decision_time, value, "decision_time", name)
        else:
            _ordered(value, decision_time, name, "decision_time")
