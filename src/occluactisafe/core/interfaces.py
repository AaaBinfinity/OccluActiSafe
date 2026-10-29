"""核心接口协议；只定义类型，不实现感知、决策或控制。"""

from typing import Any, Protocol, Sequence, Union

from occluactisafe.common.schemas import (
    ControlResult, DecisionBundle, EgoState, ObservationPacket,
    NoResult, PredictionMessage, ReceivedMessage, SourceTrack,
)


class RoadsideEdge(Protocol):
    def step(self, now: float, roadside_observations: Sequence[ObservationPacket],
             received_v2i: Sequence[ReceivedMessage], known_map: Any,
             health: Any) -> Union[PredictionMessage, NoResult]: ...


class VehicleDecisionCore(Protocol):
    def step(self, now: float, ego_state: EgoState, local_tracks: Sequence[SourceTrack],
             received_roadside_messages: Sequence[ReceivedMessage], known_map: Any,
             visibility_state: Any, health: Any) -> DecisionBundle: ...


class ControlExecutor(Protocol):
    def step(self, now: float, decision: DecisionBundle, ego_state: EgoState,
             current_safety_context: Any) -> ControlResult: ...
