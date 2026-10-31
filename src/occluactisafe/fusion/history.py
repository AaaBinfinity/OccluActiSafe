"""来源独立后验、跨源统一身份和有界因果历史；没有风险决策。"""

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import numpy as np

from occluactisafe.common.clock import Timestamp
from occluactisafe.common.schemas import ObservationMessage, ObservationPacket, ReceivedMessage
from occluactisafe.communication.protocol import finite, integer
from occluactisafe.communication.validation import ReceiverPolicy, ReceiverState, validate_received
from .association import AssociationConfig, AssociationState, associate
from .state_filter import FilterConfig, initialize, correct


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True, order=True)
class TrackKey:
    episode_id: str
    source_id: str
    session_id: str
    sensor_id: str
    local_track_id: str

    def as_tuple(self):
        return (self.episode_id, self.source_id, self.session_id, self.sensor_id, self.local_track_id)

    @property
    def namespace(self):
        return self.as_tuple()[:-1]


@dataclass(frozen=True)
class HistoryConfig:
    max_points: int = 20
    history_seconds: float = 2.
    retention_seconds: float = 3.
    max_source_tracks: int = 256

    def __post_init__(self):
        integer(self.max_points, "max_points"); integer(self.max_source_tracks, "max_source_tracks")
        finite(self.history_seconds, "history_seconds", minimum=1e-12)
        finite(self.retention_seconds, "retention_seconds", minimum=1e-12)
        if self.retention_seconds < self.history_seconds:
            raise ValueError("状态保留期不能短于历史时间窗")


@dataclass(frozen=True)
class FusionResult:
    episode_id: str
    clock_domain: str
    state_time: float
    agents: tuple
    events: tuple
    associations: tuple
    input_sha256: str

    @property
    def feature_sha256(self):
        return _digest(self.agents)

    def to_dict(self):
        return deepcopy({"episode_id": self.episode_id, "clock_domain": self.clock_domain,
                         "state_time": self.state_time, "agents": list(self.agents), "events": list(self.events),
                         "associations": list(self.associations), "input_sha256": self.input_sha256,
                         "feature_sha256": self.feature_sha256})


class FusionHistory:
    def __init__(self, *, episode_id, clock_domain, filter_config=FilterConfig(),
                 association_config=AssociationConfig(), history_config=HistoryConfig(),
                 map_id="map-A", map_version="1", coordinate_frame_id="map-right-handed",
                 receiver_policy: ReceiverPolicy | None = None):
        if not isinstance(filter_config, FilterConfig) or not isinstance(association_config, AssociationConfig) or not isinstance(history_config, HistoryConfig):
            raise ValueError("滤波/关联/历史配置类型不正确")
        for value in (map_id, map_version, coordinate_frame_id):
            if not isinstance(value, str) or not value.strip(): raise ValueError("地图/坐标标识不能为空")
        self.filter_config, self.association_config, self.history_config = filter_config, association_config, history_config
        self.map_id, self.map_version, self.coordinate_frame_id = map_id, map_version, coordinate_frame_id
        self.begin_episode(episode_id, clock_domain, receiver_policy=receiver_policy)

    def begin_episode(self, episode_id, clock_domain, *, receiver_policy=None):
        for value in (episode_id, clock_domain):
            if not isinstance(value, str) or not value.strip(): raise ValueError("episode/时钟域不能为空")
        if hasattr(self, "episode_id") and (episode_id == self.episode_id or clock_domain == self.clock_domain):
            raise ValueError("新episode必须使用不同的episode_id和clock_domain，防止旧包跨episode重放")
        if receiver_policy is not None:
            if (not isinstance(receiver_policy, ReceiverPolicy) or receiver_policy.clock_domain != clock_domain
                    or receiver_policy.map_id != self.map_id or receiver_policy.map_version != self.map_version
                    or receiver_policy.coordinate_frame_id != self.coordinate_frame_id):
                raise ValueError("接收策略与episode地图/时钟不一致")
        self.episode_id, self.clock_domain = episode_id, clock_domain
        self.receiver_policy = receiver_policy
        self.receiver_state = ReceiverState(receiver_policy) if receiver_policy is not None else None
        self._entries = {}
        self._bindings = {}
        self._counter = 0
        self._now = None

    def _key(self, track):
        return TrackKey(self.episode_id, track.source_id, track.session_id, track.sensor_id, track.source_track_id)

    def step(self, now: Timestamp, packets=(), *, received=()):
        if not isinstance(now, Timestamp) or now.clock_domain != self.clock_domain or now.seconds < 0:
            raise ValueError("融合时钟域或时间非法")
        if self._now is not None and now.seconds < self._now:
            raise ValueError("在线融合时钟不得倒退")
        self._now = now.seconds
        events, reports, consumed = [], [], []
        old_ids = set(self._bindings.values())

        def event(reason, key=None, **values):
            events.append({"reason": reason, "key": list(key.as_tuple()) if key else None, **values})

        incoming = list(packets)
        valid_received = []
        for record in received:
            if not isinstance(record, ReceivedMessage):
                event("invalid_received_type"); continue
            valid_received.append(record)
        valid_received.sort(key=lambda r: (r.usable_time, r.arrival_time, r.payload.transmitted_time,
                                            r.payload.source_id, r.payload.session_id, r.payload.sequence))
        message_fingerprints = defaultdict(set)
        for record in valid_received:
            if record.arrival_time <= now.seconds and record.usable_time <= now.seconds:
                p = record.payload
                try:
                    message_fingerprints[(p.source_id, p.session_id, p.sequence)].add(_digest(p.to_dict()))
                except (ValueError, TypeError):
                    pass  # 仍交N08验证器给出明确的invalid_message原因。
        for record in valid_received:
            # 先查接收头部，不触碰晚到/未处理消息里的目标或轨迹数值。
            if record.arrival_time > now.seconds or record.usable_time > now.seconds:
                event("received_not_available"); continue
            if self.receiver_policy is None:
                event("receiver_policy_required"); continue
            p = record.payload
            if len(message_fingerprints[(p.source_id, p.session_id, p.sequence)]) > 1:
                event("conflicting_message_duplicate", source_id=p.source_id, session_id=p.session_id, sequence=p.sequence)
                continue
            checked = validate_received(record, now, self.receiver_policy, state=self.receiver_state)
            if not checked.accepted:
                event("receive_" + checked.reason); continue
            payload = checked.received.payload
            event("received_accepted", source_id=payload.source_id, session_id=payload.session_id, sequence=payload.sequence,
                  arrival_time=record.arrival_time, usable_time=record.usable_time, valid_until=payload.valid_until)
            tracks = payload.tracks if isinstance(payload, ObservationMessage) else tuple(t.track for t in payload.targets)
            # 只提取合法状态；绝不把预测的未来坐标作为Kalman测量。
            observed = payload.observation_time if isinstance(payload, ObservationMessage) else max((t.observation_time for t in tracks), default=now.seconds)
            incoming.append(ObservationPacket(payload.map_id, payload.map_version, payload.coordinate_frame_id,
                                              payload.frame_id, payload.clock_domain, observed, record.usable_time, tracks))
        candidates = defaultdict(list)
        for packet in incoming:
            if not isinstance(packet, ObservationPacket):
                event("invalid_packet_type"); continue
            if packet.clock_domain != self.clock_domain:
                event("packet_clock_mismatch"); continue
            if packet.source_available_time > now.seconds:
                event("packet_not_available"); continue
            if (packet.map_id, packet.map_version, packet.coordinate_frame_id) != (self.map_id, self.map_version, self.coordinate_frame_id):
                event("packet_map_mismatch"); continue
            try:
                packet.to_dict()  # 可用后再验证完整N02白名单和因果字段。
            except (ValueError, TypeError):
                event("invalid_packet"); continue
            for track in packet.tracks:
                key = self._key(track)
                if track.role == "ego": event("ego_excluded", key); continue
                if not track.state_valid_mask: event("invalid_position", key); continue
                if track.state_time > now.seconds or track.last_seen_time > now.seconds:
                    event("future_track", key); continue
                if now.seconds - track.last_seen_time > self.history_config.retention_seconds:
                    event("stale_input", key); continue
                candidates[(key, track.last_seen_time)].append((track, packet.source_available_time, packet.frame_id))
        # 先按原观测寿命清理，不能因重发或新的计算任务刷新寿命。
        for key, entry in list(self._entries.items()):
            if now.seconds - entry["state"].last_seen_time > self.history_config.retention_seconds:
                del self._entries[key]; event("retention_expired", key)
        for (key, _), observations in sorted(candidates.items(), key=lambda item: (item[0][1], item[0][0])):
            fingerprints = {_digest(track.to_dict()) for track, _, _ in observations}
            if len(fingerprints) > 1:
                event("conflicting_duplicate", key, count=len(observations)); continue
            track, packet_available, packet_frame = min(observations, key=lambda v: (v[1], v[2]))
            if len(observations) > 1: event("duplicate_in_batch", key, count=len(observations) - 1)
            old = self._entries.get(key)
            if old is not None:
                previous = old["state"]
                if track.last_seen_time <= previous.last_seen_time:
                    event("duplicate_or_out_of_order", key); continue
                if not track.observed_mask:
                    event("projection_not_measurement", key); continue
                if track.state_time < previous.state_time:
                    event("source_state_time_regression", key); continue
                old_track = old["track"]
                if ((old_track.category != track.category and "unknown" not in (old_track.category, track.category))
                        or old_track.reference_point != track.reference_point):
                    event("source_identity_reset", key)
                    old = None; self._bindings.pop(key, None)
            if old is None:
                if key not in self._entries and len(self._entries) >= self.history_config.max_source_tracks:
                    event("source_capacity_limit", key); continue
                state, reason = initialize(track, self.filter_config)
                old = {"state": state, "track": track, "history": deque(maxlen=self.history_config.max_points)}
                self._entries[key] = old
                action = "source_initialized"
            else:
                state, reason = correct(old["state"], track, self.filter_config)
                old.update(state=state, track=track)
                action = "measurement_updated"
            old["history"].append({"accepted_time": now.seconds, "kind": "raw_observation" if track.observed_mask and track.state_time == track.last_seen_time else "derived_state",
                                   "track": track.to_dict(), "packet_available_time": packet_available,
                                   "packet_frame_id": packet_frame, "R_reason": reason})
            consumed.append({"key": list(key.as_tuple()), "track": track.to_dict(),
                             "packet_available_time": packet_available, "packet_frame_id": packet_frame})
            event(action, key, R_reason=reason, updates=state.updates)
        views = {}
        for key, entry in sorted(self._entries.items()):
            while entry["history"] and entry["history"][0]["track"]["last_seen_time"] < now.seconds - self.history_config.history_seconds:
                entry["history"].popleft()
            track, state = entry["track"], entry["state"]
            z, p = state.snapshot(now.seconds, self.filter_config)
            yaw = track.yaw if state.velocity_valid and np.linalg.norm(z[2:]) >= self.filter_config.low_speed_mps and track.yaw_valid else None
            views[key] = {"key": list(key.as_tuple()), "category": track.category, "reference_point": track.reference_point,
                          "role": track.role, "state": z.tolist(), "covariance": p.tolist(), "state_time": now.seconds,
                          "source_state_time": state.state_time, "observation_time": track.observation_time,
                          "source_available_time": track.source_available_time, "last_seen_time": state.last_seen_time,
                          "age_s": now.seconds - state.last_seen_time, "velocity_valid": state.velocity_valid,
                          "yaw": yaw, "yaw_valid": yaw is not None, "quality": track.quality,
                          "measurement_updates": state.updates, "history": list(entry["history"])}

        def as_association(key):
            v = views[key]
            return AssociationState(key.as_tuple(), v["category"], v["role"], v["reference_point"],
                                    v["state"], v["covariance"], v["velocity_valid"], v["yaw"])

        def representative(group):
            return min(group, key=lambda k: (views[k]["age_s"], np.trace(np.asarray(views[k]["covariance"])[:2, :2]),
                                             -(views[k]["quality"] if views[k]["quality"] is not None else -1), k))

        namespaces = defaultdict(list)
        for key in views: namespaces[key.namespace].append(key)
        groups = []
        for namespace in sorted(namespaces):
            rows = sorted(namespaces[namespace])
            if not groups:
                groups = [[k] for k in rows]; continue
            matched, report = associate([as_association(k) for k in rows],
                                        [as_association(representative(g)) for g in groups], self.association_config,
                                        column_members=[[as_association(k) for k in g] for g in groups])
            report["namespace"] = list(namespace); reports.append(report)
            for i, j in matched.pairs: groups[j].append(rows[i])
            groups.extend([[rows[i]] for i in matched.unmatched_rows])
        bindings, used, agents = {}, set(), []
        for group in sorted(groups, key=lambda g: min(g)):
            candidates_ids = sorted({self._bindings[k] for k in group if k in self._bindings} - used)
            if candidates_ids:
                uid = candidates_ids[0]
            else:
                self._counter += 1; uid = f"{self.episode_id}/agent_{self._counter:06d}"
                event("unified_created", unified_id=uid)
            used.add(uid)
            for key in group: bindings[key] = uid
            rep = representative(group)
            agents.append({"unified_id": uid, "representative_key": list(rep.as_tuple()),
                           "state": views[rep]["state"], "covariance": views[rep]["covariance"],
                           "state_time": now.seconds, "last_seen_time": views[rep]["last_seen_time"],
                           "sources": [views[k] for k in sorted(group)]})
        for retired in sorted(old_ids - used): event("unified_retired", unified_id=retired)
        self._bindings = bindings
        agents.sort(key=lambda a: a["unified_id"])
        return FusionResult(self.episode_id, self.clock_domain, now.seconds, tuple(deepcopy(agents)),
                            tuple(events), tuple(reports), _digest(consumed))
