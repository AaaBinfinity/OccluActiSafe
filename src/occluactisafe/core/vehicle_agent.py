"""只消费当时合法观测/接收消息，N09关联后每统一目标选择最新合法预测。"""

from dataclasses import replace
import json
from occluactisafe.common.clock import Timestamp
from occluactisafe.common.schemas import ObservationPacket, ReceivedMessage, DecisionBundle
from occluactisafe.communication.validation import ReceiverPolicy, ReceiverState, validate_received
from occluactisafe.fusion.history import FusionHistory
from occluactisafe.risk.coverage import targets_from_received, source_target_id
from occluactisafe.risk.multimodal import risk_table_for_candidates
from occluactisafe.planning.candidates import longitudinal_candidates, compute_ttc_cv
from occluactisafe.planning.decision import AEBDecision
from .roadside_edge import make_prediction


class VehicleAgent:
    def __init__(self, strategy="risk", *, domain="sim-n11"):
        self.domain = domain
        self.policy = ReceiverPolicy("vehicle-A", "map-A", "1", "map-right-handed", domain,
                                     {"roadside-1": "session-1", "vehicle-A": "session-1"})
        self.receiver = ReceiverState(self.policy); self.cache = {}; self.sequence = 0
        self.fusion = FusionHistory(episode_id="n11", clock_domain=domain)
        self.decision = AEBDecision(strategy); self.last_audit = None

    def step(self, now, ego, executor, *, local_tracks=(), received=(), knowledge=None, unknown=True):
        if ego.state_time != now or ego.clock_domain != self.domain: raise ValueError("决策时刻/自车域不一致")
        records = list(received); accepted = []; gates = []
        if local_tracks:
            message = make_prediction(local_tracks, now, self.sequence, source_id="vehicle-A", domain=self.domain)
            self.sequence += 1; records.append(ReceivedMessage(message, now, now, self.domain))
        for record in records:
            if record.arrival_time > now or record.usable_time > now:
                gates.append("not_available"); continue
            checked = validate_received(record, Timestamp(now, self.domain), self.policy, state=self.receiver,
                                        plan_start=Timestamp(now, self.domain), horizon_s=0.)
            gates.append(checked.reason)
            if checked.accepted:
                self.cache[record.payload.source_id] = record
                tracks = tuple(t.track for t in record.payload.targets)
                if tracks:
                    accepted.append(ObservationPacket("map-A", "1", ego.coordinate_frame_id, record.payload.frame_id,
                                                      self.domain, max(t.observation_time for t in tracks), now, tracks))
        snapshot = self.fusion.step(Timestamp(now, self.domain), accepted)
        binding = {json.dumps(s["key"][1:], separators=(",", ":")): a["unified_id"] for a in snapshot.agents for s in a["sources"]}
        choices = {}; current_tracks = {}
        for record in self.cache.values():
            try: targets = targets_from_received(record, now=now, policy=self.policy, identity_map=binding)
            except ValueError: continue
            for forecast, raw in zip(targets, record.payload.targets):
                if source_target_id(raw.track) not in binding: continue
                previous = choices.get(forecast.target_id)
                if previous is None or forecast.last_seen > previous.last_seen:
                    choices[forecast.target_id] = forecast; current_tracks[forecast.target_id] = raw.track
        candidates = longitudinal_candidates(executor, now, self.domain, ego.coordinate_frame_id)
        expected = [a["unified_id"] for a in snapshot.agents]
        table = risk_table_for_candidates(candidates, tuple(choices.values()), expected_target_ids=expected,
                                         now=now, plan_start=now + executor.delay, knowledge=knowledge)
        ttc = compute_ttc_cv(ego, tuple(current_tracks.values()), now=now)
        action = self.decision.choose(table, ttc, unknown=unknown, prediction_missing=bool(set(expected) - choices.keys()))
        row = next(r for r in table["candidates"] if r["candidate_id"] == ("SLOW" if action["action"] == "DEGRADED" else action["action"]))
        mode = {"NORMAL": "Normal", "SLOW": "Caution", "AEB": "AEB", "DEGRADED": "Degraded"}[action["action"]]
        selected = next(c for c in candidates if c.candidate_id == row["candidate_id"])
        reference = (selected.path.at(now + executor.delay), *(p for t,p in zip(selected.path.times, selected.path.poses) if t > now + executor.delay))
        bundle = DecisionBundle(f"n11-{now:.6f}", mode, action["reason"], self.domain, now + executor.delay,
                                now + .05, now + executor.delay + 3., tuple((p.x,p.y) for p in reference),
                                "uncalibrated_score" if row["R_max"] is not None else "unknown_region_constraint",
                                row["R_max"], 3., row["reference_policy"], tuple(row["block_reasons"]))
        bundle.to_dict()
        self.last_audit = {"gates": gates, "fusion": snapshot.to_dict(), "risk": table, "ttc": ttc, "decision": action,
                           "bundle": bundle.to_dict(), "inputs": {"local_tracks": [t.to_dict() for t in local_tracks],
                           "accepted_sources": sorted(self.cache), "unknown": unknown}}
        return bundle, action["action"]
