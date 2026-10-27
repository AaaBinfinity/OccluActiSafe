"""N02：严格契约、金样、时序与 NumPy 内部预测。"""

from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np

from occluactisafe.common.errors import ContractError
from occluactisafe.common.clock import Timestamp
from occluactisafe.common.schemas import (
    AdvanceRequest, ControlResult, DecisionBundle, EgoState, ObservationMessage,
    ObservationPacket, PredictionBundle, PredictionMessage, ReceivedMessage, TickEvent,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "n02"


def fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class SchemaTests(unittest.TestCase):
    def test_golden_json_roundtrip_preserves_fields_for_both_types(self):
        for cls, name in ((ObservationMessage, "golden_observation.json"),
                          (PredictionMessage, "golden_prediction.json")):
            with self.subTest(name=name):
                data = fixture(name)
                msg = cls.from_dict(data)
                self.assertEqual(json.loads(msg.to_json()), data)
                self.assertEqual(cls.from_json(msg.to_json()), msg)

    def test_unknown_missing_duplicate_nan_and_bad_version_rejected(self):
        data = fixture("golden_observation.json")
        for change in (
            lambda d: d.update({"arrival_time": 1.1}),
            lambda d: d.update({"truth_actor_id": "hidden"}),
            lambda d: d["tracks"][0].update({"future_gt": [1, 2]}),
            lambda d: d.pop("session_id"),
            lambda d: d.update({"schema_version": "2.0"}),
            lambda d: d["tracks"][0].update({"x": float("nan")}),
            lambda d: d["tracks"][0].update({"yaw": float("inf")}),
            lambda d: d["tracks"][0].update({"covariance_xy": [[1.0, 2.0], [2.0, 1.0]]}),
            lambda d: d["tracks"][0].update({"size_uncertainty": -1}),
            lambda d: d.update({"sequence": True}),
        ):
            bad = deepcopy(data); change(bad)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ObservationMessage.from_dict(bad)
        with self.assertRaises(ValueError):
            ObservationMessage.from_json('{"schema_version":"1.0", "schema_version":"1.0"}')
        with self.assertRaises(ValueError):
            ObservationMessage.from_json('{"x":NaN}')

    def test_zero_speed_is_not_missing_speed(self):
        data = fixture("golden_observation.json")
        self.assertEqual(ObservationMessage.from_dict(data).tracks[0].vx, 0.0)
        data["tracks"][0].update(vx=None, vy=None, velocity_valid=False)
        self.assertIsNone(ObservationMessage.from_dict(data).tracks[0].vx)
        data["tracks"][0]["velocity_valid"] = True
        with self.assertRaises(ValueError):
            ObservationMessage.from_dict(data)
        data = fixture("golden_observation.json")
        data["tracks"][0].update(x=None, y=None, observed_mask=False, state_valid_mask=False,
                                 length=None, width=None, size_valid=False, size_source="unknown")
        self.assertIsNone(ObservationMessage.from_dict(data).tracks[0].x)
        self.assertIsNone(ObservationMessage.from_dict(data).tracks[0].length)

    def test_observation_and_prediction_causality(self):
        data = fixture("golden_observation.json")
        for key, value in (("observation_time", 1.1), ("source_available_time", 1.2),
                           ("clock_domain", "wall"), ("coordinate_frame_id", "other")):
            bad = deepcopy(data); bad["tracks"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                ObservationMessage.from_dict(bad)
        bad = deepcopy(data); bad["tracks"][0]["state_time"] = 1.07
        with self.assertRaises(ValueError):
            ObservationMessage.from_dict(bad)
        data = fixture("golden_prediction.json")
        for key, value in (("observation_time", 1.01), ("source_available_time", 1.01),
                           ("state_time", 1.01), ("clock_domain", "wall")):
            bad = deepcopy(data); bad["targets"][0]["track"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                PredictionMessage.from_dict(bad)

    def test_prediction_time_shape_probability_and_partial_coverage(self):
        good = fixture("golden_prediction.json")
        self.assertEqual(PredictionMessage.from_dict(good).targets[0].valid_mask[1], (True, False))
        for change in (
            lambda d: d.update(future_end=1.4),
            lambda d: d.update(future_times=[1.2, 1.1]),
            lambda d: d["targets"][0].update(weights=[0.2, 0.2]),
            lambda d: d["targets"][0].update(weights=[-0.1, 1.1]),
            lambda d: d["targets"][0].update(valid_mask=[[True], [True]]),
            lambda d: d["targets"][0]["trajectories"][0][0].append(1.0),
            lambda d: d.update(completed_time=1.2, transmitted_time=1.1),
            lambda d: d.update(valid_until=1.11),
        ):
            bad = deepcopy(good); change(bad)
            with self.assertRaises(ValueError):
                PredictionMessage.from_dict(bad)
        # 未来覆盖终点与可采用期限独立：二者允许任一先后。
        good["valid_until"] = 1.15
        self.assertEqual(PredictionMessage.from_dict(good).future_end, 1.2)
        obs = fixture("golden_observation.json"); obs["valid_until"] = 1.06
        with self.assertRaises(ValueError):
            ObservationMessage.from_dict(obs)

    def test_received_metadata_is_separate_and_has_valid_order(self):
        payload = PredictionMessage.from_dict(fixture("golden_prediction.json"))
        received = ReceivedMessage(payload, 1.2, 1.3, "sim-episode-1")
        self.assertEqual(ReceivedMessage.from_json(received.to_json()), received)
        with self.assertRaises(ValueError):
            received.require_available(Timestamp(1.25, "sim-episode-1"))
        received.require_available(Timestamp(1.3, "sim-episode-1"))
        received.require_available(Timestamp(1.4, "sim-episode-1"))
        with self.assertRaises(ValueError):
            received.require_available(Timestamp(1.41, "sim-episode-1"))
        with self.assertRaises(ValueError):
            received.require_available(Timestamp(1.3, "wall"))
        self.assertNotIn("arrival_time", payload.to_dict())
        self.assertNotIn("usable_time", payload.to_dict())
        for arrival, usable, domain in ((1.1, 1.3, "sim-episode-1"), (1.3, 1.2, "sim-episode-1"),
                                        (1.2, 1.3, "wall")):
            with self.assertRaises(ValueError):
                ReceivedMessage(payload, arrival, usable, domain)

    def test_internal_bundle_shape_masks_and_probability(self):
        p = np.zeros((2, 3, 6, 40, 2))
        w = np.zeros((2, 3, 6)); w[:, :2, :] = 1 / 6
        mask = np.zeros((2, 3, 6, 40), dtype=bool); mask[:, :2, :, :] = True
        times = np.arange(1, 41) / 10
        bundle = PredictionBundle(p, w, mask, times, 0.0, 0.0, "m", "map", "sim")
        self.assertEqual(bundle.trajectories.shape, (2, 3, 6, 40, 2))
        self.assertFalse(hasattr(bundle, "to_json"))
        self.assertFalse(bundle.trajectories.flags.writeable)
        p[0, 0, 0, 0, 0] = 5
        self.assertEqual(bundle.trajectories[0, 0, 0, 0, 0], 0)
        for vals in ((p[..., 0], w, mask), (p, w, mask.astype(int)),
                     (p, np.ones((2, 3, 6)), mask),
                     (np.full_like(p, np.nan), w, mask)):
            with self.assertRaises(ValueError):
                PredictionBundle(*vals, times, 0.0, 0.0, "m", "map", "sim")

    def test_bridge_event_roundtrip_and_hidden_truth_rejected(self):
        obs = ObservationMessage.from_dict(fixture("golden_observation.json"))
        packet = ObservationPacket(obs.map_id, obs.map_version, obs.coordinate_frame_id, obs.frame_id,
                                   obs.clock_domain, obs.observation_time, obs.source_available_time, obs.tracks)
        event = TickEvent("1.0", "run-1", "episode-1", 0, 20, obs.clock_domain, 1.05, (packet,))
        self.assertEqual(TickEvent.from_json(event.to_json()), event)
        bad = event.to_dict(); bad["hidden_actor_truth"] = {"x": 100}
        with self.assertRaises(ValueError):
            TickEvent.from_dict(bad)
        bad = packet.to_dict(); bad["tracks"][0]["state_time"] = 1.1
        with self.assertRaises(ValueError):
            ObservationPacket.from_dict(bad)
        bad = event.to_dict(); bad["sim_time"] = 0.5
        with self.assertRaises(ValueError):
            TickEvent.from_dict(bad)
        bad = event.to_dict(); bad["sim_time"] = 1.0
        with self.assertRaises(ValueError):
            TickEvent.from_dict(bad)

    def test_decision_control_and_advance_contract(self):
        decision = DecisionBundle("req-1", "AEB", "collision risk", "sim", 1., 1.1, 1.2,
                                  ((0., 0.),), "uncalibrated_score", 0.4, 3., "reference", ("road",))
        self.assertEqual(DecisionBundle.from_json(decision.to_json()), decision)
        earlier_deadline = decision.to_dict(); earlier_deadline["deadline"] = 0.9
        self.assertEqual(DecisionBundle.from_dict(earlier_deadline).deadline, 0.9)
        control = ControlResult("req-1", "plan-1", "success", True, "sim", 1., 1.2, 0., 0.7, 0., None)
        request = AdvanceRequest("1.0", "run", "episode", 0, 21, "sim", 1.1, 1.2, control)
        self.assertEqual(AdvanceRequest.from_json(request.to_json()), request)
        with self.assertRaises(ValueError):
            ControlResult("req-1", "plan-1", "success", False, "sim", 1., 1.2, 0., 0.7, 0., None)
        with self.assertRaises(ValueError):
            ControlResult("req-1", "plan-1", "timeout", False, "sim", 1., 1.2, 0., 0.7, 0., None)
        with self.assertRaises(ValueError):
            AdvanceRequest("1.0", "run", "episode", 0, 21, "sim", 1.3, 1.4, control)
        with self.assertRaises(ValueError):
            AdvanceRequest("1.0", "run", "episode", 0, 21, "sim", 1.1, 1.4, control)

    def test_ego_state_roundtrip(self):
        ego = EgoState("vehicle-A", "sim", "map", "rear_axle", 1., 1.1, 1.0,
                       0., 0., None, None, 0., 4.2, 1.8, False, True)
        self.assertEqual(EgoState.from_json(ego.to_json()), ego)
        data = ego.to_dict(); data["vx"] = 0.
        with self.assertRaises(ValueError):
            EgoState.from_dict(data)


if __name__ == "__main__":
    unittest.main()
