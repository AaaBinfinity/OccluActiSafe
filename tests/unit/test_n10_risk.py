"""N10 R01～R03、绝对时间/缺口、N02/N08/N09衔接的机制检查。"""

from dataclasses import replace
import json
from pathlib import Path
import unittest
import numpy as np
from occluactisafe.common.clock import Timestamp
from occluactisafe.common.coordinates import Pose2D
from occluactisafe.common.schemas import PredictionMessage, ReceivedMessage, ObservationPacket
from occluactisafe.communication.validation import ReceiverPolicy, ReceiverState
from occluactisafe.fusion.history import FusionHistory
from occluactisafe.perception.participants import ParticipantGeometry, geometry_from_track
from occluactisafe.perception.visibility import GridSpec, SensorView, FreeSpaceEvidence, build_visibility
from occluactisafe.risk.occupancy import PoseTrajectory
from occluactisafe.risk.coverage import ModeForecast, TargetForecast, Candidate, targets_from_received, source_target_id
from occluactisafe.risk.multimodal import evaluate_candidates

ROOT = Path(__file__).resolve().parents[2]
CAR = ParticipantGeometry("vehicle", 4, 2)


def stationary(x=0, y=0, end=3, start=0, domain="sim-n10", frame="map"):
    return PoseTrajectory((start, end), (Pose2D(x, y, 0),) * 2, domain, frame)


def target(identity="t1", weight=.7, end=3):
    return TargetForecast(identity, CAR, (ModeForecast("hit", weight, stationary(end=end)),
                                         ModeForecast("miss", 1 - weight, stationary(y=10, end=end))), 0, 0, 0, 0, .5)


def candidate(identity="c1", path=None, feasible=True):
    return Candidate(identity, "declared_fixture_path", path or stationary(end=4), CAR, feasible)


def free_knowledge():
    spec = GridSpec(); ones = np.ones(spec.shape, bool)
    view = SensorView("sensor", Pose2D(0, 0, 0), 2 * np.pi, 50, 1, 0, 0, 1, "sim-n10", "map")
    proof = FreeSpaceEvidence("sensor", ones, 0, 0, 1, "sim-n10", "map")
    return build_visibility(spec, now=0, clock_domain="sim-n10", coordinate_frame_id="map", views=(view,),
                            free_evidence=(proof,), drivable=ones, closed_domain=True)


def evaluate(cs=None, ts=None, **kwargs):
    ts = (target(),) if ts is None else tuple(ts)
    return evaluate_candidates(cs or (candidate(),), ts, expected_target_ids=kwargs.pop("expected_target_ids", [t.target_id for t in ts]),
                               now=kwargs.pop("now", 0), plan_start=kwargs.pop("plan_start", 0), **kwargs)


def received_fixture():
    d = json.loads((ROOT / "tests/fixtures/n02/golden_prediction.json").read_text(encoding="utf-8"))
    d.update(future_times=[1.1, 4.], future_end=4.)
    t = d["targets"][0]; t["track"].update(yaw=0., yaw_valid=True)
    t.update(trajectories=[[[10, 2], [10, 2]], [[30, 2], [30, 2]]], weights=[.7, .3], valid_mask=[[True, True], [True, True]])
    return ReceivedMessage(PredictionMessage.from_dict(d), 1.2, 1.2, d["clock_domain"])


def receiver_policy():
    return ReceiverPolicy("vehicle-A", "map-A", "1", "map-right-handed", "sim-episode-1", {"roadside-1": "edge-1"})


class RiskTests(unittest.TestCase):
    def test_r01_mode_counted_once_across_many_times(self):
        p = PoseTrajectory((0., 1., 2., 3.), (Pose2D(0, 0, 0),) * 4, "sim-n10", "map")
        t = replace(target(), modes=(ModeForecast("hit", .7, p), target().modes[1]))
        row = evaluate(ts=(t,))["candidates"][0]
        self.assertEqual(row["R_max"], .7)
        self.assertEqual(row["R_sum"], .7)
        self.assertEqual([m["hit"] for m in row["modes"]], [True, False])
        self.assertEqual(row["risk_kind"], "uncalibrated_score")
        for name in ("calibrated_event_probability", "actual_collision", "near_conflict_proxy_event", "simulation_counterfactual_event", "action_approval"):
            self.assertIsNone(row[name])

    def test_r02_distinct_marginal_aggregates(self):
        row = evaluate(ts=(target("a", .7), target("b", .4)))["candidates"][0]
        self.assertEqual([t["raw_score"] for t in row["targets"]], [.7, .4])
        self.assertEqual(row["R_max"], .7)
        self.assertEqual(row["R_sum"], 1.)

    def test_r03_all_candidates_share_2_8_second_window(self):
        result = evaluate(cs=(candidate("a"), candidate("b", stationary(y=6, end=4))), ts=(target("long", end=4), target("short", end=2.8)))
        self.assertEqual(result["coverage"]["effective_horizon_s"], 2.8)
        self.assertFalse(result["coverage"]["complete"])
        for row in result["candidates"]:
            self.assertEqual(row["window_end"], 2.8)
            self.assertIn("target_coverage_incomplete", row["block_reasons"])
            self.assertTrue(row["normal_blocked"])

    def test_zero_score_incomplete_is_still_blocked(self):
        result = evaluate(cs=(candidate(path=stationary(y=6, end=4)),), ts=(target(end=2.8),))
        row = result["candidates"][0]
        self.assertEqual(row["R_max"], 0.)
        self.assertTrue(row["normal_blocked"])

    def test_invalid_weights_empty_and_unregistered(self):
        for weights in ((.7, .4), (-.1, 1.1), (float("nan"), .3), (True, 0)):
            with self.assertRaises(ValueError):
                replace(target(), modes=tuple(ModeForecast(str(i), w, stationary()) for i, w in enumerate(weights)))
        with self.assertRaises(ValueError): replace(target(), modes=())
        with self.assertRaises(ValueError): evaluate(expected_target_ids=[])
        with self.assertRaises(ValueError): evaluate(expected_target_ids=None)
        with self.assertRaises(ValueError): evaluate(ts=(target(), target()))

    def test_positive_modes_holes_shorten_all_and_no_extrapolation(self):
        p = PoseTrajectory((0., 1., 2., 3.), (Pose2D(0, 0, 0),) * 4, "sim-n10", "map", (True, True, False, True))
        t = replace(target(), modes=(ModeForecast("hole", 1, p),))
        self.assertEqual(evaluate(ts=(t,))["coverage"]["common_end"], 1)
        t = replace(target(), modes=(ModeForecast("late", 1, stationary(start=.1)),))
        result = evaluate(ts=(t,))
        self.assertEqual(result["coverage"]["common_end"], 0)
        self.assertIsNone(result["candidates"][0]["R_max"])

    def test_zero_weight_invalid_tail_not_in_denominator(self):
        t = replace(target(), modes=(ModeForecast("hit", 1, stationary()), ModeForecast("unused", 0, stationary(end=.1))))
        self.assertTrue(evaluate(ts=(t,))["coverage"]["complete"])

    def test_ttl_adoption_does_not_shorten_future_horizon(self):
        result = evaluate(ts=(target(end=4),), now=.1, plan_start=.1)
        self.assertTrue(result["coverage"]["complete"])
        self.assertEqual(result["coverage"]["effective_horizon_s"], 3.)
        result = evaluate(now=.6, plan_start=.6)
        self.assertEqual(result["coverage"]["gaps"][0]["reason"], "message_expired")
        self.assertIsNone(result["candidates"][0]["R_max"])

    def test_stale_memory_not_refreshed_by_projection(self):
        t = replace(target(end=4), valid_until=20, max_observation_age_s=.2)
        result = evaluate(ts=(t,), now=.3, plan_start=.3)
        self.assertEqual(result["coverage"]["gaps"][0]["reason"], "stale_last_seen")
        result = evaluate(ts=(), expected_target_ids=["known"])
        self.assertEqual(result["coverage"]["gaps"], [{"target_id": "known", "reason": "known_target_prediction_missing"}])

    def test_prefix_hazard_retained_when_future_clear(self):
        p = PoseTrajectory((0., .2, .5, 4.), (Pose2D(0, 0, 0), Pose2D(0, 0, 0), Pose2D(10, 0, 0), Pose2D(10, 0, 0)), "sim-n10", "map")
        t = replace(target(end=4), modes=(ModeForecast("prefix_only", 1, p),))
        row = evaluate(ts=(t,), plan_start=.5)["candidates"][0]
        self.assertEqual(row["R_max"], 0.)
        self.assertIn("prefix_potential_contact", row["block_reasons"])
        self.assertIsNotNone(row["prefix"][0]["event"])

    def test_missing_prefix_is_not_normal(self):
        t = replace(target(), modes=(ModeForecast("future_only", 1, stationary(start=.5, end=4)),))
        result = evaluate(ts=(t,), plan_start=.5)
        self.assertTrue(result["coverage"]["complete"])
        self.assertFalse(result["coverage"]["prefix_complete"])
        self.assertIn("prefix_coverage_incomplete", result["candidates"][0]["block_reasons"])

    def test_b07_infeasible_candidate_and_short_ego_rejected(self):
        result = evaluate(cs=(candidate("bad", feasible=False), candidate("short", stationary(end=1))))
        for row in result["candidates"]:
            self.assertTrue(row["normal_blocked"])
            self.assertIsNone(row["R_max"])

    def test_empty_registered_scene_needs_explicit_domain_and_free_evidence(self):
        row = evaluate(ts=())["candidates"][0]
        self.assertEqual(row["R_max"], 0)
        self.assertTrue(row["normal_blocked"])
        row = evaluate(ts=(), knowledge=free_knowledge())["candidates"][0]
        self.assertFalse(row["normal_blocked"])
        self.assertIsNone(row["action_approval"])

    def test_domain_mismatch_and_invalid_times(self):
        with self.assertRaises(ValueError): evaluate(cs=(candidate(path=stationary(domain="other")),))
        with self.assertRaises(ValueError): evaluate(now=1, plan_start=0)
        with self.assertRaises(ValueError): evaluate(horizon_s=float("nan"))
        with self.assertRaises(ValueError): replace(target(), last_seen=1)

    def test_n02_n08_adapter_absolute_2_8_and_expiry(self):
        record = received_fixture(); policy = receiver_policy()
        ts = targets_from_received(record, now=1.2, policy=policy)
        self.assertEqual(ts[0].anchor_time, 1.)
        self.assertEqual(ts[0].last_seen, .95)
        self.assertEqual(ts[0].heading_assumption, "constant_source_heading")
        c = candidate(path=stationary(start=1.2, end=4.2, domain=policy.clock_domain, frame=policy.coordinate_frame_id))
        result = evaluate_candidates((c,), ts, expected_target_ids=[ts[0].target_id], now=1.2, plan_start=1.2)
        self.assertAlmostEqual(result["coverage"]["effective_horizon_s"], 2.8)
        with self.assertRaisesRegex(ValueError, "not_arrived"): targets_from_received(record, now=1.19, policy=policy)
        with self.assertRaisesRegex(ValueError, "expired"): targets_from_received(record, now=1.41, policy=policy)
        state = ReceiverState(policy)
        targets_from_received(record, now=1.2, policy=policy, state=state)
        with self.assertRaisesRegex(ValueError, "duplicate"): targets_from_received(record, now=1.2, policy=policy, state=state)

    def test_adapter_missing_yaw_and_explicit_pedestrian_radius(self):
        r = received_fixture(); d = r.payload.to_dict()
        d["targets"][0]["track"].update(yaw=None, yaw_valid=False)
        r = ReceivedMessage(PredictionMessage.from_dict(d), 1.2, 1.2, r.clock_domain)
        with self.assertRaisesRegex(ValueError, "航向"): targets_from_received(r, now=1.2, policy=receiver_policy())
        d["targets"][0]["track"].update(category="pedestrian")
        r = ReceivedMessage(PredictionMessage.from_dict(d), 1.2, 1.2, r.clock_domain)
        with self.assertRaises(ValueError): targets_from_received(r, now=1.2, policy=receiver_policy())
        ts = targets_from_received(r, now=1.2, policy=receiver_policy(), pedestrian_radius=.3)
        self.assertEqual(ts[0].geometry.category, "pedestrian")

    def test_n09_lost_known_id_is_explicit_gap(self):
        r = received_fixture(); t = r.payload.targets[0].track
        packet = ObservationPacket("map-A", "1", t.coordinate_frame_id, 0, t.clock_domain, .95, 1., (t,))
        engine = FusionHistory(episode_id="n10-integration", clock_domain=t.clock_domain)
        initial = engine.step(Timestamp(1., t.clock_domain), (packet,))
        retained = engine.step(Timestamp(1.2, t.clock_domain))
        self.assertEqual(len(initial.agents), 1)
        self.assertEqual(len(retained.agents), 1)
        ids = [a["unified_id"] for a in retained.agents]
        c = candidate(path=stationary(start=1.2, end=4.2, domain=t.clock_domain, frame=t.coordinate_frame_id))
        result = evaluate_candidates((c,), (), expected_target_ids=ids, now=1.2, plan_start=1.2)
        self.assertEqual(result["coverage"]["gaps"][0]["target_id"], ids[0])
        self.assertTrue(result["candidates"][0]["normal_blocked"])


if __name__ == "__main__": unittest.main()
