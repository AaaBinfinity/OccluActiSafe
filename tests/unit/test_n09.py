"""N09构造观测与可手算矩阵；不是视频融合、真实数据或避撞实验。"""

from dataclasses import replace
import itertools
import unittest
import numpy as np

from occluactisafe.common.clock import Timestamp
from occluactisafe.common.schemas import ObservationPacket, ObservationMessage, SourceTrack, ReceivedMessage
from occluactisafe.communication.validation import ReceiverPolicy
from occluactisafe.fusion.state_filter import (
    FilterConfig, cv_predict, position_update, measurement_covariance, initialize, correct,
)
from occluactisafe.fusion.association import (
    AssociationConfig, AssociationState, associate, minimum_cost_assignment,
)
from occluactisafe.fusion.history import FusionHistory, HistoryConfig

DOMAIN = "sim-n09"


def track(source="vehicle", local_id="1", *, t=1., x=0., y=0., vx=10., vy=0., sensor="front", **overrides):
    values = dict(source_track_id=local_id, source_id=source, session_id="session-1", sensor_id=sensor,
                  category="vehicle", role="target", clock_domain=DOMAIN, coordinate_frame_id="map-right-handed",
                  reference_point="geometric_center", observation_time=t, last_seen_time=t,
                  source_available_time=t, state_time=t, x=x, y=y, vx=vx, vy=vy, yaw=0., length=4.2, width=1.8,
                  size_source="fixture", size_uncertainty=None, covariance_xy=None, observed_mask=True,
                  state_valid_mask=True, size_valid=True, velocity_valid=vx is not None and vy is not None,
                  yaw_valid=True, quality=.9)
    values.update(overrides)
    return SourceTrack(**values)


def packet(tracks, *, at=None, available=None):
    tracks = tuple(tracks)
    observed = max((t.observation_time for t in tracks), default=1.) if at is None else at
    ready = max([observed, *(t.source_available_time for t in tracks)]) if available is None else available
    return ObservationPacket("map-A", "1", "map-right-handed", 0, DOMAIN, observed, ready, tracks)


def engine(**kwargs):
    return FusionHistory(episode_id="episode-1", clock_domain=DOMAIN, **kwargs)


def now(t): return Timestamp(t, DOMAIN)


def flattened(result):
    return [s for a in result.agents for s in a["sources"]]


class KalmanTests(unittest.TestCase):
    def test_k01_exact_cv_and_nonnegative_time(self):
        z, p = cv_predict([0, 0, 10, 0], np.eye(4), .1)
        np.testing.assert_allclose(z, [1, 0, 10, 0], atol=1e-12, rtol=0)
        np.testing.assert_allclose(p, p.T, atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(p)[0], -1e-8)
        self.assertFalse(z.flags.writeable)
        with self.assertRaises(ValueError): cv_predict(z, p, -.1)
        with self.assertRaises(ValueError): cv_predict([np.nan, 0, 0, 0], p, .1)
        with self.assertRaises(ValueError): cv_predict(z, np.diag([-1, 1, 1, 1]), .1)

    def test_k02_joseph_small_nonzero_r_and_independent_scalar_reference(self):
        z, p = position_update(np.zeros(4), np.eye(4), [2, 4], np.eye(2))
        np.testing.assert_allclose(z, [1, 2, 0, 0], atol=1e-12)
        np.testing.assert_allclose(np.diag(p), [.5, .5, 1, 1], atol=1e-12)
        for i in range(200):
            z, p = cv_predict(z, p, .1)
            z, p = position_update(z, p, [i / 10, 2 * i / 10], np.eye(2) * 1e-6)
            self.assertTrue(np.isfinite(z).all() and np.isfinite(p).all())
            np.testing.assert_allclose(p, p.T, atol=1e-12)
            self.assertGreaterEqual(np.linalg.eigvalsh(p)[0], -1e-8)
        with self.assertRaises(ValueError): position_update(z, p, [0, 0], np.zeros((2, 2)))

    def test_unknown_r_velocity_masks_and_explicit_covariance_floor(self):
        cfg = FilterConfig()
        tr = track(vx=None, vy=None)
        r, reason = measurement_covariance(tr, cfg)
        np.testing.assert_array_equal(r, np.eye(2))
        self.assertEqual(reason, "unknown_R_configured")
        state, _ = initialize(tr, cfg)
        self.assertFalse(state.velocity_valid)
        self.assertEqual(state.p[2, 2], 25)
        state, _ = correct(state, track(t=1.1, x=1, vx=None, vy=None), cfg)
        self.assertTrue(state.velocity_valid)
        self.assertGreater(state.z[2], 0)
        r, reason = measurement_covariance(track(covariance_xy=((0, 0), (0, 0))), cfg)
        self.assertEqual(reason, "provided_R_regularized")
        self.assertGreaterEqual(np.linalg.eigvalsh(r)[0], 1e-6)
        with self.assertRaises(ValueError): correct(state, tr, cfg)

    def test_k03_missing_three_frames_does_not_refresh_age_or_covariance(self):
        e = engine(); e.step(now(1), [packet([track(vx=0, vy=0)])])
        traces = []
        for t in [1.1, 1.2, 1.3]:
            view = flattened(e.step(now(t)))[0]
            self.assertEqual(view["last_seen_time"], 1)
            self.assertEqual(view["measurement_updates"], 1)
            self.assertEqual(len(view["history"]), 1)
            self.assertFalse(view["yaw_valid"]); self.assertIsNone(view["yaw"])
            traces.append(np.trace(view["covariance"]))
        self.assertTrue(all(a < b for a, b in zip(traces, traces[1:])))
        self.assertAlmostEqual(view["age_s"], .3)


class AssociationTests(unittest.TestCase):
    def test_a01_a02_dummy_columns_and_no_duplicate_ownership(self):
        assignment = minimum_cost_assignment([[1, 8], [7, 2]], unmatched_cost=5)
        self.assertEqual(assignment.pairs, ((0, 0), (1, 1)))
        self.assertEqual(assignment.objective, 3)
        assignment = minimum_cost_assignment([[1, 8], [7, 2]], np.zeros((2, 2), bool), unmatched_cost=5)
        self.assertEqual(assignment.pairs, ())
        self.assertEqual(assignment.unmatched_rows, (0, 1)); self.assertEqual(assignment.unmatched_columns, (0, 1))
        self.assertEqual(assignment.objective, 10)
        assignment = minimum_cost_assignment([[1], [2]], unmatched_cost=5)
        self.assertEqual(assignment.pairs, ((0, 0),)); self.assertEqual(assignment.unmatched_rows, (1,))
        self.assertEqual(minimum_cost_assignment(np.empty((0, 2))).unmatched_columns, (0, 1))
        self.assertEqual(minimum_cost_assignment(np.empty((2, 0))).unmatched_rows, (0, 1))
        with self.assertRaises(ValueError): minimum_cost_assignment([[float("nan")]])
        with self.assertRaises(ValueError): minimum_cost_assignment([[1]], [[1]])

    def test_gates_category_position_reference_and_mahalanobis(self):
        def state(key, x, category="vehicle", p=.1, reference="geometric_center"):
            return AssociationState((key,), category, "target", reference, [x, 0, 0, 0], np.eye(4) * p, True, None)
        a = state("a", 0)
        columns = [state("b", 1, category="pedestrian"), state("c", 5), state("d", 3),
                   state("e", 0, reference="rear_axle")]
        matched, report = associate([a], columns)
        self.assertFalse(matched.pairs)
        self.assertEqual(report["reasons"][0], ["category_mismatch", "position_gate", "mahalanobis_gate", "role_or_reference_mismatch"])

    def test_ambiguous_people_are_retained_as_separate_objects(self):
        e = engine()
        p = packet([track("road", "one", x=-.02, category="pedestrian"),
                    track("road", "two", x=.02, category="pedestrian"),
                    track("vehicle", "different-local-id", x=0, category="pedestrian")])
        result = e.step(now(1), [p])
        self.assertEqual(len(result.agents), 3)
        self.assertTrue(any("ambiguous" in reason for report in result.associations for row in report["reasons"] for reason in row))

    def test_group_unknown_category_does_not_bridge_vehicle_to_pedestrian(self):
        e = engine()
        p = packet([track("a", "car", x=0, category="vehicle"),
                    track("b", "unknown", x=0, category="unknown"),
                    track("c", "person", x=0, category="pedestrian")])
        result = e.step(now(1), [p])
        self.assertEqual(len(result.agents), 2)
        categories = [{s["category"] for s in a["sources"]} for a in result.agents]
        self.assertTrue(all(not {"vehicle", "pedestrian"} <= c for c in categories))


class CausalHistoryTests(unittest.TestCase):
    def test_a02_roadside_only_preserved_and_no_local_id_shortcut(self):
        e = engine()
        result = e.step(now(1), [packet([track("road", "1", x=50), track("vehicle", "1", x=0)])])
        self.assertEqual(len(result.agents), 2)
        self.assertEqual(len(flattened(result)), 2)
        self.assertEqual({s["key"][1] for s in flattened(result)}, {"road", "vehicle"})
        self.assertEqual(sum(len(a["sources"]) for a in result.agents), 2)

    def test_correlated_sources_keep_separate_posteriors_and_one_representative(self):
        e = engine()
        result = e.step(now(1), [packet([track("road", "r", x=.05), track("vehicle", "v", x=0)])])
        self.assertEqual(len(result.agents), 1)
        a = result.agents[0]
        self.assertEqual(len(a["sources"]), 2)
        self.assertTrue(all(s["measurement_updates"] == 1 for s in a["sources"]))
        self.assertEqual(np.trace(a["covariance"]), np.trace(a["sources"][0]["covariance"]))
        self.assertEqual(a["covariance"], next(s for s in a["sources"] if s["key"] == a["representative_key"])["covariance"])

    def test_duplicates_out_of_order_conflicts_and_projection_no_new_history(self):
        e = engine(); first = track()
        e.step(now(1), [packet([first])])
        result = e.step(now(1.1), [packet([first]), packet([track(t=.9, x=-1)])])
        self.assertEqual(len(flattened(result)[0]["history"]), 1)
        self.assertEqual(flattened(result)[0]["measurement_updates"], 1)
        projected = replace(first, state_time=1.1, x=1., observed_mask=False)
        result = e.step(now(1.1), [packet([projected], available=1.1)])
        self.assertEqual(flattened(result)[0]["last_seen_time"], 1)
        self.assertEqual(len(flattened(result)[0]["history"]), 1)
        conflict = [track(t=1.2, x=2), track(t=1.2, x=100)]
        result = e.step(now(1.2), [packet(conflict)])
        self.assertEqual(flattened(result)[0]["last_seen_time"], 1)
        self.assertTrue(any(v["reason"] == "conflicting_duplicate" for v in result.events))

    def test_namespace_sensor_session_and_episode_reset(self):
        e = engine()
        records = [track("vehicle", "1", sensor="front", x=0), track("vehicle", "1", sensor="rear", x=50),
                   track("vehicle", "1", sensor="front", session_id="session-2", x=100)]
        a = e.step(now(1), [packet(records)])
        self.assertEqual(len(a.agents), 3)
        self.assertEqual(len({tuple(s["key"]) for s in flattened(a)}), 3)
        e.begin_episode("episode-2", DOMAIN)
        self.assertEqual(e.step(now(0)).agents, ())
        b = e.step(now(1), [packet([records[0]])])
        self.assertTrue(b.agents[0]["unified_id"].startswith("episode-2/"))
        self.assertEqual(len(flattened(b)[0]["history"]), 1)

    def test_delayed_fresh_measurement_is_independent_of_snapshot_frequency(self):
        initial = packet([track()])
        delayed = packet([track(t=1.2, x=2)], available=1.6)
        a, b = engine(), engine()
        a.step(now(1), [initial]); b.step(now(1), [initial])
        a.step(now(1.1)); a.step(now(1.3)); a.step(now(1.5))
        left = a.step(now(1.6), [delayed]); right = b.step(now(1.6), [delayed])
        self.assertEqual(left.feature_sha256, right.feature_sha256)
        self.assertEqual(flattened(left)[0]["last_seen_time"], 1.2)
        self.assertEqual(len(flattened(left)[0]["history"]), 2)
        derived = replace(track(), state_time=1.5, x=5., observed_mask=False)
        c = engine(); c.step(now(1.5), [packet([derived], available=1.5)])
        rejected = c.step(now(1.6), [packet([track(t=1.1, x=1)], available=1.6)])
        self.assertTrue(any(e["reason"] == "source_state_time_regression" for e in rejected.events))
        self.assertEqual(flattened(rejected)[0]["last_seen_time"], 1.)

    def test_m04_future_available_packet_cannot_add_target_or_change_features(self):
        histories = [packet([track(t=1, x=0)])]
        future = packet([track("road", "new-hidden", t=2, x=99999)], available=2.2)
        a, b = engine(), engine()
        r1 = a.step(now(1.1), histories)
        r2 = b.step(now(1.1), [future, *histories])
        self.assertEqual(r1.feature_sha256, r2.feature_sha256)
        self.assertEqual(r1.input_sha256, r2.input_sha256)
        self.assertEqual(len(r2.agents), 1)
        # 未来标签仅存在于离线变量，不传入step；改变它不改变任何合法输入。
        future_labels = np.full((50, 2), 1e9); future_labels[:] = -1e9
        self.assertEqual(a.step(now(1.1)).feature_sha256, b.step(now(1.1)).feature_sha256)
        bad = track().to_dict(); bad["truth_actor_id"] = "oracle"
        with self.assertRaises(ValueError): SourceTrack.from_dict(bad)

    def test_m05_permutation_of_packets_and_target_order_is_identical(self):
        a = [track("road", "r1", x=.05), track("road", "r2", x=20.05)]
        b = [track("vehicle", "v9", x=0), track("vehicle", "v4", x=20)]
        expected = None
        for swap_packets, swap_a, swap_b in itertools.product([False, True], repeat=3):
            packets = [packet(a[::-1] if swap_a else a), packet(b[::-1] if swap_b else b)]
            if swap_packets: packets.reverse()
            result = engine().step(now(1), packets)
            self.assertEqual(len(result.agents), 2)
            signature = (result.feature_sha256, result.input_sha256)
            if expected is None: expected = signature
            self.assertEqual(signature, expected)

    def test_bounded_history_retention_and_capacity_do_not_claim_disappearance(self):
        e = engine(history_config=HistoryConfig(max_points=2, history_seconds=.2, retention_seconds=.5, max_source_tracks=1))
        for i in range(5): e.step(now(1 + i / 10), [packet([track(t=1 + i / 10, x=i)])])
        result = e.step(now(1.4), [packet([track("road", "other", t=1.4, x=100)])])
        self.assertEqual(len(result.agents), 1)
        self.assertEqual(len(flattened(result)[0]["history"]), 2)
        self.assertTrue(any(v["reason"] == "source_capacity_limit" for v in result.events))
        retained = e.step(now(1.8)); self.assertEqual(len(retained.agents), 1)
        self.assertEqual(flattened(retained)[0]["history"], [])
        expired = e.step(now(2.0)); self.assertEqual(len(expired.agents), 0)
        self.assertTrue(any(v["reason"] == "retention_expired" for v in expired.events))

    def test_source_type_reset_derived_bootstrap_and_invalid_context(self):
        e = engine(); e.step(now(1), [packet([track()])])
        changed = track(t=1.1, category="pedestrian")
        r = e.step(now(1.1), [packet([changed])])
        self.assertEqual(len(flattened(r)[0]["history"]), 1)
        self.assertTrue(any(v["reason"] == "source_identity_reset" for v in r.events))
        self.assertTrue(any(v["reason"] == "unified_retired" for v in r.events))
        derived = replace(track(), state_time=1.2, x=2., observed_mask=False)
        r = engine().step(now(1.2), [packet([derived], available=1.2)])
        self.assertEqual(flattened(r)[0]["history"][0]["kind"], "derived_state")
        self.assertEqual(flattened(r)[0]["last_seen_time"], 1.)
        with self.assertRaises(ValueError): e.step(now(1.))
        with self.assertRaises(ValueError): e.step(Timestamp(1.2, "wrong-episode"))
        r = engine().step(now(1), [packet([track(role="ego")])])
        self.assertFalse(r.agents); self.assertTrue(any(v["reason"] == "ego_excluded" for v in r.events))

    def test_n08_receive_gate_and_new_episode_clears_receiver_highwater(self):
        policy = ReceiverPolicy("edge", "map-A", "1", "map-right-handed", DOMAIN, {"vehicle": "session-1"})
        tr = track()
        msg = ObservationMessage("1.0", "observation", "vehicle", "session-1", 0, "edge", "map-A", "1",
                                 "map-right-handed", 0, DOMAIN, 1, 1, 1, 1, 2, (tr,))
        record = ReceivedMessage(msg, 1.1, 1.2, DOMAIN)
        e = engine(receiver_policy=policy)
        self.assertFalse(e.step(now(1.15), received=[record]).agents)
        self.assertEqual(len(e.step(now(1.2), received=[record]).agents), 1)
        first = e.step(now(1.2))
        self.assertEqual(flattened(first)[0]["history"][0]["packet_available_time"], 1.2)
        repeated = e.step(now(1.3), received=[record])
        self.assertEqual(flattened(repeated)[0]["measurement_updates"], 1)
        self.assertTrue(any(v["reason"] == "receive_duplicate_or_out_of_order" for v in repeated.events))
        e.begin_episode("episode-2", DOMAIN, receiver_policy=policy)
        self.assertEqual(len(e.step(now(1.2), received=[record]).agents), 1)

    def test_conflicting_received_replays_reject_all_independent_of_order(self):
        policy = ReceiverPolicy("edge", "map-A", "1", "map-right-handed", DOMAIN, {"vehicle": "session-1"})
        first = ObservationMessage("1.0", "observation", "vehicle", "session-1", 0, "edge", "map-A", "1",
                                   "map-right-handed", 0, DOMAIN, 1, 1, 1, 1, 2, (track(),))
        second = replace(first, tracks=(track(x=100),))
        records = [ReceivedMessage(msg, 1.1, 1.2, DOMAIN) for msg in (first, second)]
        for order in (records, records[::-1]):
            e = engine(receiver_policy=policy)
            result = e.step(now(1.2), received=order)
            self.assertFalse(result.agents)
            self.assertFalse(e.receiver_state.sequences)
            self.assertEqual(sum(v["reason"] == "conflicting_message_duplicate" for v in result.events), 2)


if __name__ == "__main__": unittest.main()
