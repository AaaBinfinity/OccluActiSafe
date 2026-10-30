"""N08机制断言；全部是构造消息，不是真实数据/链路或避撞实验。"""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import unittest

from occluactisafe.common.clock import Timestamp
from occluactisafe.common.schemas import ObservationMessage, PredictionMessage, ReceivedMessage
from occluactisafe.communication.channel import SimChannel
from occluactisafe.communication.protocol import (
    ChannelConfig, Disturbance, decode_message, disturbance_for, encode_message, message_key, service_seconds,
)
from occluactisafe.communication.validation import (
    ReceiverPolicy, ReceiverState, query_prediction, validate_received,
)

DOMAIN = "sim-n08-episode"
ROOT = Path(__file__).resolve().parents[2]


def stamp(t):
    return Timestamp(t, DOMAIN)


def observation(sequence=0, sent=12., observed=12.):
    # 独立复用N02金样并显式替换时序，与运行诊断的构造器分开。
    d = json.loads((ROOT / "tests/fixtures/n02/golden_observation.json").read_text(encoding="utf-8"))
    d.update(sequence=sequence, frame_id=0, clock_domain=DOMAIN, observation_time=observed,
             source_available_time=observed, generated_time=sent, transmitted_time=sent, valid_until=100.)
    d["tracks"][0].update(clock_domain=DOMAIN, observation_time=observed, last_seen_time=observed,
                          source_available_time=observed, state_time=observed, x=0., y=0., vx=0., vy=10.)
    return ObservationMessage.from_dict(d)


def prediction(sequence=0, sent=12., anchor=12.):
    d = json.loads((ROOT / "tests/fixtures/n02/golden_prediction.json").read_text(encoding="utf-8"))
    times = [anchor + (i + 1) / 10 for i in range(40)]
    d.update(sequence=sequence, frame_id=0, clock_domain=DOMAIN, anchor_time=anchor, input_cutoff=anchor,
             completed_time=sent, generated_time=sent, transmitted_time=sent, valid_until=100.,
             future_times=times, future_end=times[-1])
    d["targets"] = [{"track": observation(observed=anchor, sent=anchor).tracks[0].to_dict(),
                      "trajectories": [[[0., 10 * (t - anchor)] for t in times]],
                      "weights": [1.], "valid_mask": [[True] * 40]}]
    return PredictionMessage.from_dict(d)


def policy(**kwargs):
    return ReceiverPolicy("vehicle-A", "map-A", "1", "map-right-handed", DOMAIN,
                          {"roadside-1": "edge-1"}, **kwargs)


def received(p=None, arrival=12.2, usable=12.2):
    return ReceivedMessage(p or prediction(), arrival, usable, DOMAIN)


def padded_message(sequence, sent):
    p = prediction(sequence, sent)
    d = p.to_dict()
    size = len(encode_message(p))
    d["model_hash"] += "X" * (38400 - size)
    p = PredictionMessage.from_dict(d)
    assert len(encode_message(p)) == 38400
    return p


class ProtocolTests(unittest.TestCase):
    def test_utf8_roundtrip_and_strict_invalid_wire(self):
        for p in (observation(), prediction()):
            d = p.to_dict(); d["session_id"] = "会话一"
            p = type(p).from_dict(d)
            raw = encode_message(p)
            self.assertEqual(decode_message(raw), p)
            self.assertGreater(len(raw), len(raw.decode("utf-8")))
        for raw in (b'\xff', b'[]', b'{}', b'{"message_type":"health"}',
                    b'{"message_type":"observation","message_type":"prediction"}',
                    b'{"message_type":"prediction","x":NaN}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                decode_message(raw)
        for mutation in ({"truth_actor_id": "hidden"}, {"arrival_time": 12.2}, {"schema_version": "2"}):
            d = prediction().to_dict(); d.update(mutation)
            with self.assertRaises(ValueError):
                decode_message(json.dumps(d))

    def test_service_units_and_configuration_rejections(self):
        self.assertEqual(service_seconds(38400, 1_000_000), .3072)
        self.assertEqual(service_seconds(38400, None), 0.)
        for kwargs in ({"capacity_bps": 0}, {"loss_rate": 1.1}, {"jitter_s": -1},
                       {"max_pending_messages": True}, {"max_queue_wait_s": float("nan")},
                       {"blackout_intervals": [[2, 1]]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ChannelConfig("I2V", DOMAIN, **kwargs)
        for args in ((True, 1e6), (38400, -1), (38400, float("inf"))):
            with self.assertRaises(ValueError): service_seconds(*args)


class ChannelTests(unittest.TestCase):
    def test_v01_exact_boundary_once_and_v02_stable_ties(self):
        c = SimChannel(ChannelConfig("I2V", DOMAIN))
        for seq in (3, 1, 2): self.assertTrue(c.send(prediction(seq), stamp(12)))
        self.assertEqual(c.deliver(stamp(12.19)), [])
        self.assertEqual([r.payload.sequence for r in c.deliver(stamp(12.20))], [3, 1, 2])
        self.assertEqual(c.deliver(stamp(12.20)), [])
        self.assertEqual(c.pending_count, 0)
        self.assertEqual(c.stats["delivered_targets"], 3)

    def test_v02_total_loss_and_lost_frames_consume_capacity(self):
        config = ChannelConfig("I2V", DOMAIN, capacity_bps=1e6, loss_rate=1, max_queue_wait_s=1)
        c = SimChannel(config)
        self.assertFalse(c.send(padded_message(0, 12), stamp(12)))
        self.assertFalse(c.send(padded_message(1, 12.1), stamp(12.1)))
        self.assertAlmostEqual(c.events[-1]["queue_wait_s"], .2072)
        self.assertEqual(c.deliver(stamp(20)), [])
        self.assertEqual(c.stats["transmitted_bytes"], 76800)
        self.assertEqual(c.stats["random_loss"], 2)
        self.assertEqual(c.pending_count, 0)

    def test_identity_disturbance_and_explicit_table_replay(self):
        cfg = ChannelConfig("I2V", DOMAIN, jitter_s=.15, loss_rate=.3)
        keys = [message_key(prediction(seq)) for seq in range(20)]
        a = {key: disturbance_for(cfg, key) for key in keys}
        b = {key: disturbance_for(cfg, key) for key in reversed(keys)}
        self.assertEqual(a, b)
        self.assertNotEqual(a[keys[0]], disturbance_for(replace(cfg, direction="V2I"), keys[0]))
        table = {keys[0]: Disturbance(.4, False), keys[1]: Disturbance(.1, False)}
        for _ in range(2):
            c = SimChannel(cfg, perturbations=table)
            c.send(prediction(0), stamp(12)); c.send(prediction(1), stamp(12))
            self.assertEqual([r.payload.sequence for r in c.deliver(stamp(12.1))], [1])
            self.assertEqual([r.payload.sequence for r in c.deliver(stamp(12.4))], [0])

    def test_capacity_10hz_actual_bytes_queue_timeout_and_bound(self):
        c = SimChannel(ChannelConfig("I2V", DOMAIN, capacity_bps=1e6))
        for i in range(20):
            now = 12 + i / 10
            c.deliver(stamp(now))
            c.send(padded_message(i, now), stamp(now))
        c.deliver(stamp(30))
        scheduled = [e for e in c.events if e["event"] == "scheduled"]
        self.assertTrue(any(e["queue_wait_s"] > 0 for e in scheduled))
        self.assertGreater(c.stats["queue_timeout"], 0)
        self.assertTrue(all(e["queue_wait_s"] <= .5 for e in scheduled))
        self.assertTrue(all(e["service_s"] == .3072 for e in scheduled))
        self.assertEqual(c.stats["transmitted_bytes"], 38400 * len(scheduled))
        self.assertEqual(c.stats["offered_bytes"], 38400 * 20)
        c = SimChannel(ChannelConfig("I2V", DOMAIN, max_pending_messages=1))
        self.assertTrue(c.send(prediction(), stamp(12)))
        self.assertFalse(c.send(prediction(1), stamp(12)))
        self.assertEqual(c.stats["queue_full"], 1)
        self.assertEqual(c.pending_count, 1)

    def test_deadline_drop_does_not_reserve_capacity(self):
        c = SimChannel(ChannelConfig("I2V", DOMAIN, capacity_bps=1e6))
        p = padded_message(0, 12).to_dict(); p["valid_until"] = 12.1
        self.assertFalse(c.send(PredictionMessage.from_dict(p), stamp(12)))
        c.send(prediction(1), stamp(12))
        self.assertEqual(c.events[-1]["queue_wait_s"], 0)
        self.assertEqual(c.stats["deadline_unreachable"], 1)

    def test_invalid_direction_time_and_payload_never_enter_heap(self):
        c = SimChannel(ChannelConfig("V2I", DOMAIN))
        for p, now in ((prediction(), stamp(12)), (observation(), stamp(12.1)),
                       (b'{}', stamp(12.1))):
            with self.assertRaises(ValueError): c.send(p, now)
            self.assertEqual(c.pending_count, 0)
        with self.assertRaises(ValueError): c.deliver(Timestamp(13, "other-episode"))
        with self.assertRaises(ValueError): c.deliver(stamp(11))

    def test_v04_channel_subscope_blackout_and_no_invented_targets(self):
        c = SimChannel(ChannelConfig("I2V", DOMAIN, blackout_intervals=((12.1, 12.3),)))
        for i in range(4):
            c.send(prediction(i, 12 + i / 10), stamp(12 + i / 10))
        out = c.deliver(stamp(13))
        self.assertEqual([r.payload.sequence for r in out], [0, 3])
        self.assertEqual(c.stats["blackout"], 2)
        self.assertTrue(all(len(r.payload.targets) == 1 for r in out))
        self.assertTrue(all(r.payload.targets[0].track.last_seen_time == 12 for r in out))


class ReceiveTests(unittest.TestCase):
    def test_arrival_usable_separate_and_reject_does_not_consume_sequence(self):
        p = policy(); s = ReceiverState(p); r = received(arrival=12.0, usable=12.2)
        self.assertEqual(validate_received(r, stamp(12.1), p, state=s).reason, "not_usable")
        self.assertFalse(s.sequences)
        self.assertTrue(validate_received(r, stamp(12.2), p, state=s).accepted)
        self.assertEqual(validate_received(r, stamp(12.2), p, state=s).reason, "duplicate_or_out_of_order")
        r = received(arrival=12.2, usable=12.3)
        self.assertEqual(validate_received(r, stamp(12.1), p).reason, "not_arrived")

    def test_identity_episode_map_fields_ttl_and_future_anchor_rejections(self):
        p = policy()
        for field, value, reason in (("source_id", "stranger", "unknown_source"),
                                     ("session_id", "old-session", "session_mismatch"),
                                     ("recipient_id", "other", "recipient_id_mismatch"),
                                     ("map_id", "other", "map_id_mismatch"),
                                     ("map_version", "2", "map_version_mismatch"),
                                     ("coordinate_frame_id", "other", "coordinate_frame_id_mismatch")):
            d = prediction().to_dict(); d[field] = value
            if field == "coordinate_frame_id": d["targets"][0]["track"][field] = value
            r = received(PredictionMessage.from_dict(d))
            result = validate_received(r, stamp(12.2), p)
            self.assertEqual(result.reason, reason); self.assertIsNone(result.received)
        self.assertEqual(validate_received(received(), Timestamp(12.2, "other-episode"), p).reason,
                         "clock_domain_mismatch")
        d = prediction().to_dict(); d["valid_until"] = 12.2
        r = received(PredictionMessage.from_dict(d))
        self.assertTrue(validate_received(r, stamp(12.2), p).accepted)
        self.assertEqual(validate_received(r, stamp(12.2001), p).reason, "expired")
        d = prediction().to_dict(); d["anchor_time"] = 13
        d["future_times"] = [13 + i / 10 for i in range(1, 41)]; d["future_end"] = 17
        self.assertEqual(validate_received(received(PredictionMessage.from_dict(d)), stamp(12.2), p).reason,
                         "future_information")

    def test_c06_newer_then_old_and_explicit_session_restart(self):
        p = policy(); s = ReceiverState(p)
        self.assertTrue(validate_received(received(prediction(9)), stamp(12.2), p, state=s).accepted)
        self.assertEqual(validate_received(received(prediction(8)), stamp(12.3), p, state=s).reason,
                         "duplicate_or_out_of_order")
        d = prediction(10).to_dict(); d["anchor_time"] = 11.9; d["input_cutoff"] = 11.9
        d["targets"][0]["track"].update(observation_time=11.9, last_seen_time=11.9,
                                         source_available_time=11.9, state_time=11.9)
        self.assertEqual(validate_received(received(PredictionMessage.from_dict(d)), stamp(12.3), p, state=s).reason,
                         "reference_time_regression")
        s.activate_session("roadside-1", "edge-2")
        self.assertEqual(validate_received(received(), stamp(12.3), p, state=s).reason, "session_mismatch")
        d = prediction().to_dict(); d["session_id"] = "edge-2"
        self.assertTrue(validate_received(received(PredictionMessage.from_dict(d)), stamp(12.3), p, state=s).accepted)
        with self.assertRaises(ValueError): s.activate_session("stranger", "x")

    def test_v03_c06_absolute_query_and_ttl_is_not_coverage(self):
        r = received(); p = policy()
        x, y = query_prediction(r, now=stamp(12.2), when=stamp(12.5), policy=p)
        self.assertEqual((x, y), (0, 5))
        self.assertEqual(r.payload.anchor_time, 12)
        short = validate_received(r, stamp(12.9), p, plan_start=stamp(13.1))
        self.assertEqual(short.reason, "coverage_insufficient")
        self.assertFalse(short.coverage["complete"])
        self.assertGreater(r.payload.valid_until, 16.1)
        self.assertEqual(validate_received(r, stamp(12.2), p, plan_start=stamp(12.1)).reason, "plan_in_past")
        with self.assertRaises(ValueError): query_prediction(r, now=stamp(12.1), when=stamp(12.5), policy=p)

    def test_partial_modes_holes_empty_duplicate_targets_and_anchor_state(self):
        p = policy()
        d = json.loads(prediction().to_json()); t = d["targets"][0]
        t["trajectories"].append(deepcopy(t["trajectories"][0]))
        t["weights"] = [.9, .1]; t["valid_mask"].append([True] * 20 + [False] * 20)
        self.assertEqual(validate_received(received(PredictionMessage.from_dict(d)), stamp(12.2), p).reason,
                         "coverage_insufficient")
        t["weights"] = [1., 0.]
        self.assertTrue(validate_received(received(PredictionMessage.from_dict(d)), stamp(12.2), p).accepted)
        t["valid_mask"][0][10] = False
        self.assertEqual(validate_received(received(PredictionMessage.from_dict(d)), stamp(12.2), p).reason,
                         "coverage_insufficient")
        d = prediction().to_dict(); d["targets"] = []
        self.assertEqual(validate_received(received(PredictionMessage.from_dict(d)), stamp(12.2), p).reason,
                         "coverage_insufficient")
        d = prediction().to_dict(); d["targets"] *= 2
        self.assertEqual(validate_received(received(PredictionMessage.from_dict(d)), stamp(12.2), p).reason,
                         "duplicate_target_identity")
        r = received(arrival=12, usable=12)
        point = query_prediction(r, now=stamp(12), when=stamp(12.05), policy=p)
        self.assertEqual(point[0], 0)
        self.assertAlmostEqual(point[1], .5, delta=1e-12)
        d = prediction().to_dict(); d["targets"][0]["track"]["state_time"] = 11.9
        d["targets"][0]["track"].update(observation_time=11.9, last_seen_time=11.9)
        with self.assertRaises(ValueError):
            query_prediction(received(PredictionMessage.from_dict(d), 12, 12), now=stamp(12), when=stamp(12.05), policy=p)

    def test_retransmission_never_refreshes_old_observation_or_state(self):
        d = prediction(sequence=99, sent=15).to_dict()
        r = received(PredictionMessage.from_dict(d), 15.2, 15.2)
        self.assertEqual(validate_received(r, stamp(15.2), policy()).reason, "observation_too_old")
        self.assertEqual(r.payload.targets[0].track.last_seen_time, 12)
        self.assertEqual(validate_received(received(), stamp(12.2), policy(max_state_age_s=.1)).reason,
                         "state_too_old")
        self.assertEqual(validate_received(received(), stamp(12.2), policy(max_anchor_age_s=.1)).reason,
                         "anchor_too_old")

    def test_validation_rejects_forged_message_and_no_highwater_on_short_coverage(self):
        p = policy(); s = ReceiverState(p)
        short = validate_received(received(prediction(100)), stamp(12.9), p, state=s, plan_start=stamp(13.1))
        self.assertFalse(short.accepted); self.assertFalse(s.sequences)
        self.assertTrue(validate_received(received(prediction(2)), stamp(12.9), p, state=s).accepted)
        self.assertEqual(validate_received(received(prediction(3)), stamp(12.8), p, state=s).reason,
                         "decision_time_regression")
        r = received(); object.__setattr__(r.payload, "model_hash", "")
        self.assertEqual(validate_received(r, stamp(12.2), p).reason, "invalid_message")

    def test_two_directions_independent_service_and_processing_statistics(self):
        up = SimChannel(ChannelConfig("V2I", DOMAIN, propagation_delay_s=.1, processing_delay_s=.05, capacity_bps=1e6))
        down = SimChannel(ChannelConfig("I2V", DOMAIN, processing_delay_s=.2, capacity_bps=1e6))
        up.send(observation(), stamp(12)); down.send(prediction(), stamp(12))
        self.assertEqual(up.events[0]["queue_wait_s"], 0)
        self.assertEqual(down.events[0]["queue_wait_s"], 0)
        ru = up.deliver(stamp(13))[0]; rd = down.deliver(stamp(13))[0]
        self.assertAlmostEqual(ru.usable_time - ru.arrival_time, .05)
        self.assertAlmostEqual(rd.usable_time - rd.arrival_time, .2)
        self.assertEqual(up.stats["delivered_bytes"], len(encode_message(observation())))
        self.assertEqual(down.stats["delivered_bytes"], len(encode_message(prediction())))
        self.assertEqual(up.stats["delivered_targets"], 1)
        self.assertGreater(up.stats["decode_wall_seconds"], 0)
        self.assertTrue(validate_received(ru, stamp(13), ReceiverPolicy("roadside-1", "map-A", "1",
            "map-right-handed", DOMAIN, {"vehicle-A": "session-1"})).accepted)


if __name__ == "__main__":
    unittest.main()
