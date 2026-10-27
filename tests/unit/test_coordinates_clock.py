"""N02 坐标与时钟的可手算及边界断言。"""

import math
import random
import unittest

from occluactisafe.common.clock import (
    Timestamp, require_available, require_same_domain,
    validate_observation_times, validate_prediction_times, validate_receive_times,
)
from occluactisafe.common.coordinates import (
    Pose2D, carla_to_project, carla_vector_to_project, local_to_world,
    normalize_yaw, project_to_carla, shift_reference_point, world_to_local,
)


def ts(value, domain="simulation:episode-1"):
    return Timestamp(value, domain)


class CoordinateChecks(unittest.TestCase):
    def test_known_world_local_transform_and_axes(self):
        pose = Pose2D(10.0, 20.0, math.pi / 2)
        x, y = local_to_world(3.0, 2.0, pose)
        self.assertAlmostEqual(x, 8.0, delta=1e-12)
        self.assertAlmostEqual(y, 23.0, delta=1e-12)
        self.assertAlmostEqual(world_to_local(x, y, pose)[0], 3.0, delta=1e-12)
        self.assertAlmostEqual(world_to_local(x, y, pose)[1], 2.0, delta=1e-12)

    def test_round_trip_error_at_metre_scale(self):
        rng = random.Random(20261027)
        max_error = 0.0
        for _ in range(1000):
            origin = Pose2D(rng.uniform(-1000, 1000), rng.uniform(-1000, 1000), rng.uniform(-math.pi, math.pi))
            original = rng.uniform(-1000, 1000), rng.uniform(-1000, 1000)
            recovered = world_to_local(*local_to_world(*original, origin), origin)
            max_error = max(max_error, *(abs(a - b) for a, b in zip(original, recovered)))
        self.assertLessEqual(max_error, 1e-8, f"最大误差 {max_error} m")

    def test_rear_axle_to_centre_and_inverse(self):
        rear = Pose2D(10, 20, math.pi / 2)
        center = shift_reference_point(rear, 1.5)
        self.assertAlmostEqual(center.x, 10.0, delta=1e-12)
        self.assertAlmostEqual(center.y, 21.5, delta=1e-12)
        recovered = shift_reference_point(center, -1.5)
        self.assertAlmostEqual(recovered.x, rear.x, delta=1e-12)
        self.assertAlmostEqual(recovered.y, rear.y, delta=1e-12)
        self.assertEqual(recovered.yaw, rear.yaw)

    def test_carla_position_velocity_and_yaw_signs(self):
        pose = carla_to_project(4, 5, 90)
        self.assertEqual((pose.x, pose.y), (4, -5))
        self.assertAlmostEqual(pose.yaw, -math.pi / 2, delta=1e-12)
        self.assertEqual(carla_vector_to_project(2, -3), (2, 3))
        x, y, degrees = project_to_carla(pose)
        self.assertEqual((x, y), (4, 5))
        self.assertAlmostEqual(degrees, 90, delta=1e-12)
        self.assertAlmostEqual(normalize_yaw(3 * math.pi), -math.pi, delta=1e-12)

    def test_non_finite_and_invalid_coordinate_values_rejected(self):
        for value in (math.nan, math.inf, -math.inf, True, "1"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError): Pose2D(value, 0, 0)
                with self.assertRaises(ValueError): local_to_world(value, 0, Pose2D(0, 0, 0))
                with self.assertRaises(ValueError): carla_to_project(0, 0, value)


class ClockChecks(unittest.TestCase):
    def test_timestamps_require_finite_seconds_and_nonempty_domain(self):
        for value in (math.nan, math.inf, -math.inf, True, "1"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError): Timestamp(value, "sim")
        for value in ("", "  ", None):
            with self.subTest(domain=value):
                with self.assertRaises(ValueError): Timestamp(0, value)
        with self.assertRaises(ValueError): require_same_domain()
        with self.assertRaises(ValueError): require_same_domain(ts(0), ts(0, "wall"))

    def test_observation_fields_remain_distinct(self):
        validate_observation_times(ts(0), ts(0.1), ts(0.3), ts(0.2))
        for args in ((0.2, 0.1, 0.3, 0.3), (0, 0.2, 0.1, 0.3), (0, 0.2, 0.3, 0.1)):
            with self.subTest(args=args):
                with self.assertRaises(ValueError): validate_observation_times(*(ts(x) for x in args))
        with self.assertRaises(ValueError): validate_observation_times(ts(0), ts(0), ts(0), ts(0, "wall"))

    def test_prediction_fields_and_independent_expiry_and_coverage(self):
        validate_prediction_times(ts(11.9), ts(12), ts(12.1), ts(12.2), ts(16), ts(12.5))
        validate_prediction_times(ts(11.9), ts(12), ts(12.1), ts(12.2), ts(16), ts(20))
        for args in ((12.1, 12, 12.2, 12.3, 16, 13), (12, 12, 12.1, 12.0, 16, 13), (12, 12, 12.1, 12.2, 12, 13)):
            with self.subTest(args=args):
                with self.assertRaises(ValueError): validate_prediction_times(*(ts(x) for x in args))

    def test_receive_order_and_decision_availability(self):
        validate_receive_times(ts(0.8), ts(1.0), ts(1.2))
        with self.assertRaises(ValueError): validate_receive_times(ts(0.8), ts(1.2), ts(1.0))
        with self.assertRaises(ValueError): validate_receive_times(ts(1.1), ts(1.0), ts(1.2))
        # 消息已到达并不等于完成解码；1.1 秒不得读取。
        with self.assertRaises(ValueError): require_available(ts(1.1), usable_time=ts(1.2))
        require_available(ts(1.2), observation_time=ts(1), source_available_time=ts(1.1), state_time=ts(1.2), usable_time=ts(1.2), valid_until=ts(1.2))
        with self.assertRaises(ValueError): require_available(ts(1.2), observation_time=ts(1.3))
        with self.assertRaises(ValueError): require_available(ts(1.2), source_available_time=ts(1.3))
        with self.assertRaises(ValueError): require_available(ts(1.2), state_time=ts(1.3))
        with self.assertRaises(ValueError): require_available(ts(1.3), usable_time=ts(1.2), valid_until=ts(1.2))
        with self.assertRaises(ValueError): require_available(ts(1.2), usable_time=ts(1, "wall"))


if __name__ == "__main__":
    unittest.main()
