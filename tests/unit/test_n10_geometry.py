"""N10 G01～G05及B05纯几何断言；机制夹具无真值事件。"""

import math
import unittest
from occluactisafe.common.coordinates import Pose2D
from occluactisafe.perception.participants import ParticipantGeometry
from occluactisafe.risk.obb import OBB, Circle, intersects, segment_obb_interval
from occluactisafe.risk.occupancy import PoseTrajectory, IntervalConfig, first_conflict_time, boundary_violation


def path(points, times=(0., 3.), mask=None, domain="sim-n10"):
    return PoseTrajectory(times, tuple(Pose2D(*p) for p in points), domain, "map", mask)


CAR = ParticipantGeometry("vehicle", length=4, width=2)
SMALL = ParticipantGeometry("vehicle", length=2, width=2)


class GeometryTests(unittest.TestCase):
    def test_g01_identical_and_symmetry(self):
        a = OBB((0, 0), 0, 4, 2)
        for b in (a, OBB((1, .5), .4, 4, 2), OBB((6, 6), .8, 4, 2)):
            self.assertEqual(intersects(a, b), intersects(b, a))
        self.assertTrue(intersects(a, a))

    def test_g02_separation_and_edge(self):
        a = OBB((0, 0), 0, 4, 2)
        self.assertFalse(intersects(a, OBB((4.1, 0), 0, 4, 2)))
        self.assertTrue(intersects(a, OBB((4., 0), 0, 4, 2)))

    def test_g03_rotated(self):
        a = OBB((0, 0), 0, 4, 2)
        self.assertTrue(intersects(a, OBB((0, 0), math.pi / 2, 4, 2)))
        self.assertFalse(intersects(a, OBB((10, 10), math.pi / 2, 4, 2)))

    def test_g04_invalid_geometry(self):
        for values in (((0, 0), 0, 0, 2), ((0, 0), 0, -1, 2), ((0, 0), 0, 4, float("nan")),
                       ((0, 0), float("inf"), 4, 2), ((float("nan"), 0), 0, 4, 2), ((0, 0), 0, 4, 2, -.1)):
            with self.subTest(values=values), self.assertRaises(ValueError): OBB(*values)
        for radius in (0, -1, float("nan")):
            with self.assertRaises(ValueError): Circle((0, 0), radius)
        with self.assertRaises(ValueError): intersects(Circle((0, 0), 1), Circle((0, 0), 1), -1)

    def test_g05_between_samples_crossing(self):
        a = path(((0, 0, 0), (0, 0, 0)), (0., .5))
        b = path(((-5, 0, 0), (5, 0, 0)), (0., .5))
        self.assertFalse(intersects(SMALL.at(a.at(0)), SMALL.at(b.at(0))))
        self.assertFalse(intersects(SMALL.at(a.at(.5)), SMALL.at(b.at(.5))))
        result = first_conflict_time(a, SMALL, b, SMALL, 0., .5)
        self.assertTrue(result.potential_contact)
        self.assertLessEqual(result.event["interval_start"], .15)
        self.assertLessEqual(result.event["interval_end"] - result.event["interval_start"], .01)
        self.assertIsNone(result.event["actual_collision"])
        self.assertTrue(result.algorithm_complete)

    def test_b05_circle_corner_and_cyclist(self):
        box = OBB((0, 0), 0, 4, 2)
        self.assertTrue(intersects(box, Circle((2.3, 1.4), .5)))
        self.assertFalse(intersects(box, Circle((2.31, 1.4), .5)))
        self.assertTrue(intersects(Circle((0, 0), .5), Circle((1, 0), .5)))
        self.assertFalse(intersects(Circle((0, 0), .5), Circle((1.01, 0), .5)))
        cyclist = ParticipantGeometry("cyclist", length=1.8, width=.6)
        self.assertIsInstance(cyclist.at(Pose2D(0, 0, 0)), OBB)
        self.assertIsInstance(ParticipantGeometry("pedestrian", radius=.3).at(Pose2D(0, 0, 0)), Circle)
        still = path(((0, 0, 0),) * 2, (0., .5))
        crossing = path(((-5, 0, 0), (5, 0, 0)), (0., .5))
        for geometry, half in ((cyclist, .9), (ParticipantGeometry("pedestrian", radius=.3), .3)):
            with self.subTest(category=geometry.category):
                self.assertFalse(intersects(CAR.at(still.at(0)), geometry.at(crossing.at(0))))
                self.assertFalse(intersects(CAR.at(still.at(.5)), geometry.at(crossing.at(.5))))
                event = first_conflict_time(still, CAR, crossing, geometry, 0., .5).event
                analytic = (5. - 2. - half) / 20.
                self.assertLessEqual(event["interval_start"], analytic)
                self.assertGreaterEqual(event["interval_end"], analytic)
                self.assertIsNone(event["actual_collision"])

    def test_b05_explicit_reference_offset(self):
        rear = ParticipantGeometry("vehicle", 4, 2, reference_point="rear_axle", center_offset_m=1.5)
        shape = rear.at(Pose2D(2, 3, math.pi / 2))
        self.assertAlmostEqual(shape.center[0], 2)
        self.assertAlmostEqual(shape.center[1], 4.5)
        self.assertAlmostEqual(rear.motion_radius, CAR.motion_radius + 1.5)
        for kwargs in ({"reference_point": "rear_axle"}, {"center_offset_m": 1}, {"category": "unknown"}):
            with self.assertRaises(ValueError): ParticipantGeometry(**({"category": "vehicle", "length": 4, "width": 2} | kwargs))

    def test_trajectory_shortest_yaw_and_mask_hole(self):
        p = path(((0, 0, math.radians(170)), (2, 0, math.radians(-170))), (0., 2.))
        self.assertAlmostEqual(abs(p.at(1).yaw), math.pi)
        self.assertEqual(p.at(1).x, 1)
        p = path(((0, 0, 0),) * 4, (0., 1., 2., 3.), (True, True, False, True))
        self.assertEqual(p.contiguous_end(.5), 1.)
        self.assertIsNone(p.contiguous_end(1.5))
        with self.assertRaises(ValueError): p.at(1.5)
        with self.assertRaises(ValueError): p.at(4.)
        with self.assertRaises(ValueError): first_conflict_time(p, CAR, p, CAR, 0., 3.)

    def test_trajectory_invalid_and_domain(self):
        for times, mask in (((0, 0), None), ((0, float("nan")), None), ((0, 1), (1, 1))):
            with self.assertRaises(ValueError): path(((0, 0, 0),) * 2, times, mask)
        a = path(((0, 0, 0),) * 2)
        b = path(((0, 0, 0),) * 2, domain="other")
        with self.assertRaises(ValueError): first_conflict_time(a, CAR, b, CAR, 0, 3)
        with self.assertRaises(ValueError): IntervalConfig(min_interval_s=.02)

    def test_uncertain_near_miss_and_budget_are_not_collision(self):
        a = path(((0, 0, 0),) * 2, (0., .5))
        b = path(((-5, 2.001, 0), (5, 2.001, 0)), (0., .5))
        r = first_conflict_time(a, SMALL, b, SMALL, 0., .5)
        self.assertEqual(r.event["kind"], "potential_overlap")
        self.assertTrue(r.event["uncertain_interval"])
        self.assertIsNone(r.event["sampled_overlap_time"])
        r = first_conflict_time(a, SMALL, b, SMALL, 0., .5, IntervalConfig(max_nodes=1))
        self.assertFalse(r.algorithm_complete)
        self.assertEqual(r.event["reason"], "interval_budget_exhausted")

    def test_clear_interval_and_boundary_full_body(self):
        a = path(((0, 0, 0),) * 2)
        b = path(((-5, 8, 0), (5, 8, 0)))
        self.assertFalse(first_conflict_time(a, CAR, b, CAR, 0, 3).potential_contact)
        bounds = (-5, -5, 5, 5)
        self.assertIsNone(boundary_violation(a, CAR, 0, 3, bounds).event)
        c = path(((4, 0, 0),) * 2)
        self.assertIsNotNone(boundary_violation(c, CAR, 0, 3, bounds).event)

    def test_rotating_long_body_and_ray(self):
        a = path(((0, 0, 0), (0, 0, math.pi / 2)), (0., 1.))
        b = path(((1.8, 1.8, 0),) * 2, (0., 1.))
        long = ParticipantGeometry("vehicle", 6, .2)
        dot = ParticipantGeometry("pedestrian", radius=.2)
        self.assertTrue(first_conflict_time(a, long, b, dot, 0, 1).potential_contact)
        self.assertEqual(segment_obb_interval((-5, 0), (5, 0), OBB((0, 0), 0, 2, 2)), (.4, .6))
        self.assertIsNone(segment_obb_interval((-5, 2), (5, 2), OBB((0, 0), 0, 2, 2)))


if __name__ == "__main__": unittest.main()
