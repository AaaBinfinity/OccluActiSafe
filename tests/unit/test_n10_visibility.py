"""B02/B05/B07在N10的未知区、地图几何与输入异常子范围。"""

from dataclasses import replace
import math
import unittest
import numpy as np
from occluactisafe.common.coordinates import Pose2D
from occluactisafe.perception.visibility import (
    GridSpec, KnownObstacle, SensorView, FreeSpaceEvidence, OcclusionRiskRegion, build_visibility, assess_constraints,
)
from occluactisafe.perception.participants import ParticipantGeometry
from occluactisafe.risk.obb import OBB
from occluactisafe.risk.occupancy import PoseTrajectory

SPEC = GridSpec((-4, -4, 4, 4), .5)
CAR = ParticipantGeometry("vehicle", 2, 1)


def view(**kwargs):
    values = dict(source_id="V", pose=Pose2D(-3, 0, 0), fov_rad=math.pi / 2, range_m=10, quality=1,
                  observation_time=0, available_time=0, valid_until=1, clock_domain="sim", coordinate_frame_id="map")
    return SensorView(**(values | kwargs))


def obstacle(**kwargs):
    return KnownObstacle(**(dict(obstacle_id="known-wall", box=OBB((0, 0), 0, .5, 4), available_time=0,
                                 clock_domain="sim", coordinate_frame_id="map") | kwargs))


def build(**kwargs):
    return build_visibility(SPEC, now=0, clock_domain="sim", coordinate_frame_id="map", **kwargs)


def proof(source="V", **kwargs):
    return FreeSpaceEvidence(**(dict(source_id=source, mask=np.ones(SPEC.shape, bool), observation_time=0, available_time=0,
                                   valid_until=1, clock_domain="sim", coordinate_frame_id="map") | kwargs))


def ego(x=-2, y=0, end=.5):
    return PoseTrajectory((0., end), (Pose2D(x, y, 0),) * 2, "sim", "map")


class VisibilityTests(unittest.TestCase):
    def test_fixed_wall_separates_visible_occluded_and_outside(self):
        k = build(views=(view(),), obstacles=(obstacle(),))
        self.assertEqual(k.visibility[8, 2], 1)  # (-2.75,.25)
        self.assertEqual(k.visibility[8, 13], 2)  # (2.75,.25)墙后
        self.assertEqual(k.visibility[15, 0], 0)  # FOV外
        self.assertTrue(k.occupied.any())
        self.assertFalse(k.certified_free.any())
        self.assertEqual(len(k.regions), 3 * np.prod(SPEC.shape))

    def test_b02_unavailable_future_observations_do_not_change_current_output(self):
        current = build(views=(view(),), obstacles=(obstacle(),))
        for x in (-1e9, 0, 1e9):
            future = view(source_id="future", pose=Pose2D(x, -x, 0), observation_time=2, available_time=2, valid_until=3)
            future_wall = obstacle(obstacle_id="future-wall", available_time=2, box=OBB((x, x), 0, 1, 1))
            k = build(views=(view(), future), obstacles=(obstacle(), future_wall), free_evidence=(proof("future", observation_time=2, available_time=2, valid_until=3),))
            np.testing.assert_array_equal(k.visibility, current.visibility)
            np.testing.assert_array_equal(k.occupied, current.occupied)
            np.testing.assert_array_equal(k.certified_free, current.certified_free)
            self.assertEqual(k.regions, current.regions)
            self.assertEqual(assess_constraints(ego(), CAR, k, 0, .5), assess_constraints(ego(), CAR, current, 0, .5))

    def test_b02_hidden_gt_is_not_an_input(self):
        with self.assertRaises(TypeError): build(hidden_targets=[(1, 2)])

    def test_no_detections_or_geometric_visibility_cannot_certify_free(self):
        k = build(views=(view(fov_rad=2 * math.pi),), drivable=np.ones(SPEC.shape, bool), closed_domain=True)
        self.assertTrue(np.all(k.visibility == 1))
        self.assertFalse(k.certified_free.any())
        self.assertIn("unknown_reachable_region", assess_constraints(ego(), CAR, k, 0, .5)["reasons"])

    def test_both_ends_blind_retain_constraints(self):
        k = build(views=(), drivable=np.ones(SPEC.shape, bool))
        result = assess_constraints(ego(), CAR, k, 0, .5)
        self.assertIn("unknown_reachable_region", result["reasons"])
        self.assertIn("outside_domain_unknown", result["reasons"])
        self.assertGreater(result["regions_total"], 0)
        self.assertEqual(result["regions_tested"] + result["regions_unexamined"], result["regions_total"])

    def test_free_evidence_bound_to_source_time_scope_and_quality(self):
        v = view(); wall = obstacle()
        k = build(views=(v,), obstacles=(wall,), free_evidence=(proof(),))
        self.assertTrue(np.all(~k.certified_free | ((k.visibility == 1) & ~k.occupied)))
        self.assertTrue(k.certified_free.any())
        for invalid in (proof("other"), proof(observation_time=-.1), proof(available_time=.1), proof(clock_domain="other")):
            self.assertFalse(build(views=(v,), free_evidence=(invalid,)).certified_free.any())
        for bad_view in (replace(v, quality=.1), replace(v, valid_until=-0.01, observation_time=-1, available_time=-1),
                         replace(v, clock_domain="other")):
            self.assertFalse(build(views=(bad_view,), free_evidence=(proof(),)).certified_free.any())

    def test_multisource_visibility_union_not_double_count(self):
        a = view(); b = view(source_id="I", pose=Pose2D(3, 0, math.pi))
        k = build(views=(a, b), obstacles=(obstacle(),))
        self.assertEqual(k.visibility[8, 13], 1)
        self.assertEqual(k.accepted_sources, ("I", "V"))
        self.assertEqual(len(k.regions), 3 * np.prod(SPEC.shape))

    def test_unknown_reach_growth_is_hand_calculable(self):
        r = OcclusionRiskRegion("r", (0, 0), .5, 0, "pedestrian", 2, 1, .3, "fixture")
        self.assertAlmostEqual(r.reachable_radius(2), .5 + .3 + 4 + 2)
        with self.assertRaises(ValueError): r.reachable_radius(-.1)
        with self.assertRaises(ValueError): replace(r, speed_bound_mps=-1)
        with self.assertRaises(ValueError): replace(r, body_radius_m=float("nan"))

    def test_b07_known_occupied_full_body_and_drivable_boundary(self):
        v = view(fov_rad=2 * math.pi); ones = np.ones(SPEC.shape, bool)
        k = build(views=(v,), obstacles=(obstacle(),), free_evidence=(proof(),), drivable=ones, closed_domain=True)
        self.assertIn("known_occupied_or_non_drivable", assess_constraints(ego(x=0), CAR, k, 0, .5)["reasons"])
        k = build(views=(v,), free_evidence=(proof(),), drivable=ones, closed_domain=True)
        self.assertIn("map_boundary", assess_constraints(ego(x=3.5), CAR, k, 0, .5)["reasons"])
        drive = ones.copy(); drive[8, 4] = False
        k = build(views=(v,), free_evidence=(proof(),), drivable=drive, closed_domain=True)
        self.assertIn("known_occupied_or_non_drivable", assess_constraints(ego(x=-2), CAR, k, 0, .5)["reasons"])

    def test_invalid_grid_masks_and_domain(self):
        for spec_args in (((0, 0, 1, 1), .3), ((0, 0, 1000, 1000), .5), ((1, 0, 0, 1), .5)):
            with self.assertRaises(ValueError): GridSpec(*spec_args)
        with self.assertRaises(ValueError): build(drivable=np.ones(SPEC.shape))
        with self.assertRaises(ValueError): build(views=(view(), view()))
        with self.assertRaises(ValueError): view(fov_rad=7)
        with self.assertRaises(ValueError): assess_constraints(ego(), CAR, build(), -.1, .5)
        with self.assertRaisesRegex(ValueError, "陈旧证空"): assess_constraints(ego(end=2), CAR, build(), 1, 2)
        other = replace(ego(), clock_domain="other")
        with self.assertRaises(ValueError): assess_constraints(other, CAR, build(), 0, .5)


if __name__ == "__main__": unittest.main()
