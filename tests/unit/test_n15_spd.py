"""N15 SPD 索引路径与地理组泄漏拒绝夹具。"""
from pathlib import Path
import tempfile
import unittest

from occluactisafe.data.adapters.spd import (audit_spd_split, build_spd_split,
                                             normalize_visible_box, resolve_index_path,
                                             resolve_missing_vehicle_locations)
from occluactisafe.data.adapters.carla import extract_visible_instances
from scripts.audit_spd import review_font_identity
from scripts.capture_n15_carla import (calibrate_fixed_roadside_transform,
                                       project_world_points_to_image, summarize_scene_labels,
                                       validate_scene_groups)
from scripts.verify_n15_carla import synchronized_frame, validate_box


class SpdPathTests(unittest.TestCase):
    def test_flat_rgb_uses_only_index_basename_and_keeps_original_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "dataset"
            side = root / "V2X-Seq-SPD" / "vehicle-side"
            image_pkg = root / "V2X-Seq-SPD-vehicle-side-image"
            side.mkdir(parents=True)
            image_pkg.mkdir()
            image = image_pkg / "000123.jpg"
            image.write_bytes(b"fixture-image")
            resolved, method = resolve_index_path(root, "vehicle-side", "image/000123.jpg", kind="image")
            self.assertEqual(resolved, image.resolve())
            self.assertEqual(method, "flat_package_basename")
            self.assertFalse((side / "image").exists())

    def test_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                resolve_index_path(Path(temp), "vehicle-side", "../../outside.jpg", kind="image")


class SpdSplitTests(unittest.TestCase):
    def records(self):
        return [{"side": side, "sequence_id": seq, "frame_id": frame,
                 "intersection_loc": loc, "source_file": path}
                for side, seq, frame, loc, path in (
                    ("vehicle-side", "v1", "001", "road-a", "vehicle/001.jpg"),
                    ("infrastructure-side", "i1", "001", "road-a", "infra/001.jpg"),
                    ("vehicle-side", "v2", "002", "road-b", "vehicle/002.jpg"),
                    ("infrastructure-side", "i2", "002", "road-b", "infra/002.jpg"),
                    ("vehicle-side", "v3", "003", "road-c", "vehicle/003.jpg"),
                    ("infrastructure-side", "i3", "003", "road-c", "infra/003.jpg"),
                    ("vehicle-side", "v4", "004", "", "vehicle/004.jpg"),)]

    def test_three_roles_deterministic_and_unknown_quarantined(self):
        rows = self.records()
        a = build_spd_split(rows, seed=7)
        b = build_spd_split(list(reversed(rows)), seed=7)
        self.assertEqual(a, b)
        self.assertEqual(audit_spd_split(a)["cross_split_group_intersection"], 0)
        self.assertEqual(set(a["role_groups"]), {"train", "model_val", "test"})
        self.assertEqual(a["quarantined_unknown_location_records"], 1)
        role_by_loc = {}
        for row in a["files"]:
            if row["intersection_loc"]:
                role_by_loc.setdefault(row["intersection_loc"], set()).add(row["project_role"])
        self.assertTrue(all(len(roles) == 1 for roles in role_by_loc.values()))
        self.assertFalse(next(row for row in a["files"] if not row["intersection_loc"])["project_input_allowed"])

    def test_double_camera_and_overlapping_derived_window_leakage_rejected(self):
        split = build_spd_split(self.records(), seed=7)
        row = split["files"][0]
        split["files"].append({**row, "side": "derived-window", "source_file": "window/001.json",
                                "project_role": "test", "group_id": row["group_id"]})
        with self.assertRaisesRegex(ValueError, "跨split泄漏"):
            audit_spd_split(split)

    def test_fewer_than_three_locations_is_not_silently_split(self):
        rows = [row for row in self.records() if row["intersection_loc"] in ("road-a", "road-b")]
        with self.assertRaisesRegex(ValueError, "至少3个独立地理组"):
            build_spd_split(rows)

    def test_missing_location_resolves_only_unique_official_mapping(self):
        rows = [
            {"side": "vehicle-side", "sequence_id": "v1", "intersection_loc": ""},
            {"side": "vehicle-side", "sequence_id": "v2", "intersection_loc": ""},
            {"side": "vehicle-side", "sequence_id": "v3", "intersection_loc": ""},
        ]
        pairs = [
            {"vehicle_sequence": "v1", "infrastructure_sequence": "i1"},
            {"vehicle_sequence": "v2", "infrastructure_sequence": "i2"},
            {"vehicle_sequence": "v2", "infrastructure_sequence": "i3"},
        ]
        locations = {"infrastructure-side:i1": {"road-a"},
                     "infrastructure-side:i2": {"road-b"},
                     "infrastructure-side:i3": {"road-c"}}
        counts = resolve_missing_vehicle_locations(rows, pairs, locations)
        self.assertEqual(counts, {"raw_blank_vehicle_records": 3,
                                  "resolved_blank_vehicle_records": 1,
                                  "unresolved_blank_vehicle_records": 2})
        self.assertEqual(rows[0]["intersection_loc"], "road-a")
        self.assertEqual(rows[0]["intersection_loc_raw"], "")
        self.assertEqual(rows[1]["location_resolution_method"], "unresolved_quarantined")
        self.assertEqual(rows[2]["location_resolution_method"], "unresolved_quarantined")

    def test_roundoff_clip_is_derived_and_large_outlier_rejected(self):
        raw = {"2d_box": {"xmin": 0.0, "ymin": 20.0, "xmax": 1920.000061, "ymax": 40.0}}
        normalized = normalize_visible_box(raw, 1920, 1080)
        self.assertEqual(normalized["visible_box"]["xmax"], 1920.0)
        self.assertTrue(normalized["boundary_roundoff_clipped"])
        self.assertEqual(raw["2d_box"]["xmax"], 1920.000061)
        with self.assertRaisesRegex(ValueError, "outside_image"):
            normalize_visible_box({"2d_box": {"xmin": 0, "ymin": 0, "xmax": 1921, "ymax": 10}}, 1920, 1080)


class CarlaVisibleLabelTests(unittest.TestCase):
    def test_n15_pair_integrity_requires_identical_carla_frame(self):
        frame = {"carla_frame_id": 12, "source_time_s": 0.6,
                 "views": {"车端": {"frame_id": 12}, "路侧": {"frame_id": 12}}}
        self.assertEqual(synchronized_frame(frame), (12, 0.6))
        frame["views"]["路侧"]["frame_id"] = 13
        with self.assertRaisesRegex(ValueError, "frame不一致"):
            synchronized_frame(frame)

    def test_n15_visible_box_bounds_are_not_relaxed(self):
        label = {"track_id": "e:a", "bbox_xyxy_visible_pixels": [0, 5, 1920, 1080],
                 "visible_pixel_count": 1}
        validate_box(label, 1920, 1080)
        label["bbox_xyxy_visible_pixels"] = [0, 5, 1921, 1080]
        with self.assertRaisesRegex(ValueError, "越界"):
            validate_box(label, 1920, 1080)

    def test_fixed_roadside_pose_uses_carla_resolved_world_transform(self):
        class Location:
            def __init__(self, x=0, y=0, z=0):
                self.x, self.y, self.z = x, y, z

        class Rotation:
            def __init__(self, yaw=0, pitch=0):
                self.yaw, self.pitch, self.roll = yaw, pitch, 0

        class Transform:
            def __init__(self, location, rotation):
                self.location, self.rotation = location, rotation

        class Carla:
            pass
        Carla.Location, Carla.Rotation, Carla.Transform = Location, Rotation, Transform

        class BlueprintLibrary:
            def find(self, name):
                return name

        class Probe:
            def __init__(self):
                self.destroyed = False
                self.transform = Transform(Location(101, 202, 303), Rotation(44, -10))

            def get_transform(self):
                return self.transform

            def destroy(self):
                self.destroyed = True

        class World:
            def get_blueprint_library(self):
                return BlueprintLibrary()

            def tick(self):
                self.ticked = True

            def spawn_actor(self, blueprint, transform, attach_to):
                self.probe = Probe()
                self.attach_to = attach_to
                return self.probe

        world, ego = World(), object()
        fixed = calibrate_fixed_roadside_transform(world, ego, Carla, [0, 1.5, 3],
                                                    yaw_deg=0, pitch_deg=-10)
        self.assertEqual((fixed.location.x, fixed.location.y, fixed.location.z), (101, 202, 303))
        self.assertEqual((fixed.rotation.yaw, fixed.rotation.pitch), (44, -10))
        self.assertIs(world.attach_to, ego)
        self.assertTrue(world.ticked)
        self.assertTrue(world.probe.destroyed)

    def test_carla_layout_groups_are_split_before_frames(self):
        config = {"maps": [{"map": "Town03_Opt", "project_role": "train"},
                           {"map": "Town05_Opt", "project_role": "model_val"},
                           {"map": "Town10HD_Opt", "project_role": "test"}],
                  "scenes": ["vehicle", "vehicle_occlusion", "cyclist", "pedestrian", "negative"],
                  "frames_per_scene": 30, "moving_target_speed_mps": 3.0,
                  "pedestrian_speed_mps": 1.2, "vehicle_occlusion_distance_m": [12, 24],
                  "roadside_camera_local_xyz": [-3, 3.5, 6]}
        validate_scene_groups(config)
        config["vehicle_occlusion_distance_m"] = [24, 12]
        with self.assertRaisesRegex(ValueError, "严格递增"):
            validate_scene_groups(config)
        config["vehicle_occlusion_distance_m"] = [12, 24]
        config["pedestrian_speed_mps"] = 0
        with self.assertRaisesRegex(ValueError, "必须为正数"):
            validate_scene_groups(config)
        config["pedestrian_speed_mps"] = 1.2
        config["maps"].append({"map": "Town03_Opt", "project_role": "test"})
        with self.assertRaisesRegex(ValueError, "地图重复"):
            validate_scene_groups(config)

    def test_carla_scene_summary_counts_nested_view_labels(self):
        scene = {"frames": [{"views": {"车端": {"negative_frame": False},
                                       "路侧": {"negative_frame": True}}}],
                 "labels": [{"labels": [{"category": "vehicle", "camera_distance_m": 5,
                                             "bbox_xyxy_visible_pixels": [0, 0, 20, 30],
                                             "occluded_state": "visible_occlusion_proxy_estimated",
                                             "occlusion_bin": "unknown",
                                             "truncated_state": "not_truncated_by_image_edge"}],
                             "hidden_actor_review": [{"actor_id": 8}]},
                            {"labels": [{"category": "vehicle", "camera_distance_m": 15,
                                          "bbox_xyxy_visible_pixels": [1, 2, 61, 60],
                                          "occluded_state": "visible_occlusion_proxy_estimated",
                                          "occlusion_bin": "unknown",
                                          "truncated_state": "touches_image_edge"},
                                         {"category": "pedestrian", "camera_distance_m": 30,
                                          "bbox_xyxy_visible_pixels": [0, 0, 120, 100],
                                          "occluded_state": "visible_occlusion_proxy_estimated",
                                          "occlusion_bin": "unknown",
                                          "truncated_state": "not_truncated_by_image_edge"}],
                             "hidden_actor_review": []}]}
        self.assertEqual(summarize_scene_labels(scene), {
            "labels_by_category": {"pedestrian": 1, "vehicle": 2},
            "camera_distance_bins_m": {"0_10m": 1, "10_25m": 1, "over_25m": 1},
            "bbox_max_dimension_bins_px": {"small_le_32px": 1, "medium_33_96px": 1, "large_over_96px": 1},
            "occlusion_proxy_bins": {"low_0_10pct": 0, "medium_10_50pct": 0,
                                     "high_50_100pct": 0, "unknown": 3},
            "occlusion_proxy_unknown": 3,
            "truncated_by_image_edge": 1,
            "negative_view_frames": 1,
            "hidden_actor_review_records": 1})

    def test_visible_boxes_come_from_instance_pixels_and_hidden_target_has_no_box(self):
        import numpy as np
        image = np.zeros((5, 7, 3), dtype=np.uint8)
        # RGB semantic tag in R; actor id is G + (B << 8), matching saved CARLA PNG.
        image[1:4, 2:5, 0] = 10
        image[1:4, 2:5, 1] = 7
        result = extract_visible_instances(image, [
            {"actor_id": 7, "track_id": "episode:a7", "category": "vehicle",
             "blueprint_id": "vehicle.tesla.model3"},
            {"actor_id": 8, "track_id": "episode:a8", "category": "pedestrian",
             "blueprint_id": "walker.pedestrian.0001"}], frame_id=11, episode_id="episode",
            source_time_s=0.5, rgb_path="rgb.png", instance_path="instance.png",
            projected_actor_boxes={7: [1, 1, 6, 4]})
        self.assertEqual(len(result["labels"]), 1)
        self.assertEqual(result["labels"][0]["bbox_xyxy_visible_pixels"], [2, 1, 5, 4])
        self.assertEqual(result["labels"][0]["visible_pixel_count"], 9)
        self.assertEqual(result["labels"][0]["occlusion_fraction_proxy"], 0.4)
        self.assertEqual(result["labels"][0]["occlusion_bin"], "medium_10_50pct")
        self.assertEqual(len(result["hidden_actor_review"]), 1)
        self.assertFalse(result["projected_3d_boxes_used_as_visible_gt"])
        self.assertFalse(result["negative_frame"])

    def test_world_points_project_to_expected_camera_rectangle(self):
        class Point:
            def __init__(self, x, y, z):
                self.x, self.y, self.z = x, y, z
        points = [Point(10, y, z) for y in (-1, 1) for z in (-1, 1)]
        matrix = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        box = project_world_points_to_image(points, matrix, 200, 100, 90)
        self.assertEqual(box, [90.0, 40.0, 110.0, 60.0])
        self.assertIsNone(project_world_points_to_image(
            [Point(-1, 0, 0)] * 4, matrix, 200, 100, 90))

    def test_bicycle_blueprint_input_normalizes_to_cyclist(self):
        import numpy as np
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        image[1:3, 1:4, 1] = 9
        result = extract_visible_instances(image, [
            {"actor_id": 9, "track_id": "e:bicycle", "category": "bicycle",
             "blueprint_id": "vehicle.bh.crossbike"}], frame_id=4, episode_id="e", source_time_s=.2,
            rgb_path="bike.png", instance_path="mask.png")
        self.assertEqual(result["labels"][0]["category"], "cyclist")

    def test_empty_mask_is_a_negative_frame_and_duplicate_actor_ids_rejected(self):
        import numpy as np
        image = np.zeros((3, 4, 3), dtype=np.uint8)
        result = extract_visible_instances(image, [], frame_id=1, episode_id="e", source_time_s=0,
                                           rgb_path="rgb.png", instance_path="instance.png")
        self.assertTrue(result["negative_frame"])
        actors = [{"actor_id": 1, "track_id": "a", "category": "vehicle", "blueprint_id": "vehicle.a"},
                  {"actor_id": 1, "track_id": "b", "category": "vehicle", "blueprint_id": "vehicle.b"}]
        with self.assertRaisesRegex(ValueError, "actor_id"):
            extract_visible_instances(image, actors, frame_id=1, episode_id="e", source_time_s=0,
                                      rgb_path="rgb.png", instance_path="instance.png")

    def test_review_font_contains_chinese_and_is_hash_pinned(self):
        from PIL import ImageFont
        identity = review_font_identity()
        font = ImageFont.truetype(identity["path"], identity["size_px"])
        self.assertTrue(Path(identity["path"]).is_file())
        self.assertEqual(len(identity["sha256"]), 64)
        self.assertGreater(font.getlength("车端分层抽查"), 0)


if __name__ == "__main__":
    unittest.main()
