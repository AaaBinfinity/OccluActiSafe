"""N15 SPD 索引路径与地理组泄漏拒绝夹具。"""
from pathlib import Path
import tempfile
import unittest

from occluactisafe.data.adapters.spd import audit_spd_split, build_spd_split, resolve_index_path


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


if __name__ == "__main__":
    unittest.main()
