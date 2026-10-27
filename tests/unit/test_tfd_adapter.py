"""N03 只读来源适配：构造夹具不是官方数据或实测研究结果。"""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from occluactisafe.data.adapters.tfd import (
    OFFICIAL_TYPES, discover_tfd, iter_maps, iter_traffic_lights, iter_trajectories, iter_trajectory_file,
)
from occluactisafe.data.quality import TYPE_TO_CATEGORY, audit_trajectories


class TFDAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_csv(self, relative, header, rows):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def test_official_directory_views_and_original_values(self):
        header = ["city", "timestamp", "id", "type", "sub_type", "tag", "x", "y", "intersect_id", "from_side", "car_side_id", "road_side_id"]
        row = ["A", "1000000000000000001", "007", "Pedestrain", "Pedestrian", "1", "2.000", "-0.0", "01", "infrastructure", "17", "8"]
        for rel in ("single-vehicle/trajectories/001.csv", "single-infrastructure/trajectories/001.csv",
                    "cooperative-vehicle-infrastructure/infrastructure-trajectories/001.csv",
                    "cooperative-vehicle-infrastructure/vehicle-trajectories/001.csv",
                    "cooperative-vehicle-infrastructure/cooperative-trajectories/001.csv"):
            self.write_csv(rel, header, [row])
        data = list(iter_trajectories(self.root))
        self.assertEqual(len(data), 5)
        self.assertEqual({item["source"] for item in data}, {"vehicle", "infrastructure", "cooperative"})
        for item in data:
            self.assertEqual(item["source_row"], 2)
            self.assertEqual(item["timestamp"], "1000000000000000001")
            self.assertEqual(item["original"]["id"], "007")
            self.assertEqual(item["original"]["x"], "2.000")
            self.assertEqual(item["normalized_category"], "pedestrian")
            self.assertEqual(item["map_status"], "missing")
            self.assertEqual(item["time_unit"], "unverified")
            self.assertEqual(item["coordinate_system"], "unverified")
        coop = [item for item in data if item["source"] == "cooperative"][0]
        self.assertEqual(coop["usage_boundary"], "offline_only_future_leakage_unverified")
        self.assertEqual(coop["original"]["car_side_id"], "17")
        self.assertEqual(coop["original"]["road_side_id"], "8")

    def test_maps_signals_and_per_intersection_status(self):
        header = ["timestamp", "id", "type", "x", "y", "intersect_id"]
        self.write_csv("train/single-vehicle/trajectories/x.csv", header, [["1", "v", "Vehicle", "0", "1", "01"], ["2", "v", "Vehicle", "0", "2", "02"]])
        self.write_csv("train/single-infrastructure/traffic-light/x.csv", ["timestamp", "lane_id", "color_1", "remain_1", "intersect_id"], [["1", "A", "red", "3", "01"]])
        maps = self.root / "train/maps"
        maps.mkdir(parents=True)
        (maps / "hdmap01.json").write_text(json.dumps({"LANE": {"A": {"centerline": [[1, 2]]}}, "STOPLINE": {}, "CROSSWALK": {}}), encoding="utf-8")
        (maps / "hdmap02.json").write_text("{broken", encoding="utf-8")
        (maps / "hdmap03.json").write_text("{}", encoding="utf-8")
        inventory = discover_tfd(self.root)
        self.assertEqual(inventory["map_status"], "present")
        self.assertEqual(len(inventory["trajectory_files"]), 1)
        self.assertEqual(len(inventory["traffic_light_files"]), 1)
        self.assertEqual([row["map_status"] for row in iter_trajectories(self.root)], ["present", "invalid"])
        signal = list(iter_traffic_lights(self.root))[0]
        self.assertEqual(signal["lane_id"], "A")
        self.assertEqual(signal["original"]["remain_1"], "3")
        self.assertEqual(signal["availability"], "unknown")
        map_rows = list(iter_maps(self.root))
        self.assertEqual(map_rows[0]["original"]["LANE"]["A"]["centerline"], [[1, 2]])
        self.assertEqual(map_rows[1]["status"], "invalid")
        self.assertEqual(map_rows[2]["status"], "incomplete")
        self.assertIn("LANE", map_rows[2]["missing_sections"])

    def test_missing_columns_extra_fields_unknown_category_and_multiline_location(self):
        path = self.root / "single-vehicle/trajectories/example.csv"
        path.parent.mkdir(parents=True)
        path.write_text('timestamp,id,type,x,y,note\n1,a,Bicycle,1,2,"two\nlines"\n2,b,UNKNOWN,3\n3,c,Vehicle,1,2,x,extra\n', encoding="utf-8")
        rows = list(iter_trajectories(self.root))
        self.assertEqual([r["source_row"] for r in rows], [2, 4, 5])
        self.assertEqual(rows[0]["normalized_category"], "bicycle")
        self.assertEqual(rows[1]["normalized_category"], "unknown")
        self.assertEqual(rows[1]["category_status"], "official_unknown")
        self.assertIn("y", rows[1]["missing_columns"])
        self.assertEqual(rows[2]["extra_values"], ["extra"])

    def test_absent_root_and_duplicate_csv_headers_are_rejected(self):
        with self.assertRaises(FileNotFoundError):
            discover_tfd(self.root / "missing")
        self.assertEqual(discover_tfd(self.root)["map_status"], "missing")
        self.write_csv("single-vehicle/trajectories/x.csv", ["id", "id"], [["1", "2"]])
        with self.assertRaisesRegex(ValueError, "重复"):
            list(iter_trajectories(self.root))

    def test_official_split_data_layout_and_episode_namespacing(self):
        header = ["timestamp", "id", "type", "x", "y"]
        for split in ("train", "val", "test_obs"):
            for view in ("vehicle-trajectories", "infrastructure-trajectories", "cooperative-trajectories"):
                self.write_csv(f"cooperative-vehicle-infrastructure/{view}/{split}/data/001.csv", header,
                               [["1", "007", "Vehicle", "2", "3"]])
        items = list(iter_trajectories(self.root))
        self.assertEqual(len(items), 9)
        self.assertEqual({x["episode_id"] for x in items}, {f"cooperative/{s}/001" for s in ("train", "val", "test_obs")})
        self.assertEqual({x["split"] for x in items}, {"train", "val", "test_obs"})
        selected = "cooperative-vehicle-infrastructure/vehicle-trajectories/train/data/001.csv"
        self.assertEqual(list(iter_trajectory_file(self.root, selected))[0]["episode_id"], "cooperative/train/001")
        with self.assertRaises(ValueError):
            list(iter_trajectory_file(self.root, "../outside.csv"))

    def test_official_tool_uppercase_types_and_unknown_are_not_invented(self):
        header = ["timestamp", "id", "type", "x", "y"]
        values = ["VEHICLE", "BICYCLE", "PEDESTRIAN", "UNKNOWN_UNMOVABLE", "UNKNOWN", "NOT_A_TYPE", ""]
        self.write_csv("single-vehicle/trajectories/001.csv", header,
                       [[str(i), str(i), value, "2", "3"] for i, value in enumerate(values)])
        rows = list(iter_trajectories(self.root))
        self.assertEqual([r["normalized_category"] for r in rows],
                         ["vehicle", "bicycle", "pedestrian", "unknown_unmovable", "unknown", "unknown", "unknown"])
        self.assertEqual([r["category_status"] for r in rows],
                         ["recognized", "recognized", "recognized", "official_unknown", "official_unknown", "unrecognized", "missing"])

    def test_adapter_and_quality_share_category_and_raw_time_semantics(self):
        self.assertEqual(OFFICIAL_TYPES, TYPE_TO_CATEGORY)
        self.write_csv("single-vehicle/trajectories/001.csv", ["timestamp", "id", "type", "x", "y"],
                       [["1.0000", "v1", " VEHICLE ", "0.0", "2"],
                        ["1.1000", "v1", " VEHICLE ", "1.0", "2"]])
        original = list(iter_trajectories(self.root))
        valid, quality = audit_trajectories(original)
        self.assertEqual(len(valid), 2)
        self.assertEqual(valid[0]["object_type"], " VEHICLE ")
        self.assertEqual(valid[0]["normalized_category"], "vehicle")
        self.assertEqual(quality["valid"]["category_track_counts"]["vehicle"], 1)
        self.assertEqual(quality["time_unit"], "unverified_raw_units")
        self.assertEqual(valid[0]["timestamp"], "1.0000")
        self.assertEqual(quality["valid"]["time_spans"][0]["span_raw_time_units"], "0.1000")


if __name__ == "__main__":
    unittest.main()
