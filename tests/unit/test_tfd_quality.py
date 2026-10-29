"""N03 质量审计机制夹具，不代替真实 TFD 数据检查。"""

import unittest

from occluactisafe.data.quality import audit_trajectories, audit_trajectory_stream


def row(number, track="1", timestamp="1", x="1", y="2", kind="Vehicle", episode="ep1", source="vehicle"):
    return {"source_file": f"{source}/{episode}.csv", "source_row": number,
            "episode_id": episode, "source": source, "track_id": track,
            "timestamp": timestamp, "x": x, "y": y, "object_type": kind,
            "original": {"track_id": track, "timestamp": timestamp, "type": kind}}


class TrajectoryQualityTests(unittest.TestCase):
    def test_d01_and_missing_invalid_rows_keep_original_denominators(self):
        rows = [row(2, timestamp="2"), row(3, timestamp="1"), row(4, timestamp="1"),
                row(5, track="2", timestamp="0", kind="Bicycle"),
                row(6, track="", timestamp="4"), row(7, track="3", x=""),
                row(8, track="4", kind="hovercraft"),
                row(9, track="5", kind="Pedestrain"),
                row(10, track="6", timestamp="NaN"),
                row(11, track="7", y="Infinity"), row(12, track="8", kind="")]
        valid, report = audit_trajectories(rows)
        self.assertEqual(report["raw"]["row_count"], 11)
        self.assertEqual(report["valid"]["row_count"], 4)
        self.assertEqual(report["excluded_row_count"], 7)
        self.assertEqual(report["raw"]["track_count"], 8)
        self.assertEqual(report["valid"]["track_count"], 3)
        self.assertEqual(report["valid"]["category_track_counts"],
                         {"bicycle": 1, "pedestrian": 1, "vehicle": 1})
        self.assertEqual([r["source_row"] for r in valid], [3, 2, 5, 9])
        self.assertEqual(report["anomaly_counts"],
                         {"duplicate": 1, "out_of_order": 1, "missing_track_id": 1,
                          "invalid_timestamp": 1, "missing_coordinates": 1, "invalid_coordinates": 1,
                          "missing_category": 1, "invalid_category": 1, "extra_fields": 0, "short_row": 0})
        for event in report["anomalies"]:
            self.assertTrue(event["source_file"])
            self.assertGreaterEqual(event["source_row"], 2)
            self.assertTrue(event["reason"])
        duplicate = next(e for e in report["anomalies"] if e["kind"] == "duplicate")
        self.assertEqual(duplicate["source_row"], 4)
        self.assertEqual(duplicate["first"]["source_row"], 3)
        self.assertEqual(report["raw"]["type_row_counts"]["hovercraft"], 1)
        self.assertEqual(report["time_unit"], "unverified_raw_units")
        self.assertEqual(report["raw"]["time_spans"][0]["span_raw_time_units"], "4")

    def test_ids_and_sources_are_isolated_and_fractional_times_keep_precision(self):
        rows = [row(2, timestamp="9007199254740993"),
                row(3, timestamp="9007199254740992"),
                row(4, track="2", timestamp="1"),
                row(5, timestamp="9007199254740992", source="infrastructure"),
                row(6, timestamp="9007199254740992", episode="ep2")]
        valid, report = audit_trajectories(rows)
        self.assertEqual(report["anomaly_counts"]["duplicate"], 0)
        self.assertEqual(report["anomaly_counts"]["out_of_order"], 1)
        self.assertEqual(report["raw"]["episode_count"], 2)
        self.assertEqual(report["raw"]["track_count"], 4)
        first_track = [r for r in valid if r["episode_id"] == "ep1" and
                       r["source"] == "vehicle" and r["track_id"] == "1"]
        self.assertEqual([r["timestamp"] for r in first_track],
                         ["9007199254740992", "9007199254740993"])
        self.assertNotIn("velocity", first_track[0])

    def test_duplicate_uses_first_valid_row_and_reports_first_raw_location(self):
        rows = [row(2, x=""), row(3), row(4)]
        valid, report = audit_trajectories(rows)
        self.assertEqual([r["source_row"] for r in valid], [3])
        events = [e for e in report["anomalies"] if e["kind"] == "duplicate"]
        self.assertEqual(len(events), 2)
        self.assertEqual([e["first"]["source_row"] for e in events], [2, 2])

    def test_missing_location_fails_and_empty_input_is_explicit(self):
        bad = row(2)
        del bad["source_file"]
        with self.assertRaisesRegex(ValueError, "原始文件路径"):
            audit_trajectories([bad])
        valid, report = audit_trajectories([])
        self.assertEqual(valid, [])
        self.assertEqual(report["raw"]["row_count"], 0)
        self.assertEqual(report["valid"]["row_count"], 0)
        self.assertEqual(report["raw"]["time_spans"], [])

    def test_stream_emits_one_file_at_a_time_and_isolates_reused_ids(self):
        first = row(2, timestamp="1", episode="train/scene")
        first["source_file"] = "train/scene.csv"
        repeated = dict(first, source_row=3)
        second = row(2, timestamp="1", episode="val/scene")
        second["source_file"] = "val/scene.csv"
        results, anomalies = [], []
        report = audit_trajectory_stream(iter([first, repeated, second]), results.append, anomalies.append)
        self.assertEqual(len(results), 2)
        self.assertEqual(report["raw"]["row_count"], 3)
        self.assertEqual(report["valid"]["row_count"], 2)
        self.assertEqual(report["raw"]["episode_count"], 2)
        self.assertEqual(report["raw"]["track_count"], 2)
        self.assertEqual(report["anomaly_counts"]["duplicate"], 1)
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0]["source_file"], "train/scene.csv")
        self.assertIsNone(report["anomalies"])
        self.assertEqual(report["anomaly_record_count"], 1)
        with self.assertRaisesRegex(ValueError, "递增"):
            audit_trajectory_stream([second, first], results.append, anomalies.append)

    def test_same_episode_source_and_id_in_distinct_files_are_not_merged(self):
        first = row(2)
        second = dict(first, source_file="other/ep1.csv")
        valid, report = audit_trajectories([first, second])
        self.assertEqual(len(valid), 2)
        self.assertEqual(report["raw"]["track_count"], 2)
        self.assertEqual(report["anomaly_counts"]["duplicate"], 0)

    def test_subtype_tracks_are_counted_separately_without_assuming_riders(self):
        a = row(2, track="1", timestamp="1", kind="Bicycle")
        a["original"]["sub_type"] = "Cyclist"
        b = row(3, track="1", timestamp="2", kind="Bicycle")
        b["original"]["sub_type"] = "Motorcyclist"
        c = row(4, track="2", kind="Bicycle")
        d = row(5, track="3", kind="Pedestrain", x="")
        d["original"]["sub_type"] = "Pedestrian"
        valid, report = audit_trajectories([a, b, c, d])
        self.assertEqual(report["raw"]["subtype_row_counts"],
                         {"<missing>": 1, "Cyclist": 1, "Motorcyclist": 1, "Pedestrian": 1})
        self.assertEqual(report["valid"]["subtype_track_counts"],
                         {"<missing>": 1, "Cyclist": 1, "Motorcyclist": 1})
        self.assertEqual(report["valid"]["track_count"], 2)
        self.assertEqual(report["raw"]["subtype_track_counts"]["Pedestrian"], 1)
        self.assertNotIn("Pedestrian", report["valid"]["subtype_track_counts"])
        self.assertIn("Bicycle 不等于 Cyclist", report["note"])

    def test_original_field_missing_and_malformed_csv_counts(self):
        first = row(2)
        first["original"] = {"id": "1", "timestamp": "1", "type": "Vehicle", "x": "1", "y": "2", "tag": ""}
        first["extra_values"] = ["unexpected"]
        second = row(3, timestamp="2")
        second["original"] = {"id": "1", "timestamp": "2", "type": "Vehicle", "x": "1", "y": "", "tag": "AGENT"}
        second["missing_columns"] = ["y"]
        results, anomalies = [], []
        report = audit_trajectory_stream(iter([first, second]), results.append, anomalies.append)
        self.assertEqual(report["raw"]["field_missing_counts"]["tag"], 1)
        self.assertEqual(report["raw"]["field_missing_counts"]["y"], 1)
        self.assertEqual(report["raw"]["field_missing_counts"]["sub_type"], 2)
        self.assertIn("sub_type", report["raw"]["file_field_summaries"][0]["missing_documented_columns"])
        self.assertEqual(report["anomaly_counts"]["extra_fields"], 1)
        self.assertEqual(report["anomaly_counts"]["short_row"], 1)
        self.assertEqual({e["source_row"] for e in anomalies}, {2, 3})

    def test_official_uppercase_unknown_is_not_fabricated_vehicle(self):
        records = [row(2, track="1", kind="UNKNOWN"), row(3, track="2", kind="UNKNOWN_UNMOVABLE"),
                   row(4, track="3", kind="PEDESTRIAN"), row(5, track="4", kind="VEHICLE")]
        valid, report = audit_trajectories(records)
        self.assertEqual(len(valid), 4)
        self.assertEqual(report["anomaly_counts"]["invalid_category"], 0)
        self.assertEqual(report["valid"]["category_track_counts"],
                         {"pedestrian": 1, "unknown": 1, "unknown_unmovable": 1, "vehicle": 1})


if __name__ == "__main__":
    unittest.main()
