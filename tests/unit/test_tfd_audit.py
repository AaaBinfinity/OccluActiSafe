"""N03 集成测试只用自建夹具；不代表官方样本或正式数据。"""
import csv
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from occluactisafe.data.tfd_audit import load_task_config, run_audit

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "prepare_data.py"


class AuditIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "fixture"
        self.output = self.root / "outputs"
        csvfile = self.data / "cooperative-vehicle-infrastructure" / "vehicle-trajectories" / "train" / "data" / "001.csv"
        csvfile.parent.mkdir(parents=True)
        with csvfile.open("w", encoding="utf8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "id", "type", "sub_type", "x", "y", "intersect_id"])
            w.writerows([["2", "a", "Vehicle", "Car", "3", "4", "01"],
                         ["1", "a", "Vehicle", "Car", "1", "2", "01"],
                         ["1", "a", "Vehicle", "Car", "1", "2", "01"],
                         ["3", "b", "Bicycle", "Cyclist", "2", "3", "01"],
                         ["4", "c", "Pedestrain", "Pedestrian", "", "3", "01"]])
        signal = self.data / "cooperative-vehicle-infrastructure" / "traffic-light" / "001.csv"
        signal.parent.mkdir(parents=True)
        signal.write_text("timestamp,lane_id,color_1\n1,L1,red\n", encoding="utf8")
        maps = self.data / "maps"
        maps.mkdir()
        (maps / "hdmap01.json").write_text('{"LANE":{"L1":{}},"STOPLINE":{},"CROSSWALK":{}}', encoding="utf8")
        self.config_file = self.root / "config.yaml"
        self.config_file.write_text("data_root: fixture\noutput_root: outputs\n", encoding="utf8")

    def config(self):
        return load_task_config(self.config_file, project_root=self.root)

    def test_run_writes_traceable_stats_without_claiming_real_or_full(self):
        config = self.config()
        first = run_audit(config, project_root=ROOT)
        second = run_audit(config, project_root=ROOT)
        self.assertNotEqual(first["run_id"], second["run_id"])
        self.assertFalse(first["formal_data_ready"])
        run_dir = Path(first["run_dir"])
        quality = json.loads((run_dir / "data_quality.json").read_text(encoding="utf8"))
        self.assertEqual((quality["raw"]["row_count"], quality["valid"]["row_count"]), (5, 3))
        self.assertEqual(quality["anomaly_counts"]["duplicate"], 1)
        self.assertEqual(quality["anomaly_counts"]["out_of_order"], 1)
        self.assertEqual(quality["raw"]["subtype_track_counts"]["Pedestrian"], 1)
        self.assertEqual(quality["valid"]["subtype_track_counts"]["Cyclist"], 1)
        self.assertEqual(quality["empty_or_header_only_trajectory_files"], [])
        self.assertEqual(quality["traffic_light_row_count"], 1)
        with gzip.open(run_dir / "trajectories.jsonl.gz", "rt", encoding="utf8") as f:
            rows = [json.loads(line) for line in f]
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["source_row"] for row in rows], [3, 2, 5])
        self.assertEqual(rows[0]["original"]["x"], "1")
        with gzip.open(run_dir / "traffic_lights.jsonl.gz", "rt", encoding="utf8") as f:
            signals = [json.loads(line) for line in f]
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["original"]["color_1"], "red")
        self.assertFalse((run_dir / "trajectories.jsonl").exists())
        self.assertEqual(quality["map_status_valid_row_counts"], {"present": 3})
        maps = json.loads((run_dir / "map_index.json").read_text(encoding="utf8"))
        self.assertEqual(maps[0]["lane_entry_count"], 1)
        self.assertEqual(quality["time_unit"], "unverified_raw_units")
        self.assertTrue((run_dir / "trajectory_preview.svg").is_file())
        anomalies = [json.loads(line) for line in (run_dir / "anomalies.jsonl").read_text(encoding="utf8").splitlines()]
        self.assertTrue(all(item["source_file"] and item["source_row"] >= 2 and item["reason"] for item in anomalies))
        duplicate = next(item for item in anomalies if item["kind"] == "duplicate")
        self.assertEqual(duplicate["original"]["id"], "a")
        self.assertEqual(duplicate["original"]["x"], "1")
        missing = next(item for item in anomalies if item["kind"] == "missing_coordinates")
        self.assertEqual(missing["original"]["x"], "")
        data_manifest = json.loads((run_dir / "data_manifest.json").read_text(encoding="utf8"))
        self.assertEqual(data_manifest["provenance_verification"], "user_provided_unverified")
        self.assertEqual(data_manifest["usage_conditions_claim"], "unverified")
        self.assertEqual(data_manifest["usage_conditions_evidence"], "not_provided_or_verified_by_audit")
        self.assertEqual(data_manifest["acquired_date_claim"], "unknown")
        self.assertEqual(data_manifest["acquired_date_evidence"], "user_supplied_unverified")
        self.assertEqual(len(data_manifest["files"]), 3)
        self.assertTrue(all(len(item["sha256"]) == 64 for item in data_manifest["files"]))

    def test_empty_and_missing_data_fail_without_fake_results(self):
        missing = self.config(); missing["data_root"] = str(self.root / "missing")
        with self.assertRaises(FileNotFoundError):
            run_audit(missing, project_root=ROOT)
        self.assertFalse(self.output.exists())
        for path in self.data.rglob("*.csv"):
            path.unlink()
        csvfile = self.data / "single-vehicle" / "trajectories" / "empty.csv"
        csvfile.parent.mkdir(parents=True)
        csvfile.write_text("timestamp,id,type,x,y\n", encoding="utf8")
        with self.assertRaisesRegex(ValueError, "不含数据行"):
            run_audit(self.config(), project_root=ROOT)
        statuses = list(self.output.glob("*/audit_status.json"))
        self.assertEqual(len(statuses), 1)
        self.assertEqual(json.loads(statuses[0].read_text(encoding="utf8"))["status"], "failed")
        quality = json.loads((statuses[0].parent / "data_quality.json").read_text(encoding="utf8"))
        self.assertEqual(len(quality["empty_or_header_only_trajectory_files"]), 1)

    def test_low_disk_space_preserves_failed_status(self):
        with mock.patch("occluactisafe.data.tfd_audit.shutil.disk_usage") as disk:
            disk.return_value = type("Usage", (), {"free": 1})()
            with self.assertRaisesRegex(OSError, "可用空间不足"):
                run_audit(self.config(), project_root=ROOT)
        statuses = list(self.output.glob("*/audit_status.json"))
        self.assertEqual(len(statuses), 1)
        status = json.loads(statuses[0].read_text(encoding="utf8"))
        self.assertEqual(status["status"], "failed")
        self.assertFalse(status["formal_data_ready"])

    def test_prefixed_map_is_indexed_without_fabricated_association(self):
        original = self.data / "maps" / "hdmap01.json"
        original.rename(self.data / "maps" / "yizhuang_hdmap01.json")
        result = run_audit(self.config(), project_root=ROOT)
        run_dir = Path(result["run_dir"])
        quality = json.loads((run_dir / "data_quality.json").read_text(encoding="utf8"))
        maps = json.loads((run_dir / "map_index.json").read_text(encoding="utf8"))
        self.assertEqual(quality["map_status_valid_row_counts"], {"unresolved": 3})
        self.assertEqual(quality["map_file_count"], 1)
        self.assertEqual(maps[0]["map_id"], "yizhuang_hdmap01")
        self.assertEqual(maps[0]["lane_entry_count"], 1)
        self.assertEqual(len(maps[0]["sha256"]), 64)
        self.assertNotIn("original", maps[0])

    def test_path_precedence_and_config_validation(self):
        old = Path.cwd()
        try:
            before = self.config()
            os.chdir("/")
            self.assertEqual(load_task_config(self.config_file, project_root=self.root), before)
        finally:
            os.chdir(old)
        merged = load_task_config(self.config_file, project_root=self.root,
                                  overrides={"dataset_scope": "full"}, cli_overrides={"dataset_scope": "example"})
        self.assertEqual(merged["dataset_scope"], "example")
        for bad in ({"bogus": 1}, {"dataset_scope": "missing"},
                    {"source_url": "https://user:pass@example.com/"},
                    {"source_url": "https://example.com/?token=secret"},
                    {"source_url": "https://example.com/#token=secret"},
                    {"acquired_date": "2026-02-30"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                load_task_config(self.config_file, overrides=bad, project_root=self.root)
        default = self.root / "default.yaml"
        default.write_text("dataset_scope: full\n", encoding="utf8")
        with self.assertRaisesRegex(ValueError, "完整数据须明确指定"):
            load_task_config(default, project_root=self.root)
        self.assertEqual(load_task_config(self.config_file, project_root=self.root,
                                         overrides={"acquired_date": "2026-10-27"})["acquired_date"], "2026-10-27")
        inside = self.config(); inside["output_root"] = str(self.data / "writes")
        with self.assertRaisesRegex(ValueError, "原始数据"):
            run_audit(inside, project_root=ROOT)

    def test_entry_reports_missing_data_with_isolated_windows_encoding(self):
        missing = self.root / "unavailable"
        self.config_file.write_text(f"data_root: {missing}\noutput_root: {self.output}\n", encoding="utf8")
        result = subprocess.run([sys.executable, "-I", "-B", str(SCRIPT), "--config", str(self.config_file)],
                                cwd=self.root, capture_output=True, text=True, encoding="utf8")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("TFD 数据目录不存在", result.stderr)
        self.assertNotIn("UnicodeEncodeError", result.stderr)
        self.assertFalse(self.output.exists())

    def test_entry_no_args_fails_clearly_and_import_does_not_run(self):
        missing = self.root / "unavailable"
        self.config_file.write_text(f"data_root: {missing}\noutput_root: {self.output}\n", encoding="utf8")
        entry = subprocess.run([sys.executable, "-B", str(SCRIPT), "--config", str(self.config_file)],
                               cwd=self.root, capture_output=True, text=True, encoding="utf8")
        self.assertNotEqual(entry.returncode, 0)
        self.assertIn("TFD 数据目录不存在", entry.stderr)
        self.assertNotIn("TFD 审计完成", entry.stdout)
        imported = subprocess.run([sys.executable, "-B", "-c",
                                   "import runpy,sys; runpy.run_path(sys.argv[1],run_name='imported')", str(SCRIPT)],
                                  cwd=self.root, capture_output=True, text=True, encoding="utf8")
        self.assertEqual(imported.returncode, 0, imported.stderr)
        self.assertEqual(imported.stdout, "")
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
