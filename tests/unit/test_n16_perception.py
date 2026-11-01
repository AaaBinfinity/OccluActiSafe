import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from occluactisafe.data.spd_perception import load_spd_perception_frame
from occluactisafe.data.adapters.spd import sha256_file
from occluactisafe.evaluation.detection_metrics import (
    GroundTruthObject, MetricFrame, score_detection_strata, score_detections, score_tracking,
)
from occluactisafe.perception.adapters.ultralytics import UltralyticsDetector
from occluactisafe.perception.registry import create_detector, registered_detectors
from occluactisafe.perception.tracker import ByteTrack, ByteTrackConfig
from occluactisafe.perception.types import Detection, Detections, PixelTrack, TrackNamespace


def detections(frame, boxes, *, side="vehicle", episode="ep", timestamp=None):
    ts = frame/10 if timestamp is None else timestamp
    return Detections(str(frame), ts, 200, 100,
                      TrackNamespace("fixture", side, episode), tuple(boxes),
                      "ok" if boxes else "empty")


class DetectorAdapterTests(unittest.TestCase):
    def test_non_square_input_returns_original_xyxy_and_empty_frame(self):
        class Tensor:
            def __init__(self, value): self.value = value
            def detach(self): return self
            def cpu(self): return self
            def tolist(self): return self.value

        class Boxes:
            xyxy = Tensor([[5, 7, 95, 55]])
            conf = Tensor([0.8])
            cls = Tensor([2])
            def __len__(self): return 1

        class Result:
            orig_shape = (60, 100)
            boxes = Boxes()

        class Model:
            def __init__(self, result): self.result = result; self.kwargs = None; self.source = None
            def predict(self, **kwargs): self.kwargs = kwargs; self.source = kwargs["source"]; return [self.result]

        model = Model(Result())
        detector = UltralyticsDetector(model, class_map={2: "vehicle"}, imgsz=960)
        namespace = TrackNamespace("fixture", "camera", "ep")
        output = detector.detect(np.zeros((60, 100, 3), dtype=np.uint8), frame_id="f1",
                                 timestamp_s=0.1, namespace=namespace)
        self.assertEqual(output.items[0].xyxy, (5.0, 7.0, 95.0, 55.0))
        self.assertEqual((output.width, output.height), (100, 60))
        self.assertEqual(output.items[0].class_id, "vehicle")
        self.assertEqual(model.kwargs["imgsz"], 960)
        rgb = np.zeros((60, 100, 3), dtype=np.uint8)
        rgb[0, 0] = (1, 2, 3)
        detector.detect(rgb, frame_id="f2", timestamp_s=0.2, namespace=namespace)
        self.assertEqual(tuple(model.source[0, 0]), (3, 2, 1))

    def test_registry_is_explicit_and_weight_does_not_autodownload(self):
        self.assertIn("ultralytics", registered_detectors())
        with self.assertRaisesRegex(FileNotFoundError, "权重文件不存在"):
            create_detector("ultralytics", weights=Path("missing-n16-weight.pt"))

    def test_input_shape_mismatch_is_rejected(self):
        class Model:
            def predict(self, **kwargs):
                class Result: orig_shape = (50, 100); boxes = None
                return [Result()]
        detector = UltralyticsDetector(Model(), class_map={0: "vehicle"})
        with self.assertRaisesRegex(ValueError, "尺寸"):
            detector.detect(np.zeros((60, 100, 3)), frame_id="f", timestamp_s=0,
                            namespace=TrackNamespace("s", "c", "e"))

    def test_prediction_failure_is_explicit_and_unmapped_classes_are_rejected(self):
        class Model:
            def predict(self, **kwargs): raise RuntimeError("fixture inference failure")
        detector = UltralyticsDetector(Model(), class_map={0: "vehicle"})
        result = detector.detect(np.zeros((40, 80, 3)), frame_id="f", timestamp_s=0,
                                 namespace=TrackNamespace("s", "c", "e"))
        self.assertEqual(result.status, "error")
        self.assertIn("fixture inference failure", result.error_message)
        with self.assertRaises(ValueError):
            UltralyticsDetector(Model(), class_map={})


class ByteTrackTests(unittest.TestCase):
    def setUp(self):
        self.tracker = ByteTrack(ByteTrackConfig(min_hits=2, max_lost_frames=3))

    def test_low_score_frame_five_recovers_existing_identity(self):
        seen = []
        for frame in range(1, 11):
            score = 0.15 if frame == 5 else 0.9
            result = self.tracker.track(detections(frame, [Detection((10+frame, 20, 40+frame, 50), score, "vehicle")]))
            seen.append(result)
        self.assertEqual(len(seen[0]), 0)
        self.assertEqual(len(seen[1]), 1)
        ids = {tracks[0].track_id for tracks in seen[1:]}
        self.assertEqual(len(ids), 1)
        self.assertTrue(seen[4][0].confirmed)
        self.assertAlmostEqual(seen[4][0].score, 0.15)

    def test_isolated_low_score_does_not_create_confirmed_track(self):
        output = self.tracker.track(detections(1, [Detection((10, 10, 20, 20), 0.15, "vehicle")]))
        self.assertEqual(output, ())
        output = self.tracker.track(detections(2, []))
        self.assertEqual(output, ())

    def test_camera_and_episode_namespaces_never_share_ids_or_state(self):
        vehicle = ()
        road = ()
        new_episode = ()
        for frame in (1, 2):
            vehicle = self.tracker.track(detections(frame, [Detection((10, 10, 30, 30), .9, "vehicle")], side="v"))
            road = self.tracker.track(detections(frame, [Detection((10, 10, 30, 30), .9, "vehicle")], side="r"))
            new_episode = self.tracker.track(detections(frame, [Detection((10, 10, 30, 30), .9, "vehicle")], episode="ep2"))
        self.assertNotEqual(vehicle[0].track_id, road[0].track_id)
        self.assertNotEqual(vehicle[0].track_id, new_episode[0].track_id)
        self.tracker.reset(source_id="fixture", sensor_id="v", episode_id="ep")
        self.tracker.track(detections(1, [Detection((10, 10, 30, 30), .9, "vehicle")], side="v"))
        restarted = self.tracker.track(detections(2, [Detection((10, 10, 30, 30), .9, "vehicle")], side="v"))
        self.assertTrue(restarted[0].track_id.endswith("track-1"))

    def test_empty_frame_ages_tracks_and_timestamps_must_increase(self):
        for frame in (1, 2):
            self.tracker.track(detections(frame, [Detection((10, 10, 30, 30), .9, "vehicle")]))
        self.assertEqual(self.tracker.track(detections(3, [])), ())
        with self.assertRaisesRegex(ValueError, "严格递增"):
            self.tracker.track(detections(3, []))


class MetricTests(unittest.TestCase):
    def test_one_correct_prediction_and_one_missing_gt(self):
        frame = MetricFrame("frame", (
            GroundTruthObject((0, 0, 10, 10), "vehicle", "g1"),
            GroundTruthObject((20, 20, 30, 30), "vehicle", "g2"),
        ), (Detection((0, 0, 10, 10), .9, "vehicle"),))
        report = score_detections([frame])
        self.assertEqual((report["tp"], report["fp"], report["fn"]), (1, 0, 1))
        self.assertEqual(report["ground_truth_count"], 2)
        self.assertEqual(report["recall"], .5)
        self.assertAlmostEqual(report["ap_by_class"]["vehicle"], .5)

    def test_no_prediction_frame_retains_all_misses(self):
        frame = MetricFrame("empty-prediction", (GroundTruthObject((0, 0, 10, 10), "pedestrian", "g1"),), ())
        report = score_detections([frame])
        self.assertEqual((report["tp"], report["fp"], report["fn"]), (0, 0, 1))
        self.assertEqual(report["ap_by_class"]["pedestrian"], 0.0)

    def test_side_and_target_strata_keep_explicit_recall_denominators(self):
        gt = GroundTruthObject((0, 0, 10, 10), "vehicle", "g1", "Car", "small_le_32px", "2")
        frame = MetricFrame("f", (gt,), (Detection((0, 0, 10, 10), .9, "vehicle"),),
                            metadata=(("camera_side", "vehicle-side"),))
        report = score_detection_strata([frame])
        self.assertEqual(report["by_camera_side"]["vehicle-side"]["frames"], 1)
        self.assertEqual(report["target_recall_by_class_scale_official_occlusion"][
            "vehicle|small_le_32px|2"]["gt"], 1)
        self.assertEqual(report["target_recall_by_class_scale_official_occlusion"][
            "vehicle|small_le_32px|2"]["recall"], 1.0)

    def test_tracking_id_switches_and_idf1_are_counted(self):
        ns = TrackNamespace("s", "camera", "ep")
        gt = GroundTruthObject((0, 0, 10, 10), "vehicle", "g1")
        frames = [MetricFrame("f1", (gt,), tracks=(PixelTrack("p1", gt.xyxy, "vehicle", .9, 0, "f1", ns, True, 2, 0),)),
                  MetricFrame("f2", (gt,), tracks=(PixelTrack("p2", gt.xyxy, "vehicle", .9, 1, "f2", ns, True, 3, 0),))]
        report = score_tracking(frames)
        self.assertEqual(report["id_switches"], 1)
        self.assertEqual(report["idf1"], .5)


class SPDAdapterTests(unittest.TestCase):
    def test_reads_real_shape_timestamp_namespace_and_gt_separately(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            side = "vehicle-side"
            (root / "V2X-Seq-SPD" / side).mkdir(parents=True)
            (root / "V2X-Seq-SPD-vehicle-side-image").mkdir()
            Image.fromarray(np.zeros((60, 100, 3), dtype=np.uint8)).save(
                root / "V2X-Seq-SPD-vehicle-side-image" / "frame.jpg")
            row = {"sequence_id": "seq1", "frame_id": "000001", "image_path": "nested/frame.jpg",
                   "image_timestamp": "1626155123939720", "label_camera_std_path": "label.json",
                   "calib_camera_intrinsic_path": "intrinsic.json"}
            (root / "V2X-Seq-SPD" / side / "data_info.json").write_text(json.dumps([row]), encoding="utf-8")
            (root / "V2X-Seq-SPD" / side / "label.json").write_text(json.dumps([
                {"type": "Car", "track_id": "vehicle-1", "2d_box": {"xmin": 1, "ymin": 2, "xmax": 30, "ymax": 40},
                 "occluded_state": "1", "truncated_state": "0"}]), encoding="utf-8")
            (root / "V2X-Seq-SPD" / side / "intrinsic.json").write_text(json.dumps({"cam_K": [1,0,0,0,1,0,0,0,1]}), encoding="utf-8")
            split_row = {"side": side, "sequence_id": "seq1", "frame_id": "000001",
                         "project_role": "model_val", "width": 100, "height": 60,
                         "intersection_loc": "road-a"}
            expected_hashes = {
                "V2X-Seq-SPD-vehicle-side-image/frame.jpg": sha256_file(root / "V2X-Seq-SPD-vehicle-side-image" / "frame.jpg"),
                f"V2X-Seq-SPD/{side}/label.json": sha256_file(root / "V2X-Seq-SPD" / side / "label.json"),
                f"V2X-Seq-SPD/{side}/intrinsic.json": sha256_file(root / "V2X-Seq-SPD" / side / "intrinsic.json"),
            }
            frame = load_spd_perception_frame(root, split_row, expected_source_sha256=expected_hashes)
            self.assertEqual(frame.image_rgb.shape, (60, 100, 3))
            self.assertEqual(frame.timestamp_us, 1626155123939720)
            self.assertEqual(frame.namespace.sensor_id, side)
            self.assertEqual(frame.ground_truth[0].class_id, "vehicle")
            self.assertEqual(frame.ground_truth[0].raw_category, "Car")
            self.assertFalse(frame.unsupported_labels)
            self.assertEqual(frame.official_occlusion_counts, {"1": 1})
            with self.assertRaises(ValueError):
                load_spd_perception_frame(root, split_row, expected_source_sha256={})


if __name__ == "__main__":
    unittest.main()
