import copy
from pathlib import Path
import tempfile
import unittest

from occluactisafe.experiments.detector_data import class_box_line, load_config, materialize
from occluactisafe.experiments.detector_matrix import validate_matrix
from occluactisafe.experiments.run_store import write_json, read_json, sha256
from occluactisafe.experiments.selection import select_detector
from occluactisafe.experiments.detector_evaluation import metric_summary
from occluactisafe.evaluation.detection_metrics import GroundTruthObject, MetricFrame, score_detections
from occluactisafe.perception.types import Detection


class N17Tests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()

    def candidate(self, name, recall=.97, latency=20):
        return dict(candidate_id=name, split='model_val', status='completed',
                    spd_recall={k: recall for k in ('vehicle', 'pedestrian', 'cyclist')},
                    carla_empty_frame_fp=0.02, detector_p95_ms=latency, parameter_count=3000000)

    def test_full_matrix_and_same_frozen_budget(self):
        pairs = validate_matrix(self.cfg)
        self.assertEqual(len(pairs), 9)
        self.assertEqual(set(seed for _, seed in pairs), {17,29,43})
        invalid = copy.deepcopy(self.cfg)
        invalid['models'].pop()
        with self.assertRaises(ValueError):
            validate_matrix(invalid)

    def test_performance_winner_can_be_any_family(self):
        for family in ('YOLOv8', 'YOLO11', 'YOLO26'):
            rows = [self.candidate(name, .99 if name == family else .96)
                    for name in ('YOLOv8', 'YOLO11', 'YOLO26')]
            self.assertEqual(select_detector(rows, self.cfg['gates'])['selected_candidate'], family)

    def test_latency_missing_negative_and_failure_are_retained(self):
        rows = [self.candidate('latency', latency=101), self.candidate('empty'), self.candidate('failed')]
        rows[1]['carla_empty_frame_fp'] = None
        rows[2]['status'] = 'failed'
        result = select_detector(rows, self.cfg['gates'])
        self.assertEqual(result['status'], 'no_qualified_candidate')
        self.assertEqual(len(result['decisions']), 3)
        self.assertTrue(all(d['reasons'] for d in result['decisions']))

    def test_test_split_and_injected_test_metric_are_rejected(self):
        row = self.candidate('x')
        row['split'] = 'test'
        with self.assertRaises(ValueError):
            select_detector([row], self.cfg['gates'])
        row['split'] = 'model_val'
        row['test_map'] = 1
        with self.assertRaises(ValueError):
            select_detector([row], self.cfg['gates'])

    def test_box_export_original_coordinate_and_bounds(self):
        gt = dict(class_id='vehicle', xyxy=(10,20,30,40))
        self.assertEqual(class_box_line(gt, ['vehicle'], 100, 50),
                         '0 0.2000000000 0.6000000000 0.2000000000 0.4000000000')
        with self.assertRaises(ValueError):
            class_box_line(gt, ['vehicle'], 20, 50)

    def test_materialization_refuses_test_media(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                materialize({'role':'test'}, Path(temp), ['vehicle'])

    def test_ap_uses_confidence_order_instead_of_max_iou_assignment(self):
        gt = GroundTruthObject((0,0,10,10), 'vehicle', 'one')
        preds = (Detection((1,0,11,10), .9, 'vehicle'), Detection((0,0,10,10), .1, 'vehicle'))
        result = score_detections([MetricFrame('a', (gt,), preds), MetricFrame('missing', (gt,))])
        self.assertEqual((result['tp'], result['fp'], result['fn']), (1,1,1))
        self.assertAlmostEqual(result['ap_by_class']['vehicle'], .5)

    def test_failure_state_and_json_hash_can_be_audited(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'失败状态.json'
            write_json(path, dict(status='failed', error='worker interrupted', family='YOLO11'))
            before = sha256(path)
            self.assertEqual(read_json(path)['status'], 'failed')
            self.assertEqual(before, sha256(path))

    def test_full_metrics_keep_missing_gt_and_negative_denominators(self):
        gt = GroundTruthObject((0,0,10,10), 'vehicle', 'one', pixel_scale='small', official_occlusion_state='0')
        meta = (('namespace','a'), ('camera_side','vehicle'))
        frames = [MetricFrame('a', (gt,), (Detection((0,0,10,10),.9,'vehicle'),), metadata=meta),
                  MetricFrame('b', (gt,), metadata=meta), MetricFrame('negative', (), metadata=meta)]
        result = metric_summary(frames, .25)
        self.assertEqual((result['tp'], result['fn'], result['negative_frames']), (1,1,1))
        self.assertAlmostEqual(result['map50_95'], .5)
        self.assertEqual(result['conditional_strata']['pixel_scale']['small']['gt'], 2)
        self.assertEqual(result['false_positives_per_empty_frame'], 0)


if __name__ == '__main__':
    unittest.main()
