"""M06与统一指标手算、缺失和状态时间机制测试。"""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from occluactisafe.prediction.baselines.cv import predict, predict_state
from occluactisafe.evaluation.prediction_metrics import score_target, MetricAccumulator
from occluactisafe.prediction.evaluate import load_evaluation_config


class CVTests(unittest.TestCase):
    def test_real_gap_and_masked_values(self):
        x = np.array([[[-3., 0], [9999, np.nan], [0, 0]]])
        r = predict(x, [-.3, -.1, 0], np.array([[True, False, True]]), future_times=[.1, 3])
        np.testing.assert_allclose(r['velocity'], [[10, 0]], atol=1e-12)
        np.testing.assert_allclose(r['positions'], [[[[1, 0], [30, 0]]]], atol=1e-12)
        self.assertEqual(r['positions'].shape, (1, 1, 2, 2))
        self.assertEqual(r['K'], 1)

    def test_stationary_turn_short_empty_and_stale(self):
        x = np.array([[[0, 0], [0, 0], [0, 0]], [[0, 0], [1, 0], [1, 1]],
                      [[0, 0], [0, 0], [4, 4]], [[0, 0], [0, 0], [0, 0]]], float)
        m = np.array([[1, 1, 1], [1, 1, 1], [0, 0, 1], [0, 0, 0]], bool)
        r = predict(x, [-2, -1, 0], m, future_times=[1])
        np.testing.assert_array_equal(r['valid'], [True, True, False, False])
        np.testing.assert_allclose(r['positions'][:2, 0, 0], [[0, 0], [1, 2]])
        stale = predict([[[-3, 0], [0, 0]]], [-.6, -.3], np.ones((1, 2), bool), future_times=[.1])
        np.testing.assert_allclose(stale['positions'], [[[[4, 0]]]])

    def test_state_time_not_observation_time(self):
        r = predict_state([2, 0], [10, 0], state_time=.2, last_seen_time=0, anchor_time=.3, future_times=[0, .1])
        np.testing.assert_allclose(r['positions'], [[3, 0], [4, 0]])
        self.assertEqual(r['last_seen_time'], 0)

    def test_bad_times_and_finite_history_rejected(self):
        for times in ([0, 0], [.1, .2], [-1, float('nan')]):
            with self.assertRaises(ValueError):
                predict([[[0, 0], [1, 0]]], times, np.ones((1, 2), bool), future_times=[1])
        with self.assertRaises(ValueError):
            predict([[[np.nan, 0], [1, 0]]], [-1, 0], np.ones((1, 2), bool), future_times=[1])

    def test_metrics_hand_calculation_mask_and_topk(self):
        y = np.array([[0., 0], [np.nan, np.nan], [0, 0]])
        p = np.array([[[3., 4], [9999, 9999], [0, 10]], [[0, 0], [0, 0], [3, 4]]])
        r = score_target(p, y, np.array([1, 0, 1], bool), top_k=2, logits=[1, 0])
        self.assertEqual((r['ADE_m'], r['FDE_m'], r['minADE_m'], r['minFDE_m']), (7.5, 10, 2.5, 5))
        self.assertTrue(r['miss'])
        self.assertEqual(score_target(p, y, np.array([1, 0, 1], bool), top_k=1, logits=[0, 1])['ADE_m'], 2.5)
        with self.assertRaises(ValueError):
            score_target(p, y, np.ones(3, bool))

    def test_missing_prediction_denominator_and_no_labels(self):
        a = MetricAccumulator()
        for valid, mask in ((True, [1, 1]), (False, [1, 1]), (False, [0, 0])):
            a.add(score_target(np.zeros((1, 2, 2)), np.zeros((2, 2)), np.array(mask, bool), prediction_valid=valid))
        s = a.summary()
        self.assertEqual((s['targets'], s['scored'], s['missing_prediction'], s['no_labels']), (3, 1, 1, 1))
        self.assertEqual((s['MissRate'], s['MissRate_denominator'], s['error_denominator']), (.5, 2, 1))

    def test_rigid_world_restoration_keeps_meters(self):
        from occluactisafe.common.coordinates import local_to_world, Pose2D
        p = np.array([[[1., 2], [4, 6]]]); y = np.array([[0., 0], [0, 0]])
        pose = Pose2D(700000, 3000000, 1.2)
        world_p = np.array([[local_to_world(*point, pose) for point in p[0]]])
        world_y = np.array([local_to_world(*point, pose) for point in y])
        local = score_target(p, y, np.ones(2, bool)); world = score_target(world_p, world_y, np.ones(2, bool))
        self.assertAlmostEqual(local['ADE_m'], world['ADE_m'], places=8)

    def test_config_precedence_and_sealed_test(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / 'cv.yaml'
            file.write_text('windows_dir: input\noutput_root: output\nmax_windows: 3\n', encoding='utf-8')
            c = load_evaluation_config(file, overrides={'max_windows': 2}, cli_overrides={'max_windows': 0})
            self.assertEqual(c['max_windows'], 0)
            for change in ({'role': 'test'}, {'top_k': 6}, {'role': 'risk_calibration'}, {'unknown': 0}):
                with self.assertRaises(ValueError):
                    load_evaluation_config(file, overrides=change)


if __name__ == '__main__':
    unittest.main()
