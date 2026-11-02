import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from occluactisafe.experiments.detector_models import expanded_models, verify_weights, prepare_weights, ASSETS
from occluactisafe.experiments.detector_stopping import PlateauStopper, replay_stopping


class ScalingTests(unittest.TestCase):
    def test_all_sizes_have_unique_weights_and_three_architectures(self):
        models = expanded_models(['n', 's', 'm', 'l', 'x'])
        self.assertEqual(len(models), 15)
        self.assertEqual(len({m['family'] for m in models}), 15)
        self.assertEqual({m['weights'] for m in models}, set(ASSETS))
        self.assertEqual([m['family'] for m in models[:3]], ['YOLO26n', 'YOLOv8n', 'YOLO11n'])
        for invalid in ([], ['x', 'x'], ['huge'], 'nsmlx'):
            with self.assertRaises(ValueError): expanded_models(invalid)

    def test_missing_or_changed_asset_rejected_without_network_or_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); target = root/'yolo26n.pt'; target.write_bytes(b'changed fixture')
            with patch('urllib.request.urlopen') as network:
                with self.assertRaises(ValueError): verify_weights(root, expanded_models(['n']))
                with self.assertRaises(ValueError): prepare_weights(root, expanded_models(['n']))
                network.assert_not_called()
            self.assertEqual(target.read_bytes(), b'changed fixture')
            self.assertEqual(len(list(root.glob('准备_*.json'))), 1)

    def test_plateau_waits_for_minimum_and_replays_actual_termination(self):
        stopper = PlateauStopper(5, 2); records = []
        for epoch, fitness in enumerate([.5, .4, .4, .4, .4], 1):
            decision = stopper(epoch, fitness); records.append(copy.deepcopy(stopper.last_decision))
            self.assertEqual(decision, epoch == 5)
        result = replay_stopping(records, {'minimum_epochs': 5, 'patience': 2}, 10)
        self.assertEqual((result['actual_epochs'], result['stop_reason']), (5, 'early_stopping'))
        changed = copy.deepcopy(records); changed[-1]['best_epoch'] = 2
        with self.assertRaises(ValueError): replay_stopping(changed, {'minimum_epochs': 5, 'patience': 2}, 10)
        with self.assertRaises(ValueError): replay_stopping(records[:-1], {'minimum_epochs': 5, 'patience': 2}, 10)

    def test_improvement_resets_patience_zero_fitness_does_not_claim_convergence(self):
        stopper = PlateauStopper(1, 2)
        for epoch, value in enumerate([.1, .1, .2, .2, .2], 1):
            self.assertEqual(stopper(epoch, value), epoch == 5)
        zero = PlateauStopper(1, 2); records = []
        for epoch in range(1, 6):
            self.assertFalse(zero(epoch, 0)); records.append(copy.deepcopy(zero.last_decision))
        result = replay_stopping(records, {'minimum_epochs': 1, 'patience': 2}, 5)
        self.assertEqual(result['stop_reason'], 'maximum_epochs'); self.assertFalse(result['convergence_guaranteed'])
        with self.assertRaises(ValueError): zero(6, float('nan'))
        with self.assertRaises(ValueError): PlateauStopper(1, 2)(2, .5)


if __name__ == '__main__': unittest.main()
