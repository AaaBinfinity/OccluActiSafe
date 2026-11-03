import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from occluactisafe.experiments.detector_controlled_trial import (
    validate_trial, controlled_termination, pause_after_committed_epoch,
)
from occluactisafe.experiments.detector_stopping import PlateauStopper, replay_stopping
from occluactisafe.experiments.detector_local_checkpoints import publish_weights


class ControlledTrialTests(unittest.TestCase):
    def config(self):
        return {'training': {'epochs': 300, 'batch': 2, 'nbs': 2, 'workers': 0,
                            'amp': True, 'imgsz': 960, 'optimizer': 'SGD', 'warmup_epochs': 0., 'lr0': .001},
                'budget_policy': {'criterion': 'native_fitness', 'minimum_epochs': 50, 'patience': 30},
                'local_profile': {'model_id': 'YOLO26n', 'seed': 17, 'pause_after_epoch': None,
                                  'controlled_trial': {'protocol': 'n17-lr-6x10-v1',
                                                       'stop_after_epoch': 10, 'learning_rate': .001}}}

    def test_only_six_approved_conditions_are_allowed(self):
        cfg = self.config()
        for family in ('YOLO26n', 'YOLOv8n', 'YOLO11n'):
            for lr in (.001, .0003):
                cfg['local_profile']['model_id'] = family
                cfg['training']['lr0'] = cfg['local_profile']['controlled_trial']['learning_rate'] = lr
                self.assertEqual(validate_trial(cfg, family, 17)['learning_rate'], lr)
        for key, value in [('batch', 8), ('epochs', 10), ('imgsz', 1280), ('amp', False)]:
            bad = copy.deepcopy(cfg); bad['training'][key] = value
            with self.assertRaises(ValueError): validate_trial(bad, 'YOLO11n', 17)
        with self.assertRaises(ValueError): validate_trial(cfg, 'YOLO26s', 17)
        with self.assertRaises(ValueError): validate_trial(cfg, 'YOLO11n', 29)

    def test_partial_trace_is_paused_and_ordinary_completion_stays_strict(self):
        policy = self.config()['budget_policy']; stopper = PlateauStopper(50, 30); records = []
        for epoch in range(1, 11):
            stopper(epoch, .5 if epoch == 3 else .4); records.append(dict(stopper.last_decision))
        result = controlled_termination(records, policy, 300, 10)
        self.assertEqual((result['actual_epochs'], result['best_epoch']), (10, 3))
        self.assertFalse(result['full_training_complete'])
        self.assertEqual(result['stop_reason'], 'declared_development_pause')
        with self.assertRaises(ValueError): replay_stopping(records, policy, 300)
        for trace, maximum, stop in [(records[:-1], 300, 10), (records, 10, 10), (records, 300, 9)]:
            with self.assertRaises(ValueError): controlled_termination(trace, policy, maximum, stop)
        records[5]['best_fitness'] = .99
        with self.assertRaises(ValueError): controlled_termination(records, policy, 300, 10)

    def test_pause_requires_committed_weights_and_final_validation_is_not_an_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory); trainer = SimpleNamespace(save_dir=path, epoch=8, stop=False)
            self.assertIsNone(pause_after_committed_epoch(trainer))
            trainer.epoch = 9
            with self.assertRaises((ValueError, FileNotFoundError)): pause_after_committed_epoch(trainer)
            publish_weights(path, b'controlled mechanism fixture', True,
                            {'epoch': 10, 'applied_updates': 9, 'batch_attempts': 10})
            self.assertEqual(pause_after_committed_epoch(trainer)['epoch'], 10)
            self.assertTrue(trainer.stop)
            trainer._n17_final_eval = True; trainer.epoch = 300
            self.assertIsNone(pause_after_committed_epoch(trainer))


if __name__ == '__main__': unittest.main()
