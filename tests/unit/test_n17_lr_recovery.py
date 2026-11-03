"""恢复编排机制：不运行CUDA/训练，不改变原六项预算或失败证据。"""
import copy
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[2]/'scripts/recover_n17_lr_contrast.py'
spec = importlib.util.spec_from_file_location('n17_recovery_entry', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RecoveryTests(unittest.TestCase):
    def test_only_exact_checkpoint_and_one_replayed_epoch_are_approved(self):
        checkpoint = {'status':'completed', 'epoch':6}
        batches = {'image_exposures':119021, 'per_epoch':[{'epoch':i,'images':17003} for i in range(1,8)],
                   'batch_attempts':59514}
        result = module.validate_budget(checkpoint, batches)
        self.assertEqual(result['maximum_total_exposures'], 1037183)
        self.assertEqual(result['approved_extra_exposures'], 17003)
        for epoch in (5,7):
            with self.assertRaises(ValueError): module.validate_budget({**checkpoint,'epoch':epoch},batches)
        with self.assertRaises(ValueError):
            module.validate_budget(checkpoint,{**batches,'image_exposures':119022})

    def test_recovery_keeps_recipe_schedule_and_original_config_unchanged(self):
        config={'output_root':'old','training':{'epochs':300,'lr0':.001,'batch':2},
                'local_profile':{'model_id':'YOLO26n','seed':17,'recipe_sha256':'original'},
                'gates':{'pedestrian':.9},'_config_sha256':'input-only'}
        original=copy.deepcopy(config)
        checkpoint={'epoch':6,'last_sha256':'last','applied_updates':50989,'batch_attempts':51012}
        result=module.recovery_configuration(config,Path('source'),checkpoint,
                                              {'class_transfer':{'aliases':{'person':'pedestrian'}}},Path('new'))
        self.assertEqual(config,original)
        self.assertEqual(result['training'],config['training'])
        self.assertEqual(result['gates'],config['gates'])
        self.assertEqual(result['local_profile']['recipe_sha256'],'original')
        self.assertEqual(result['local_profile']['resume_epoch'],6)
        self.assertNotIn('_config_sha256',result)
        result['local_profile']['seed']=29
        with self.assertRaises(ValueError):
            module.recovery_configuration(result,Path('source'),checkpoint,{'class_transfer':{}},Path('new'))

    def test_refuses_second_recovery_or_another_task(self):
        first={'id':'YOLO26n_lr_0p001','family':'YOLO26n','seed':17,'learning_rate':.001,'status':'training'}
        state={'source_fingerprint_sha256':module.FINGERPRINT,'tasks':[first]+[{'status':'planned'} for _ in range(5)]}
        self.assertEqual(module.validate_state(state),first)
        for changed in ({**state,'recovery':{'started':True}},
                        {**state,'source_fingerprint_sha256':'different'},
                        {**state,'tasks':state['tasks'][:-1]}):
            with self.assertRaises(ValueError):module.validate_state(changed)
        state['tasks'][0]['status']='failed'
        with self.assertRaises(ValueError):module.validate_state(state)


if __name__ == '__main__': unittest.main()
