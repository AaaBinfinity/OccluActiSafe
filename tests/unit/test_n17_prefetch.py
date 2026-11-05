from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from occluactisafe.experiments.detector_local import local_options,build_local_config
from occluactisafe.experiments.detector_local_training import rich_trainer_class


class PrefetchTests(unittest.TestCase):
    def test_optional_settings_validate_and_preserve_legacy_recipe(self):
        old=local_options();self.assertIsNone(old['VALIDATION_WORKERS']);self.assertIsNone(old['PREFETCH_FACTOR'])
        for key,values in [('VALIDATION_WORKERS',[-1,True,1.5,129]),('PREFETCH_FACTOR',[0,True,1.5,9])]:
            for value in values:
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):local_options(overrides={key:value})
        env={'font_path':'fixture','source':{'source_fingerprint_sha256':'fixed'},'input_verification':{'source_index_sha256':'fixed'}}
        snapshot={'index_sha256':'fixed'}
        one=build_local_config(old,env,snapshot,Path('unused'))
        two=build_local_config({**old,'VALIDATION_WORKERS':0,'PREFETCH_FACTOR':1},env,snapshot,Path('unused'))
        self.assertNotIn('validation_workers',one['local_profile'])
        self.assertNotIn('prefetch_factor',one['local_profile'])
        self.assertEqual(one['training'],two['training']);self.assertEqual(one['gates'],two['gates'])
        self.assertNotEqual(one['local_profile']['recipe_sha256'],two['local_profile']['recipe_sha256'])
        self.assertEqual((two['local_profile']['validation_workers'],two['local_profile']['prefetch_factor']),(0,1))

    def test_prefetch_is_applied_before_native_iterator_starts(self):
        cls=rich_trainer_class(16,None,{'prefetch_factor':1});trainer=object.__new__(cls)
        loader=SimpleNamespace(iterator=None,num_workers=1,prefetch_factor=4)
        with patch.object(cls.__bases__[0],'get_dataloader',return_value=loader):
            self.assertIs(trainer.get_dataloader('fixture',16,mode='train'),loader)
        self.assertEqual(loader.prefetch_factor,1)
        loader.iterator=object()
        with patch.object(cls.__bases__[0],'get_dataloader',return_value=loader),self.assertRaisesRegex(RuntimeError,'已启动'):
            trainer.get_dataloader('fixture',16,mode='train')

    def test_zero_training_workers_has_no_phantom_prefetch_queue(self):
        cls=rich_trainer_class(16,None,{'prefetch_factor':1});trainer=object.__new__(cls)
        loader=SimpleNamespace(iterator=None,num_workers=0,prefetch_factor=None)
        with patch.object(cls.__bases__[0],'get_dataloader',return_value=loader):trainer.get_dataloader('fixture',16,mode='train')
        self.assertIsNone(loader.prefetch_factor)

    def test_validation_uses_own_workers_and_batch_without_mutating_training_args(self):
        cls=rich_trainer_class(16,None,{'validation_workers':0,'validation_batch':4})
        trainer=object.__new__(cls);trainer.args=SimpleNamespace(workers=1);trainer.device='cpu'
        with patch.object(cls,'build_dataset',return_value='real_contract_fixture') as dataset,patch('ultralytics.data.build.build_dataloader',return_value='loader') as build:
            self.assertEqual(trainer.get_dataloader('fixture',32,rank=-1,mode='val'),'loader')
        dataset.assert_called_once_with('fixture','val',4)
        self.assertEqual(build.call_args.kwargs,{'batch':4,'workers':0,'shuffle':False,'rank':-1,'drop_last':False,'device':'cpu'})
        self.assertEqual(trainer.args.workers,1)

    def test_final_checkpoint_validator_carries_the_explicit_worker_limit(self):
        cls=rich_trainer_class(16,None,{'validation_workers':0,'validation_batch':4});trainer=object.__new__(cls)
        class Validator:
            args=SimpleNamespace(workers=1,batch=32)
        with patch.object(cls.__bases__[0],'get_validator',return_value=Validator()):v=trainer.get_validator()
        self.assertEqual((v.args.workers,v.args.batch),(0,4))

    def test_real_windows_cpu_loader_snapshots_prefetch_one_and_closes_worker(self):
        # 明确为机制夹具，不是本机真实数据训练或GPU利用率验收。
        import torch
        from ultralytics.data.build import build_dataloader
        loader=build_dataloader(torch.utils.data.TensorDataset(torch.arange(8)),batch=2,workers=1,shuffle=False,device='cpu')
        try:
            self.assertIsNone(loader.iterator);loader.prefetch_factor=1
            first=next(iter(loader))[0];self.assertEqual(first.tolist(),[0,1])
            self.assertEqual(loader.iterator._prefetch_factor,1)
            workers=list(loader.iterator._workers);self.assertTrue(workers[0].is_alive())
        finally:loader.close()
        self.assertFalse(any(w.is_alive() for w in workers))


if __name__=='__main__':unittest.main()
