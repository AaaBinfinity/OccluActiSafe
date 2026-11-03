import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from occluactisafe.experiments import detector_lr_contrast as module
from occluactisafe.experiments.run_store import write_json, read_json, sha256


class LearningRateContrastTests(unittest.TestCase):
    def settings(self, root):
        root = root.resolve()  # 与公共入口一致，规范Windows临时目录的8.3别名。
        return {'INPUT_ROOT': root/'input', 'OUTPUT_ROOT': root/'output',
                'WAIT_FOR_RUN': root/'output'/'previous', 'RUN_DIR': '', 'DEVICE': 0,
                'ACTION': 'plan', 'WAIT_TIMEOUT_HOURS': 1, '_bytes': b'unit fixture',
                '_path': 'fixture', '_sha256': 'fixture'}

    def test_exact_six_tasks_and_no_extra_models_or_seeds(self):
        tasks = module.tasks()
        self.assertEqual(len(tasks), 6)
        self.assertEqual({(r['family'],r['seed'],r['learning_rate']) for r in tasks},
                         {(m,17,lr) for m in module.MODELS for lr in module.LEARNING_RATES})
        self.assertEqual(module.PROTOCOL['planned_image_exposures'], 17003*10*6)

    def test_default_check_does_not_import_or_launch_gpu_training(self):
        cfg = module.options()
        self.assertEqual(cfg['ACTION'], 'check')
        self.assertEqual(cfg['WAIT_TIMEOUT_HOURS'], 24)
        self.assertEqual(cfg['INPUT_ROOT'].drive, 'H:')
        with self.assertRaises(ValueError): module.options(action='train')

    def test_user_run_must_finish_successfully_before_gpu_is_ready(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write_json(root/'本机训练任务.json', {'status':'running'})
            self.assertFalse(module.prior_run_ready(root))
            write_json(root/'本机训练任务.json', {'status':'failed'})
            with self.assertRaises(RuntimeError): module.prior_run_ready(root)
            write_json(root/'本机训练任务.json', {'status':'completed','worker_returncode':0,'audit':{'status':'passed'}})
            self.assertTrue(module.prior_run_ready(root))

    def test_one_authorization_pointer_reuses_plan_and_rejects_source_change(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg=self.settings(Path(temp))
            source={'source_commit':'a'*40,'source_fingerprint_sha256':'fixture'}
            with patch.object(module,'source_info',return_value=source):
                job, first=module.prepare_study(cfg)
                same, second=module.prepare_study(cfg)
            self.assertEqual(job,same); self.assertEqual(first['tasks'],second['tasks'])
            self.assertEqual(len(list((cfg['OUTPUT_ROOT']/'04_开发与复核/学习率对照').iterdir())),1)
            with patch.object(module,'source_info',return_value={**source,'source_fingerprint_sha256':'changed'}):
                with self.assertRaises(ValueError): module.prepare_study(cfg)

    def test_all_six_configs_differ_only_in_lr_and_keep_300_epoch_schedule(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg=self.settings(Path(temp));environment={'font_path':'fixture',
                'source':{'source_fingerprint_sha256':'fixture'},'input_verification':{'source_index_sha256':'fixture'}}
            snapshot={'index_sha256':'fixture'}
            configs=[module.trial_config(cfg,t,environment,snapshot,Path(temp)/t['id']) for t in module.tasks()]
            self.assertTrue(module.verify_fair_configs(configs))
            self.assertTrue(all(r['training']['epochs']==300 for r in configs))
            wrong=copy.deepcopy(configs);wrong[-1]['training']['weight_decay']=.1
            with self.assertRaises(ValueError):module.verify_fair_configs(wrong)
            wrong=copy.deepcopy(configs);wrong[-1]['gates']['minimum_recall_iou50']['pedestrian']=.2
            with self.assertRaises(ValueError):module.verify_fair_configs(wrong)
            with self.assertRaises(ValueError):module.verify_fair_configs(configs[:-1])

    def test_failed_attempt_cannot_silently_restart_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);config=root/'config.yaml';config.write_text('fixture',encoding='utf-8')
            task={'family':'YOLO26n','directory':str(root),'config':str(config),
                  'config_sha256':sha256(config),'status':'training'}
            with patch.object(module,'launch') as launch:
                with self.assertRaises(RuntimeError):module.run_tasks({},root,{'tasks':[task]})
            launch.assert_not_called()

    def test_prior_run_failure_is_recorded_without_gpu_preflight(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();cfg=self.settings(root);cfg['ACTION']='run'
            job=root/'study';job.mkdir();state={'status':'planned','tasks':module.tasks()}
            with patch.object(module,'prepare_study',return_value=(job,state)), \
                    patch.object(module,'prior_run_ready',side_effect=RuntimeError('prior failed')), \
                    patch.object(module,'preflight') as preflight:
                with self.assertRaises(RuntimeError):module.run(cfg)
            preflight.assert_not_called()
            self.assertEqual(read_json(job/'对照任务.json')['status'],'failed')
            self.assertTrue(all(t['status']=='planned' for t in state['tasks']))

    def test_wait_timeout_is_bounded_and_never_starts_gpu(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();cfg=self.settings(root);cfg['ACTION']='run'
            job=root/'study';job.mkdir();state={'status':'planned','tasks':module.tasks()}
            with patch.object(module,'prepare_study',return_value=(job,state)), \
                    patch.object(module,'prior_run_ready',return_value=False), \
                    patch.object(module.time,'monotonic',side_effect=[0.,3601.]), \
                    patch.object(module,'preflight') as preflight:
                with self.assertRaises(TimeoutError):module.run(cfg)
            preflight.assert_not_called()
            self.assertIn('超过',read_json(job/'对照任务.json')['error'])

    def test_completion_marker_requires_six_tasks_and_unchanged_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);file=root/'result.json';file.write_text('{}',encoding='utf-8')
            state={'development_comparison_complete':True,'tasks':[{'status':'completed'} for _ in range(6)]}
            write_json(root/'完成产物清单.json',[{'path':file.name,'sha256':sha256(file)}])
            module.verify_completed_study(root,state)
            state['tasks'][0]['status']='training'
            with self.assertRaises(ValueError):module.verify_completed_study(root,state)
            state['tasks'][0]['status']='completed';file.write_text('{"changed":true}',encoding='utf-8')
            with self.assertRaises(ValueError):module.verify_completed_study(root,state)


if __name__ == '__main__': unittest.main()
