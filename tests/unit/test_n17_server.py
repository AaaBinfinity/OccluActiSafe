import copy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import yaml

from occluactisafe.experiments.detector_data import load_config
from occluactisafe.experiments.detector_transfer import apply_class_aliases
from occluactisafe.experiments.detector_server import (
    server_settings,plan_tasks,execution_lock,final_alias,validate_completed,
    collect_results,run_job,preflight,
    worker_cuda_selector,
)
from occluactisafe.experiments.run_store import write_json,read_json,sha256


class ServerTests(unittest.TestCase):
    def settings(self,root):
        cfg=yaml.safe_load(Path('configs/experiments/n17_server.yaml').read_text(encoding='utf-8'))
        cfg.update(input_root=str(root/'input'),output_root=str(root/'results'))
        p=root/'server.yaml';p.write_text(yaml.safe_dump(cfg),encoding='utf-8')
        return server_settings(p)

    def test_formal_requires_nine_training_and_one_complete_evaluation(self):
        cfg=load_config()
        formal=plan_tasks('formal',['person_alias'],cfg['models'],cfg['seeds'])
        self.assertEqual(sum(t['kind']=='train' for t in formal),9)
        self.assertEqual({t.get('seed') for t in formal if t['kind']=='train'},{17,29,43})
        self.assertEqual(sum(t['kind']=='evaluate' for t in formal),1)
        pilot=plan_tasks('pilot',['baseline','person_alias'],cfg['models'],cfg['seeds'])
        self.assertEqual(len(pilot),8)
        self.assertEqual({t.get('seed') for t in pilot if t['kind']=='train'},{17})

    def test_gpu_selection_is_resolved_before_worker_imports_torch(self):
        self.assertEqual(worker_cuda_selector(2),'2')
        self.assertEqual(worker_cuda_selector(1,'2,5'),'5')
        self.assertEqual(worker_cuda_selector(0,'GPU-abc'),'GPU-abc')
        with self.assertRaises(ValueError):worker_cuda_selector(2,'2,5')

    def test_settings_use_config_paths_instead_of_cwd_and_reject_unknown_gates(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();settings=self.settings(root);cwd=Path.cwd()
            try:
                os.chdir(root)
                self.assertEqual(server_settings(settings['_path'])['input_root'],root/'input')
            finally:os.chdir(cwd)
            cfg=yaml.safe_load(Path(settings['_path']).read_text());cfg['minimum_recall']=.1
            Path(settings['_path']).write_text(yaml.safe_dump(cfg))
            with self.assertRaises(ValueError):server_settings(settings['_path'])

    def test_person_alias_preserves_other_classes_and_rejects_many_to_one(self):
        model=SimpleNamespace(model=SimpleNamespace(names={0:'person',1:'car',2:'truck'}))
        before=copy.deepcopy(model.model.names)
        transfer=apply_class_aliases(model,{'person':'pedestrian'})
        self.assertEqual(model.model.names,{0:'pedestrian',1:'car',2:'truck'})
        self.assertEqual(transfer['source_names'],before)
        with self.assertRaises(ValueError):apply_class_aliases(model,{'car':'vehicle','truck':'vehicle'})

    def test_offline_font_cache_uses_framework_subdirectory(self):
        from occluactisafe.experiments.visual_fonts import prepare_font_cache
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);font=root/'local_fixture_font.ttc';font.write_bytes(b'font cache mechanism fixture')
            prepare_font_cache(root/'settings',str(font))
            for name in ('Arial.ttf','Arial.Unicode.ttf'):
                self.assertEqual((root/'settings/Ultralytics'/name).read_bytes(),font.read_bytes())
            self.assertFalse((root/'settings/Arial.ttf').exists())

    def test_lock_is_exclusive_and_released_after_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'.lock'
            with execution_lock(path):
                with self.assertRaises(RuntimeError):
                    with execution_lock(path):pass
            with execution_lock(path):pass

    def test_completed_result_with_bad_returncode_or_changed_manifest_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);task={'kind':'evaluate','attempts':[{'directory':str(root),'returncode':1}]}
            with self.assertRaises(ValueError):validate_completed(task,'unused','fixture')
            m=root/'manifest.json';write_json(m,[])
            write_json(root/'worker_result.json',{'status':'completed','manifest':str(m),'manifest_sha256':sha256(m),'directory':str(root)})
            task['attempts'][0]['returncode']=0;validate_completed(task,'unused','fixture')
            write_json(m,[{'path':'changed'}])
            with self.assertRaises(ValueError):validate_completed(task,'unused','fixture')

    def test_failed_task_retry_keeps_old_attempt_and_skips_completed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();settings=self.settings(root)
            rows=[{'key':'spd_vehicle_1','role':'train','domain':'SPD','image':'H:/old.jpg','source_image':'H:/source.jpg','gt':[]}]
            env={'source':{'source_commit':'a'*40,'source_fingerprint_sha256':'fixture'},
                 'input_verification':{'manifest_sha256':'fixture-input','source_index_sha256':'fixture-index'},
                 'gpu':{'uuid':'fixture-gpu'},'worker_cuda_visible_devices':'0'}
            calls=[]
            def execute(params,log):
                task=read_json(params);calls.append(task['id']);Path(log).write_text('mechanism failed attempt\n')
                if task['id'].endswith('YOLO26_17'):
                    write_json(Path(task['attempt'])/'worker_result.json',{'status':'completed'});return 0
                return 1
            with patch('occluactisafe.experiments.detector_server.preflight',return_value=(rows,env,root)),patch('occluactisafe.experiments.detector_server.execute_worker',side_effect=execute),patch('occluactisafe.experiments.detector_server.validate_completed',return_value={}):
                with self.assertRaises(RuntimeError):run_job(settings,'pilot')
                job=next((settings['output_root']/'02_运行产物').iterdir());first=read_json(job/'任务状态.json')
                self.assertEqual(first['status'],'failed');self.assertEqual(len(first['tasks']),8)
                old=Path(first['tasks'][1]['attempts'][0]['directory'])/'控制台.txt';content=old.read_bytes()
                with self.assertRaises(RuntimeError):run_job(settings,'pilot',job)
                second=read_json(job/'任务状态.json')
            self.assertEqual(calls.count('train_baseline_YOLO26_17'),1)
            self.assertEqual(len(second['tasks'][0]['attempts']),1)
            self.assertEqual(len(second['tasks'][1]['attempts']),2)
            self.assertEqual(old.read_bytes(),content)
            self.assertFalse((settings['output_root']/'01_最终结果').exists())

    def test_failed_job_can_be_collected_without_input_images_or_success_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();settings=self.settings(root);job=settings['output_root']/'02_运行产物/job_fixture';job.mkdir(parents=True)
            write_json(job/'任务状态.json',{'status':'failed','device':0,'source_commit':'a'*40,'run_id':'job_fixture'})
            (job/'真实失败说明.txt').write_text('mechanism fixture, no real model',encoding='utf-8')
            original=root/'input';original.mkdir();(original/'not_to_include.jpg').write_bytes(b'input fixture')
            output=collect_results(settings,job)
            with zipfile.ZipFile(output) as z:
                entries=z.namelist();self.assertFalse(any('not_to_include' in n for n in entries))
                meta=json.loads(z.read('job_fixture/回传文件清单.json'))
                self.assertFalse(meta['complete']);self.assertEqual(meta['status_at_collection'],'failed')
            self.assertEqual(output.with_suffix('.zip.sha256').read_text().split()[0],sha256(output))

    def test_formal_orchestrator_completes_all_seeds_before_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();settings=self.settings(root);calls=[]
            env={'source':{'source_commit':'a'*40,'source_fingerprint_sha256':'fixture'},
                 'input_verification':{'manifest_sha256':'fixture-input','source_index_sha256':'fixture-index'},
                 'gpu':{'uuid':'fixture-gpu'},'worker_cuda_visible_devices':'0'}
            rows=[{'key':'spd_1','role':'train','domain':'SPD','image':'H:/old.jpg','source_image':'H:/old.jpg','gt':[]}]
            def execute(params,log):
                p=read_json(params);calls.append(p['kind']);folder=Path(p['attempt'])
                Path(log).write_text('mocked orchestration, no real training')
                if p['kind']=='train':
                    write_json(folder/'worker_result.json',{'status':'completed','family':p['family'],'seed':p['seed'],
                               'config_sha256':'fixture-config','data_index_sha256':'fixture-index'})
                else:
                    matrix=read_json(p['training_matrix']);self.assertEqual(len(matrix['runs']),9)
                    self.assertEqual({r['seed'] for r in matrix['runs']},{17,29,43})
                    result=folder/'metrics';result.mkdir();write_json(result/'离线预选.json',{'status':'no_qualified_candidate','selected_candidate':None})
                    write_json(folder/'worker_result.json',{'status':'completed','directory':str(result)})
                return 0
            with patch('occluactisafe.experiments.detector_server.preflight',return_value=(rows,env,root)),patch('occluactisafe.experiments.detector_server.execute_worker',side_effect=execute),patch('occluactisafe.experiments.detector_server.validate_completed',return_value={}),patch('occluactisafe.experiments.detector_server.verify_input',return_value=(rows,{'status':'passed'})):
                job=run_job(settings,'formal');before=(job/'任务状态.json').read_bytes()
                self.assertEqual(calls,['train']*9+['evaluate'])
                self.assertEqual(read_json(job/'任务状态.json')['status'],'completed')
                run_job(settings,'formal',job)
                self.assertEqual((job/'任务状态.json').read_bytes(),before)
                self.assertEqual(len(calls),10)

    @unittest.skipUnless(os.name=='nt','此项实际验证Windows目录联接')
    def test_last_entry_is_same_directory_and_does_not_rename_job(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();job=root/"02_运行产物/含空格和'引号";job.mkdir(parents=True)
            (job/'证据.txt').write_text('stable',encoding='utf-8');before=sha256(job/'证据.txt')
            alias=final_alias(root,job)
            self.assertTrue(os.path.samefile(alias,job));self.assertEqual(sha256(alias/'证据.txt'),before)
            self.assertTrue(job.is_dir())

    def test_cpu_only_environment_fails_before_data_or_model_load(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();settings=self.settings(root);settings['input_root'].mkdir()
            with patch('torch.cuda.is_available',return_value=False),patch('occluactisafe.experiments.detector_server.verify_input') as data:
                with self.assertRaisesRegex(RuntimeError,'CUDA'):preflight(settings)
                data.assert_not_called()
            reports=list((settings['output_root']/'03_验收证据').glob('*/服务器预检.json'))
            self.assertEqual(read_json(reports[0])['status'],'failed')


if __name__=='__main__':unittest.main()
