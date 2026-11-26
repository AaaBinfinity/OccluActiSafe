from collections import Counter
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from occluactisafe.experiments.detector_local import local_options,build_local_config,audit_local_batches,DEFAULT_OPTIONS
from occluactisafe.experiments.detector_local_checkpoints import publish_weights,verify_checkpoint,carry_committed_logs
from occluactisafe.experiments.run_store import read_json,write_json,sha256
from occluactisafe.experiments.source_version import protected_source


class LocalTaskTests(unittest.TestCase):
    def test_options_execute_hash_and_archive_the_same_captured_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'选项.py'
            original=DEFAULT_OPTIONS.read_bytes()+b"\nSEED = 17\nACTION = 'check'\n"
            path.write_bytes(original)
            options=local_options(path)
            path.write_bytes(original.replace(b'SEED = 17',b'SEED = 29'))
            self.assertNotEqual(path.read_bytes(),original)
            self.assertEqual(options['options_source_bytes'],original)
            self.assertEqual(options['options_sha256'],hashlib.sha256(original).hexdigest())
            self.assertEqual(options['SEED'],17)

    def test_worker_rejects_source_change_before_loading_gpu_or_training(self):
        from occluactisafe.experiments import detector_server as server
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);task=root/'任务.json'
            write_json(task,{'config':'fixture','attempt':str(root)})
            with patch.object(server,'load_config',return_value={'local_profile':{'source_fingerprint_sha256':'old'}}),patch.object(server,'source_info',return_value={'source_fingerprint_sha256':'new'}):
                with self.assertRaisesRegex(ValueError,'源码改变'):server.worker(task)
            self.assertEqual(read_json(root/'worker_failure.json')['status'],'failed')

    def test_model_seed_are_selectable_without_changing_protected_algorithm(self):
        options=local_options(overrides={'MODEL':'yolov8m.pt','SEED':29,'ACTION':'check'})
        self.assertEqual((options['MODEL'],options['SEED'],options['ACTION']),('YOLOv8m',29,'check'))
        self.assertFalse(protected_source('configs/experiments/n17_local.py'))
        self.assertTrue(protected_source('scripts/train_n17_local.py'))
        for invalid in ({'MAX_EPOCHS':301},{'SEED':-1},{'SEED':True},{'MODEL':'YOLO_fake'},{'ACTION':'resume'},{'BATCH':0}):
            with self.assertRaises(ValueError):local_options(overrides=invalid)

    def test_selected_model_seed_do_not_change_common_training_recipe(self):
        one=local_options();two=local_options(overrides={'MODEL':'YOLO11x','SEED':43})
        environment={'font_path':'fixture','source':{'source_fingerprint_sha256':'fixture'},
                     'input_verification':{'source_index_sha256':'fixture'}}
        snapshot={'index_sha256':'fixture'}
        a=build_local_config(one,environment,snapshot,Path('unused'))
        b=build_local_config(two,environment,snapshot,Path('unused'))
        self.assertEqual(a['local_profile']['recipe_sha256'],b['local_profile']['recipe_sha256'])
        self.assertEqual((a['training']['epochs'],a['budget_policy']['minimum_epochs'],a['budget_policy']['patience']),(300,50,30))
        self.assertEqual(a['checkpoint_period'],-1)
        self.assertNotEqual(a['local_profile']['model_id'],b['local_profile']['model_id'])

    def test_optional_cpu_threads_preserve_legacy_defaults_and_freeze_new_recipe(self):
        options=local_options(overrides={'CPU_THREADS':0})
        environment={'font_path':'fixture','source':{'source_fingerprint_sha256':'fixture'},
                     'input_verification':{'source_index_sha256':'fixture'}}
        snapshot={'index_sha256':'fixture'}
        old=build_local_config(options,environment,snapshot,Path('unused'))
        new=build_local_config({**options,'CPU_THREADS':4},environment,snapshot,Path('unused'))
        self.assertNotIn('cpu_threads',old['local_profile'])
        self.assertEqual(new['local_profile']['cpu_threads'],4)
        self.assertNotEqual(old['local_profile']['recipe_sha256'],new['local_profile']['recipe_sha256'])
        self.assertEqual(old['training'],new['training'])
        self.assertEqual(old['gates'],new['gates'])
        for invalid in (-1,129,True,1.5):
            with self.assertRaises(ValueError):local_options(overrides={'CPU_THREADS':invalid})

    def test_fast_configuration_is_safe_and_independent_of_user_options(self):
        fast=local_options(DEFAULT_OPTIONS.with_name('n17_local_fast.py'))
        self.assertEqual((fast['ACTION'],fast['CPU_THREADS']),('check',4))
        self.assertEqual((fast['MAX_EPOCHS'],fast['MIN_EPOCHS'],fast['PATIENCE']),(300,15,10))
        self.assertEqual((fast['WORKERS'],fast['PREFETCH_FACTOR'],fast['VALIDATION_WORKERS']),(2,1,0))
        self.assertEqual((fast['BATCH'],fast['VALIDATION_BATCH'],fast['LR0'],fast['AMP']),(32,4,.001,True))
        self.assertEqual(fast['PAUSE_AFTER_EPOCH'],None)

    def test_larger_batch_probe_keeps_three_real_validation_batch_examples(self):
        from occluactisafe.experiments.detector_local import development_rows
        rows=[{'role':role,'domain':domain,'key':f'{role}_{domain}_{i}',
               'gt':[{'class_id':'vehicle'}]} for role in ('train','model_val')
              for domain in ('SPD','CARLA') for i in range(100)]
        selected=development_rows(rows,16)
        self.assertGreaterEqual(sum(r['role']=='model_val' for r in selected),3*2*16)
        self.assertGreaterEqual(sum(r['role']=='train' for r in selected),3*16)
        self.assertEqual(len({r['key'] for r in selected}),len(selected))
        self.assertEqual(Counter(r['role'] for r in development_rows(rows)),{'train':32,'model_val':16})

    def test_validation_batch_is_frozen_separately_without_changing_training(self):
        opts=local_options(overrides={'VALIDATION_BATCH':0})
        env={'font_path':'fixture','source':{'source_fingerprint_sha256':'fixture'},
             'input_verification':{'source_index_sha256':'fixture'}}
        a=build_local_config(opts,env,{'index_sha256':'fixture'},Path('unused'))
        b=build_local_config({**opts,'VALIDATION_BATCH':4},env,{'index_sha256':'fixture'},Path('unused'))
        self.assertEqual(a['training'],b['training'])
        self.assertNotEqual(a['local_profile']['recipe_sha256'],b['local_profile']['recipe_sha256'])
        self.assertEqual(b['local_profile']['validation_batch'],4)
        for invalid in (-1,129,True,1.5):
            with self.assertRaises(ValueError):local_options(overrides={'VALIDATION_BATCH':invalid})

    def test_validation_memory_cap_leaves_train_loader_batch_unchanged(self):
        from occluactisafe.experiments.detector_local_training import rich_trainer_class
        cls=rich_trainer_class(8,None,{'validation_batch':4})
        trainer=object.__new__(cls)
        with patch.object(cls.__bases__[0],'get_dataloader',return_value='loader') as native:
            self.assertEqual(trainer.get_dataloader('fixture',batch_size=16,rank=-1,mode='val'),'loader')
            self.assertEqual(native.call_args.kwargs['batch_size'],4)
            trainer.get_dataloader('fixture',batch_size=8,rank=-1,mode='train')
            self.assertEqual(native.call_args.kwargs['batch_size'],8)

    def test_isolated_preflight_validates_returned_source_and_input_index(self):
        from occluactisafe.experiments import detector_local as m
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();settings={'input_root':root/'input','output_root':root,'weights_root':root/'weights'}
            def child(command,**kwargs):
                task=Path(command[-1]);index=task.parent/'预检原始索引.jsonl'
                index.write_text(json.dumps({'key':'mechanism_only'})+'\n',encoding='utf-8')
                write_json(task.parent/'独立预检结果.json',{
                    'environment':{'source':{'source_fingerprint_sha256':'fixed'},'input_verification':{'assets':1}},
                    'checkdir':str(task.parent),'rows':1,'index':str(index),'index_sha256':sha256(index),'worker_pid':12345})
                return SimpleNamespace(returncode=0)
            with patch('occluactisafe.experiments.source_version.source_info',return_value={'source_fingerprint_sha256':'fixed'}),patch.object(m.subprocess,'run',side_effect=child):
                rows,env,directory=m.isolated_preflight(settings,[{'family':'mechanism'}])
            self.assertEqual(rows,[{'key':'mechanism_only'}])
            boundary=read_json(directory/'父子进程资源边界.json')
            self.assertTrue(boundary['child_has_exited_before_training'])
            self.assertTrue(boundary['no_training_started_by_preflight'])

    def test_isolated_preflight_failure_never_returns_fake_success(self):
        from occluactisafe.experiments import detector_local as m
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()
            with patch('occluactisafe.experiments.source_version.source_info',return_value={'source_fingerprint_sha256':'fixed'}),patch.object(m.subprocess,'run',return_value=SimpleNamespace(returncode=1)):
                with self.assertRaisesRegex(RuntimeError,'预检失败'):
                    m.isolated_preflight({'input_root':root/'input','output_root':root,'weights_root':root/'weights'},[])

    def test_only_two_weights_are_published_and_existing_unexpected_file_is_not_deleted(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);publish_weights(root,b'epoch1 mechanism fixture',True,{'epoch':1,'applied_updates':1,'batch_attempts':1})
            previous=(root/'weights/best.pt').read_bytes()
            publish_weights(root,b'epoch2 mechanism fixture',False,{'epoch':2,'applied_updates':2,'batch_attempts':2})
            self.assertEqual((root/'weights/best.pt').read_bytes(),previous)
            self.assertEqual({p.name for p in (root/'weights').iterdir()},{'last.pt','best.pt'})
            self.assertEqual(verify_checkpoint(root)['epoch'],2)
            (root/'weights/epoch_old.pt').write_bytes(b'existing, do not delete')
            with self.assertRaises(ValueError):publish_weights(root,b'new',True,{'epoch':3})
            self.assertTrue((root/'weights/epoch_old.pt').exists())

    def test_interrupted_best_publication_is_recoverable_without_mutating_source(self):
        from occluactisafe.experiments import detector_local_checkpoints as m
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);original=m.atomic_bytes
            def publish(path,data):
                if Path(path).name=='best.pt':raise OSError('injected interrupted publication')
                return original(path,data)
            with patch.object(m,'atomic_bytes',side_effect=publish):
                with self.assertRaises(OSError):publish_weights(root,b'complete binary fixture',True,{'epoch':1,'applied_updates':2,'batch_attempts':3})
            before=sha256(root/'检查点发布事务.json');result=verify_checkpoint(root)
            self.assertEqual(result['best_source_file'],'last.pt');self.assertEqual(result['status'],'recoverable_pending')
            self.assertEqual(sha256(root/'检查点发布事务.json'),before)
            self.assertFalse((root/'weights/best.pt').exists())

    def test_resume_carries_only_committed_prefix_and_retains_failed_tail(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'old';source.mkdir();target=root/'new'
            p=source/'逐批次优化尝试.csv';p.write_text('epoch,attempt\n1,1\n1,2\n2,3\n',encoding='utf-8')
            for name in ('早停判定.jsonl','逐轮验证指标.jsonl'):
                (source/name).write_text(json.dumps({'epoch':1})+'\n'+json.dumps({'epoch':2})+'\n',encoding='utf-8')
            before=sha256(p);carry_committed_logs(source,target,1,{'逐批次优化尝试.csv':2})
            self.assertEqual(sha256(p),before)
            self.assertEqual((target/p.name).read_text(encoding='utf-8').splitlines(),['epoch,attempt','1,1','1,2'])
            self.assertEqual(len((target/'早停判定.jsonl').read_text().splitlines()),1)

    def test_amp_skips_are_attempts_not_fabricated_parameter_updates(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);files=[str((root/f'{i}.jpg').resolve()) for i in range(4)];expected=Counter(files)
            fields=['step','epoch','attempt','image_count','image_files','update_applied','amp_scale_before','amp_scale_after',
                    'loss','gradient_norm','learning_rate','step_seconds','gpu_allocated_mib','host_rss_mib']
            rows=[dict(zip(fields,[0,1,1,2,json.dumps(files[:2]),False,65536,32768,1,'inf',.001,.01,10,20])),
                  dict(zip(fields,[1,1,2,2,json.dumps(files[2:]),True,32768,32768,1,.5,.001,.01,10,20]))]
            def write():
                for name,values in [('逐批次优化尝试.csv',rows),('逐步优化指标.csv',rows[1:])]:
                    with (root/name).open('w',encoding='utf-8',newline='') as f:
                        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(values)
            write();result=audit_local_batches(root,expected,1,2,True)
            self.assertEqual((result['batch_attempts'],result['applied_updates'],result['amp_skips'],result['image_exposures']),(2,1,1,4))
            rows[0]['amp_scale_after']=65536;write()
            with self.assertRaises(ValueError):audit_local_batches(root,expected,1,2,True)

    def test_final_best_validation_does_not_fabricate_extra_training_epoch(self):
        from ultralytics.models.yolo.detect import DetectionTrainer
        from occluactisafe.experiments.detector_training import trainer_class
        cls=trainer_class(2);trainer=object.__new__(cls);seen=[]
        def native(self):seen.append(self._n17_final_eval);return 'done'
        with patch.object(DetectionTrainer,'final_eval',native):self.assertEqual(trainer.final_eval(),'done')
        self.assertEqual(seen,[True]);self.assertFalse(trainer._n17_final_eval)


if __name__=='__main__':unittest.main()
