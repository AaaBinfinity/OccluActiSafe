import copy
import csv
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from occluactisafe.experiments.detector_models import expanded_models, verify_weights, prepare_weights, ASSETS
from occluactisafe.experiments.detector_stopping import PlateauStopper, replay_stopping
from occluactisafe.experiments.detector_server import server_settings,server_base,plan_tasks,run_job
from occluactisafe.experiments.detector_matrix import validate_matrix
from occluactisafe.experiments.detector_data import load_config
from occluactisafe.experiments.detector_timing import timing_budget
from occluactisafe.experiments.run_store import read_json,write_json,sha256
import yaml


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

    def settings(self, root):
        cfg=yaml.safe_load(Path('configs/experiments/n17_server_a100.yaml').read_text(encoding='utf-8'))
        cfg.update(input_root=str(root/'input'),output_root=str(root/'output'))
        p=root/'site.yaml';p.write_text(yaml.safe_dump(cfg),encoding='utf-8')
        return server_settings(p)

    def test_a100_matrix_is_45_training_and_60_evaluation_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            settings=self.settings(Path(temp).resolve());cfg=server_base(settings)
            self.assertEqual(len(validate_matrix(cfg)),45)
            tasks=plan_tasks('formal',['person_alias'],cfg['models'],cfg['seeds'])
            self.assertEqual((sum(t['kind']=='train' for t in tasks),len(tasks)),(45,46))
            self.assertEqual(len(cfg['models'])*(1+len(cfg['seeds'])),60)
            self.assertEqual(len(plan_tasks('pilot',['baseline','person_alias'],cfg['models'],cfg['seeds'])),32)
            cfg['models'].pop()
            with self.assertRaises(ValueError):validate_matrix(cfg)
            bad=yaml.safe_load(Path(settings['_path']).read_text());bad['minimum_epochs']=301
            Path(settings['_path']).write_text(yaml.safe_dump(bad))
            with self.assertRaises(ValueError):server_settings(settings['_path'])

    def test_stopping_agrees_with_locked_native_rule_when_minimum_is_one(self):
        from ultralytics.utils.torch_utils import EarlyStopping
        native=EarlyStopping(patience=2);ours=PlateauStopper(1,2)
        for epoch,value in enumerate([0,0,.1,.1,.2,.2,.2],1):
            self.assertEqual(native(epoch,value),ours(epoch,value))
            self.assertEqual((native.best_fitness,native.best_epoch,native.possible_stop),
                             (ours.best_fitness,ours.best_epoch,ours.possible_stop))

    def test_trainer_installs_frozen_stopper_and_rejects_silent_amp_fallback(self):
        from types import SimpleNamespace
        from ultralytics.models.yolo.detect import DetectionTrainer
        from occluactisafe.experiments.detector_training import trainer_class
        with tempfile.TemporaryDirectory() as temp:
            cls=trainer_class(2,{'criterion':'native_fitness','minimum_epochs':5,'patience':2})
            trainer=object.__new__(cls);trainer.amp=True;trainer.args=SimpleNamespace(amp=True,patience=2,epochs=10);trainer.save_dir=Path(temp)
            with patch.object(DetectionTrainer,'_setup_train',return_value=None):
                trainer._setup_train();self.assertIsInstance(trainer.stopper,PlateauStopper)
                trainer.amp=False
                with self.assertRaisesRegex(RuntimeError,'AMP'):trainer._setup_train()

    def test_audit_accepts_proven_early_stop_and_rejects_truncated_or_changed_fitness(self):
        from occluactisafe.experiments.detector_audit import audit_training_matrix
        from occluactisafe.experiments.run_store import append_json
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);train=root/'训练';train.mkdir();weights=train/'weights';weights.mkdir()
            cfg=load_config();cfg['training'].update(epochs=10,batch=2,nbs=2,patience=2)
            cfg['budget_policy']={'criterion':'native_fitness','minimum_epochs':2,'patience':2}
            stopper=PlateauStopper(2,2,train/'早停判定.jsonl')
            for epoch,fitness in enumerate([.5,.4,.4],1):
                stopper(epoch,fitness);append_json(train/'逐轮验证指标.jsonl',{'epoch':epoch,'native_fitness':fitness})
            trace=[json.loads(s) for s in (train/'早停判定.jsonl').read_text().splitlines()]
            path=train/'逐步优化指标.csv';fields=['step','epoch','image_count','loss','gradient_norm','learning_rate','step_seconds','gpu_allocated_mib','host_rss_mib']
            with path.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
                for i in range(1,4):w.writerow(dict(zip(fields,[i,i,2,1,.1,.001,.01,10,10])))
            graph={'points':3,'validation_epochs':3,'step_csv_sha256':sha256(path),'epoch_jsonl_sha256':sha256(train/'逐轮验证指标.jsonl'),'figures':[]}
            write_json(train/'训练图表来源_完整分项.json',graph)
            for name in ('best.pt','last.pt'):(weights/name).write_bytes(b'mechanism fixture, not a trained model')
            run={'family':'YOLO26','seed':17,'status':'completed','config_sha256':cfg['_config_sha256'],'data_index_sha256':'fixture',
                 'initial_weight_sha256':'fixture','directory':str(root),'source_commit':'fixture','checkpoint':str(weights/'best.pt'),
                 'checkpoint_sha256':sha256(weights/'best.pt'),'last_checkpoint':str(weights/'last.pt'),'last_checkpoint_sha256':sha256(weights/'last.pt'),
                 'checkpoint_rule':'native_model_val_best_fitness','termination':replay_stopping(trace,cfg['budget_policy'],10),
                 'amp_requested':False,'amp_actual':False}
            matrix=root/'matrix.json';write_json(matrix,{'runs':[run]})
            source='def trainer_class(): pass\ndef train_one(): pass\n'
            with patch('occluactisafe.experiments.detector_audit.load_index',return_value=([{'role':'train'}]*2,{'index_sha256':'fixture'})),patch('occluactisafe.experiments.detector_audit.historical_source',return_value=source):
                result=audit_training_matrix(cfg,matrix)
                self.assertEqual(result['runs'][0]['status'],'passed');self.assertEqual(result['runs'][0]['actual_epochs'],3)
                self.assertEqual(result['total_optimizer_steps'],3);self.assertEqual(result['total_image_exposures'],6)
                with path.open('a') as f:f.write('4,4,2,1,.1,.001,.01,10,10\n')
                with self.assertRaises(ValueError):audit_training_matrix(cfg,matrix)

    def test_time_estimate_is_conditional_and_failed_pilot_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();settings=self.settings(root)
            estimate=timing_budget(settings)
            self.assertFalse(estimate['actual_a100_measurement']);self.assertFalse(estimate['convergence_guaranteed'])
            self.assertEqual([r['assumed_epochs_per_run'] for r in estimate['scenarios']],[50,100,300])
            self.assertAlmostEqual(estimate['scenarios'][1]['days_lower'],9.28125)
            pilot=root/'pilot';pilot.mkdir();write_json(pilot/'任务状态.json',{'status':'planned','mode':'pilot','configuration_sha256':settings['_sha256']})
            with self.assertRaises(ValueError):timing_budget(settings,pilot)

    def test_time_extrapolation_binds_each_model_actual_epoch_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();settings=self.settings(root);pilot=root/'pilot';pilot.mkdir();tasks=[]
            for model in expanded_models(settings['model_sizes']):
                folder=pilot/model['family'];train=folder/'运行/训练';train.mkdir(parents=True)
                (train/'逐轮耗时.jsonl').write_text(''.join(json.dumps({'epoch':i,'seconds_including_validation_and_save':v})+'\n' for i,v in enumerate([10.,2.,3.],1)))
                write_json(folder/'worker_result.json',{'directory':str(folder/'运行')})
                tasks.append({'kind':'train','policy':'person_alias','family':model['family'],'status':'completed','attempts':[{'returncode':0,'directory':str(folder)}]})
            env=pilot/'环境.json';write_json(env,{'gpu':{'name':'mechanism timing fixture','uuid':'fixture'}})
            state={'mode':'pilot','status':'completed','configuration_sha256':settings['_sha256'],'tasks':tasks,'environment_report':str(env)}
            write_json(pilot/'任务状态.json',state);result=timing_budget(settings,pilot)
            self.assertEqual(len(result['pilot_sources']),15)
            self.assertEqual(result['pilot_sources'][0]['used_seconds'],[2.,3.])
            self.assertEqual(result['per_model_epoch_seconds_interval']['YOLO26x'],(2.,3.25))
            state['tasks'].pop();write_json(pilot/'任务状态.json',state)
            with self.assertRaises(ValueError):timing_budget(settings,pilot)

    def test_full_orchestrator_does_not_mix_sizes_or_skip_seeds(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();settings=self.settings(root);calls=[]
            env={'source':{'source_commit':'a'*40,'source_fingerprint_sha256':'fixture'},
                 'input_verification':{'manifest_sha256':'fixture','source_index_sha256':'fixture'},
                 'gpu':{'uuid':'fixture'},'worker_cuda_visible_devices':'0'}
            rows=[{'key':'spd_1','role':'train','domain':'SPD','image':'H:/old.jpg','source_image':'H:/old.jpg','gt':[]}]
            def execute(params,log):
                task=read_json(params);calls.append(task['kind']);folder=Path(task['attempt']);cfg=load_config(task['config'])
                self.assertEqual((cfg['training']['epochs'],cfg['training']['amp'],cfg['training']['batch']),(300,True,16))
                self.assertEqual(cfg['budget_policy'],{'criterion':'native_fitness','minimum_epochs':50,'patience':30})
                if task['kind']=='train':
                    write_json(folder/'worker_result.json',{'status':'completed','family':task['family'],'seed':task['seed'],'config_sha256':'fixture','data_index_sha256':'fixture'})
                else:
                    matrix=read_json(task['training_matrix']);self.assertEqual(len(matrix['runs']),45)
                    self.assertEqual(len({(r['family'],r['seed']) for r in matrix['runs']}),45)
                    metrics=folder/'metrics';metrics.mkdir();write_json(metrics/'离线预选.json',{'status':'no_qualified_candidate','selected_candidate':None})
                    write_json(folder/'worker_result.json',{'status':'completed','directory':str(metrics)})
                return 0
            prefix='occluactisafe.experiments.detector_server.'
            with patch(prefix+'preflight',return_value=(rows,env,root)),patch(prefix+'execute_worker',side_effect=execute),patch(prefix+'validate_completed',return_value={}),patch(prefix+'verify_input',return_value=(rows,{'status':'passed'})):
                job=run_job(settings,'formal')
            self.assertEqual(calls,['train']*45+['evaluate'])
            self.assertEqual(read_json(job/'任务状态.json')['status'],'completed')


if __name__ == '__main__': unittest.main()
