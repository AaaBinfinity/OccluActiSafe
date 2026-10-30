"""N07构造场景的完整训练/验证/恢复及IDE入口，不冒充真实数据。"""
import json
from pathlib import Path
import subprocess
import sys
import unittest
import yaml
import test_n04_pipeline as n04
from occluactisafe.data.preprocess import load_preprocess_config,run_preprocess
from occluactisafe.prediction.train import load_training_config,run,enforce_budget


class TrainingPipelineTests(unittest.TestCase):
    def setUp(self):
        self.fixture=n04.PreprocessTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root
        result=run_preprocess(load_preprocess_config(self.fixture.config))
        model=self.root/'gru.yaml';model.write_text('hidden_size: 8\nmodes: 6\ndropout: 0.0\n',encoding='utf-8')
        template=yaml.safe_load((n04.ROOT/'configs/experiments/tfd_gru.yaml').read_text(encoding='utf-8'))
        template.update(windows_dir=result['run_dir'],output_root=str(self.root/'train'),model_config=str(model),device='cpu',seeds=[17],epochs=2,batch_size=4)
        self.config=self.root/'task.yaml';self.config.write_text(yaml.safe_dump(template),encoding='utf-8')

    def test_complete_train_validation_resume_and_missing_labels_rejection(self):
        config=load_training_config(self.config);result=run(config)
        self.assertEqual(result['status'],'completed');self.assertTrue(result['results'][0]['checkpoint_reload_equal'])
        directory=Path(result['run_dir']);checkpoint=directory/'seed_17/latest.pt'
        import csv
        seed_dir=directory/'seed_17'
        with (seed_dir/'step_metrics.csv').open(encoding='utf-8',newline='') as stream:steps=list(csv.DictReader(stream))
        self.assertEqual([int(r['global_step']) for r in steps],list(range(1,result['results'][0]['steps']+1)))
        self.assertTrue(all(float(r['target_count'])>0 for r in steps))
        plots=json.loads((seed_dir/'plot_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(plots['step_rows'],len(steps));self.assertEqual(plots['epoch_rows'],2)
        self.assertEqual(len(plots['images']),8);self.assertFalse(plots['smoothing'])
        self.assertTrue(all((seed_dir/p['file']).is_file() for p in plots['images']))
        resumed=run(load_training_config(self.config,overrides={'resume':str(checkpoint)}))
        self.assertEqual(resumed['results'][0]['metrics'],result['results'][0]['metrics'])
        self.assertEqual((Path(resumed['run_dir'])/'seed_17/step_metrics.csv').read_bytes(),(seed_dir/'step_metrics.csv').read_bytes())
        self.assertTrue((directory/'seed_17/validation_predictions.npz').is_file())
        for file in Path(config['windows_dir']).rglob('*.labels.npz'):file.unlink()
        with self.assertRaises(FileNotFoundError):run(config)

    def test_no_cli_entry_from_three_cwds_and_budget_guard(self):
        for i,cwd in enumerate((n04.ROOT,n04.ROOT/'scripts',self.root)):
            code=("import runpy; from pathlib import Path; "
                f"m=runpy.run_path({str(n04.ROOT/'scripts/train_prediction.py')!r},run_name='entry'); "
                f"m['main'].__globals__['CONFIG_FILE']=Path({str(self.config)!r}); "
                f"m['main'].__globals__['OVERRIDES']={{'output_root':{str(self.root/f'cwd_{i}')!r}}}; "
                "raise SystemExit(m['main']([]))")
            p=subprocess.run([sys.executable,'-B','-c',code],cwd=cwd,capture_output=True)
            self.assertEqual(p.returncode,0,p.stderr.decode('utf-8'))
        config=load_training_config(self.config)
        with self.assertRaisesRegex(RuntimeError,'工作集'):
            enforce_budget({**config['budget'],'max_host_working_set_mib':1},__import__('time').perf_counter(),output_dir=self.root)
        with self.assertRaisesRegex(RuntimeError,'墙钟'):
            enforce_budget(config['budget'],__import__('time').perf_counter()-8000,output_dir=self.root)


if __name__=='__main__':unittest.main()
