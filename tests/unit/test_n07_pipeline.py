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
        resumed=run(load_training_config(self.config,overrides={'resume':str(checkpoint)}))
        self.assertEqual(resumed['results'][0]['metrics'],result['results'][0]['metrics'])
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
