"""N04分组与窗口完整入口夹具，不代表真实数据实验。"""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from occluactisafe.data.split import audit_group_split
from occluactisafe.prediction.dataset import PredictionDataset
from occluactisafe.data.preprocess import load_preprocess_config, run_preprocess
from occluactisafe.data.tfd_audit import load_task_config, run_audit

ROOT=Path(__file__).resolve().parents[2]


class PreprocessTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.data=self.root/'tfd'
        for i in range(8):
            file=self.data/f'single-vehicle/trajectories/train/{i}.csv';file.parent.mkdir(parents=True,exist_ok=True)
            with file.open('w',encoding='utf-8',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['city','intersect_id','timestamp','id','type','tag','x','y','theta'])
                for t in range(5):
                    writer.writerow(['fixture',str(i),str(t/10),'ego','VEHICLE','AV',str(t),0,0])
                    writer.writerow(['fixture',str(i),str(t/10),'target','VEHICLE','TARGET_AGENT',str(2*t),2,0])
        cfg=self.root/'n03.yaml';cfg.write_text(f'data_root: {self.data}\noutput_root: {self.root / "audit/full"}\ndataset_scope: full\n',encoding='utf-8')
        n03=run_audit(load_task_config(cfg),project_root=ROOT);self.audit=Path(n03['run_dir'])
        manifest=json.loads((self.audit/'data_manifest.json').read_text(encoding='utf-8'))
        (self.audit.parent.parent/'readiness_review.json').write_text(json.dumps({'formal_data_ready_for_registered_local_scope':True,'dataset_sha256':manifest['dataset_sha256'],'scope':'mechanism fixture only'}),encoding='utf-8')
        self.config=self.root/'n04.yaml';self.config.write_text(f'audit_dir: {self.audit}\noutput_root: {self.root / "output"}\nhistory_steps: 3\nfuture_steps: 2\nworkers: 1\n',encoding='utf-8')

    def test_complete_fixture_pipeline_and_source_hash_guard(self):
        config=load_preprocess_config(self.config);result=run_preprocess(config)
        self.assertEqual(result['status'],'completed');self.assertEqual(result['counts']['written'],8)
        self.assertEqual(result['counts']['eligible_targets'],8)
        run=Path(result['run_dir']);manifest=json.loads((run/'split_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(audit_group_split(manifest)['cross_split_shared_keys'],0)
        self.assertEqual(sum(len(PredictionDataset(run,r)) for r in ('train','model_val','risk_calibration','test')),8)
        source=next(self.data.rglob('*.csv'));source.write_text(source.read_text(encoding='utf-8')+'\n',encoding='utf-8')
        with self.assertRaises(ValueError):run_preprocess(config)

    def test_config_precedence_and_import_does_not_execute(self):
        config=load_preprocess_config(self.config,overrides={'max_files':2},cli_overrides={'max_files':0})
        self.assertEqual(config['max_files'],0)
        with self.assertRaises(ValueError):load_preprocess_config(self.config,overrides={'unknown':1})
        with self.assertRaises(ValueError):load_preprocess_config(self.config,overrides={'workers':True})
        script=ROOT/'scripts/prepare_windows.py'
        result=subprocess.run([sys.executable,'-I','-B','-c',"import runpy,sys;runpy.run_path(sys.argv[1],run_name='imported')",str(script)],cwd=self.root,capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(result.stdout,'')

    def test_ide_no_cli_and_three_cwd(self):
        wrapper="import runpy,sys;from pathlib import Path; n=runpy.run_path(sys.argv[1],run_name='ide');n['main'].__globals__['CONFIG_FILE']=Path(sys.argv[2]);raise SystemExit(n['main']([]))"
        script=ROOT/'scripts/prepare_windows.py'
        for cwd in (ROOT,ROOT/'scripts',self.root):
            result=subprocess.run([sys.executable,'-I','-B','-c',wrapper,str(script),str(self.config)],cwd=cwd,capture_output=True,text=True,encoding='utf-8',timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr);self.assertIn('N04完成',result.stdout)


if __name__=='__main__':unittest.main()
