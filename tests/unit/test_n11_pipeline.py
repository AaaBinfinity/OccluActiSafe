"""默认IDE入口、严格配置、失败证据与图表重建；过程产物保留在硬盘。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4
from occluactisafe.core.analytic_diagnostics import load_config,run,render_plots

ROOT=Path(__file__).resolve().parents[2]
ENTRY=ROOT/'scripts/run_analytic_demo.py'
EVIDENCE=Path('H:/OccluActiSafeDataset/artifacts/n11_pipeline_checks')


class PipelineTests(unittest.TestCase):
    def test_strict_config_and_missing_data_path(self):
        with self.assertRaises(ValueError):load_config(overrides={'truth':True})
        with self.assertRaises(ValueError):load_config(overrides={'output_root':'https://example.com'})
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'duplicate.yaml';p.write_text('schema_version: a\nschema_version: b\n',encoding='utf-8')
            with self.assertRaises(ValueError):load_config(p)
            c=load_config(overrides={'output_root':str(Path(temp)/'not-mounted/out')})
            with self.assertRaises(FileNotFoundError):run(c)

    def test_ide_three_cwds_and_plot_rebuild(self):
        if not EVIDENCE.parent.is_dir():raise FileNotFoundError('N11过程证据硬盘未接入')
        parent=EVIDENCE/uuid4().hex;parent.mkdir(parents=True)
        for index,cwd in enumerate((ROOT,ROOT/'scripts',parent)):
            output=parent/f'cwd-{index}'
            code=("import importlib.util,pathlib,sys; "
                  f"spec=importlib.util.spec_from_file_location('n11_entry',{str(ENTRY)!r}); "
                  "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m); "
                  f"assert not pathlib.Path({str(output)!r}).exists();assert 'torch' not in sys.modules; "
                  f"m.OVERRIDES={{'output_root':{str(output)!r}}};raise SystemExit(m.main([]))")
            proc=subprocess.run([sys.executable,'-B','-W','error','-c',code],cwd=cwd,capture_output=True,text=True,encoding='utf-8',timeout=90)
            (parent/f'cwd-{index}-stdout.txt').write_text(proc.stdout,encoding='utf-8')
            (parent/f'cwd-{index}-stderr.txt').write_text(proc.stderr,encoding='utf-8')
            self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
            directory=next(output.iterdir());result=json.loads((directory/'result.json').read_text(encoding='utf-8'))
            self.assertEqual(result['episodes'],12);self.assertEqual(result['state_rows'],1452);self.assertEqual(result['checks_passed'],21)
            manifest=json.loads((directory/'plot_manifest.json').read_text(encoding='utf-8'))
            for name,sha in {**manifest['sources'],**manifest['figures']}.items():self.assertEqual(hashlib.sha256((directory/name).read_bytes()).hexdigest(),sha)
            if index==0:
                rebuilt=render_plots(directory,output_dir=parent/'plot_rebuild')
                for name in manifest['figures']:
                    if name.endswith('.png'):self.assertEqual(rebuilt['figures'][name],manifest['figures'][name])

    def test_injected_plot_failure_retains_raw(self):
        parent=EVIDENCE/uuid4().hex;parent.mkdir(parents=True)
        c=load_config(overrides={'output_root':str(parent/'injected')})
        with patch('occluactisafe.core.analytic_diagnostics.render_plots',side_effect=RuntimeError('explicit injected N11 plot failure')):
            with self.assertRaisesRegex(RuntimeError,'explicit injected'):run(c)
        directory=next((parent/'injected').iterdir())
        status=json.loads((directory/'status.json').read_text(encoding='utf-8'))
        self.assertEqual(status['status'],'failed')
        self.assertTrue((directory/'paired_episode_metrics.json').is_file())
        self.assertEqual(len(list(directory.glob('*.jsonl'))),12)
        self.assertTrue((directory/'evidence_index.json').is_file())


if __name__=='__main__':unittest.main()
