"""明确构造数据上的评测入口验证，真实TFD结果另存硬盘。"""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import test_n04_pipeline as n04
from occluactisafe.data.preprocess import load_preprocess_config, run_preprocess
from occluactisafe.prediction.evaluate import load_evaluation_config, run, sha256


class EvaluationPipelineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = n04.PreprocessTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        result = run_preprocess(load_preprocess_config(self.fixture.config))
        self.config = self.root / 'cv.yaml'
        self.config.write_text(f'windows_dir: {result["run_dir"]}\noutput_root: {self.root / "cv"}\n', encoding='utf-8')

    def test_full_fixture_repeat_and_label_separation(self):
        a = run(load_evaluation_config(self.config))
        b = run(load_evaluation_config(self.config))
        self.assertEqual(a['overall'], b['overall'])
        self.assertEqual(a['overall']['ADE_m'], 0)
        for name in ('predictions.csv', 'per_agent_metrics.csv', 'input_inventory.jsonl'):
            self.assertEqual(sha256(Path(a['run_dir']) / name), sha256(Path(b['run_dir']) / name))
        # 删除监督明确失败，不能静默跳过样本。
        config = load_evaluation_config(self.config)
        for file in Path(config['windows_dir']).rglob('*.labels.npz'):
            file.unlink()
        with self.assertRaises(FileNotFoundError):
            run(config)

    def test_import_and_three_cwds_no_required_cli(self):
        spec = importlib.util.spec_from_file_location('cv_entry', n04.ROOT / 'scripts/evaluate_prediction.py')
        entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
        self.assertEqual(entry.OVERRIDES, {})
        # 用机制配置替换入口常量，main()无CLI；真实默认入口另做全量运行。
        code = ("import importlib.util; from pathlib import Path; "
                f"s=importlib.util.spec_from_file_location('e',{str(n04.ROOT / 'scripts/evaluate_prediction.py')!r}); "
                "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                f"m.CONFIG_FILE=Path({str(self.config)!r}); raise SystemExit(m.main([]))")
        for cwd in (n04.ROOT, n04.ROOT / 'scripts', self.root):
            result = subprocess.run([sys.executable, '-B', '-c', code], cwd=cwd, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr.decode('utf-8'))


if __name__ == '__main__':
    unittest.main()
