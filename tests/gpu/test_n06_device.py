"""必须在真实GPU实际执行；缺设备/依赖视为失败，不skip。"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from occluactisafe.common.device_smoke import load_smoke_config, run

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/runtime/n06.yaml'


class RealDeviceTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(os.environ.get('N06_EVIDENCE_DIR', 'H:/OccluActiSafeDataset/artifacts/n06_20261030/gpu_tests'))
        self.root.mkdir(parents=True, exist_ok=True)

    def test_real_gpu_forward_backward_and_three_seed_repeat(self):
        for seed in (17, 29, 43):
            result = run(load_smoke_config(CONFIG, overrides={'seed': seed, 'output_root': str(self.root / 'seeds')}))
            self.assertEqual(result['mode'], 'gpu_acceptance')
            self.assertTrue(result['deterministic'])
            self.assertEqual(result['device']['capability'], [12, 0])
            self.assertGreater(result['memory']['peak_allocated_bytes'], 0)
            self.assertLessEqual(result['memory']['peak_reserved_bytes'], 12288 * 1024**2)
            self.assertTrue(all(t['parameters_changed'] and t['tensor_device'] == 'cuda:0' for t in result['trials']))

    def test_three_cwds_no_required_cli_and_fresh_process_determinism(self):
        output = self.root / 'cwd'
        code = ("import runpy; "
                f"m=runpy.run_path({str(ROOT / 'scripts/smoke_check.py')!r},run_name='smoke_entry'); "
                f"m['main'].__globals__['OVERRIDES']={{'output_root':{str(output)!r}}}; "
                "raise SystemExit(m['main']([]))")
        with tempfile.TemporaryDirectory() as other:
            for i, cwd in enumerate((ROOT, ROOT / 'scripts', Path(other))):
                p = subprocess.run([sys.executable, '-B', '-c', code], cwd=cwd, capture_output=True)
                (self.root / f'cwd_{i}.txt').write_bytes(p.stdout + p.stderr)
                self.assertEqual(p.returncode, 0, p.stderr.decode('utf-8'))
        results = [json.loads(p.read_text(encoding='utf-8')) for p in output.glob('*/device_smoke.json')]
        self.assertEqual(len(results), 3)
        self.assertEqual(len({r['trials'][0]['output_hash'] for r in results}), 1)
        self.assertEqual(len({tuple(r['trials'][0]['parameter_hashes']) for r in results}), 1)

    def test_real_allocator_oom_and_invalid_device_fail_without_fallback(self):
        for label, args, expected in (
            ('oom', ['--allocator-limit-mib','1'], 'OutOfMemoryError'),
            ('invalid_device', ['--device','cuda:999'], 'ValueError')):
            output = self.root / label
            p = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/smoke_check.py'),
                                '--output-root', str(output), *args], capture_output=True)
            (self.root / f'{label}.txt').write_bytes(p.stdout + p.stderr)
            self.assertNotEqual(p.returncode, 0)
            statuses = [json.loads(f.read_text(encoding='utf-8')) for f in output.glob('*/status.json')]
            self.assertEqual(len(statuses), 1)
            self.assertEqual(statuses[0]['status'], 'failed')
            self.assertEqual(statuses[0]['error_type'], expected)
            self.assertFalse(list(output.glob('*/device_smoke.json')))

    def test_explicit_cpu_diagnostic_not_gpu_acceptance(self):
        result = run(load_smoke_config(CONFIG, overrides={'device': 'cpu', 'output_root': str(self.root / 'cpu')}))
        self.assertEqual(result['mode'], 'cpu_diagnostic')
        self.assertIsNone(result['memory'])
        self.assertEqual(result['trials'][0]['tensor_device'], 'cpu')


if __name__ == '__main__':
    unittest.main()
