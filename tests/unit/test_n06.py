"""N06配置与拒绝机制；实际CUDA验收在tests/gpu，禁止跳过冒充通过。"""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from occluactisafe.common.device import import_torch, select_device, seed_everything
from occluactisafe.common.device_smoke import load_smoke_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/runtime/n06.yaml'


class DeviceMechanismTests(unittest.TestCase):
    def test_config_overrides_and_path(self):
        c = load_smoke_config(CONFIG, overrides={'seed': 29}, cli_overrides={'seed': 0})
        self.assertEqual(c['seed'], 0)
        self.assertTrue(Path(c['output_root']).is_absolute())
        self.assertEqual(c['training_budget']['seeds'], [17, 29, 43])

    def test_invalid_config_and_budget_rejected(self):
        for override in ({'steps': 0}, {'steps': True}, {'seed': -1}, {'matrix_size': 100000},
                         {'unknown': 1}, {'training_budget': {}}, {'allocator_limit_mib': 0}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                load_smoke_config(CONFIG, overrides=override)

    def test_cpu_build_and_unavailable_cuda_are_explicit_failures(self):
        torch = Mock(); torch.version.cuda = None
        with self.assertRaisesRegex(RuntimeError, '不回退CPU'):
            select_device('cuda:0', torch)
        torch.version.cuda = '12.8'; torch.cuda.is_available.return_value = False
        with self.assertRaises(RuntimeError):
            select_device('cuda:0', torch)
        torch.device.assert_not_called()

    def test_invalid_gpu_index_and_auto_device_rejected(self):
        torch = Mock(); torch.version.cuda = '12.8'
        torch.cuda.is_available.return_value = True; torch.cuda.device_count.return_value = 1
        for name in ('auto', 'cuda', 'cuda:-1', 'cuda:999'):
            with self.assertRaises(ValueError):
                select_device(name, torch)
        torch.cuda.set_device.assert_not_called()

    def test_missing_torch_import_is_actionable(self):
        with patch.dict(sys.modules, {'torch': None}):
            with self.assertRaisesRegex(RuntimeError, 'group training'):
                import_torch()

    def test_import_entry_and_helpers_do_not_load_torch_or_run(self):
        code = ("import runpy,sys; import occluactisafe.common.device; "
                "import occluactisafe.common.device_smoke; "
                f"runpy.run_path({str(ROOT / 'scripts/smoke_check.py')!r},run_name='entry_import'); "
                "assert 'torch' not in sys.modules")
        result = subprocess.run([sys.executable, '-I', '-B', '-c', code], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr.decode('utf-8'))
        self.assertEqual(result.stdout, b'')


if __name__ == '__main__':
    unittest.main()
