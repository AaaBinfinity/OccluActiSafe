import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
import uuid

SCRIPTS=Path(__file__).resolve().parents[2]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import recover_n17_console_close as recovery
import n17_console_free


class HeadlessRecoveryTests(unittest.TestCase):
    def test_rejects_budget_changes_and_missing_approval(self):
        approved={'approved':True,'additional_exposures':9412,'cumulative_extra_exposures':26415,'maximum_total_exposures':1046595}
        self.assertEqual(recovery.approved_budget(approved),approved)
        for changed in ({},{**approved,'approved':False},{**approved,'additional_exposures':9413}):
            with self.assertRaises(ValueError):recovery.approved_budget(changed)

    def test_queue_launch_always_uses_verified_console_free_worker(self):
        class Queue:pass
        queue=Queue();recovery.install_console_free_launch(queue)
        with patch.object(n17_console_free,'start_console_free') as start:
            root=recovery.EVIDENCE/('机制启动_'+uuid.uuid4().hex);root.mkdir(parents=True)
            start.return_value.wait.return_value=0
            self.assertEqual(queue.launch(Path('task.json'),root/'输出.txt','0'),0)
        args=start.call_args.args
        self.assertEqual(Path(args[0][0]).name,'python.exe')
        self.assertIn('n17_lr_headless_host.py',args[0][2])
        self.assertEqual(args[2]['CUDA_VISIBLE_DEVICES'],'0')

    def test_real_pythonw_host_and_child_both_have_zero_console(self):
        root=recovery.EVIDENCE/('宿主机制_'+uuid.uuid4().hex);root.mkdir(parents=True)
        pythonw=Path(sys.executable).with_name('pythonw.exe')
        result=n17_console_free.run_console_free([str(pythonw),'-B',str(SCRIPTS/'n17_lr_headless_host.py'),
                  '--action','smoke','--smoke-root',str(root)],root/'宿主输出.txt')
        self.assertEqual(result,0)
        for file in ('无控制台宿主身份.json','无控制台子进程身份.json'):
            self.assertEqual(json.loads((root/file).read_text(encoding='utf-8'))['console_handle'],0)


if __name__=='__main__':unittest.main()
