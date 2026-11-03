"""真实关闭本测试自建的隐藏控制台；不运行模型，不关闭用户终端。"""
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import patch
import uuid

ENTRY=Path(__file__).resolve().parents[2]/'scripts/n17_console_free.py'
spec=importlib.util.spec_from_file_location('n17_console_free',ENTRY)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


@unittest.skipUnless(os.name=='nt','本组只验证Windows真实控制台事件')
class ConsoleIsolationTests(unittest.TestCase):
    def test_no_window_flag_and_closed_stdin(self):
        with patch.object(module.subprocess,'Popen') as popen:
            module.start_console_free(['fixture'],None)
        kwargs=popen.call_args.kwargs
        self.assertEqual(kwargs['creationflags'],subprocess.CREATE_NO_WINDOW)
        self.assertEqual(kwargs['stdin'],subprocess.DEVNULL)
        self.assertTrue(kwargs['close_fds'])

    def test_window_close_kills_attached_fixture_but_console_free_child_finishes(self):
        base=Path(r'H:\OccluActiSafeDataset\artifacts\节点资料\N17_检测器比较\03_验收证据\n17_lr_contrast_delivery_20261103\进程中断恢复_20261103')
        root=base/('控制台关闭机制_'+uuid.uuid4().hex);root.mkdir()
        child=root/'机制子进程.py'
        child.write_text('''import ctypes,json,os,sys,time
from pathlib import Path
from ctypes import wintypes
root=Path(sys.argv[1]);name=sys.argv[2]
kernel=ctypes.WinDLL('kernel32',use_last_error=True)
kernel.GetConsoleWindow.restype=wintypes.HWND
record={'pid':os.getpid(),'console':int(kernel.GetConsoleWindow() or 0),'fixture_only':True}
(root/(name+'_身份.json')).write_text(json.dumps(record),encoding='utf-8')
for i in range(40):
    with (root/(name+'_心跳.txt')).open('a') as stream:stream.write(str(i)+'\\n')
    time.sleep(.1)
(root/(name+'_完成.json')).write_text(json.dumps({'completed':True}),encoding='utf-8')
''',encoding='utf-8')
        parent=root/'机制父进程.py'
        parent.write_text('''import ctypes,json,os,subprocess,sys,time
from pathlib import Path
from ctypes import wintypes
root=Path(sys.argv[1]);sys.path.insert(0,sys.argv[2])
from n17_console_free import start_console_free
kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetConsoleWindow.restype=wintypes.HWND
with (root/'普通子进程.txt').open('wb') as stream:
    subprocess.Popen([sys.executable,str(root/'机制子进程.py'),str(root),'attached'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
with (root/'无控制台子进程.txt').open('wb') as stream:
    start_console_free([sys.executable,str(root/'机制子进程.py'),str(root),'isolated'],stream)
(root/'父进程身份.json').write_text(json.dumps({'pid':os.getpid(),'console':int(kernel.GetConsoleWindow() or 0)}),encoding='utf-8')
time.sleep(8)
''',encoding='utf-8')
        startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow=subprocess.SW_HIDE
        with (root/'父进程输出.txt').open('wb') as stream:
            # Windows venv的python.exe是转发器，使用基础解释器核验真实PID/窗口身份。
            process=subprocess.Popen([sys._base_executable,str(parent),str(root),str(ENTRY.parent)],
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_CONSOLE,startupinfo=startup)
        deadline=time.monotonic()+5
        files=['父进程身份.json','attached_身份.json','isolated_身份.json']
        while not all((root/f).is_file() and (root/f).stat().st_size for f in files):
            if time.monotonic()>deadline: self.fail('隐藏控制台夹具未准备就绪：'+str(root))
            time.sleep(.05)
        parent_id,attached,isolated=[json.loads((root/f).read_text(encoding='utf-8')) for f in files]
        self.assertEqual(parent_id['pid'],process.pid)
        self.assertNotEqual(parent_id['console'],0)
        self.assertEqual(attached['console'],parent_id['console'])
        self.assertEqual(isolated['console'],0)
        # HWND仅来自本测试新建父进程的身份文件，不枚举或关闭外部窗口。
        from ctypes import wintypes
        user=ctypes.WinDLL('user32',use_last_error=True)
        user.PostMessageW.argtypes=[wintypes.HWND,wintypes.UINT,wintypes.WPARAM,wintypes.LPARAM]
        self.assertTrue(user.PostMessageW(parent_id['console'],0x0010,0,0))  # WM_CLOSE
        result=process.wait(timeout=10)
        self.assertEqual(result & 0xffffffff,0xC000013A)
        deadline=time.monotonic()+6
        while not (root/'isolated_完成.json').is_file() and time.monotonic()<deadline:time.sleep(.1)
        self.assertTrue((root/'isolated_完成.json').is_file())
        self.assertFalse((root/'attached_完成.json').exists())
        (root/'控制台关闭核验.json').write_text(json.dumps({'status':'passed',
            'fixture_only':True,'parent_exit_code_hex':hex(result & 0xffffffff),
            'attached_child_finished':False,'isolated_child_finished':True,
            'isolated_console_handle':0,'closed_user_windows':0,'model_training_performed':False},indent=2),encoding='utf-8')


if __name__=='__main__':unittest.main()
