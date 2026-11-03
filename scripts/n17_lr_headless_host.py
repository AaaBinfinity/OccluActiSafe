"""N17无控制台宿主/worker边界；IDE默认只检查，不训练。"""
import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import runpy
import sys
from datetime import datetime

HERE=Path(__file__).resolve().parent
EVIDENCE=Path(r'H:\OccluActiSafeDataset\artifacts\节点资料\N17_检测器比较\03_验收证据\n17_lr_contrast_delivery_20261103\进程中断恢复_20261103\无控制台恢复_20261103')


def console_handle():
    if os.name!='nt':raise RuntimeError('仅在Windows验证此宿主')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetConsoleWindow.restype=wintypes.HWND
    return int(kernel.GetConsoleWindow() or 0)


def write_identity(path,kind):
    record={'time':datetime.now().astimezone().isoformat(),'pid':os.getpid(),
            'kind':kind,'console_handle':console_handle(),'executable':sys.executable}
    if record['console_handle']!=0:raise RuntimeError('检测到附着控制台，拒绝启动训练')
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(record,stream,ensure_ascii=False,indent=2)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('check','run','worker','smoke'),default='check')
    parser.add_argument('--worker-task',type=Path)
    parser.add_argument('--smoke-root',type=Path)
    args=parser.parse_args()
    if args.action=='check':
        if sys.stdout:print(json.dumps({'mode':'read_only_check','console_handle':console_handle(),'training_started':False}))
        return
    if args.action=='smoke':
        from n17_console_free import run_console_free
        root=args.smoke_root.resolve();write_identity(root/'无控制台宿主身份.json','mechanism_host')
        python=str(Path(sys.executable).with_name('python.exe'))
        code="import ctypes,json,os,sys; from ctypes import wintypes; from pathlib import Path; k=ctypes.WinDLL('kernel32'); k.GetConsoleWindow.restype=wintypes.HWND; Path(sys.argv[1]).write_text(json.dumps({'pid':os.getpid(),'console_handle':int(k.GetConsoleWindow() or 0),'model_training':False}),encoding='utf-8')"
        result=run_console_free([python,'-B','-c',code,str(root/'无控制台子进程身份.json')],root/'子进程输出.txt')
        if result:raise RuntimeError('无控制台宿主机制子进程失败')
        return
    if args.action=='worker':
        import recover_n17_lr_contrast as original
        task=args.worker_task.resolve()
        if not task.is_relative_to(original.STUDY.resolve()):raise ValueError('worker任务越界')
        data=json.loads(task.read_text(encoding='utf-8'));attempt=Path(data['attempt']).resolve()
        if not attempt.is_relative_to(original.STUDY.resolve()):raise ValueError('worker输出越界')
        write_identity(attempt/'无控制台worker身份.json',data['kind'])
        original.runtime()
        entry=original.RUNTIME/'scripts/run_n17_lr_contrast.py'
        sys.argv=[str(entry),'--worker-task',str(task)]
        runpy.run_path(str(entry),run_name='__main__');return
    # pythonw宿主没有终端；显式设置日志，避免GUI解释器的None标准输出。
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    write_identity(EVIDENCE/'无控制台宿主身份.json','production_host')
    stream=(EVIDENCE/'无控制台队列日志.txt').open('x',encoding='utf-8',buffering=1)
    sys.stdout=sys.stderr=stream;sys.stdin=open(os.devnull)
    os.environ.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1')
    try:
        import recover_n17_console_close
        recover_n17_console_close.main(['--action','run'])
        result={'completed':True,'exit_code':0}
    except BaseException as exc:
        import traceback
        traceback.print_exc();result={'completed':False,'error':repr(exc),'exit_code':1}
        raise
    finally:
        (EVIDENCE/'无控制台队列退出.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        stream.flush()


if __name__=='__main__':main()
