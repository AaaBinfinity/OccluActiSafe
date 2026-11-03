"""Windows无控制台子进程：N17恢复编排使用；直接运行只显示机制，不启动训练。"""
import os
from pathlib import Path
import subprocess


def start_console_free(command, output, environment=None):
    if os.name != 'nt':
        raise RuntimeError('本入口只验证Windows无控制台运行，不替换其他平台契约')
    return subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output,
        stderr=subprocess.STDOUT, close_fds=True, creationflags=subprocess.CREATE_NO_WINDOW,
        env={**os.environ, 'PYTHONUTF8':'1', 'PYTHONIOENCODING':'utf-8',
             'PYTHONUNBUFFERED':'1', **(environment or {})})


def run_console_free(command, log, environment=None):
    # 不使用DETACHED_PROCESS/CREATE_NEW_CONSOLE，以免使CREATE_NO_WINDOW失效。
    with Path(log).open('xb') as stream:
        return start_console_free(command, stream, environment).wait()


if __name__ == '__main__':
    print('只读检查：Windows使用CREATE_NO_WINDOW，stdin关闭，日志独占创建；没有启动训练')
