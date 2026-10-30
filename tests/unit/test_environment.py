"""N01 轻量安装与环境入口检查；在已安装项目的干净虚拟环境中运行。"""

import contextlib
import importlib.metadata
import io
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_environment.py"


class EnvironmentChecks(unittest.TestCase):
    def test_installed_package_in_virtual_environment_and_lightweight_import(self):
        self.assertNotEqual(sys.prefix, sys.base_prefix, "请在干净虚拟环境安装本项目后运行")
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c",
             "import importlib.metadata,sys,occluactisafe; "
             "assert importlib.metadata.version('occluactisafe') == occluactisafe.__version__; "
             "assert not any(x in sys.modules for x in ('torch','carla','casadi')); "
             "print(occluactisafe.__version__)"],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(result.stdout.strip())

    def test_default_entry_at_three_working_directories(self):
        with tempfile.TemporaryDirectory() as other:
            for cwd in (ROOT, ROOT / "scripts", Path(other)):
                with self.subTest(cwd=str(cwd)):
                    result = subprocess.run(
                        [sys.executable, "-I", "-B", str(SCRIPT)], cwd=cwd,
                        capture_output=True, text=True, encoding="utf-8",
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn(f"项目根目录：{ROOT}", result.stdout)
                    self.assertIn("轻量工程环境检查通过", result.stdout)

    def test_running_entry_is_read_only_and_does_not_launch_or_connect(self):
        # 审计钩子阻止写文件、网络连接和子进程；临时工作目录也必须保持为空。
        wrapper = r'''
import os, runpy, sys
sys.dont_write_bytecode = True
def audit(event, args):
    if event in {"socket.connect", "socket.bind", "subprocess.Popen", "os.system", "os.spawn", "os.exec"}:
        raise RuntimeError("禁止操作: " + event)
    if event == "open":
        mode, flags = args[1], args[2]
        if (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
            raise RuntimeError("禁止写文件: " + str(args[0]))
    if event in {"os.mkdir", "os.remove", "os.rename", "os.rmdir"}:
        raise RuntimeError("禁止修改文件系统: " + event)
sys.addaudithook(audit)
script = sys.argv[1]
sys.argv = [script]
runpy.run_path(script, run_name="__main__")
'''
        with tempfile.TemporaryDirectory() as other:
            result = subprocess.run(
                [sys.executable, "-I", "-B", "-c", wrapper, str(SCRIPT)], cwd=other,
                capture_output=True, text=True, encoding="utf-8",
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(list(Path(other).iterdir()), [])
            self.assertIn("轻量工程环境检查通过", result.stdout)

    def test_import_script_does_not_run_main(self):
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c",
             "import runpy,sys; runpy.run_path(sys.argv[1], run_name='environment_import_test')",
             str(SCRIPT)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")

    def test_redirected_ascii_streams_keep_utf8_success_and_missing_package_errors(self):
        # 显式覆盖输出编码，不能依赖父进程 UTF-8 模式或系统代码页。
        env = {**os.environ, "PYTHONIOENCODING": "ascii"}
        for flags, expected_code in (([], 0), (["-S"], 1)):
            with self.subTest(flags=flags):
                result = subprocess.run(
                    [sys.executable, "-B", *flags, str(SCRIPT)], cwd=ROOT,
                    env=env, capture_output=True, text=True, encoding="utf-8",
                )
                self.assertEqual(result.returncode, expected_code, result.stdout + result.stderr)
                self.assertIn(f"项目根目录：{ROOT}", result.stdout)
                if expected_code == 0:
                    self.assertIn("轻量工程环境检查通过", result.stdout)
                else:
                    self.assertIn("缺少已安装的 occluactisafe 项目包", result.stderr)
                    self.assertIn("环境检查未通过", result.stderr)
                    self.assertNotIn("轻量工程环境检查通过", result.stdout + result.stderr)

    def test_missing_package_and_unsupported_python_are_failures(self):
        namespace = runpy.run_path(str(SCRIPT), run_name="environment_unit_test")
        main = namespace["main"]
        real_version = importlib.metadata.version
        def missing_project(name):
            if name == "occluactisafe":
                raise importlib.metadata.PackageNotFoundError(name)
            return real_version(name)
        cases = (
            (mock.patch("importlib.metadata.version", side_effect=missing_project),
             "缺少已安装的 occluactisafe 项目包"),
            (mock.patch.object(sys, "version_info", (3, 10, 0)), "Python 版本不支持"),
            (mock.patch.object(sys, "version_info", (3, 12, 0)), "Python 版本不支持"),
        )
        for patcher, expected in cases:
            with self.subTest(expected=expected), patcher:
                stdout, stderr = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    code = main([])
                self.assertNotEqual(code, 0)
                self.assertIn(expected, stderr.getvalue())
                self.assertIn("环境检查未通过", stderr.getvalue())
                self.assertNotIn("轻量工程环境检查通过", stdout.getvalue())
                self.assertNotIn("轻量工程环境检查通过", stderr.getvalue())

    def test_missing_runtime_dependency_is_failure(self):
        main = runpy.run_path(str(SCRIPT), run_name="environment_unit_test")["main"]
        real_version = importlib.metadata.version
        def version(name):
            if name == "PyYAML":
                raise importlib.metadata.PackageNotFoundError(name)
            return real_version(name)
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch("importlib.metadata.version", side_effect=version), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main([])
        self.assertNotEqual(code, 0)
        self.assertIn("缺少或无法导入依赖 PyYAML", stderr.getvalue())
        self.assertNotIn("轻量工程环境检查通过", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
