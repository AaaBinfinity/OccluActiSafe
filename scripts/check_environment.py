"""只读检查轻量工程环境；首次安装另行执行，运行时不会安装依赖。"""

import argparse
import importlib
import importlib.metadata
import importlib.util
import os
import platform
from pathlib import Path
import sys
import sysconfig


# 本节点的默认任务是检查已安装的轻量工程；不要求训练或仿真依赖。
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main(argv=None):
    """检查解释器、项目安装和仓库路径，失败时返回非零状态码。"""
    # 重定向管道在 Windows 上可能使用系统代码页；与调用方约定 UTF-8。
    # IDE/测试替代流（如 StringIO）没有 reconfigure，仍可直接使用。
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="检查 OccluActiSafe 轻量工程环境")
    parser.parse_args(argv)
    # 导入项目包时不在本次检查中写入字节码缓存。
    sys.dont_write_bytecode = True
    errors = []
    print("默认任务：轻量工程环境检查（不安装、不下载）")
    print(f"项目根目录：{PROJECT_ROOT}")
    # platform.uname() 在 Windows 上也可能运行 ver 命令。
    # 直接读取系统接口，保持检查过程不启动子进程。
    if sys.platform == "win32":
        version = sys.getwindowsversion()
        system, release = "Windows", f"{version.major}.{version.minor}.{version.build}"
        machine = sysconfig.get_platform()
    elif hasattr(os, "uname"):
        info = os.uname()
        system, release, machine = info.sysname, info.release, info.machine
    else:
        system, release, machine = sys.platform, "未查询", "未查询"
    print(f"操作系统：{system} {release}")
    print(f"解释器：{sys.executable}")
    print(f"Python：{platform.python_version()}（需要 3.11）")
    print(f"系统／解释器平台：{machine}")

    if sys.version_info[:2] != (3, 11):
        errors.append("Python 版本不支持：需要 Python 3.11。")
    if not (PROJECT_ROOT / "pyproject.toml").is_file():
        errors.append(f"找不到项目文件：{PROJECT_ROOT / 'pyproject.toml'}")

    try:
        distribution_version = importlib.metadata.version("occluactisafe")
    except importlib.metadata.PackageNotFoundError:
        errors.append("缺少已安装的 occluactisafe 项目包：请先在当前解释器环境安装本项目。")
    else:
        try:
            package = importlib.import_module("occluactisafe")
        except ImportError as exc:
            errors.append(f"occluactisafe 项目包无法导入：{exc}")
        else:
            print(f"项目包：{distribution_version}，位置：{package.__file__}")

    for distribution, module in (("numpy", "numpy"), ("PyYAML", "yaml")):
        try:
            version = importlib.metadata.version(distribution)
            importlib.import_module(module)
        except (importlib.metadata.PackageNotFoundError, ImportError) as exc:
            errors.append(f"缺少或无法导入依赖 {distribution}：{exc}；请重新安装本项目依赖。")
        else:
            print(f"依赖 {distribution}：{version}")

    for label, module in (("PyTorch", "torch"), ("CARLA", "carla"), ("CasADi", "casadi")):
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            available = False
        print(f"可选能力 {label}：{'发现模块' if available else '未安装'}（未验证兼容性）")
    print("GPU、驱动和显存：本节点未检查")

    if errors:
        for error in errors:
            print(f"缺项：{error}", file=sys.stderr)
        print("环境检查未通过", file=sys.stderr)
        return 1
    print("轻量工程环境检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
