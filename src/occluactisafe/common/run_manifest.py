"""记录可审计的运行上下文；不读取系统环境变量或保存凭据。"""

import hashlib
import json
import math
import platform
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone
from uuid import uuid4

import yaml

from .config import PROJECT_ROOT


_SENSITIVE_KEY = re.compile(r"(?:password|passwd|secret|token|credential|api[_-]?key|private[_-]?key|authorization)", re.I)
_SENSITIVE_VALUE = re.compile(r"(?:bearer\s+\S+|(?:password|passwd|secret|token|api[_-]?key|credential)\s*[=:]\s*\S+|://)", re.I)
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")


def _check_public(value, context="manifest"):
    """Reject suspicious nested keys/strings rather than silently saving secrets."""
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or _SENSITIVE_KEY.search(key):
                raise ValueError(f"{context} 含不允许的凭据字段")
            _check_public(item, context + "." + key)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_public(item, context)
    elif isinstance(value, str):
        if _SENSITIVE_VALUE.search(value):
            raise ValueError(f"{context} 含疑似凭据或 URL，须先移除")
    elif value is None or type(value) in (bool, int):
        pass
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{context} 含非有限数")
    else:
        raise ValueError(f"{context} 含不可序列化字段类型")


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _source_identity(root):
    """Hash the actual current source and config contents, including untracked files."""
    files = []
    for folder in ("src", "scripts", "tests", "configs"):
        directory = root / folder
        if directory.exists():
            files.extend(p for p in directory.rglob("*") if p.is_file() and p.suffix in (".py", ".yaml", ".yml", ".json"))
    files.extend(p for p in (root / "pyproject.toml", root / "uv.lock", root / ".python-version") if p.is_file())
    if not files:
        raise ValueError("项目根目录不含可哈希的源码或配置文件")
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    def git(*args):
        try:
            result = subprocess.run(["git", "-C", str(root), *args], stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, check=True, text=True)
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    commit = git("rev-parse", "--verify", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=normal")
    return {"git_commit": commit, "working_tree_dirty": None if status is None else bool(status),
            "source_sha256": digest.hexdigest(), "source_file_count": len(files),
            "hash_scope": "src,scripts,tests,configs 的 Python/YAML/JSON 文件及项目元数据"}


def create_run_manifest(*, config, input_mode, data, output_root, project_root=None):
    """创建一个唯一 run 目录并写入 manifest.json，返回清单字典。

    data 是包含 name/version/sha256 的映射，或明确的 None（无数据）。
    config 记录已解析配置及其哈希；拒绝凭据、URL、非有限数和任意对象。
    """
    if not isinstance(config, dict):
        raise ValueError("config 必须是已解析配置映射")
    if not isinstance(input_mode, str) or not input_mode.strip():
        raise ValueError("input_mode 必须是非空字符串")
    if data is not None:
        if not isinstance(data, dict) or set(data) != {"name", "version", "sha256"}:
            raise ValueError("data 必须包含且仅包含 name、version 和 sha256，或为 None")
        if not all(isinstance(data[key], str) and data[key] for key in ("name", "version")):
            raise ValueError("data name/version 必须为非空字符串")
        if not isinstance(data["sha256"], str) or not _SHA256.fullmatch(data["sha256"]):
            raise ValueError("data sha256 必须是64位十六进制字符串")
    _check_public(config, "config")
    _check_public(input_mode, "input_mode")
    _check_public(data, "data")
    config_bytes = _canonical(config)
    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    source = _source_identity(root)
    output = Path(output_root).expanduser()
    output = (output if output.is_absolute() else root / output).resolve()
    # 保证并发创建冲突不会覆盖已有 run。
    output.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        run_id = uuid4().hex
        run_dir = output / run_id
        try:
            run_dir.mkdir()
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError("无法创建唯一 run 目录")
    manifest = {
        "schema_version": "1.0", "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(), "input_mode": input_mode,
        "source": source, "config": {"resolved": config, "sha256": hashlib.sha256(config_bytes).hexdigest(),
                                      "file": "resolved_config.yaml"},
        "data": data, "seed": config.get("runtime", {}).get("seed"),
        "environment": {"os": platform.system(), "release": platform.release(),
                        "architecture": platform.machine(), "python": platform.python_version(),
                        "implementation": platform.python_implementation()},
    }
    _check_public(manifest)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=True), encoding="utf-8")
    (run_dir / "manifest.json").write_bytes(_canonical(manifest) + b"\n")
    return manifest
