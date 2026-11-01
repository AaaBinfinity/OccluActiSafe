"""Git工作区或可校验源码包的不可变版本；运行参数不混入源码身份。"""
import hashlib
import json
from pathlib import Path
import subprocess

from occluactisafe.common.config import PROJECT_ROOT

VERSION_FILE = '.n17_source_version.json'


def protected_source(path):
    return (path.startswith(('src/', 'scripts/')) and path.endswith(('.py', '.ps1'))
            or path in ('pyproject.toml', 'uv.lock', 'configs/experiments/detector_train.yaml'))


def source_info(root=PROJECT_ROOT):
    root = Path(root).resolve()
    packaged = root / VERSION_FILE
    if packaged.is_file():
        meta = json.loads(packaged.read_text(encoding='utf-8'))
        files = meta['files']
        for name, expected in files.items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or not protected_source(name):
                raise ValueError('源码包清单含越界或未知受保护文件')
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f'源码包已修改：{name}；请用新版本重新建任务')
        actual = {p.relative_to(root).as_posix() for folder in ('src', 'scripts')
                  for p in (root/folder).rglob('*') if p.is_file()
                  and protected_source(p.relative_to(root).as_posix())}
        if not actual.issubset(files):
            raise ValueError('源码包含未登记的Python/PowerShell源码')
        commit = meta['source_commit']
        kind = 'verified_source_package'
    else:
        try:
            commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                             text=True, stderr=subprocess.PIPE).strip()
            names = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z'],
                                            encoding='utf-8').split('\0')
            files = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                     for name in names if name and protected_source(name)}
            dirty = subprocess.check_output(
                ['git', '-C', str(root), 'status', '--porcelain', '--',
                 'src', 'scripts', 'pyproject.toml', 'uv.lock',
                 'configs/experiments/detector_train.yaml'], text=True)
            if dirty.strip():
                raise ValueError('受保护源码有未提交修改；先提交代码，站点参数请只改n17_server.yaml')
            kind = 'git'
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise RuntimeError('需完整Git工作区或带.n17_source_version.json的服务器代码包') from exc
    if not files or not isinstance(commit, str) or len(commit) != 40:
        raise ValueError('源码身份不完整')
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'source_commit': commit, 'source_fingerprint_sha256': digest,
            'source_kind': kind, 'protected_files': files}


def historical_source(commit, relative, root=PROJECT_ROOT):
    root = Path(root)
    if (root/VERSION_FILE).is_file():
        info = source_info(root)
        if commit != info['source_commit'] or relative not in info['protected_files']:
            raise ValueError('源码包不含请求的历史被测版本')
        return (root/relative).read_text(encoding='utf-8')
    return subprocess.check_output(['git', '-C', str(root), 'show', commit+':'+relative], encoding='utf-8')
