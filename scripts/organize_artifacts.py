"""整理移动硬盘既有产物；IDE默认预览，--apply实际归类并保留隐藏旧路径。"""
import argparse
import base64
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path('H:/OccluActiSafeDataset/artifacts')
CATALOG = ROOT/'docs/最终结果目录.json'
NODES = dict(enumerate(('工程安装', '数据契约', '轨迹审计', '窗口预处理', 'CV基线', 'GPU环境',
                        'GRU训练', '通信机制', '状态融合', '几何风险', '解析闭环', '仿真桥接',
                        '执行器标定', '盲区路口'), 1))


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write((json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))
    try: os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()


def bounded(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts: raise ValueError('产物路径必须是根目录内相对路径')
    path = root/relative
    if path == root or not path.resolve().is_relative_to(root): raise ValueError('产物路径越界')
    return path


def node_folder(node):
    number = int(node[1:])
    if number not in NODES: raise ValueError(f'尚未登记的节点：{node}')
    return Path('节点资料')/f'{node}_{NODES[number]}'


def is_link(path):
    return path.is_symlink() or bool(getattr(path.lstat(), 'st_reparse_tag', 0))


def make_link(path, target, directory=True):
    """只创建新入口；绝不覆盖任何目录、文件或已有不同目标。"""
    if os.path.lexists(path):
        if not is_link(path) or not os.path.samefile(path, target): raise FileExistsError(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if not directory:
        path.symlink_to(target)
        return
    quote = lambda text: "'"+str(text).replace("'", "''")+"'"
    command = "$ErrorActionPreference='Stop'; New-Item -ItemType Junction -Path "+quote(path)+" -Target "+quote(target)+" | Out-Null"
    encoded = base64.b64encode(command.encode('utf-16-le')).decode()
    shell = shutil.which('pwsh') or shutil.which('powershell')
    if not shell: raise RuntimeError('缺少Windows PowerShell，不能建立目录联接')
    result = subprocess.run([shell, '-NoProfile', '-NonInteractive', '-EncodedCommand', encoded],
                            capture_output=True, timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode: raise RuntimeError(result.stderr.decode('utf-8', errors='replace'))
    if not os.path.samefile(path, target): raise RuntimeError('目录联接目标不符')


def hide_link(path):
    if not is_link(path): raise ValueError('仅允许隐藏兼容链接，不能隐藏实际结果')
    library = ctypes.WinDLL('kernel32', use_last_error=True)
    library.SetFileAttributesW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32]
    library.SetFileAttributesW.restype = ctypes.c_int
    attributes = path.lstat().st_file_attributes
    if not library.SetFileAttributesW(str(path), attributes | 2): raise ctypes.WinError(ctypes.get_last_error())
    if not path.lstat().st_file_attributes & 2: raise RuntimeError('兼容入口未设为隐藏')


def build_plan(root, catalog):
    root = Path(root).resolve()
    if root.is_relative_to(ROOT) or not root.is_dir(): raise ValueError('必须使用已接入的仓库外产物根')
    final_sources = {x['original_artifact_path'].split('/')[0] for x in catalog['entries']}
    moves = []
    for source in sorted(root.iterdir()):
        if is_link(source) or source.name in ('节点资料', '公共管理', '产物导航.md'): continue
        match = re.match(r'n(\d{2})(?:_|$)', source.name)
        if source.is_dir() and match:
            node = 'N'+match[1]; folder = node_folder(node)
            category = '04_开发与复核'
            if re.fullmatch(r'n\d{2}', source.name) or source.name in final_sources: category = '02_运行产物'
            if re.fullmatch(r'n\d{2}_\d{8}', source.name) or source.name.startswith(('n07_step_', 'n14_budget_', 'n14_revision_')): category = '03_验收证据'
            if source.name in ('n12_preview', 'n12_recovery'): category = '02_运行产物'
            if source.name.startswith('n14_discovery_'): category = '05_地图发现'
            target = folder/category/source.name
        elif source.name == 'uv_cache' and source.is_dir():
            target = Path('公共管理/03_工具缓存')/source.name
        elif source.is_file():
            target = Path('公共管理')/('02_目录管理' if source.name.startswith(('最终结果目录', '目录整理')) else '01_图片与迁移核验')/source.name
        else: raise ValueError(f'未识别的根目录内容，需先明确用途：{source.name}')
        destination = bounded(root, target)
        if os.path.lexists(destination): raise FileExistsError(destination)
        info = source.stat()
        moves.append({'original': source.name, 'physical': target.as_posix(), 'directory': source.is_dir(),
                      'file_identity': [info.st_dev, info.st_ino], 'sha256': None if source.is_dir() else sha(source)})
    return {'root': str(root), 'moves': moves}


def apply_plan(plan, catalog, report_path):
    if os.name != 'nt': raise RuntimeError('仅实际验证Windows；当前系统不执行迁移')
    root = Path(plan['root']).resolve()
    report = {'scope': '既有产物目录整理，不重跑训练或仿真、不改变验收', 'moves': plan['moves'],
              'source_sha256': sha(Path(__file__)),
              'completed': [], 'final_directories': [], 'status': 'running'}
    write(report_path, report)
    for item in plan['moves']:
        source = bounded(root, item['original']); target = bounded(root, item['physical'])
        # 执行时再次核对范围和身份；仅同卷原子改名，不复制/递归删除。
        if [source.stat().st_dev, source.stat().st_ino] != item['file_identity'] or is_link(source):
            raise ValueError(f'待移动来源身份已改变：{source}')
        if os.path.lexists(target): raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        try: make_link(source, target, item['directory'])
        except BaseException:
            if not os.path.lexists(source): target.rename(source)
            raise
        hide_link(source)
        if [target.stat().st_dev, target.stat().st_ino] != item['file_identity'] or not os.path.samefile(source, target):
            raise RuntimeError('移动后文件身份不一致')
        if item['sha256'] and sha(source) != item['sha256']: raise RuntimeError('原文件内容改变')
        report['completed'].append(item['original']); write(report_path, report)
        if len(report['completed']) % 10 == 0: print(f"已归类 {len(report['completed'])}/{len(plan['moves'])}", flush=True)
    updated = dict(catalog); updated['entries'] = []
    for entry in catalog['entries']:
        source = bounded(root, entry['original_artifact_path']).resolve()
        evidence = source/entry['evidence_file']
        if sha(evidence) != entry['evidence_sha256']: raise RuntimeError('最终结果证据哈希改变')
        legacy = entry.get('legacy_directory', entry['directory'])
        alias = node_folder(entry['node'])/'01_最终结果'/Path(entry['directory']).name
        final = bounded(root, alias); make_link(final, source)
        old_alias = bounded(root, legacy)
        make_link(old_alias, source); hide_link(old_alias)
        if sha(final/entry['evidence_file']) != entry['evidence_sha256']: raise RuntimeError('新入口证据不匹配')
        updated['entries'].append({**entry, 'directory': alias.as_posix(), 'legacy_directory': legacy,
                                   'physical_artifact_path': source.relative_to(root).as_posix()})
        report['final_directories'].append(alias.as_posix())
    updated['storage'] = '按节点和用途归类的真实目录；last联接集中在各节点01_最终结果，根目录旧路径为隐藏兼容链接'
    report['status'] = 'passed'; report['moves_completed'] = len(report['completed'])
    report['last_verified'] = len(report['final_directories']); write(report_path, report)
    return updated


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts-root', type=Path, default=ARTIFACTS)
    parser.add_argument('--apply', action='store_true', help='实际归类；不带参数仅预览')
    args = parser.parse_args(argv)
    catalog = read(CATALOG); plan = build_plan(args.artifacts_root, catalog)
    if not args.apply:
        print(json.dumps(plan, ensure_ascii=False, indent=2)); return 0
    report = args.artifacts_root/'公共管理/02_目录管理'/('目录整理_'+uuid.uuid4().hex)/'核验.json'
    try:
        updated = apply_plan(plan, catalog, report); write(CATALOG, updated)
    except BaseException as exc:
        if report.exists():
            failed = read(report); failed.update(status='failed', error=repr(exc)); write(report, failed)
        raise
    print(json.dumps({'moved': len(plan['moves']), 'last_verified': len(updated['entries']), 'report': str(report)}, ensure_ascii=False))
    return 0


if __name__ == '__main__': raise SystemExit(main())
