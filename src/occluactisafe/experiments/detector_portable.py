"""N17冻结输入的便携包；只携带train/model_val，旧索引原字节保留。"""
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath, PurePosixPath
import re
import shutil
import subprocess
import uuid
import zipfile

import yaml

from occluactisafe.common.config import PROJECT_ROOT
from .detector_data import load_config, DEFAULT_CONFIG
from .run_store import read_json, write_json, sha256, now
from .source_version import VERSION_FILE, protected_source, source_info

INDEX_SHA = '83e9454340af6857bb7ccb5bf5b608560a764c9a8406a57cc123316b7f722870'
CONFIG_SHA = 'f8a74eea2957c4fe1a9d2dfad7e8fc6ae073dad4d3ec2fb0af3f906316d569b4'
WEIGHTS = {
    'yolo26n.pt': '9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef',
    'yolov8n.pt': 'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36',
    'yolo11n.pt': '0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1',
}


def bounded(root, relative):
    """清单始终用POSIX相对路径，拒绝Windows绝对路径与链接逃逸。"""
    if not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative:
        raise ValueError('输入清单路径非法')
    p = PurePosixPath(relative)
    if p.is_absolute() or '..' in p.parts or str(p) != relative:
        raise ValueError('输入清单路径越界或非规范')
    root = Path(root).resolve()
    result = (root/relative).resolve()
    if not result.is_relative_to(root) or result == root:
        raise ValueError('输入文件逃逸包根目录')
    return result


def row_paths(row):
    if row['role'] not in ('train', 'model_val'):
        raise ValueError('便携输入禁止test/risk_calibration角色')
    if not re.fullmatch(r'[A-Za-z0-9_]+', row['key']):
        raise ValueError('冻结帧key不能用于安全文件路径')
    ext = PureWindowsPath(row['image']).suffix.lower()
    if ext not in ('.png', '.jpg', '.jpeg'):
        raise ValueError('未知冻结RGB格式')
    return (f"data/images/{row['role']}/{row['key']}{ext}",
            f"data/labels/{row['role']}/{row['key']}.txt")


def link_or_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    try:
        os.link(source, target)
        return 'hardlink'
    except OSError:
        shutil.copy2(source, target)
        return 'copy'


def export_code_zip(destination, root=PROJECT_ROOT):
    """导出已提交源码；没有Git的服务器也能核验源码身份。"""
    root = Path(root)
    info = source_info(root)
    if info['source_kind'] != 'git':
        raise ValueError('代码打包须从原始Git工作区执行')
    destination = Path(destination)
    subprocess.run(['git', '-C', str(root), 'archive', '--format=zip',
                    '--prefix=OccluActiSafe/', '-o', str(destination), info['source_commit']], check=True)
    with zipfile.ZipFile(destination, 'a', compression=zipfile.ZIP_DEFLATED) as z:
        hashes = {name.removeprefix('OccluActiSafe/'): hashlib.sha256(z.read(name)).hexdigest()
                  for name in z.namelist() if protected_source(name.removeprefix('OccluActiSafe/'))}
        z.writestr('OccluActiSafe/'+VERSION_FILE, json.dumps(
            {'source_commit': info['source_commit'], 'files': hashes}, ensure_ascii=False, indent=2))
    return {'file': destination.name, 'sha256': sha256(destination), 'source_commit': info['source_commit']}


def export_input(source_root=None, destination_parent=None, include_code=True):
    cfg = load_config()
    source = Path(source_root or cfg['output_root']).resolve()
    data = source/'02_运行产物/冻结数据'
    original = read_json(data/'数据清单.json')
    if original['config_sha256'] != CONFIG_SHA or sha256(data/'逐帧数据索引.jsonl') != INDEX_SHA:
        raise ValueError('仅允许打包N17已封存输入；不猜测新数据或改写旧索引')
    parent = Path(destination_parent) if destination_parent else source.parents[2]/'服务器输入/N17'
    if parent.resolve().is_relative_to(PROJECT_ROOT):
        raise ValueError('数据/权重打包目录必须在Git仓库外')
    target = parent.resolve()/('n17_input_'+uuid.uuid4().hex)
    target.mkdir(parents=True)
    state = {'status': 'running', 'started_at': now(), 'source_root': str(source), 'directory': str(target)}
    write_json(target/'准备状态.json', state)
    try:
        rows = [json.loads(line) for line in (data/'逐帧数据索引.jsonl').read_text(encoding='utf-8').splitlines()]
        assets = []
        methods = Counter()
        for i, row in enumerate(rows, 1):
            image_rel, label_rel = row_paths(row)
            for src, relative, expected in ((Path(row['image']), image_rel, row['image_sha256']),
                                           (data/'labels'/row['role']/(row['key']+'.txt'), label_rel, row['label_sha256'])):
                if sha256(src) != expected:
                    raise ValueError(f'冻结输入SHA改变：{src}')
                method = link_or_copy(src, bounded(target, relative))
                if method == 'copy' and sha256(bounded(target, relative)) != expected:
                    raise ValueError('跨卷复制后SHA不符')
                methods[method] += 1
                assets.append({'path': relative, 'sha256': expected, 'bytes': src.stat().st_size})
            if i % 2000 == 0:
                print(f'服务器输入打包 {i}/{len(rows)}', flush=True)
        for filename in ('逐帧数据索引.jsonl', '数据清单.json'):
            src = data/filename; relative = 'data/'+filename
            link_or_copy(src, bounded(target, relative))
            assets.append({'path': relative, 'sha256': sha256(src), 'bytes': src.stat().st_size})
        for filename, expected in WEIGHTS.items():
            src = source/'02_运行产物/官方预训练权重'/filename
            if sha256(src) != expected:
                raise ValueError('官方初始权重SHA不符')
            method = link_or_copy(src, bounded(target, 'weights/'+filename)); methods[method] += 1
            if sha256(bounded(target, 'weights/'+filename)) != expected:
                raise ValueError('便携权重SHA不符')
            assets.append({'path': 'weights/'+filename, 'sha256': expected, 'bytes': src.stat().st_size})
        package = {'schema_version': 'n17-portable-input-1', 'status': 'completed', 'created_at': now(),
                   'source_index_sha256': INDEX_SHA, 'source_config_sha256': CONFIG_SHA,
                   'classes': cfg['classes'], 'counts': original['counts'], 'assets': assets,
                   'images': len(rows), 'test_media_included': 0, 'transfer_methods': dict(methods)}
        write_json(target/'输入清单.json', package)
        if include_code:
            write_json(target/'代码包来源.json', export_code_zip(target/'N17服务器代码.zip'))
        (target/'使用说明.md').write_text(
            '# N17服务器输入\n\n整目录复制到服务器，包含data、weights与输入清单。'
            '不要只复制目录联接，不携带旧虚拟环境。原索引含H盘旧地址仅作来源；运行入口按key重新定位。\n\n'
            '解压N17服务器代码.zip，阅读其中docs/N17服务器运行说明.md；在n17_server.yaml填写本目录绝对路径、'
            '输出目录和GPU编号。先Setup再check，pilot为开发对照，formal为完整三seed训练。\n'
            '原始数据与权重不入Git，包不含test。复制后入口会逐文件SHA核验。\n', encoding='utf-8')
        state.update(status='completed', finished_at=now(), images=len(rows), assets=len(assets),
                     manifest_sha256=sha256(target/'输入清单.json'))
        write_json(target/'准备状态.json', state)
        return target
    except BaseException as exc:
        state.update(status='failed', finished_at=now(), error=repr(exc))
        write_json(target/'准备状态.json', state)
        raise


def verify_input(root, hash_files=True):
    root = Path(root).resolve()
    package = read_json(root/'输入清单.json')
    if package.get('schema_version') != 'n17-portable-input-1' or package.get('status') != 'completed':
        raise ValueError('服务器输入包不完整')
    if package['source_index_sha256'] != INDEX_SHA or package['source_config_sha256'] != CONFIG_SHA:
        raise ValueError('输入包不是约定的N17冻结版本')
    index = root/'data/逐帧数据索引.jsonl'
    if sha256(index) != INDEX_SHA:
        raise ValueError('原索引SHA改变')
    rows = [json.loads(line) for line in index.read_text(encoding='utf-8').splitlines()]
    original = read_json(root/'data/数据清单.json')
    if original['status'] != 'completed' or original['index_sha256'] != INDEX_SHA or original['config_sha256'] != CONFIG_SHA:
        raise ValueError('原始数据清单身份改变')
    if package['classes'] != load_config()['classes']:
        raise ValueError('便携输入类别顺序改变')
    expected = {'data/逐帧数据索引.jsonl': INDEX_SHA,
                'data/数据清单.json': sha256(root/'data/数据清单.json')}
    seen_keys = set()
    for row in rows:
        if row['key'] in seen_keys:
            raise ValueError('冻结输入含重复帧key')
        seen_keys.add(row['key'])
        image, label = row_paths(row)
        expected.update({image: row['image_sha256'], label: row['label_sha256']})
    expected.update({'weights/'+name: digest for name, digest in WEIGHTS.items()})
    actual, seen_paths = {}, set()
    for item in package['assets']:
        key = item['path']
        path = bounded(root, key)
        if key.casefold() in seen_paths:
            raise ValueError('输入清单路径重复')
        seen_paths.add(key.casefold())
        actual[key] = item['sha256']
        if not path.is_file() or path.stat().st_size != item['bytes']:
            raise ValueError(f'输入文件缺失/大小改变：{key}')
        if hash_files and sha256(path) != item['sha256']:
            raise ValueError(f'输入文件SHA改变：{key}')
    if actual != expected:
        raise ValueError('输入清单漏文件、多文件或SHA与原索引不一致')
    for directory in ('data', 'weights'):
        for current, dirs, files in os.walk(root/directory, followlinks=False):
            for name in dirs:
                p = Path(current)/name
                if p.is_symlink() or getattr(p.lstat(), 'st_reparse_tag', 0):
                    raise ValueError('输入子目录含目录联接，请复制实际数据')
            for name in files:
                relative = (Path(current)/name).relative_to(root).as_posix()
                # Ultralytics只允许在标签旁新增这两个衍生cache，不允许新增test媒体。
                if relative not in expected and relative not in ('data/labels/train.cache', 'data/labels/model_val.cache'):
                    raise ValueError(f'输入目录含未登记文件：{relative}')
    counts = dict(Counter(r['domain']+':'+r['role'] for r in rows))
    if counts != package['counts']:
        raise ValueError('输入帧数分母改变')
    return rows, {'status': 'passed', 'images': len(rows), 'assets': len(actual),
                  'source_index_sha256': INDEX_SHA, 'manifest_sha256': sha256(root/'输入清单.json'),
                  'test_media_opened': 0, 'counts': counts, 'full_hash_verification': hash_files}


def stage_snapshot(input_root, rows, destination):
    """只写路径重绑定的元数据；RGB/标签仍在便携输入目录，不复制20GB图像。"""
    input_root, destination = Path(input_root).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    cfg = load_config()
    moved = []
    for original in rows:
        row = copy.deepcopy(original)
        image, _ = row_paths(row)
        row.update(original_image=row['image'], original_source_image=row['source_image'],
                   image=str(bounded(input_root, image)), source_image=str(bounded(input_root, image)))
        moved.append(row)
    index = destination/'逐帧数据索引.jsonl'
    index.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in moved), encoding='utf-8')
    for role in ('train', 'model_val'):
        (destination/(role+'.txt')).write_text(''.join(r['image']+'\n' for r in moved if r['role']==role), encoding='utf-8')
    spec = {'path': str(input_root/'data'), 'train': str(destination/'train.txt'),
            'val': str(destination/'model_val.txt'), 'names': dict(enumerate(cfg['classes']))}
    (destination/'dataset.yaml').write_text(yaml.safe_dump(spec, allow_unicode=True), encoding='utf-8')
    meta = {'schema_version': 'n17-relocated-data-1', 'status': 'completed', 'root': str(destination),
            'source_config_sha256': CONFIG_SHA, 'source_index_sha256': INDEX_SHA,
            'index_sha256': sha256(index), 'classes': cfg['classes'],
            'counts': dict(Counter(r['domain']+':'+r['role'] for r in moved))}
    write_json(destination/'数据清单.json', meta)
    return {'root': str(destination), 'manifest_sha256': sha256(destination/'数据清单.json'),
            'index_sha256': sha256(index), 'source_index_sha256': INDEX_SHA}
