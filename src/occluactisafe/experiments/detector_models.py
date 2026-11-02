"""N17三族五尺寸；官方v8.4.0资产SHA固定，下载仅在显式准备时发生。"""
from pathlib import Path
import os
import shutil
import urllib.request
import uuid

from .run_store import write_json, sha256, now

SCALES = ('n', 's', 'm', 'l', 'x')
PREFIXES = {'YOLO26': 'yolo26', 'YOLOv8': 'yolov8', 'YOLO11': 'yolo11'}
# 2026-11-02核对GitHub官方assets/releases/tags/v8.4.0的digest字段。
ASSETS = {
    'yolo26n.pt': '9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef',
    'yolo26s.pt': '646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b',
    'yolo26m.pt': '401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7',
    'yolo26l.pt': '9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26',
    'yolo26x.pt': '9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92',
    'yolov8n.pt': 'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36',
    'yolov8s.pt': '1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a',
    'yolov8m.pt': '5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5',
    'yolov8l.pt': '64c9115303f6a25575f82200d1b22ec409fa6bd7d08d0313884fc20d919478cd',
    'yolov8x.pt': '3df4ada6b4dad6d657868f2fdf7faecfb34dcfccf3a25c4b82079064718524c8',
    'yolo11n.pt': '0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1',
    'yolo11s.pt': '85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5',
    'yolo11m.pt': 'd5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95',
    'yolo11l.pt': '9ebd0e09d59811db4b1d61e2bc6730649608b1ac47f8dd01e2da6bca7c20023f',
    'yolo11x.pt': '7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24',
}
RELEASE = 'https://github.com/ultralytics/assets/releases/download/v8.4.0/'


def expanded_models(scales):
    if (not isinstance(scales, list) or not scales or len(set(scales)) != len(scales)
            or any(s not in SCALES for s in scales)):
        raise ValueError('model_sizes须为不重复的n/s/m/l/x列表')
    return [{'family': family+scale, 'architecture': family, 'scale': scale,
             'weights': prefix+scale+'.pt', 'end2end': family == 'YOLO26'}
            for scale in SCALES if scale in scales for family, prefix in PREFIXES.items()]


def verify_weights(directory, models):
    directory = Path(directory).resolve()
    records = []
    for model in models:
        name = model['weights']
        if name not in ASSETS:
            raise ValueError('权重不在冻结官方资产表')
        path = directory/name
        if not path.is_file() or not path.resolve().is_relative_to(directory) or sha256(path) != ASSETS[name]:
            raise ValueError(f'缺少或改变官方权重：{path}；先显式prepare-weights')
        records.append({'model': model['family'], 'file': name, 'bytes': path.stat().st_size,
                        'sha256': ASSETS[name], 'url': RELEASE+name})
    return {'status': 'passed', 'release': 'v8.4.0', 'records': records}


def prepare_weights(directory, models, original=None):
    """不覆盖现有文件；失败part和准备记录保留，发布前逐文件校验固定SHA。"""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    report = directory/('准备_'+uuid.uuid4().hex+'.json')
    state = {'status': 'running', 'started_at': now(), 'completed': []}
    write_json(report, state)
    try:
        for model in models:
            name = model['weights']; target = directory/name
            if name not in ASSETS:
                raise ValueError('未知权重资产')
            if not target.exists():
                old = Path(original)/name if original else None
                if old and old.is_file() and sha256(old) == ASSETS[name]:
                    try: os.link(old, target)
                    except OSError: shutil.copy2(old, target)
                else:
                    temporary = directory/(name+'.'+uuid.uuid4().hex+'.part')
                    print('显式下载官方权重：'+name, flush=True)
                    with urllib.request.urlopen(RELEASE+name, timeout=90) as source, temporary.open('xb') as out:
                        shutil.copyfileobj(source, out, length=1024*1024)
                    if sha256(temporary) != ASSETS[name]:
                        raise ValueError('下载SHA不符，保留part：'+str(temporary))
                    if target.exists(): raise FileExistsError(target)
                    temporary.rename(target)
            verify_weights(directory, [model]); state['completed'].append(name); write_json(report, state)
        result = verify_weights(directory, models)
        state.update(status='completed', finished_at=now(), **{'assets': result})
        write_json(report, state)
        return result
    except BaseException as exc:
        state.update(status='failed', finished_at=now(), error=repr(exc)); write_json(report, state)
        raise
