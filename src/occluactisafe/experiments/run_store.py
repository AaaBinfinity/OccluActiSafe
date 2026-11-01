"""N17原子JSON与逐行落盘；失败保留。"""
from pathlib import Path
import hashlib
import json
import os
import tempfile
from datetime import datetime


def now():
    return datetime.now().astimezone().isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        f.write((json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())
    try:
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def append_json(path, data):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()


def manifest(directory):
    directory = Path(directory)
    return [{'path': p.relative_to(directory).as_posix(), 'bytes': p.stat().st_size,
             'sha256': sha256(p)} for p in sorted(directory.rglob('*'))
            if p.is_file() and p.name != '产物清单.json']
