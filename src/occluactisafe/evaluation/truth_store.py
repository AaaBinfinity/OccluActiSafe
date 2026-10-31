"""仅供桥接侧写入评测真值；不提供给在线核心。"""
import json
from pathlib import Path


class TruthStore:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open('x', encoding='utf-8')

    def append(self, record):
        self._file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        self._file.flush()

    def close(self): self._file.close()
