"""N17首次准备：真实数据导出及显式官方权重下载；IDE无必填参数。"""
from pathlib import Path
import argparse
import sys
import os
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from occluactisafe.experiments.detector_data import load_config, DEFAULT_CONFIG, prepare_dataset
from occluactisafe.experiments.run_store import write_json, sha256, now


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    args = p.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg['output_root'])
    os.environ['YOLO_CONFIG_DIR'] = str(root/'04_开发与复核'/'Ultralytics设置')
    print(prepare_dataset(cfg), flush=True)
    from ultralytics.utils.downloads import safe_download
    weights = root/'02_运行产物'/'官方预训练权重'
    weights.mkdir(parents=True, exist_ok=True)
    records = []
    for item in cfg['models']:
        target = weights/item['weights']
        url = 'https://github.com/ultralytics/assets/releases/download/v8.4.0/'+item['weights']
        try:
            if not target.exists():
                safe_download(url=url, file=target, min_bytes=100000, retry=2)
            if not target.is_file():
                raise FileNotFoundError(target)
            records.append({'model': item['family'], 'url': url, 'path': str(target), 'sha256': sha256(target), 'status': 'downloaded'})
        except BaseException as exc:
            records.append({'model': item['family'], 'url': url, 'status': 'failed', 'error': repr(exc)})
            write_json(weights/'权重来源.json', {'date': now(), 'records': records})
            raise
    write_json(weights/'权重来源.json', {'date': now(), 'records': records})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
