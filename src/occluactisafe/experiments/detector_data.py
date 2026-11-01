"""N17只导出冻结训练/模型验证组，测试媒体不打开。"""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import json
import os

import yaml

from occluactisafe.common.config import PROJECT_ROOT
from occluactisafe.data.spd_perception import (
    iter_spd_perception_frames, load_frozen_spd_split, load_frozen_spd_audit,
)
from .run_store import sha256, read_json, write_json, append_json, now

DEFAULT_CONFIG = PROJECT_ROOT / 'configs/experiments/detector_train.yaml'


def load_config(path=DEFAULT_CONFIG):
    path = Path(path)
    path = (path if path.is_absolute() else PROJECT_ROOT/path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    cfg['_config_path'] = str(path)
    cfg['_config_sha256'] = sha256(path)
    return cfg


def class_box_line(gt, names, width, height):
    x1, y1, x2, y2 = gt['xyxy']
    if gt['class_id'] not in names or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError('标签类别/框不符合冻结像素契约')
    return '%d %.10f %.10f %.10f %.10f' % (
        names.index(gt['class_id']), (x1+x2)/2/width, (y1+y2)/2/height,
        (x2-x1)/width, (y2-y1)/height)


def materialize(row, root, names):
    role = row['role']
    if role not in ('train', 'model_val'):
        raise ValueError('N17导出禁止测试角色')
    image = Path(row['source_image'])
    if sha256(image) != row['image_sha256']:
        raise ValueError(f'RGB哈希改变: {image}')
    stem = row['key']
    target = root/'images'/role/(stem+image.suffix.lower())
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    os.link(image, target)  # 同H卷硬链接，不复制RGB，不写原件。
    label = root/'labels'/role/(stem+'.txt')
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text('\n'.join(class_box_line(gt, names, row['width'], row['height'])
                               for gt in row['gt'])+'\n', encoding='utf-8')
    row['image'] = str(target)
    row['label_sha256'] = sha256(label)
    return row


def prepare_dataset(cfg):
    root = Path(cfg['output_root'])/'02_运行产物'/'冻结数据'
    if (root/'数据清单.json').exists():
        saved = read_json(root/'数据清单.json')
        if saved['config_sha256'] != cfg['_config_sha256']:
            raise ValueError('已有数据配置哈希不同，禁止隐式覆盖')
        return saved
    if root.exists():
        raise ValueError('数据目录存在但未完成，保留失败后请使用明确的新目录')
    root.mkdir(parents=True)
    status = {'status': 'running', 'started_at': now(), 'config_sha256': cfg['_config_sha256']}
    write_json(root/'准备状态.json', status)
    rows = []
    try:
        spd_cfg = yaml.safe_load((PROJECT_ROOT/cfg['spd_config']).read_text(encoding='utf-8-sig'))
        spd = spd_cfg['dataset']
        split = load_frozen_spd_split(Path(spd['frozen_split_manifest']))
        hashes, _ = load_frozen_spd_audit(Path(spd['frozen_audit_report']), split)
        for role in ('train', 'model_val'):
            for frame in iter_spd_perception_frames(Path(spd['root']), split,
                                                    project_role=role, expected_source_sha256=hashes):
                ns = frame.namespace.key
                gt = [asdict(g) for g in frame.ground_truth]
                duplicate_ids = {k for k, n in Counter(g['track_id'] for g in gt if g['track_id']).items() if n > 1}
                excluded = 0
                for g in gt:
                    if g['track_id'] in duplicate_ids:
                        g['track_id'] = None
                        excluded += 1
                    elif g['track_id']:
                        g['track_id'] = ns+':'+g['track_id']
                row = dict(key='spd_'+frame.frame_id.replace(':', '_'), domain='SPD', role=role,
                           source_image=str(frame.image_path), image_sha256=frame.image_sha256,
                           width=frame.width, height=frame.height, gt=gt,
                           timestamp_s=frame.timestamp_s, namespace=asdict(frame.namespace),
                           camera_side=frame.namespace.sensor_id, frame_id=frame.frame_id,
                           ambiguous_gt_ids_excluded=excluded)
                rows.append(materialize(row, root, cfg['classes']))
                if len(rows) % 1000 == 0:
                    print(f'真实SPD导出 {len(rows)}帧', flush=True)
        carla_path = Path(cfg['carla_manifest'])
        carla = read_json(carla_path)
        for group in carla['groups']:
            if group['project_role'] not in ('train', 'model_val'):
                continue
            path = Path(group['manifest'])
            if sha256(path) != group['manifest_sha256']:
                raise ValueError('CARLA场景清单SHA不一致')
            manifest = read_json(path)
            for frame in manifest['frames']:
                for view, item in frame['views'].items():
                    label_path = path.parent/item['label']
                    if sha256(label_path) != item['label_sha256']:
                        raise ValueError('CARLA可见标注SHA不一致')
                    label = read_json(label_path)
                    ns = dict(source_id='carla', sensor_id='vehicle' if view == '车端' else 'roadside',
                              episode_id=group['episode_id'])
                    gt = []
                    for g in label['labels']:
                        box = g['bbox_xyxy_visible_pixels']
                        size = max(box[2]-box[0], box[3]-box[1])
                        gt.append(dict(xyxy=box, class_id=g['category'],
                                       track_id=ns['sensor_id']+':'+g['track_id'],
                                       raw_category=g['category'],
                                       pixel_scale='small_le_32px' if size <= 32 else 'medium_33_96px' if size <= 96 else 'large_over_96px',
                                       official_occlusion_state='not_available_proxy_only',
                                       truncation_state=g['truncated_state']))
                    key=f"carla_{group['map']}_{group['scene_kind']}_{ns['sensor_id']}_{frame['carla_frame_id']}"
                    row=dict(key=key, domain='CARLA', role=group['project_role'], source_image=str(path.parent/item['rgb']),
                             image_sha256=item['rgb_sha256'], width=label['image_size'][0], height=label['image_size'][1],
                             gt=gt, timestamp_s=frame['source_time_s'], namespace=ns, camera_side=ns['sensor_id'],
                             frame_id=key, ambiguous_gt_ids_excluded=0)
                    rows.append(materialize(row, root, cfg['classes']))
        for role in ('train', 'model_val'):
            (root/(role+'.txt')).write_text('\n'.join(r['image'] for r in rows if r['role'] == role)+'\n', encoding='utf-8')
        data = {'path': str(root), 'train': str(root/'train.txt'), 'val': str(root/'model_val.txt'),
                'names': dict(enumerate(cfg['classes']))}
        (root/'dataset.yaml').write_text(yaml.safe_dump(data, allow_unicode=True), encoding='utf-8')
        for row in rows:
            append_json(root/'逐帧数据索引.jsonl', row)
        result = {'status': 'completed', 'started_at': status['started_at'], 'finished_at': now(),
                  'root': str(root), 'config_sha256': cfg['_config_sha256'],
                  'split_sha256': sha256(spd['frozen_split_manifest']),
                  'carla_manifest_sha256': sha256(carla_path),
                  'index_sha256': sha256(root/'逐帧数据索引.jsonl'),
                  'counts': dict(Counter(r['domain']+':'+r['role'] for r in rows)),
                  'ground_truth_counts': dict(Counter(g['class_id'] for r in rows for g in r['gt'])),
                  'ambiguous_gt_instances_excluded_from_identity_metrics': sum(r['ambiguous_gt_ids_excluded'] for r in rows),
                  'test_images_opened': 0}
        write_json(root/'数据清单.json', result)
        write_json(root/'准备状态.json', result)
        return result
    except BaseException as exc:
        write_json(root/'准备状态.json', {**status, 'status': 'failed', 'error': repr(exc), 'finished_at': now()})
        raise


def load_index(cfg):
    snapshot = cfg.get('dataset_snapshot')
    if snapshot is not None:
        root = Path(snapshot['root'])
        if not root.is_absolute():
            raise ValueError('运行数据快照必须使用绝对路径')
        audit = read_json(root/'数据清单.json')
        if (sha256(root/'数据清单.json') != snapshot['manifest_sha256']
                or audit.get('schema_version') != 'n17-relocated-data-1'
                or audit.get('status') != 'completed'
                or audit['source_index_sha256'] != snapshot['source_index_sha256']
                or audit['classes'] != cfg['classes']
                or sha256(root/'逐帧数据索引.jsonl') != snapshot['index_sha256']
                or audit['index_sha256'] != snapshot['index_sha256']):
            raise ValueError('服务器数据快照SHA/类别/来源不一致')
        rows = [json.loads(line) for line in (root/'逐帧数据索引.jsonl').read_text(encoding='utf-8').splitlines()]
        if any(r['role'] not in ('train','model_val') for r in rows):
            raise ValueError('服务器数据快照含禁止的测试角色')
        return rows, audit
    root = Path(cfg['output_root'])/'02_运行产物'/'冻结数据'
    audit = read_json(root/'数据清单.json')
    if audit['config_sha256'] != cfg['_config_sha256'] or sha256(root/'逐帧数据索引.jsonl') != audit['index_sha256']:
        raise ValueError('冻结数据配置/索引哈希改变')
    return [json.loads(line) for line in (root/'逐帧数据索引.jsonl').read_text(encoding='utf-8').splitlines()], audit
