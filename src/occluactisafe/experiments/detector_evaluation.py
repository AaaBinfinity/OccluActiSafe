"""N17真实RGB验证与统一离线指标；模型输入无GT。"""
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import json
import os
import subprocess
import time
import uuid

import numpy as np

from occluactisafe.common.config import PROJECT_ROOT
from occluactisafe.evaluation.detection_metrics import (
    GroundTruthObject, MetricFrame, score_detections, score_tracking, score_detection_strata, _match,
)
from occluactisafe.perception.types import Detection, Detections, PixelTrack, TrackNamespace
from occluactisafe.perception.tracker import ByteTrack, ByteTrackConfig
from occluactisafe.perception.adapters.ultralytics import UltralyticsDetector
from .detector_data import load_index
from .run_store import sha256, read_json, write_json, append_json, now, manifest
from .selection import select_detector, family_candidates
from .source_version import source_info


def gt_from_row(row):
    return tuple(GroundTruthObject(**g) for g in row['gt'])


def sum_tracking(frames):
    groups = defaultdict(list)
    for f in frames:
        groups[dict(f.metadata)['namespace']].append(f)
    metrics = [score_tracking(items) for items in groups.values()]
    keys = ['gt_instances', 'predicted_tracks', 'idtp', 'idfp', 'idfn', 'id_switches', 'matched_instances']
    total = {k: sum(m[k] for m in metrics) for k in keys}
    denom = 2*total['idtp']+total['idfp']+total['idfn']
    total['idf1'] = 2*total['idtp']/denom if denom else None
    total['track_coverage'] = total['matched_instances']/total['gt_instances'] if total['gt_instances'] else None
    total['frames'] = len(frames)
    return total


def metric_summary(frames, operating_confidence):
    operating = [MetricFrame(f.frame_id, f.ground_truth,
                             tuple(p for p in f.detections if p.score >= operating_confidence), f.tracks, f.metadata)
                 for f in frames]
    point = score_detections(operating)
    aps = [score_detections(frames, iou_threshold=t) for t in np.linspace(.5, .95, 10)]
    point['map50'] = aps[0]['map_at_iou_threshold']
    point['map50_95'] = float(np.mean([r['map_at_iou_threshold'] for r in aps]))
    point['ap50_by_class'] = aps[0]['ap_by_class']
    point['ap50_95_by_class'] = {k: float(np.mean([r['ap_by_class'][k] for r in aps]))
                                if aps[0]['ap_by_class'][k] is not None else None for k in aps[0]['ap_by_class']}
    point['tracking'] = sum_tracking(operating)
    point['strata'] = score_detection_strata(operating)
    by_side = {}
    for side in sorted({dict(f.metadata)['camera_side'] for f in frames}):
        subset = [f for f in frames if dict(f.metadata)['camera_side'] == side]
        values = [score_detections(subset, iou_threshold=t) for t in np.linspace(.5, .95, 10)]
        by_side[side] = {'frames': len(subset), 'map50': values[0]['map_at_iou_threshold'],
                         'map50_95': float(np.mean([v['map_at_iou_threshold'] for v in values])),
                         'tracking': sum_tracking(subset)}
    point['by_camera_side'] = by_side
    # GT条件分层：匹配其他分层GT的预测忽略；无法归属遮挡状态的背景FP共享且显式标记。
    conditional = {}
    for field in ('class_id', 'pixel_scale', 'official_occlusion_state'):
        conditional[field] = {}
        values = sorted({getattr(g, field) for f in frames for g in f.ground_truth})
        for value in values:
            subset = []
            for f in frames:
                matches, _, _ = _match(f.ground_truth, f.detections, .5)
                excluded_pred = {pi for gi, pi in matches if getattr(f.ground_truth[gi], field) != value}
                gt = tuple(g for g in f.ground_truth if getattr(g, field) == value)
                det = tuple(p for pi, p in enumerate(f.detections) if pi not in excluded_pred
                            and (field != 'class_id' or p.class_id == value))
                track_matches, _, _ = _match(f.ground_truth, f.tracks, .5)
                excluded_tracks = {pi for gi, pi in track_matches if getattr(f.ground_truth[gi], field) != value}
                tracks = tuple(t for pi, t in enumerate(f.tracks) if pi not in excluded_tracks
                               and (field != 'class_id' or t.class_id == value))
                subset.append(MetricFrame(f.frame_id, gt, det, tracks, f.metadata))
            ap = score_detections(subset)
            op = score_detections([MetricFrame(f.frame_id, f.ground_truth,
                                               tuple(p for p in f.detections if p.score >= operating_confidence)) for f in subset])
            conditional[field][value] = {'gt': op['ground_truth_count'], 'tp': op['tp'], 'fp': op['fp'],
                                         'fn': op['fn'], 'recall': op['recall'],
                                         'ap50_gt_conditioned': ap['map_at_iou_threshold'],
                                         'tracking_gt_conditioned': sum_tracking(subset),
                                         'background_fp_shared_between_gt_strata': field != 'class_id'}
    point['conditional_strata'] = conditional
    negative = [f for f in operating if not f.ground_truth]
    point['negative_frames'] = len(negative)
    point['false_positives_in_negative_frames'] = sum(len(f.detections) for f in negative)
    point['false_positives_per_empty_frame'] = point['false_positives_in_negative_frames']/len(negative) if negative else None
    return point


def overlay(rgb, row, detections, directory, index):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    from .visual_fonts import chinese_font
    font = ImageFont.truetype(str(chinese_font()), 22)
    for gt in row['gt']:
        draw.rectangle(gt['xyxy'], outline=(255,90,90), width=2)
    for pred in detections:
        if pred.score >= .25:
            draw.rectangle(pred.xyxy, outline=(0,255,120), width=2)
    draw.rectangle((5,5,1100,85), fill='black')
    draw.text((12,10), f"真实检测结果 | 红色GT，绿色预测(score≥0.25) | {row['domain']} {row['camera_side']}", font=font, fill='white')
    draw.text((12,45), f"帧 {row['frame_id']} | GT={len(row['gt'])} 预测={sum(d.score >= .25 for d in detections)}", font=font, fill='white')
    image.save(directory/f'检测叠加_{row["domain"]}_{index:04d}.png')


def evaluate_one(cfg, model_cfg, weights, stage, seed, root):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('N17验证必须使用本机CUDA')
    device = int(cfg.get('device', 0))
    torch.cuda.set_device(device)
    settings = Path(cfg['output_root'])/'04_开发与复核'/'Ultralytics设置'
    settings.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(settings)
    os.environ['YOLO_OFFLINE'] = 'true'
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    from .visual_fonts import prepare_font_cache
    prepare_font_cache(settings,cfg.get('font_path'))
    if cfg.get('font_path'):
        os.environ['OAS_CJK_FONT'] = cfg['font_path']
    from ultralytics import YOLO, __version__
    from ultralytics.utils.torch_utils import get_flops
    rows, data = load_index(cfg)
    rows = [r for r in rows if r['role'] == 'model_val']
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    candidate_id = model_cfg['family']+'_'+stage+('_'+str(seed) if seed is not None else '')
    result = dict(candidate_id=candidate_id, family=model_cfg['family'], stage=stage, seed=seed,
                  split='model_val', status='running', started_at=now(), config_sha256=cfg['_config_sha256'],
                  data_index_sha256=data['index_sha256'], weight_sha256=sha256(weights),
                  weights=str(weights), ultralytics=__version__, expected_frames=len(rows),
                  **source_info(), device=device, gpu_name=torch.cuda.get_device_name(device),
                  gpu_uuid=str(getattr(torch.cuda.get_device_properties(device),'uuid','unknown')))
    result.pop('protected_files')
    write_json(root/'评测状态.json', result)
    try:
        model = YOLO(str(weights))
        result['parameter_count'] = sum(p.numel() for p in model.model.parameters())
        result['gflops_960'] = float(get_flops(model.model, imgsz=cfg['evaluation']['imgsz']))
        e = cfg['evaluation']
        class_map = ({0:'pedestrian', 1:'cyclist', 2:'vehicle', 3:'cyclist', 5:'vehicle', 7:'vehicle'}
                     if stage == 'pretrained' else dict(enumerate(cfg['classes'])))
        detector = UltralyticsDetector(model=model, class_map=class_map, device=str(device), confidence=e['confidence_keep'],
                                       imgsz=e['imgsz'], predict_kwargs={'classes': list(class_map), 'quantize': 16 if e['half'] else 32,
                                       'iou': e['iou_nms'], 'max_det': e['max_det'], 'nms': not model_cfg['end2end']})
        result['postprocessing'] = 'end2end_nms_false' if model_cfg['end2end'] else 'standard_nms'
        from PIL import Image
        rgb = np.asarray(Image.open(rows[0]['image']).convert('RGB'))
        for _ in range(e['warmup_frames']):
            warm = detector.detect(rgb, frame_id='warmup', timestamp_s=0, namespace=TrackNamespace(**rows[0]['namespace']))
            if warm.status == 'error':
                raise RuntimeError(warm.error_message)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        tracker = ByteTrack(ByteTrackConfig())
        result['tracker_config'] = asdict(ByteTrackConfig())
        by_domain, timings = defaultdict(list), defaultdict(list)
        errors = []
        visualized = defaultdict(int)
        for index, row in enumerate(rows):
            if sha256(row['image']) != row['image_sha256']:
                raise ValueError('评测RGB哈希改变')
            t_io = time.perf_counter()
            with Image.open(row['image']) as im:
                rgb = np.asarray(im.convert('RGB'))
            io_ms = (time.perf_counter()-t_io)*1000
            ns = TrackNamespace(**row['namespace'])
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            det = detector.detect(rgb, frame_id=row['frame_id'], timestamp_s=row['timestamp_s'], namespace=ns)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            tracks = tracker.track(det, row['timestamp_s'])
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            timing = dict(detector_ms=(t1-t0)*1000, detection_tracking_ms=(t2-t0)*1000, image_read_ms=io_ms)
            timings[row['domain']].append(timing)
            if det.status == 'error':
                errors.append({'key': row['key'], 'error': det.error_message})
            meta = (('camera_side',row['camera_side']), ('namespace',ns.key))
            frame = MetricFrame(row['key'], gt_from_row(row), det.items, tracks, meta)
            by_domain[row['domain']].append(frame)
            append_json(root/'逐帧预测跟踪与计时.jsonl',
                        dict(key=row['key'], domain=row['domain'], status=det.status, error=det.error_message,
                             detections=[asdict(d) for d in det.items], tracks=[asdict(t) for t in tracks], **timing))
            if visualized[(row['domain'],row['camera_side'])] < 2:
                overlay(rgb, row, det.items, root, index)
                visualized[(row['domain'],row['camera_side'])] += 1
            _, missed, _ = _match(frame.ground_truth, tuple(p for p in det.items if p.score >= e['confidence_operating']), .5)
            if missed and visualized[(row['domain'],'failure')] < 2:
                overlay(rgb, row, det.items, root, index)
                append_json(root/'漏检样例索引.jsonl', {'key':row['key'], 'gt_count':len(row['gt']),
                            'false_negative_count':len(missed), 'image_sha256':row['image_sha256'],
                            'file':f'检测叠加_{row["domain"]}_{index:04d}.png'})
                visualized[(row['domain'],'failure')] += 1
            if (index+1)%500 == 0:
                print(f'{candidate_id}真实验证 {index+1}/{len(rows)}', flush=True)
        domains = {}
        for domain, frames in by_domain.items():
            domains[domain] = metric_summary(frames, e['confidence_operating'])
            domains[domain]['latency'] = {name: {'p50': float(np.percentile([r[name] for r in timings[domain]],50)),
                                               'p95': float(np.percentile([r[name] for r in timings[domain]],95)),
                                               'count': len(timings[domain]), 'unit': 'ms'}
                                           for name in ('detector_ms','detection_tracking_ms','image_read_ms')}
        result.update(status='completed' if not errors else 'failed', finished_at=now(), errors=errors,
                      actual_frames=sum(len(f) for f in by_domain.values()), domains=domains,
                      peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                      spd_recall={c: domains['SPD']['by_class'][c]['recall'] for c in cfg['gates']['minimum_recall_iou50']},
                      carla_empty_frame_fp=domains['CARLA']['false_positives_per_empty_frame'],
                      detector_p95_ms=max(d['latency']['detector_ms']['p95'] for d in domains.values()))
        write_json(root/'验证指标.json', result)
        write_json(root/'评测状态.json', result)
        write_json(root/'产物清单.json', manifest(root))
        return result
    except BaseException as exc:
        result.update(status='failed', finished_at=now(), error=repr(exc))
        raise
    finally:
        write_json(root/'评测状态.json', result)


def run_evaluation_matrix(cfg, training_matrix=None):
    root = Path(cfg['output_root'])/'02_运行产物'/('n17_eval_'+uuid.uuid4().hex)
    root.mkdir(parents=True)
    candidates = []
    for m in cfg['models']:
        weights = Path(cfg.get('weights_root', Path(cfg['output_root'])/'02_运行产物'/'官方预训练权重'))/m['weights']
        candidates.append((m, weights, 'pretrained', None))
    if training_matrix is not None:
        from .detector_audit import audit_training_matrix
        budget_audit=audit_training_matrix(cfg,training_matrix)
        write_json(root/'训练预算复核.json',budget_audit)
        if budget_audit['status']!='passed':
            raise ValueError('训练矩阵未完成冻结预算，保留训练预算复核，不冒充完整评测')
        trained = read_json(training_matrix)
        for run in trained['runs']:
            m = next(m for m in cfg['models'] if m['family'] == run['family'])
            candidates.append((m, Path(run.get('checkpoint','')), 'finetuned', run['seed']))
    outputs = []
    for m, weight, stage, seed in candidates:
        directory = root/(m['family']+'_'+stage+'_'+str(seed))
        try:
            outputs.append(evaluate_one(cfg, m, weight, stage, seed, directory))
        except Exception as exc:
            outputs.append({'candidate_id':directory.name, 'split':'model_val', 'status':'failed',
                            'family':m['family'], 'stage':stage, 'seed':seed, 'error':repr(exc)})
        write_json(root/'比较矩阵.json', {'run_id':root.name, 'runs':outputs})
    families = family_candidates(outputs, cfg['seeds'])
    selection = select_detector(families, cfg['gates'])
    selection['family_metrics'] = families
    write_json(root/'离线预选.json', selection)
    from .detector_audit import audit_evaluation_matrix
    write_json(root/'矩阵公平性复核.json',audit_evaluation_matrix(cfg,outputs,root,len(candidates)))
    comparison_plots(root, outputs)
    return root


def comparison_plots(root, runs):
    import csv
    import statistics
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .visual_fonts import configure_plot_fonts
    configure_plot_fonts()
    rows = []
    for run in runs:
        for domain, m in run.get('domains',{}).items():
            row = dict(candidate=run['candidate_id'], family=run['family'], stage=run['stage'], seed=run['seed'],
                       domain=domain, frames=m['frames'], gt=m['ground_truth_count'], tp=m['tp'], fp=m['fp'], fn=m['fn'],
                       map50=m['map50'], map50_95=m['map50_95'], recall=m['recall'], idf1=m['tracking']['idf1'],
                       id_switches=m['tracking']['id_switches'], track_coverage=m['tracking']['track_coverage'],
                       detector_p50_ms=m['latency']['detector_ms']['p50'], detector_p95_ms=m['latency']['detector_ms']['p95'],
                       detection_tracking_p95_ms=m['latency']['detection_tracking_ms']['p95'],
                       peak_gpu_mib=run['peak_gpu_allocated_mib'], parameter_count=run['parameter_count'])
            rows.append(row)
    if not rows:
        return
    path = root/'检测跟踪资源对照.csv'
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for stage,label in (('pretrained','预训练'),('finetuned','微调')):
        with (root/(label+'检测跟踪资源.csv')).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(r for r in rows if r['stage']==stage)
    grouped=defaultdict(list)
    for row in rows:grouped[(row['family'],row['stage'],row['domain'])].append(row)
    summary=[]
    for (family,stage,domain),items in grouped.items():
        entry=dict(family=family,stage=stage,domain=domain,seeds=[r['seed'] for r in items],runs=len(items),metrics={})
        for name in ('map50','map50_95','recall','idf1','id_switches','track_coverage','detector_p95_ms','peak_gpu_mib'):
            values=[r[name] for r in items if r[name] is not None]
            entry['metrics'][name]={'n':len(values),'mean':statistics.mean(values) if values else None,
                                    'sample_sd':statistics.stdev(values) if len(values)>1 else None,
                                    'minimum':min(values) if values else None,'maximum':max(values) if values else None}
        summary.append(entry)
    write_json(root/'各族种子统计.json',{'rows':summary,'independent_scenario_repetitions_claimed':False})
    differences=[]
    for i,left in enumerate(rows):
        for right in rows[i+1:]:
            if left['family']==right['family'] or (left['stage'],left['seed'],left['domain']) != (right['stage'],right['seed'],right['domain']):
                continue
            differences.append(dict(left=left['candidate'],right=right['candidate'],domain=left['domain'],seed=left['seed'],
                                    frames=left['frames'],gt=left['gt'],
                                    map50_95_difference=left['map50_95']-right['map50_95'],
                                    recall_difference=left['recall']-right['recall'],
                                    latency_p95_ms_difference=left['detector_p95_ms']-right['detector_p95_ms']))
    write_json(root/'同片段配对差值.json',{'rows':differences,'statistical_significance_claimed':False,
                                             'scope':'相同验证片段与同seed的原始差值，非总体因果收益'})
    figs=[]
    for domain in ('SPD','CARLA'):
        subset=[r for r in rows if r['domain']==domain]
        fig,axes=plt.subplots(3,1,figsize=(13,11))
        labels=[r['candidate'] for r in subset]
        for ax,key in zip(axes,('map50_95','recall','detector_p95_ms')):
            ax.bar(labels,[r[key] for r in subset]);ax.set(ylabel=key,title=domain+'模型验证（实际值）')
            ax.tick_params(axis='x',rotation=45);ax.grid(axis='y',alpha=.2)
        fig.tight_layout()
        for ext in ('png','svg'):
            p=root/(domain+'检测器对比图.'+ext);fig.savefig(p,dpi=160);figs.append({'path':p.name,'sha256':sha256(p)})
        plt.close(fig)
    write_json(root/'对比图表来源.json',{'raw_csv_sha256':sha256(path),'figures':figs,'smoothing':False})
