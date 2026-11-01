"""N17只依据验证域指标选择；型号名称不参与排序。"""


def select_detector(candidates, gates):
    decisions = []
    for row in candidates:
        if row.get('split') != 'model_val' or any('test' in key.lower() for key in row):
            raise ValueError('选择函数只接受model_val，禁止测试指标')
        reasons = []
        if row.get('status') != 'completed':
            reasons.append('运行未完成')
        recalls = row.get('spd_recall', {})
        for cls, floor in gates['minimum_recall_iou50'].items():
            value = recalls.get(cls)
            if value is None or value < floor:
                reasons.append(f'{cls}召回不足或不可估计')
        fp = row.get('carla_empty_frame_fp')
        if fp is None or fp > gates['maximum_false_positives_per_empty_frame']:
            reasons.append('CARLA负帧误警超限或不可估计')
        latency = row.get('detector_p95_ms')
        if latency is None or latency > gates['detector_end_to_end_p95_ms']:
            reasons.append('检测链p95时延超限或不可估计')
        decisions.append({'candidate_id': row['candidate_id'], 'eligible': not reasons,
                          'reasons': reasons})
    eligible = [r for r, d in zip(candidates, decisions) if d['eligible']]
    eligible.sort(key=lambda r: (-r.get('macro_recall_mean', sum(r['spd_recall'].values()) / len(r['spd_recall'])),
                                r['carla_empty_frame_fp'], r['detector_p95_ms'],
                                r['parameter_count']))
    return {'status': 'selected' if eligible else 'no_qualified_candidate',
            'selected_candidate': eligible[0]['candidate_id'] if eligible else None,
            'ranking': [r['candidate_id'] for r in eligible], 'decisions': decisions,
            'statistical_superiority_claimed': False,
            'scope': '离线验证入围；最终闭环选型留后续节点'}


def family_candidates(runs, seeds):
    """每族/阶段汇总，微调必须三seed全完成且各seed均过原门槛，避免挑seed。"""
    groups = {}
    for run in runs:
        groups.setdefault((run['family'], run['stage']), []).append(run)
    out = []
    for (family, stage), items in groups.items():
        expected = set(seeds) if stage == 'finetuned' else {None}
        complete = {r.get('seed') for r in items if r['status'] == 'completed'} == expected
        valid = [r for r in items if r['status'] == 'completed']
        row = dict(candidate_id=family+'_'+stage, split='model_val',
                   status='completed' if complete and len(valid) == len(expected) else 'failed',
                   member_ids=[r['candidate_id'] for r in items], seed_count=len(valid))
        if valid:
            classes = ('vehicle','pedestrian','cyclist')
            row.update(spd_recall={c: min(r['spd_recall'][c] for r in valid) for c in classes},
                       mean_spd_recall={c: sum(r['spd_recall'][c] for r in valid)/len(valid) for c in classes},
                       macro_recall_mean=sum(sum(r['spd_recall'][c] for c in classes)/3 for r in valid)/len(valid),
                       carla_empty_frame_fp=max(r['carla_empty_frame_fp'] for r in valid),
                       detector_p95_ms=max(r['detector_p95_ms'] for r in valid),
                       parameter_count=max(r['parameter_count'] for r in valid))
        out.append(row)
    return out
