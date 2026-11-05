"""为已完成训练但图集失败的本机N17任务补齐收尾，不启动模型。"""
from pathlib import Path
import argparse
import uuid

from occluactisafe.common.config import PROJECT_ROOT
from .run_store import read_json,write_json,sha256,now
from .source_version import source_info

DEFAULT_RUN=Path(r'H:/OccluActiSafeDataset/artifacts/节点资料/N17_检测器比较/02_运行产物/n17_local_yolov8n_seed17_234e51ae79a64486bbf93244c1c37c41')


def repair_gallery(training_run=DEFAULT_RUN):
    from .detector_data import load_config
    from .detector_local_gallery import detection_gallery
    from .detector_local import audit_local_training
    run=Path(training_run);run=(run if run.is_absolute() else PROJECT_ROOT/run).resolve()
    task=read_json(run/'本机训练任务.json');cfg=load_config(run/'本次训练配置.yaml')
    if task['status']!='failed' or task['frozen_config_sha256']!=cfg['_config_sha256']:
        raise ValueError('仅接受原配置SHA匹配的失败任务，不覆盖已完成任务')
    training=Path(task['training_directory']).resolve()
    if not training.is_relative_to(run):raise ValueError('训练目录不在指定run内')
    if (training/'误检漏检与检测样例').exists():raise FileExistsError('已有图集目录，保留不覆盖；请查看既有修复回执')
    # 仅新建缺失派生产物；先登记所有既有字节身份，旧状态/媒体/权重均不改写。
    prior={p.relative_to(run).as_posix():{'sha256':sha256(p),'bytes':p.stat().st_size}
           for p in run.rglob('*') if p.is_file() and not p.is_symlink()}
    out=Path(cfg['output_root'])/'03_验收证据'/('n17_gallery_repair_'+uuid.uuid4().hex);out.mkdir(parents=True)
    state={'run_id':out.name,'status':'running','started_at':now(),'training_run':str(run),'training_directory':str(training),
           'training_source_commit':task['source_commit'],'training_source_fingerprint_sha256':task['source_fingerprint_sha256'],
           'postprocess_source':source_info(),'configuration_sha256':cfg['_config_sha256'],
           'training_started':False,'model_inference_started':False,'test_media_opened':0,
           'scope':'补齐原任务缺失图集并审计已有训练；旧失败状态保留，回执单独说明修复完成'}
    write_json(out/'原任务文件身份.json',{'run':str(run),'files':prior})
    write_json(out/'修复状态.json',state)
    try:
        gallery=detection_gallery(cfg,training)
        audit=audit_local_training(cfg,training)
        for name,item in prior.items():
            if sha256(run/name)!=item['sha256']:raise ValueError('原任务文件SHA改变：'+name)
        if read_json(run/'本机训练任务.json')['status']!='failed':raise ValueError('原失败状态被改写')
        state.update(status='completed_postprocessing',finished_at=now(),gallery=gallery,training_audit=audit,
                     original_files_unchanged=len(prior),original_failure_status_preserved=True)
        write_json(out/'本次训练补充核验.json',audit)
    except BaseException as exc:
        state.update(status='failed',finished_at=now(),error=repr(exc));raise
    finally:write_json(out/'修复状态.json',state)
    (out/'结果说明.md').write_text(
        '# N17已保存结果的图集修复\n\n原训练未重跑，未执行模型推理；原任务失败记录保留。'
        '图集及完整训练审计已补齐，修复状态见本目录JSON。\n\n'
        f'原训练目录：{training}\n\n'
        f'实际训练{audit["actual_epochs"]}轮、最佳第{audit["termination"]["best_epoch"]}轮；'
        f'固定0.25/IoU0.5诊断TP={gallery["tp"]}、FP={gallery["fp"]}、FN={gallery["fn"]}，'
        f'FP中包含零面积预测{gallery["zero_area_at_threshold"]}个。'
        '这不是原生mAP复测、统一FP32/ByteTrack选型或闭环结果。\n',encoding='utf-8')
    print('图集补齐并通过已有训练审计；未重训，旧失败记录保持：'+str(out),flush=True)
    return out


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run',type=Path,default=DEFAULT_RUN,help='原本机训练任务根目录；无参数修复当前指定失败run')
    args=parser.parse_args(argv);return repair_gallery(args.training_run)
