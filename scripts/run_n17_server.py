"""N17 Windows服务器入口；IDE无参数默认check，不会启动长训。"""
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from occluactisafe.experiments.detector_server import (
    DEFAULT_SERVER_CONFIG,server_settings,preflight,run_job,collect_results,worker,execution_lock,
)


def main(default_config=DEFAULT_SERVER_CONFIG):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=default_config)
    parser.add_argument('--action',choices=['check','pilot','formal','collect','prepare-weights','estimate'])
    parser.add_argument('--run',type=Path,help='任务级续跑/回传的原run绝对路径')
    parser.add_argument('--dry-run',action='store_true',help='pilot/formal只生成并核验计划，不启动训练')
    parser.add_argument('--worker-task',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker_task:
        worker(args.worker_task);return
    settings=server_settings(args.config);action=args.action or settings['action']
    if action not in ('collect','estimate') and not settings['input_root'].is_dir():
        parser.error('先复制便携输入包，并填写n17_server.yaml的input_root；未启动训练')
    if action=='prepare-weights':
        from occluactisafe.experiments.detector_models import prepare_weights
        from occluactisafe.experiments.detector_server import server_base
        with execution_lock(settings['weights_root']/'.prepare.lock'):
            prepare_weights(settings['weights_root'],server_base(settings)['models'],settings['input_root']/'weights')
    elif action=='estimate':
        import uuid
        from occluactisafe.experiments.detector_timing import timing_budget
        destination=settings['output_root']/'03_验收证据'/('n17_timing_'+uuid.uuid4().hex)
        result=timing_budget(settings,args.run,destination)
        print('耗时预算（非收敛保证）：'+str(destination/'耗时预算.md'))
        for case in result['scenarios']:print(f"{case['training_runs']}次×{case['assumed_epochs_per_run']}轮：{case['days_lower']:.1f}–{case['days_upper']:.1f}天")
    elif action=='check':
        with execution_lock(settings['output_root']/f'.gpu_{settings["device"]}.lock'):preflight(settings)
    elif action=='collect':
        if args.run is None:parser.error('collect需要--run，防止猜错回传目录')
        collect_results(settings,args.run)
    else:run_job(settings,action,args.run,args.dry_run)


if __name__=='__main__':
    main()
