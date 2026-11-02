"""N17 Windows服务器入口；IDE无参数默认check，不会启动长训。"""
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from occluactisafe.experiments.detector_server import (
    DEFAULT_SERVER_CONFIG,server_settings,preflight,run_job,collect_results,worker,execution_lock,
)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_SERVER_CONFIG)
    parser.add_argument('--action',choices=['check','pilot','formal','collect'])
    parser.add_argument('--run',type=Path,help='任务级续跑/回传的原run绝对路径')
    parser.add_argument('--dry-run',action='store_true',help='pilot/formal只生成并核验计划，不启动训练')
    parser.add_argument('--worker-task',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker_task:
        worker(args.worker_task);return
    settings=server_settings(args.config);action=args.action or settings['action']
    if action!='collect' and not settings['input_root'].is_dir():
        parser.error('先复制便携输入包，并填写n17_server.yaml的input_root；未启动训练')
    if action=='check':
        with execution_lock(settings['output_root']/f'.gpu_{settings["device"]}.lock'):preflight(settings)
    elif action=='collect':
        if args.run is None:parser.error('collect需要--run，防止猜错回传目录')
        collect_results(settings,args.run)
    else:run_job(settings,action,args.run,args.dry_run)


if __name__=='__main__':
    main()
