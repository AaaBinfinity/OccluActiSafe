"""N12 真实服务恢复验收：IDE 默认启动本次专用服务，拒绝占用现有服务。"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import select
import socket
import socketserver
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


class RPCProxy:
    """仅转发本测试端口的 TCP 字节；中断时关闭已有连接，保留真实 CARLA 进程。"""
    def __init__(self,target_port):
        self.lock=threading.Lock(); self.blocked=False; self.connections=set(); self.events=[]
        proxy=self
        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                with proxy.lock:
                    if proxy.blocked:return
                try: upstream=socket.create_connection(('127.0.0.1',target_port),timeout=1)
                except OSError:return
                pair=(self.request,upstream)
                try:
                    with proxy.lock:
                        if proxy.blocked:return
                        proxy.connections.add(pair)
                    while True:
                        ready,_,_=select.select(pair,[],[],.1)
                        for source in ready:
                            data=source.recv(65536)
                            if not data:return
                            (upstream if source is self.request else self.request).sendall(data)
                except (OSError,ValueError):pass
                finally:
                    with proxy.lock:proxy.connections.discard(pair)
                    upstream.close()
        class Server(socketserver.ThreadingTCPServer):daemon_threads=True
        self.server=Server(('127.0.0.1',0),Handler)
        self.port=self.server.server_address[1]
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()

    def block(self,value):
        with self.lock:
            self.blocked=value; self.events.append({'blocked':value,'monotonic_s':time.monotonic()})
            pairs=list(self.connections) if value else []
        for pair in pairs:
            for sock in pair:
                try:sock.shutdown(socket.SHUT_RDWR)
                except OSError:pass
                sock.close()

    def close(self):
        self.block(True); self.server.shutdown(); self.server.server_close();self.thread.join(timeout=2)


def snapshot(client):
    world=client.get_world(); settings=world.get_settings()
    return {'world_id':world.id,'actor_ids':sorted(a.id for a in world.get_actors()),
            'settings':{key:getattr(settings,key) for key in ('synchronous_mode','fixed_delta_seconds',
              'substepping','max_substeps','max_substep_delta_time','no_rendering_mode')}}


def wait_file(root,name,process,timeout=45):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        paths=list(root.rglob(name))
        if paths:return paths[0]
        if process.poll() is not None:raise RuntimeError('子进程在证据生成前退出：'+name)
        time.sleep(.05)
    raise TimeoutError('证据等待超时：'+name)


def suite(args,output):
    import carla
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1',2000))==0:raise RuntimeError('端口2000已占用，拒绝影响现有服务')
    startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    log=(output/'服务端日志.txt').open('wb')
    server=subprocess.Popen([str(args.server),'-RenderOffScreen','-nosound','-quality-level=Low'],cwd=args.server.parent,
            stdout=log,stderr=subprocess.STDOUT,startupinfo=startup,creationflags=subprocess.CREATE_NO_WINDOW)
    client=carla.Client('127.0.0.1',2000);client.set_timeout(2)
    sentinel=None;results=[]
    env=os.environ.copy();env['PYTHONUTF8']='1'
    try:
        deadline=time.monotonic()+90
        while True:
            try:
                if client.get_server_version()!='0.9.16':raise RuntimeError('服务端版本不符')
                world=client.get_world();world.get_settings();break
            except RuntimeError:
                if server.poll() is not None or time.monotonic()>deadline:raise
                time.sleep(.5)
        client.set_timeout(5)
        bp=world.get_blueprint_library().filter('vehicle.*')[0];bp.set_attribute('role_name','foreign:'+output.name)
        for point in reversed(world.get_map().get_spawn_points()):
            sentinel=world.try_spawn_actor(bp,point)
            if sentinel:break
        if sentinel is None:raise RuntimeError('无法创建外部 actor 验证对象')
        deadline=time.monotonic()+5
        while sentinel.id not in snapshot(client)['actor_ids']:
            if time.monotonic()>deadline:raise RuntimeError('外部 actor 未进入快照')
            time.sleep(.1)
        for case in ('normal','disconnect','timeout','exception','transient_rpc','permanent_rpc','server_terminated'):
            directory=output/case;directory.mkdir();before=snapshot(client);original=client.get_world().get_settings()
            proxy=RPCProxy(2000) if case.endswith('_rpc') else None
            port=proxy.port if proxy else 2000
            command=[str(args.core_python),str(ROOT/'scripts/run_carla_bridge.py'),'--bridge-python',str(args.bridge_python),
                     '--output-root',str(directory),'--port',str(port)]
            if case!='normal':command+=['--fault-test',case if case in ('disconnect','timeout','exception') else 'server_loss']
            with (directory/'核心输出.txt').open('wb') as stdout:
                process=subprocess.Popen(command,stdout=stdout,stderr=subprocess.STDOUT,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
                entry={'case':case,'before':before,'sentinel_id':sentinel.id}
                started=None
                try:
                    ready=wait_file(directory,'握手.json',process)
                    if case in ('transient_rpc','permanent_rpc','server_terminated'):
                        started=time.monotonic();entry['frozen_frame']=json.loads(ready.read_text(encoding='utf-8'))['frame']
                        if proxy:proxy.block(True)
                        else:
                            killed=subprocess.run(['taskkill','/PID',str(server.pid),'/T','/F'],capture_output=True)
                            (directory/'中断服务.txt').write_bytes(killed.stdout+killed.stderr)
                            if killed.returncode:raise RuntimeError('本次服务端未中断')
                        if case=='transient_rpc':
                            path=wait_file(directory,'清理报告.json',process,15)
                            deadline=time.monotonic()+5
                            while True:
                                record=json.loads(path.read_text(encoding='utf-8'))
                                if any(a.get('error') for a in record['attempts']):break
                                if process.poll() is not None or time.monotonic()>deadline:raise RuntimeError('未观察到断连重试')
                                time.sleep(.05)
                            entry['during_frame']=client.get_world().get_snapshot().frame
                            proxy.block(False)
                    code=process.wait(timeout=90)
                    entry['exit_code']=code
                    if started:entry['fault_to_exit_s']=time.monotonic()-started
                    cleanup=json.loads(next(directory.rglob('清理报告.json')).read_text(encoding='utf-8'))
                    entry['cleanup']=cleanup
                    expected_failed=case in ('permanent_rpc','server_terminated')
                    passed=(code==(1 if case in ('transient_rpc','permanent_rpc','server_terminated') else 0))
                    passed=passed and cleanup['status']==('failed' if expected_failed else 'completed')
                    passed=passed and cleanup['settings_restored'] is not expected_failed and cleanup['elapsed_s']<=15
                    if started:passed=passed and entry['fault_to_exit_s']<=25
                    if case=='transient_rpc':passed=passed and entry['during_frame']==entry['frozen_frame']
                    if case!='server_terminated':
                        if expected_failed:
                            entry['before_test_reclaim']=snapshot(client)
                            # 验收证据已保存；只清理本测试留下且归属匹配的 actor，避免污染下个测试。
                            for owned in cleanup['owned']:
                                actor=client.get_world().get_actor(owned['actor_id'])
                                if actor:
                                    if actor.type_id!=owned['type_id'] or actor.attributes.get('role_name')!=owned['role_name']:
                                        raise RuntimeError('测试残留对象归属不匹配')
                                    response=client.apply_batch_sync([carla.command.DestroyActor(actor.id)],False)[0]
                                    if response.error:raise RuntimeError(response.error)
                            client.get_world().apply_settings(original)
                            write(directory/'测试残留清理.json',{'owned':cleanup['owned'],'settings_restored_by_test':True})
                        deadline=time.monotonic()+5;after=snapshot(client)
                        while before!=after and time.monotonic()<deadline:time.sleep(.1);after=snapshot(client)
                        entry['after']=after;passed=passed and before==after and sentinel.id in after['actor_ids']
                    entry['passed']=passed
                except Exception as exc:
                    entry.update(passed=False,error=repr(exc))
                finally:
                    if process.poll() is None:process.terminate();process.wait(timeout=10)
                    if proxy:entry['proxy_events']=proxy.events.copy();proxy.close()
            results.append(entry);write(output/'恢复验收.json',results);print(case,entry['passed'],flush=True)
            if not entry['passed']:raise RuntimeError('真实验收失败，保留证据：'+case)
        return 0
    finally:
        if server.poll() is None:
            if sentinel is not None:
                try:sentinel.destroy()
                except RuntimeError:pass
            subprocess.run(['taskkill','/PID',str(server.pid),'/T','/F'],capture_output=True)
        server.wait(timeout=10);log.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-python',type=Path,default=ROOT/'.venv/Scripts/python.exe')
    parser.add_argument('--bridge-python',type=Path,default=Path('H:/OccluActiSafeDataset/environments/carla_bridge_py311/Scripts/python.exe'))
    parser.add_argument('--server',type=Path,default=Path('H:/OccluActiSafeDataset/environments/carla_0.9.16_server/CarlaUE4.exe'))
    parser.add_argument('--output-root',type=Path,default=Path('H:/OccluActiSafeDataset/artifacts/n12_recovery'))
    parser.add_argument('--suite-worker',type=Path)
    args=parser.parse_args()
    if args.suite_worker:return suite(args,args.suite_worker)
    if os.name!='nt':raise RuntimeError('本入口只在当前 Windows 环境验证')
    if not args.bridge_python.is_file() or not args.server.is_file():raise FileNotFoundError('缺少桥接环境或CARLA服务端')
    output=args.output_root.resolve()/datetime.now().strftime('n12_recovery_%Y%m%d_%H%M%S_%f');output.mkdir(parents=True)
    sys.path.insert(0,str(ROOT/'src'))
    from occluactisafe.common.run_manifest import _source_identity
    write(output/'运行清单.json',{'source':_source_identity(ROOT),'recovery_budget_s':12.,'cleanup_rpc_s':.5,
                                'server':str(args.server),'bridge_python':str(args.bridge_python)})
    command=[str(args.bridge_python),str(Path(__file__).resolve()),'--suite-worker',str(output),
             '--server',str(args.server),'--core-python',str(args.core_python),'--bridge-python',str(args.bridge_python)]
    env=os.environ.copy();env['PYTHONUTF8']='1'
    result=subprocess.run(command,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
    if (output/'恢复验收.json').exists():
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        records=json.loads((output/'恢复验收.json').read_text(encoding='utf-8'))
        fig,ax=plt.subplots(figsize=(11,4));ax.bar(range(len(records)),[r.get('cleanup',{}).get('elapsed_s',0) for r in records])
        ax.axhline(15,color='red',linestyle='--');ax.set(xticks=range(len(records)),xticklabels=[r['case'] for r in records],ylabel='cleanup time (s)')
        fig.autofmt_xdate();fig.tight_layout();images={}
        for suffix in ('png','svg'):
            path=output/('断连恢复与限时收尾.'+suffix);fig.savefig(path,dpi=150);images[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        plt.close(fig);write(output/'图表来源.json',{'source':'恢复验收.json','source_sha256':hashlib.sha256((output/'恢复验收.json').read_bytes()).hexdigest(),'images':images})
    print(output);return result.returncode


if __name__=='__main__':raise SystemExit(main())
