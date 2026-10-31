"""N12真实桌面预览验收；默认启动本次专用服务，拒绝接管现存2000端口。"""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from test_n12_recovery import snapshot,wait_file

ROOT=Path(__file__).resolve().parents[1]


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def capture_window(directory):
    from PIL import ImageGrab
    user32=ctypes.windll.user32;user32.SetProcessDPIAware()
    user32.FindWindowW.argtypes=[wintypes.LPCWSTR,wintypes.LPCWSTR];user32.FindWindowW.restype=wintypes.HWND
    user32.GetWindowRect.argtypes=[wintypes.HWND,ctypes.POINTER(wintypes.RECT)]
    user32.GetWindowThreadProcessId.argtypes=[wintypes.HWND,ctypes.POINTER(wintypes.DWORD)]
    user32.PostMessageW.argtypes=[wintypes.HWND,wintypes.UINT,wintypes.WPARAM,wintypes.LPARAM]
    metadata=json.loads((directory/'预览窗口.json').read_text(encoding='utf-8'))
    window=user32.FindWindowW(None,metadata['title']);pid=wintypes.DWORD()
    if not window:raise RuntimeError('没有找到本次预览窗口')
    user32.GetWindowThreadProcessId(window,ctypes.byref(pid))
    if pid.value!=metadata['pid']:raise RuntimeError('窗口不属于本次运行')
    user32.ShowWindow(window,9);user32.SetForegroundWindow(window);time.sleep(.3)
    before=json.loads((directory/'预览帧记录.jsonl').read_text(encoding='utf-8').splitlines()[-1])
    rect=wintypes.RECT();user32.GetWindowRect(window,ctypes.byref(rect))
    path=directory/'实时预览窗口实测.png';ImageGrab.grab(bbox=(rect.left,rect.top,rect.right,rect.bottom)).save(path)
    after=json.loads((directory/'预览帧记录.jsonl').read_text(encoding='utf-8').splitlines()[-1])
    write(directory/'窗口截图来源.json',{'kind':'actual_desktop_window','image':path.name,
          'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'window':metadata,'frame_range':[before['frame'],after['frame']]})
    # 使用真实窗口消息触发 F12 绑定，不直接调用归档函数。
    user32.PostMessageW(metadata['window_id'],0x100,0x7b,0)
    user32.PostMessageW(metadata['window_id'],0x101,0x7b,0)


def pose(world):
    transform=world.get_spectator().get_transform()
    return [transform.location.x,transform.location.y,transform.location.z,
            transform.rotation.pitch,transform.rotation.yaw,transform.rotation.roll]


def suite(args,output):
    import carla
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1',2000))==0:raise RuntimeError('现存CARLA端口已占用，拒绝接管')
    startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=1
    with (output/'服务端日志.txt').open('wb') as server_log:
        server=subprocess.Popen([str(args.server),'-windowed','-ResX=960','-ResY=540','-quality-level=Low','-nosound'],
             cwd=args.server.parent,stdout=server_log,stderr=subprocess.STDOUT,startupinfo=startup)
        results=[];controls={}
        try:
            deadline=time.monotonic()+90
            while True:
                try:
                    with socket.create_connection(('127.0.0.1',2000),timeout=.5):pass
                    client=carla.Client('127.0.0.1',2000);client.set_timeout(5)
                    assert client.get_server_version()=='0.9.16';client.get_world().get_settings();break
                except (OSError,RuntimeError):
                    if server.poll() is not None or time.monotonic()>deadline:raise
                    time.sleep(.5)
            for case in ('headless','visual','close_window'):
                directory=output/case;directory.mkdir();before=snapshot(client);original_pose=pose(client.get_world())
                entry={'case':case,'before':before,'original_spectator':original_pose}
                script='run_carla_bridge.py' if case=='headless' else 'preview_carla_bridge.py'
                command=[str(args.core_python),str(ROOT/'scripts'/script),'--output-root',str(directory),'--bridge-python',str(args.bridge_python)]
                command+=['--ticks','400'] if case=='headless' else ['--preview-hold-seconds','1']
                env=os.environ.copy();env['PYTHONUTF8']='1'
                with (directory/'核心日志.txt').open('wb') as log:
                    process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                    try:
                        if case!='headless':
                            meta=wait_file(directory,'预览窗口.json',process);run=meta.parent
                            deadline=time.monotonic()+45
                            while True:
                                records=(run/'预览帧记录.jsonl').read_text(encoding='utf-8').splitlines()
                                if len(records)>=30:break
                                if process.poll() is not None or time.monotonic()>deadline:raise RuntimeError('预览未产生足够真实帧')
                                time.sleep(.05)
                            world=client.get_world();actors=[a for a in world.get_actors() if a.type_id.startswith('vehicle.') and a.attributes.get('role_name')=='oas:'+run.name]
                            if len(actors)!=1:raise RuntimeError('无法定位本次自车')
                            transform=actors[0].get_transform();forward=transform.get_forward_vector();actual=pose(world)
                            expected=[transform.location.x-7*forward.x,transform.location.y-7*forward.y,transform.location.z+4]
                            entry['following_spectator']=actual;entry['expected_follow_xyz']=expected
                            assert all(abs(a-b)<.2 for a,b in zip(actual[:3],expected))
                            if case=='visual':
                                subprocess.run([str(args.core_python),str(Path(__file__).resolve()),'--capture-window',str(run)],env=env,check=True,timeout=10)
                            else:
                                user32=ctypes.windll.user32
                                user32.FindWindowW.argtypes=[wintypes.LPCWSTR,wintypes.LPCWSTR];user32.FindWindowW.restype=wintypes.HWND
                                user32.PostMessageW.argtypes=[wintypes.HWND,wintypes.UINT,wintypes.WPARAM,wintypes.LPARAM]
                                title=json.loads(meta.read_text(encoding='utf-8'))['title']
                                user32.PostMessageW(user32.FindWindowW(None,title),0x10,0,0)
                        code=process.wait(timeout=100);assert code==0
                        run=next(directory.glob('n12_*'));rows=[json.loads(line) for line in (run/'时序.jsonl').read_text(encoding='utf-8').splitlines()]
                        cleanup=json.loads((run/'清理报告.json').read_text(encoding='utf-8'))
                        assert cleanup['status']=='completed' and cleanup['settings_restored']
                        after=snapshot(client);deadline=time.monotonic()+5
                        while before!=after and time.monotonic()<deadline:time.sleep(.1);after=snapshot(client)
                        assert before==after
                        restored=pose(client.get_world());assert all(abs(a-b)<.001 for a,b in zip(restored,original_pose))
                        entry.update(run_id=run.name,ticks=len(rows),after=after,restored_spectator=restored,exit_code=code)
                        controls[case]=[row['control'] for row in rows]
                        if case=='headless':assert len(rows)==400
                        else:
                            preview=json.loads((run/'预览结果.json').read_text(encoding='utf-8'));entry['preview']=preview
                            archive=json.loads((run/'预览图片来源.json').read_text(encoding='utf-8'))
                            assert cleanup['spectator_restored']
                            if case=='visual':
                                assert len(rows)==400 and preview['displayed_frames']==400 and not preview['stopped_by_user']
                                assert controls['headless']==controls['visual']
                                assert any(item['save_reason']=='手动' for item in archive['images'])
                                timing=[json.loads(line) for line in (run/'预览帧记录.jsonl').read_text(encoding='utf-8').splitlines()]
                                assert timing[-1]['wall_elapsed_s']-timing[0]['wall_elapsed_s']>=19.8
                                assert any(not row['valid'] for row in timing)
                            else:assert 30<=len(rows)<400 and preview['stopped_by_user']
                        entry['passed']=True
                    except Exception as exc:entry.update(passed=False,error=repr(exc))
                    finally:
                        if process.poll() is None:process.terminate();process.wait(timeout=10)
                results.append(entry);write(output/'可视验收.json',results);print(case,entry['passed'],flush=True)
                if not entry['passed']:raise RuntimeError('可视验收失败，保留证据：'+case)
        finally:
            if server.poll() is None:subprocess.run(['taskkill','/PID',str(server.pid),'/T','/F'],capture_output=True)
            server.wait(timeout=10)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-python',type=Path,default=ROOT/'.venv/Scripts/python.exe')
    parser.add_argument('--bridge-python',type=Path,default=Path('H:/OccluActiSafeDataset/environments/carla_bridge_py311/Scripts/python.exe'))
    parser.add_argument('--server',type=Path,default=Path('H:/OccluActiSafeDataset/environments/carla_0.9.16_server/CarlaUE4.exe'))
    parser.add_argument('--output-root',type=Path,default=Path('H:/OccluActiSafeDataset/artifacts/n12_preview'))
    parser.add_argument('--worker',type=Path);parser.add_argument('--capture-window',type=Path)
    args=parser.parse_args()
    if args.capture_window:return capture_window(args.capture_window)
    if args.worker:return suite(args,args.worker)
    output=args.output_root.resolve()/datetime.now().strftime('n12_preview_%Y%m%d_%H%M%S_%f');output.mkdir(parents=True)
    sys.path.insert(0,str(ROOT/'src'))
    from occluactisafe.common.run_manifest import _source_identity
    write(output/'运行清单.json',{'source':_source_identity(ROOT),'preview_hz':20,'default_visual_ticks':400})
    env=os.environ.copy();env['PYTHONUTF8']='1'
    command=[str(args.bridge_python),str(Path(__file__).resolve()),'--worker',str(output),'--core-python',str(args.core_python),
             '--bridge-python',str(args.bridge_python),'--server',str(args.server)]
    with (output/'验收进程日志.txt').open('wb') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env)
    print((output/'验收进程日志.txt').read_text(encoding='utf-8',errors='replace')[-3000:]);print(output)
    return result.returncode


if __name__=='__main__':raise SystemExit(main())
