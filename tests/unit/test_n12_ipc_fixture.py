"""真实 IPC + 模拟 CARLA 的机制测试；不是服务端验收。"""
import json
from multiprocessing.connection import Listener
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from occluactisafe.common.schemas import AdvanceRequest, ControlResult
from occluactisafe.simulation.carla_client import main
from occluactisafe.simulation.observation_adapter import decode_tick


class Actor:
    next_id = 0
    def __init__(self):
        Actor.next_id += 1; self.id = Actor.next_id; self.destroyed = False; self.callback = None
    def set_autopilot(self, value): raise AssertionError('新生手动车辆不应初始化 Traffic Manager')
    def apply_control(self, value): pass
    def listen(self, callback): self.callback = callback
    def stop(self): pass
    def destroy(self): self.destroyed = True
    def get_transform(self): return SimpleNamespace(location=SimpleNamespace(x=1, y=2, z=3))


class World:
    def __init__(self): self.frame=10; self.settings=SimpleNamespace(synchronous_mode=False,fixed_delta_seconds=None); self.actors=[]
    def get_settings(self): return SimpleNamespace(**vars(self.settings))
    def apply_settings(self, value): self.settings=value
    def get_snapshot(self): return SimpleNamespace(frame=self.frame,timestamp=SimpleNamespace(elapsed_seconds=self.frame*.05))
    def get_blueprint_library(self):
        bp=SimpleNamespace(set_attribute=lambda *args:None)
        return SimpleNamespace(filter=lambda _: [bp],find=lambda _:bp)
    def get_map(self): return SimpleNamespace(get_spawn_points=lambda:[object()])
    def try_spawn_actor(self,*args): self.actors.append(Actor()); return self.actors[-1]
    def spawn_actor(self,*args,**kwargs): self.actors.append(Actor()); return self.actors[-1]
    def tick(self):
        self.frame+=1
        image=SimpleNamespace(frame=self.frame,width=320,height=180,raw_data=bytes(320*180*4))
        self.actors[-1].callback(image)
        return self.frame


class BridgeIPCTests(unittest.TestCase):
    def test_socket_whitelist_truth_side_channel_and_cleanup(self):
        world=World(); client=SimpleNamespace(set_timeout=lambda _:None,get_client_version=lambda:'0.9.16',
             get_server_version=lambda:'0.9.16',get_world=lambda:world)
        def destroy_batch(commands, do_tick):
            self.assertFalse(do_tick)
            for actor_id in commands:
                next(actor for actor in world.actors if actor.id == actor_id).destroy()
            return [SimpleNamespace(error='') for _ in commands]
        client.apply_batch_sync=destroy_batch
        carla=SimpleNamespace(Client=lambda *args:client,Transform=lambda *args:object(),Location=lambda **kwargs:object(),
                              VehicleControl=lambda **kwargs:object(), command=SimpleNamespace(DestroyActor=lambda actor_id:actor_id))
        with tempfile.TemporaryDirectory() as directory, Listener(('127.0.0.1',0),authkey=b'test-key') as listener:
            config={'host':'127.0.0.1','port':2000,'output':directory,'run_id':'run','sensor_timeout_s':.1}
            env={'OAS_IPC_PORT':str(listener.address[1]),'OAS_IPC_AUTH':b'test-key'.hex(), 'OAS_BRIDGE_CONFIG':json.dumps(config)}
            errors=[]
            def worker():
                try: main()
                except Exception as exc: errors.append(exc)
            with patch.dict(sys.modules,{'carla':carla}), patch.dict(os.environ,env):
                thread=threading.Thread(target=worker); thread.start()
                with listener.accept() as conn:
                    ready=json.loads(conn.recv_bytes()); self.assertEqual(ready['frame'],10)
                    control=ControlResult('1','plan','success',True,'carla',.5,1.,0.,1.,0.,None)
                    request=AdvanceRequest('1.0','run','episode-1',1,11,'carla',.5,1.,control)
                    conn.send_bytes(json.dumps({'kind':'advance','request':request.to_dict()}).encode())
                    result=json.loads(conn.recv_bytes()); event,camera=decode_tick(result)
                    self.assertEqual(event.frame_id,11); self.assertTrue(camera['valid']); self.assertNotIn('truth',result)
                    conn.send_bytes(b'{"kind":"close"}'); self.assertEqual(json.loads(conn.recv_bytes()),{'kind':'closed'})
                thread.join(timeout=5); self.assertFalse(thread.is_alive()); self.assertEqual(errors,[])
            truth=json.loads((Path(directory)/'评测真值.jsonl').read_text(encoding='utf-8'))
            self.assertEqual(truth['frame'],11); self.assertTrue(all(actor.destroyed for actor in world.actors))
            cleanup=json.loads((Path(directory)/'清理报告.json').read_text(encoding='utf-8'))
            self.assertEqual(cleanup['status'],'completed'); self.assertTrue(cleanup['settings_restored'])
            self.assertFalse(world.settings.synchronous_mode); self.assertIsNone(world.settings.fixed_delta_seconds)


if __name__ == '__main__': unittest.main()
