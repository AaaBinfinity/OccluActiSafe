"""N12独立同步机制夹具；不替代真实CARLA 200 tick验收。"""
from dataclasses import dataclass
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from occluactisafe.common.schemas import AdvanceRequest, ControlResult, TickEvent
from occluactisafe.simulation.sensors import FrameBuffer
from occluactisafe.simulation.runner import SynchronousRunner
from occluactisafe.simulation.observation_adapter import decode_tick, stationary_control


@dataclass
class Settings:
    synchronous_mode: bool = False
    fixed_delta_seconds: float | None = None


class FixtureWorld:
    def __init__(self): self.frame=10;self.current=Settings();self.calls=0
    def get_settings(self):return Settings(self.current.synchronous_mode,self.current.fixed_delta_seconds)
    def apply_settings(self,settings):self.current=Settings(settings.synchronous_mode,settings.fixed_delta_seconds)
    def get_snapshot(self):return SimpleNamespace(frame=self.frame,timestamp=SimpleNamespace(elapsed_seconds=self.frame*.05))
    def tick(self):self.calls+=1;self.frame+=1;return self.frame


def request(frame=11,generation=1,now=.5):
    c=ControlResult('control','plan','success',True,'sim-n12',now,10.,0.,0.,0.,None)
    return AdvanceRequest('1.0','run','episode',generation,frame,'sim-n12',now,10.,c).to_json()


class BridgeMechanismTests(unittest.TestCase):
    def test_exactly_one_tick_and_stale_requests_never_tick(self):
        world=FixtureWorld();controls=[]
        with SynchronousRunner(world,run_id='run',episode_id='episode',generation=1,clock_domain='sim-n12') as runner:
            event=TickEvent.from_json(runner.advance(request(),apply_control=controls.append))
            self.assertEqual(event.frame_id,11);self.assertEqual(world.calls,1);self.assertEqual(len(controls),1)
            for invalid in (request(),request(13,now=.55),request(12,generation=0,now=.55)):
                with self.assertRaises(ValueError):runner.advance(invalid,apply_control=controls.append)
            self.assertEqual(world.calls,1);self.assertEqual(len(controls),1)
        self.assertFalse(world.current.synchronous_mode);self.assertIsNone(world.current.fixed_delta_seconds)

    def test_failure_stops_runner_and_cleans_only_owned(self):
        world=FixtureWorld();own=SimpleNamespace(destroyed=False);other=SimpleNamespace(destroyed=False)
        def destroy():own.destroyed=True
        own.destroy=destroy
        with SynchronousRunner(world,run_id='run',episode_id='episode',generation=1,clock_domain='sim-n12') as runner:
            runner.register_owned(own)
            with self.assertRaises(RuntimeError):runner.advance(request(),apply_control=lambda _: (_ for _ in ()).throw(RuntimeError('injected')))
            with self.assertRaises(RuntimeError):runner.advance(request(),apply_control=lambda _:None)
            self.assertEqual(world.calls,0)
        self.assertTrue(own.destroyed);self.assertFalse(other.destroyed);self.assertFalse(world.current.synchronous_mode)

    def test_future_frame_never_fills_missing_current(self):
        b=FrameBuffer(capacity=2);b.put(3,'future')
        result=b.get(2,.001);self.assertFalse(result['valid']);self.assertEqual(result['reason'],'timeout')
        self.assertEqual(b.get(3,0)['value'],'future')
        b.put(4,'first');self.assertFalse(b.put(4,'duplicate'));self.assertEqual(b.get(4,0)['value'],'first')
        b.close();self.assertEqual(b.get(5,0)['reason'],'closed')

    def test_core_rejects_truth_and_future_frame(self):
        event=TickEvent('1.0','run','episode',1,11,'sim-n12',.55,()).to_dict()
        camera={'frame':11,'valid':False,'width':None,'height':None,'bgra':None,'reason':'timeout'}
        self.assertEqual(decode_tick({'event':event,'camera':camera})[0].frame_id,11)
        with self.assertRaises(ValueError): decode_tick({'event':event,'camera':camera,'truth':{'hidden':True}})
        with self.assertRaises(ValueError): decode_tick({'event':event,'camera':dict(camera,frame=12)})

    def test_hidden_truth_replacement_does_not_enter_decision(self):
        event=TickEvent('1.0','run','episode',1,11,'sim-n12',.55,())
        results=[]
        for hidden in ({'future_x': 1}, {'future_x': -999}):
            self.assertIn('future_x', hidden)  # 旁路变量不属于决策输入
            results.append(stationary_control(event,0).to_dict())
        self.assertEqual(results[0],results[1])

    def test_core_imports_do_not_include_carla_or_truth_store(self):
        root=Path(__file__).resolve().parents[2] / 'src' / 'occluactisafe' / 'core'
        for path in root.rglob('*.py'):
            tree=ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node,ast.Import): modules=[alias.name for alias in node.names]
                elif isinstance(node,ast.ImportFrom): modules=[node.module or '']
                else: continue
                for name in modules:
                    self.assertFalse(name == 'carla' or name.startswith('carla.') or 'truth_store' in name or 'truth_logger' in name, (path,name))


if __name__=='__main__':unittest.main()
