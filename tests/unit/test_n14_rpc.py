"""N14批次确认不拥有tick，非法actor/命令不发送。"""
from types import SimpleNamespace
import unittest
from occluactisafe.simulation.scenario_rpc import execute_batch


class RPCTests(unittest.TestCase):
    def test_control_batch_has_acknowledgements_and_never_ticks(self):
        calls=[]
        def apply(commands,do_tick):
            self.assertFalse(do_tick);calls.append(commands)
            return [SimpleNamespace(error=None) for _ in commands]
        client=SimpleNamespace(apply_batch_sync=apply)
        carla=SimpleNamespace(VehicleControl=lambda **kw:kw,command=SimpleNamespace(ApplyVehicleControl=lambda actor,control:(actor,control)))
        request=[dict(kind='control',actor_id=1,value=dict(throttle=0.,brake=1.,steer=0.))]
        self.assertEqual(execute_batch(client,carla,request,{1}),1);self.assertEqual(len(calls),1)

    def test_invalid_actor_and_unknown_commands_are_not_dispatched(self):
        def unexpected(*args):raise AssertionError('非法请求不得发送')
        client=SimpleNamespace(apply_batch_sync=unexpected)
        with self.assertRaises(ValueError):execute_batch(client,None,[dict(kind='control',actor_id=9,value={})],{1})
        with self.assertRaises(ValueError):execute_batch(client,None,[dict(kind='unknown',actor_id=1,value={})],{1})


if __name__=='__main__':unittest.main()
