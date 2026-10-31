"""有限恢复与归属拒绝机制，不能代替真实 RPC 中断。"""
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from test_n12_mechanisms import FixtureWorld, request
from occluactisafe.simulation.runner import SynchronousRunner


class RecoveryTests(unittest.TestCase):
    def setup_case(self, directory, *, budget=1):
        world=FixtureWorld(); world.id=10
        actor=SimpleNamespace(id=42,type_id='vehicle.test',attributes={'role_name':'oas:run'})
        world.get_actor=lambda _:actor
        destroyed=[]
        client=SimpleNamespace(set_timeout=lambda _:None,get_world=lambda:world,
            apply_batch_sync=lambda commands,tick: destroyed.extend(commands) or [SimpleNamespace(error='')])
        runner=SynchronousRunner(world,run_id='run',episode_id='episode',generation=1,clock_domain='sim',
            reconnect=lambda _:client,destroy_command=lambda actor_id:actor_id,cleanup_path=Path(directory)/'清理.json',
            recovery_budget_s=budget,cleanup_rpc_s=min(.01,budget))
        runner.register_owned(actor)
        return runner,world,actor,client,destroyed

    def test_transient_disconnect_retries_then_restores_without_tick(self):
        with tempfile.TemporaryDirectory() as directory:
            runner,world,actor,client,destroyed=self.setup_case(directory)
            calls=[]
            def reconnect(_):
                calls.append(1)
                if len(calls)==1: raise RuntimeError('temporary disconnect')
                return client
            runner._reconnect=reconnect; runner.close()
            self.assertEqual(destroyed,[42]); self.assertFalse(world.current.synchronous_mode)
            self.assertEqual(world.calls,0)
            report=json.loads((Path(directory)/'清理.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'],'completed'); self.assertEqual(len(report['attempts']),2)
            with self.assertRaises(RuntimeError): runner.advance(request(),apply_control=lambda _:None)

    def test_changed_world_never_destroys_or_restores(self):
        with tempfile.TemporaryDirectory() as directory:
            runner,world,actor,client,destroyed=self.setup_case(directory); world.id=11
            with self.assertRaisesRegex(RuntimeError,'world_changed'):runner.close()
            self.assertEqual(destroyed,[]); self.assertTrue(world.current.synchronous_mode)

    def test_reused_actor_id_with_different_owner_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            runner,world,actor,client,destroyed=self.setup_case(directory)
            world.get_actor=lambda _:SimpleNamespace(id=42,type_id=actor.type_id,attributes={'role_name':'user.actor'})
            with self.assertRaisesRegex(RuntimeError,'ownership_mismatch'):runner.close()
            self.assertEqual(destroyed,[]); self.assertTrue(world.current.synchronous_mode)

    def test_permanent_disconnect_fails_within_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            runner,world,actor,client,destroyed=self.setup_case(directory,budget=.03)
            def unavailable(_):raise RuntimeError('offline')
            runner._reconnect=unavailable; started=time.monotonic()
            with self.assertRaisesRegex(RuntimeError,'recovery_deadline'):runner.close()
            self.assertLess(time.monotonic()-started,.2); self.assertEqual(destroyed,[])
            report=json.loads((Path(directory)/'清理.json').read_text(encoding='utf-8'))
            self.assertFalse(report['settings_restored']); self.assertEqual(report['status'],'failed')

    def test_evidence_write_failure_does_not_prevent_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            runner,world,actor,client,destroyed=self.setup_case(directory)
            with patch.object(Path,'replace',side_effect=OSError('disk offline')):
                with self.assertRaisesRegex(RuntimeError,'evidence_write_failed'):runner.close()
            self.assertEqual(destroyed,[42]); self.assertFalse(world.current.synchronous_mode)

    def test_follow_spectator_is_restored_only_during_same_world_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            runner,world,actor,client,destroyed=self.setup_case(directory)
            original=object();spectator=SimpleNamespace(transform=original)
            spectator.get_transform=lambda:spectator.transform
            spectator.set_transform=lambda transform:setattr(spectator,'transform',transform)
            world.get_spectator=lambda:spectator
            actor.get_transform=lambda:SimpleNamespace(location=SimpleNamespace(x=10,y=20,z=0),
                rotation=SimpleNamespace(yaw=0),get_forward_vector=lambda:SimpleNamespace(x=1,y=0))
            carla=SimpleNamespace(Location=lambda **kw:SimpleNamespace(**kw),Rotation=lambda **kw:SimpleNamespace(**kw),
                                  Transform=lambda location,rotation:SimpleNamespace(location=location,rotation=rotation))
            runner.follow_ego(carla,actor);self.assertEqual(spectator.transform.location.x,3)
            runner.close();self.assertIs(spectator.transform,original)
            report=json.loads((Path(directory)/'清理.json').read_text(encoding='utf-8'))
            self.assertTrue(report['spectator_restored']);self.assertEqual(world.calls,0)


if __name__=='__main__':unittest.main()
