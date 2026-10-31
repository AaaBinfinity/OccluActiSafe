import copy
from types import SimpleNamespace
import unittest
from occluactisafe.simulation.runner import SynchronousRunner


class World:
    def __init__(self):self.settings=SimpleNamespace(synchronous_mode=False,fixed_delta_seconds=None)
    def get_settings(self):return copy.deepcopy(self.settings)
    def apply_settings(self,s):self.settings=copy.deepcopy(s)
    def get_snapshot(self):return SimpleNamespace(frame=0,timestamp=SimpleNamespace(elapsed_seconds=0.))


class ReloadTests(unittest.TestCase):
    def test_sync_set_before_reload_and_original_restored(self):
        world=World()
        class Client:
            def get_world(self):return world
            def reload_world(self,reset_settings):
                self.checked=(not reset_settings and world.settings.synchronous_mode and world.settings.fixed_delta_seconds==.05)
                return world
        client=Client()
        with SynchronousRunner.reload_synchronously(client,run_id='r',episode_id='e',generation=1,clock_domain='sim'):pass
        self.assertTrue(client.checked);self.assertFalse(world.settings.synchronous_mode)

    def test_reload_error_restores_settings(self):
        world=World()
        class Client:
            def get_world(self):return world
            def reload_world(self,reset_settings):raise RuntimeError('injected')
        with self.assertRaises(RuntimeError):SynchronousRunner.reload_synchronously(Client(),run_id='r',episode_id='e',generation=1,clock_domain='sim')
        self.assertFalse(world.settings.synchronous_mode)


if __name__=='__main__':unittest.main()
