"""N14静态几何缓存避免正式循环重新调用阻塞RPC。"""
from types import SimpleNamespace
import unittest
from occluactisafe.simulation.scenarios import state,restricted_track


class Vehicle:
    @property
    def bounding_box(self):raise AssertionError('正式循环不得重新查询包围盒RPC')
    def get_transform(self):
        return SimpleNamespace(rotation=SimpleNamespace(yaw=0.),transform=lambda location:SimpleNamespace(x=2.,y=3.))
    def get_velocity(self):return SimpleNamespace(x=8.,y=0.)


class GeometryTests(unittest.TestCase):
    def test_live_measured_box_is_reused_for_state_and_legal_track(self):
        box=SimpleNamespace(location=SimpleNamespace(x=0.,y=0.,z=1.),extent=SimpleNamespace(x=2.,y=1.))
        result=state(Vehicle(),box);self.assertEqual((result['length'],result['width']),(4.,2.))
        self.assertEqual((result['x'],result['y']),(2.,-3.))
        track=restricted_track(Vehicle(),'vehicle-A',3.,box)
        self.assertEqual((track.x,track.y,track.length),(2.,-3.,4.))


if __name__=='__main__':unittest.main()
