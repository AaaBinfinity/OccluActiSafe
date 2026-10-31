import unittest
from occluactisafe.simulation.occluder_driver import OccluderDriver


class OccluderDriverTests(unittest.TestCase):
    def setUp(self):
        self.driver=OccluderDriver({'x':-40.,'y':-3.5,'yaw':0.},[0.,0.],speed_mps=8.,brake_front_m=7.)

    def state(self,x=-30.,speed=8.):
        return dict(x=x,y=3.5,vx=speed,vy=0.,yaw=0.,length=8.5,width=2.9)

    def test_own_state_cruise_mutual_and_same_direction(self):
        action=self.driver.step(self.state())
        self.assertGreater(action['throttle'],0);self.assertEqual(action['brake'],0);self.assertAlmostEqual(action['steer'],0)

    def test_stop_line_latches_before_body_enters_conflict(self):
        action=self.driver.step(self.state(x=-10.))
        self.assertEqual(action,dict(throttle=0.,brake=1.,steer=0.))
        self.assertEqual(self.driver.step(self.state(x=-20.))['brake'],1.)

    def test_initialization_does_not_trigger_route_braking(self):
        self.assertEqual(self.driver.step(self.state(x=-10.),initializing=True)['brake'],0.)
        self.assertFalse(self.driver.stopping)


if __name__=='__main__':unittest.main()
