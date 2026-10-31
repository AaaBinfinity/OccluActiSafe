"""执行器机制夹具；不能替代CARLA实测标定。"""
import unittest
from occluactisafe.control.actuator import CalibratedActuator


class ActuatorTests(unittest.TestCase):
    def setUp(self):
        self.a=CalibratedActuator({'acceleration_to_pedal':[[-3,-1],[0,0],[2,1]],
            'angle_to_steer':[[-.4,-1],[0,0],[.4,1]],'speed_min_mps':0,'speed_max_mps':15},
            kp=.5,ki=.2,steering_rate_per_s=2)

    def test_signed_commands_and_steering_rate(self):
        positive=self.a.step(speed_mps=8,target_speed_mps=8,acceleration_mps2=1,angle_rad=.4,dt_s=.1)
        self.assertAlmostEqual(positive['throttle'],.5);self.assertEqual(positive['brake'],0);self.assertAlmostEqual(positive['steer'],.2)
        negative=self.a.step(speed_mps=8,target_speed_mps=8,acceleration_mps2=-1.5,angle_rad=-.4,dt_s=.1)
        self.assertEqual(negative['throttle'],0);self.assertAlmostEqual(negative['brake'],.5);self.assertAlmostEqual(negative['steer'],0)

    def test_saturation_does_not_accumulate_integral(self):
        for _ in range(20):
            out=self.a.step(speed_mps=0,target_speed_mps=15,acceleration_mps2=2,angle_rad=0,dt_s=.1)
            self.assertTrue(out['saturated']);self.assertEqual(out['throttle'],1);self.assertEqual(out['brake'],0)
        self.assertEqual(self.a.integral,0)

    def test_outside_measured_range_rejected_without_state_change(self):
        for args in ({'speed_mps':16},{'acceleration_mps2':3},{'angle_rad':.5},{'dt_s':0}):
            values={'speed_mps':8,'target_speed_mps':8,'acceleration_mps2':0,'angle_rad':0,'dt_s':.1};values.update(args)
            with self.assertRaises(ValueError):self.a.step(**values)
            self.assertEqual(self.a.integral,0);self.assertEqual(self.a.steer,0)


if __name__=='__main__':unittest.main()
