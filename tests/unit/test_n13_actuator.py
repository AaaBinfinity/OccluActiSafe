"""执行器机制夹具；不能替代CARLA实测标定。"""
import unittest
from occluactisafe.control.actuator import CalibratedActuator


class ActuatorTests(unittest.TestCase):
    def setUp(self):
        self.a=CalibratedActuator({'acceleration_to_pedal':[[-3,-1],[0,0],[2,1]],
            'angle_to_steer':[[-.4,-1],[0,0],[.4,1]],'speed_min_mps':0,'speed_max_mps':15,
            'vehicle_blueprint':'fixture.vehicle','surface_id':'fixture.dry'},
            kp=.5,ki=.2,steering_rate_per_s=2)

    def test_signed_commands_and_steering_rate(self):
        positive=self.a.step(speed_mps=8,target_speed_mps=8,acceleration_mps2=1,angle_rad=.4,dt_s=.1,vehicle_blueprint='fixture.vehicle',surface_id='fixture.dry')
        self.assertAlmostEqual(positive['throttle'],.5);self.assertEqual(positive['brake'],0);self.assertAlmostEqual(positive['steer'],.2)
        negative=self.a.step(speed_mps=8,target_speed_mps=8,acceleration_mps2=-1.5,angle_rad=-.4,dt_s=.1,vehicle_blueprint='fixture.vehicle',surface_id='fixture.dry')
        self.assertEqual(negative['throttle'],0);self.assertAlmostEqual(negative['brake'],.5);self.assertAlmostEqual(negative['steer'],0)

    def test_saturation_does_not_accumulate_integral(self):
        for _ in range(20):
            out=self.a.step(speed_mps=0,target_speed_mps=15,acceleration_mps2=2,angle_rad=0,dt_s=.1,vehicle_blueprint='fixture.vehicle',surface_id='fixture.dry')
            self.assertTrue(out['saturated']);self.assertEqual(out['throttle'],1);self.assertEqual(out['brake'],0)
        self.assertEqual(self.a.integral,0)

    def test_outside_measured_range_rejected_without_state_change(self):
        for args in ({'speed_mps':16},{'acceleration_mps2':3},{'angle_rad':.5},{'dt_s':0},{'surface_id':'fixture.wet'},{'vehicle_blueprint':'fixture.other'}):
            values={'speed_mps':8,'target_speed_mps':8,'acceleration_mps2':0,'angle_rad':0,'dt_s':.1,'vehicle_blueprint':'fixture.vehicle','surface_id':'fixture.dry'};values.update(args)
            with self.assertRaises(ValueError):self.a.step(**values)
            self.assertEqual(self.a.integral,0);self.assertEqual(self.a.steer,0)

    def test_speed_slice_interpolation_and_no_extrapolation(self):
        a=CalibratedActuator({'acceleration_by_speed':[{'speed_mps':4,'points':[[-2,-1],[2,1]]},
            {'speed_mps':12,'points':[[-1,-1],[1,1]]}], 'angle_to_steer':[[-.4,-1],[.4,1]],
            'speed_min_mps':4,'speed_max_mps':12,'vehicle_blueprint':'v','surface_id':'dry'},
            kp=0,ki=0,steering_rate_per_s=2)
        out=a.step(speed_mps=8,target_speed_mps=8,acceleration_mps2=.5,angle_rad=0,dt_s=.1,vehicle_blueprint='v',surface_id='dry')
        self.assertAlmostEqual(out['throttle'],.375)
        with self.assertRaises(ValueError):
            a.step(speed_mps=8,target_speed_mps=8,acceleration_mps2=1.5,angle_rad=0,dt_s=.1,vehicle_blueprint='v',surface_id='dry')

    def test_calibrated_jerk_limits_request_and_exposes_delay(self):
        a=CalibratedActuator({'acceleration_to_pedal':[[-3,-1],[0,0],[2,1]],'angle_to_steer':[[-.4,-1],[0,0],[.4,1]],
            'speed_min_mps':0,'speed_max_mps':15,'vehicle_blueprint':'v','surface_id':'dry',
            'capabilities':{'command_jerk_limit_mps3':1,'observed_response_delay_s':.15}},
            kp=0,ki=0,steering_rate_per_s=2)
        result=a.step(speed_mps=8,target_speed_mps=8,acceleration_mps2=1,angle_rad=0,dt_s=.1,vehicle_blueprint='v',surface_id='dry')
        self.assertAlmostEqual(result['limited_acceleration_mps2'],.1)
        self.assertEqual(result['estimated_response_delay_s'],.15)
        with self.assertRaises(ValueError):
            a.step(speed_mps=8,target_speed_mps=8,acceleration_mps2=3,angle_rad=0,dt_s=.1,vehicle_blueprint='v',surface_id='dry')
        self.assertAlmostEqual(a.last_acceleration,.1)


if __name__=='__main__':unittest.main()
