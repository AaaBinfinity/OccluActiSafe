"""扫值矩阵与控制复核工况结构；不代替 CARLA 实测。"""
import unittest
from occluactisafe.simulation.actuator import cases


class SweepCasesTests(unittest.TestCase):
    def test_full_matrix_has_three_repetitions_for_each_condition(self):
        config={'repetitions':3,'speeds_mps':[4.,8.,12.],'pedals':[.1,.3,.6,1.]}
        result=list(cases(config))
        self.assertEqual(len(result),144)
        for speed in config['speeds_mps']:
            for kind in ('throttle','brake','steer_step','steer_ramp'):
                subset=[x for x in result if x['speed']==speed and x['kind']==kind]
                self.assertEqual(len(subset),12)
                self.assertEqual({x['repeat'] for x in subset},{0,1,2})

    def test_controller_checks_keep_their_config(self):
        check={'initial_speed_mps':8.,'target_speed_mps':8.,'acceleration_mps2':1.,'angle_rad':0.}
        result=list(cases({'repetitions':2,'controller_checks':[check]}))
        self.assertEqual([x['repeat'] for x in result],[0,1])
        self.assertTrue(all(x['kind']=='controller' and x['check']==check for x in result))


if __name__=='__main__': unittest.main()
