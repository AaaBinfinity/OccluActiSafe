"""N14因果白名单夹具；真实CARLA配对由场景实测另验。"""
import unittest
from dataclasses import replace
from occluactisafe.simulation.trajectory_input import TrajectoryAEB
from occluactisafe.common.schemas import SourceTrack,ObservationPacket,ReceivedMessage
from occluactisafe.core.roadside_edge import make_prediction
from occluactisafe.evaluation.episode_metrics import overlap,digest


def track(now=3.,source='vehicle-A'):
    return SourceTrack('target',source,'session-1',source,'vehicle','target','carla-n14','map-right-handed',
        'geometric_center',now,now,now,now,0.,-8.,0.,8.,1.5708,4.,2.,'annotation',0.,((0.,0.),(0.,0.)),True,True,True,True,True,1.)


def obs(now=3.):
    return {'time_s':now,'ego':dict(x=-8.,y=0.,vx=8.,vy=0.,yaw=0.,length=4.,width=2.),'local':None,'received':[]}


class TrajectoryTests(unittest.TestCase):
    def test_distant_conflict_warns_then_budget_triggers_same_rule(self):
        p=TrajectoryAEB('local_aeb');o=obs();o['ego']['x']=-24.
        distant=replace(track(),y=-24.)
        o['local']=ObservationPacket('map-A','1','map-right-handed',60,'carla-n14',3.,3.,(distant,)).to_dict()
        d=p.step(o)
        self.assertTrue(d['warning']);self.assertFalse(d['brake'])
        self.assertGreater(d['stopping_margin_m'],0)
        self.assertEqual(len(d['known_targets'][0]['prediction_xy']),31)
        o=obs(3.1);o['ego']['x']=-12.
        o['local']=ObservationPacket('map-A','1','map-right-handed',62,'carla-n14',3.1,3.1,(replace(track(3.1),y=-12.),)).to_dict()
        d=p.step(o);self.assertTrue(d['brake']);self.assertLessEqual(d['stopping_margin_m'],0)
        self.assertEqual(d['first_warning_s'],3.);self.assertEqual(d['first_emergency_s'],3.1)

    def test_budget_uses_measured_braking_delay_and_margin(self):
        b=dict(deceleration_mps2=8.,response_delay_s=.15,control_period_s=.05,margin_m=5.,max_track_age_s=.5)
        p=TrajectoryAEB('local_aeb',braking=b)
        self.assertAlmostEqual(p.step(obs())['stopping_budget_m'],10.6)
        o=obs();o['ego']['vx']=4.
        self.assertAlmostEqual(p.step(o)['stopping_budget_m'],6.8)
        with self.assertRaises(ValueError):TrajectoryAEB('local_aeb',braking=dict(b,deceleration_mps2=0.))

    def test_causal_roadside_cache_expires_and_local_does_not_use_it(self):
        p=TrajectoryAEB('cooperative_aeb');o=obs();o['ego']['x']=-24.
        msg=make_prediction((replace(track(2.8,'roadside-1'),y=-24.),),2.8,0,domain='carla-n14')
        o['received']=[ReceivedMessage(msg,3.,3.,'carla-n14').to_dict()]
        self.assertTrue(p.step(o)['warning'])
        self.assertEqual(TrajectoryAEB('local_aeb').step(o)['known_targets'],[])
        o['time_s']=3.1;o['received']=[]
        d=p.step(o);self.assertTrue(d['warning']);self.assertEqual(d['used_sources'],['roadside-1'])
        self.assertAlmostEqual(d['known_targets'][0]['age_s'],.3)
        o['time_s']=3.4;self.assertEqual(p.step(o)['known_targets'],[])

    def test_both_aeb_sources_have_identical_distance_threshold(self):
        for distance in (24.,12.):
            local=obs();local['ego']['x']=-distance
            local['local']=ObservationPacket('map-A','1','map-right-handed',60,'carla-n14',3.,3.,(replace(track(),y=-distance),)).to_dict()
            road=obs();road['ego']['x']=-distance
            msg=make_prediction((replace(track(3.,'roadside-1'),y=-distance),),3.,0,domain='carla-n14')
            road['received']=[ReceivedMessage(msg,3.,3.,'carla-n14').to_dict()]
            a=TrajectoryAEB('local_aeb').step(local);b=TrajectoryAEB('cooperative_aeb').step(road)
            for key in ('warning','brake','stopping_budget_m','distance_to_conflict_envelope_m'):
                self.assertEqual(a[key],b[key])

    def test_hidden_and_reference_cannot_trigger(self):
        o=obs();self.assertFalse(TrajectoryAEB('local_aeb').step(o)['brake'])
        o['local']=ObservationPacket('map-A','1','map-right-handed',60,'carla-n14',3.,3.,(track(),)).to_dict()
        self.assertFalse(TrajectoryAEB('reference').step(o)['brake'])
        self.assertTrue(TrajectoryAEB('local_aeb').step(o)['brake'])

    def test_prefix_and_latch(self):
        p=TrajectoryAEB('local_aeb');o=obs(2.)
        o['local']=ObservationPacket('map-A','1','map-right-handed',40,'carla-n14',2.,2.,(track(2.),)).to_dict()
        self.assertFalse(p.step(o)['brake']);o['time_s']=3.
        o['local']=ObservationPacket('map-A','1','map-right-handed',60,'carla-n14',3.,3.,(track(3.),)).to_dict()
        self.assertTrue(p.step(o)['brake'])
        self.assertTrue(p.step(obs(3.1))['brake'])

    def test_usable_time_and_source_strategy(self):
        o=obs();msg=make_prediction((track(2.8,'roadside-1'),),2.8,0,domain='carla-n14')
        o['received']=[ReceivedMessage(msg,3.,3.05,'carla-n14').to_dict()]
        with self.assertRaises(ValueError):TrajectoryAEB('cooperative_aeb').step(o)
        o['time_s']=3.05
        self.assertTrue(TrajectoryAEB('cooperative_aeb').step(o)['brake'])
        self.assertFalse(TrajectoryAEB('local_aeb').step(o)['brake'])

    def test_truth_and_future_fields_rejected(self):
        for field in ('truth','future','target_hidden'):
            o=obs();o[field]={'x':0.}
            with self.assertRaises(ValueError):TrajectoryAEB('cooperative_aeb').step(o)
        o=obs();o['ego']['vx']=float('nan')
        with self.assertRaises(ValueError):TrajectoryAEB('cooperative_aeb').step(o)

    def test_geometry_and_prefix_identity(self):
        a=obs()['ego'];b=dict(a,x=20.)
        self.assertFalse(overlap(a,b));self.assertTrue(overlap(a,a))
        self.assertEqual(digest(a),digest(dict(reversed(list(a.items())))))


if __name__=='__main__':unittest.main()
