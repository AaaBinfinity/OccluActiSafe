"""N11适用C03/C08/F01及TTC/消息/锁存的解析机制。"""
from dataclasses import replace
import unittest
from occluactisafe.core.runtime import observed_track, ego_state, run_episode
from occluactisafe.core.vehicle_agent import VehicleAgent
from occluactisafe.core.roadside_edge import make_prediction
from occluactisafe.common.schemas import ReceivedMessage
from occluactisafe.control.executor import AnalyticExecutor
from occluactisafe.planning.braking import MotionState
from occluactisafe.planning.candidates import compute_ttc_cv
from occluactisafe.risk.obb import OBB, intersects
from occluactisafe.risk.occupancy import PoseTrajectory, first_conflict_time
from occluactisafe.common.coordinates import Pose2D
from occluactisafe.perception.participants import ParticipantGeometry
from occluactisafe.planning.decision import AEBDecision


class AnalyticTests(unittest.TestCase):
    def test_ttc_net_gap_and_separating(self):
        ego=ego_state(MotionState(0,10,0),0)
        track=replace(observed_track(14,0,'roadside-1'),vx=5.)
        ttc=compute_ttc_cv(ego,(track,),now=0)
        self.assertAlmostEqual(ttc['ttc_s'],2,delta=.01)
        table={'coverage':{'complete':True},'candidates':[{'candidate_id':name,'R_max':0.,'normal_blocked':False,'block_reasons':[]} for name in ('NORMAL','SLOW','AEB')]}
        self.assertEqual(AEBDecision('ttc').choose(table,ttc,unknown=False,prediction_missing=False)['action'],'AEB')
        self.assertIsNone(compute_ttc_cv(ego,(replace(track,vx=20.),),now=0)['ttc_s'])
        self.assertAlmostEqual(compute_ttc_cv(ego,(track,),now=1)['ttc_s'],1.,delta=.01)

    def test_normal_requires_legal_clear_snapshot(self):
        import numpy as np
        from occluactisafe.perception.visibility import GridSpec, SensorView, FreeSpaceEvidence, build_visibility
        spec=GridSpec((-8,-8,8,8));ones=np.ones(spec.shape,bool)
        v=SensorView('clear',Pose2D(0,0,0),2*np.pi,100,1.,0,0,1.,'sim-n11','map-right-handed')
        proof=FreeSpaceEvidence('clear',ones,0,0,1,'sim-n11','map-right-handed')
        k=build_visibility(spec,now=0,clock_domain='sim-n11',coordinate_frame_id='map-right-handed',views=(v,),free_evidence=(proof,),drivable=ones,closed_domain=True)
        e=AnalyticExecutor(speed=0);v=VehicleAgent();bundle,action=v.step(0,ego_state(e.at(0),0),e,knowledge=k,unknown=False)
        self.assertEqual(action,'NORMAL');self.assertEqual(bundle.mode,'Normal')

    def test_c03_exchange_edge_rotation_and_swept_contact(self):
        self.assertTrue(intersects(OBB((0,0),0,4,2),OBB((4,0),0,4,2)))
        self.assertFalse(intersects(OBB((0,0),0,4,2),OBB((4.1,0),0,4,2)))
        self.assertTrue(intersects(OBB((0,0),0,4,2),OBB((0,0),1.57079632679,4,2)))
        a=PoseTrajectory((0.,.5),(Pose2D(0,0,0),)*2,'sim','map')
        b=PoseTrajectory((0.,.5),(Pose2D(-5,0,0),Pose2D(5,0,0)),'sim','map'); g=ParticipantGeometry('vehicle',2,2)
        f=first_conflict_time(a,g,b,g,0,.5);r=first_conflict_time(b,g,a,g,0,.5)
        self.assertEqual(f.event['interval_start'],r.event['interval_start']);self.assertTrue(f.potential_contact)

    def test_unavailable_and_expired_cannot_change_decision(self):
        base=VehicleAgent();e=AnalyticExecutor();base.step(.6,ego_state(e.at(.6),.6),e)
        expected=base.last_audit['decision']
        for x,arrival in ((-1e6,1.),(1e6,1.),(35,.2)):
            message=make_prediction((observed_track(x,0,'roadside-1'),),0,0)
            r=ReceivedMessage(message,arrival,arrival,'sim-n11');v=VehicleAgent();e=AnalyticExecutor()
            v.step(.6,ego_state(e.at(.6),.6),e,received=(r,))
            self.assertEqual(v.last_audit['decision'],expected)
            self.assertEqual(len(v.last_audit['fusion']['agents']),0)

    def test_duplicate_keeps_original_age_and_one_unified_object(self):
        track=observed_track(35,0,'roadside-1');r=ReceivedMessage(make_prediction((track,),0,0),0,0,'sim-n11')
        v=VehicleAgent();e=AnalyticExecutor();v.step(0,ego_state(e.at(0),0),e,received=(r,))
        v.step(.1,ego_state(e.at(.1),.1),e,received=(r,))
        self.assertIn('duplicate_or_out_of_order',v.last_audit['gates'])
        self.assertEqual(v.last_audit['fusion']['agents'][0]['last_seen_time'],0)
        self.assertEqual(len(v.last_audit['risk']['coverage']['expected_target_ids']),1)

    def test_c08_no_safe_candidate_is_mitigation_and_latched(self):
        table={'coverage':{'complete':True},'candidates':[{'candidate_id':name,'R_max':1.,'normal_blocked':False,'block_reasons':[]} for name in ('NORMAL','SLOW','AEB')]}
        decision=AEBDecision('risk');r=decision.choose(table,{'ttc_s':None},unknown=False,prediction_missing=False)
        self.assertTrue(r['no_safe_candidate_found']); self.assertEqual(r['action'],'AEB')
        for row in table['candidates']: row['R_max']=0.
        self.assertEqual(decision.choose(table,{'ttc_s':None},unknown=False,prediction_missing=False)['action'],'AEB')
        table['candidates'][2]['R_max']=1.
        self.assertEqual(decision.choose(table,{'ttc_s':None},unknown=False,prediction_missing=False)['action'],'SLOW')
        for row in table['candidates']:row['block_reasons']=['declared_physically_infeasible']
        self.assertEqual(decision.choose(table,{'ttc_s':None},unknown=False,prediction_missing=False)['action'],'DEGRADED')

    def test_normal_slow_unknown_and_missing_coverage_states(self):
        table={'coverage':{'complete':True},'candidates':[{'candidate_id':name,'R_max':0.,'normal_blocked':False,'block_reasons':[]} for name in ('NORMAL','SLOW','AEB')]}
        d=AEBDecision()
        self.assertEqual(d.choose(table,{'ttc_s':None},unknown=False,prediction_missing=False)['action'],'NORMAL')
        table['candidates'][0]['R_max']=.2
        self.assertEqual(d.choose(table,{'ttc_s':None},unknown=False,prediction_missing=False)['action'],'SLOW')
        self.assertEqual(d.choose(table,{'ttc_s':None},unknown=True,prediction_missing=False)['action'],'DEGRADED')
        table['coverage']['complete']=False
        self.assertEqual(d.choose(table,{'ttc_s':None},unknown=False,prediction_missing=False)['action'],'DEGRADED')

    def test_f01_paired_sufficient_and_unavoidable_and_replay(self):
        for x in (35.,8.):
            reference=run_episode(x,'none','local',duration=4.)
            cooperative=run_episode(x,'risk','cooperative',duration=4.)
            self.assertEqual(reference['initial_sha256'],cooperative['initial_sha256'])
            self.assertEqual(reference['exogenous_schedule_sha256'],cooperative['exogenous_schedule_sha256'])
            self.assertTrue(reference['analytic_contact'])
            if x==35: self.assertFalse(cooperative['analytic_contact']);self.assertEqual(cooperative['final']['v'],0)
            else:
                self.assertTrue(cooperative['analytic_contact'])
                self.assertLess(cooperative['events'][0]['ego_speed_mps'],reference['events'][0]['ego_speed_mps'])
            for row in cooperative['rows']:
                self.assertEqual(row['control']['throttle']*row['control']['brake'],0)
                if row['time']<.2 and not row['sensor_local_visible']:
                    self.assertNotEqual(row['audit']['decision']['action'],'AEB')
            first_aeb=next(r['time'] for r in cooperative['rows'] if r['audit']['decision']['action']=='AEB')
            self.assertTrue(all(r['audit']['decision']['action']=='AEB' for r in cooperative['rows'] if r['time']>=first_aeb))
            again=run_episode(x,'risk','cooperative',duration=4.)
            self.assertEqual(again['episode_replay_sha256'],cooperative['episode_replay_sha256'])


if __name__=='__main__': unittest.main()
