"""示意视图的因果来源与记录完整性，真实桌面显示另在CARLA实测。"""
import json
from pathlib import Path
import tempfile
import unittest

from occluactisafe.simulation.aeb_overview import OverviewRecorder, render_overview
from occluactisafe.simulation.trajectory_input import TrajectoryAEB


class OverviewTests(unittest.TestCase):
    def inputs(self):
        observation={'time_s':0.,'ego':dict(x=-20.,y=0.,vx=8.,vy=0.,yaw=0.,length=4.,width=2.),
                     'local':None,'received':[]}
        config={'conflict_xy':[0.,0.],'ego_route':[{'x':-40.,'y':0.},{'x':30.,'y':0.}],
                'target_route':[{'x':0.,'y':-40.},{'x':0.,'y':30.}], 'overview':{'live_window':False}}
        return observation,config

    def test_hidden_evaluation_replacement_cannot_change_view(self):
        o,c=self.inputs();d=TrajectoryAEB('cooperative_aeb').step(o)
        self.assertEqual(d['known_targets'],[])
        a=render_overview(o,d,c,'cooperative_aeb')
        b=render_overview(o,d,dict(c,hidden_truth={'x':999.,'future':[12.,30.]}),'cooperative_aeb')
        self.assertEqual(a.size,(1280,720));self.assertEqual(a.tobytes(),b.tobytes())

    def test_saved_frames_match_causal_audit_without_carla_or_truth_file(self):
        o,c=self.inputs();p=TrajectoryAEB('cooperative_aeb')
        with tempfile.TemporaryDirectory() as tmp:
            viewer=OverviewRecorder(tmp,c,'cooperative_aeb')
            for t in (0.,.05):o['time_s']=t;viewer.update(o,p.step(o))
            viewer.close();root=Path(tmp)
            index=json.loads((root/'示意帧索引.json').read_text(encoding='utf-8'))
            audit=[json.loads(line) for line in (root/'示意在线状态.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual([x['time_s'] for x in index],[0.,.05])
            self.assertEqual([x['index'] for x in audit],[0,1])
            self.assertTrue(all((root/x['image']).is_file() for x in index))
            self.assertTrue(all(x['decision']['known_targets']==[] for x in audit))


if __name__=='__main__':unittest.main()
