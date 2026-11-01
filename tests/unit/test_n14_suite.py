"""N14多对象、真实walker命令和原生1080p机制；不替代实际CARLA场景。"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
import hashlib
from unittest.mock import patch

import yaml
from occluactisafe.simulation.n14_suite import expand_cases
from occluactisafe.simulation.n14_participants import participant_specs,walker_action
from occluactisafe.simulation.scenario_rpc import execute_batch
from occluactisafe.simulation.trajectory_input import TrajectoryAEB
from occluactisafe.simulation.aeb_overview import render_overview
from occluactisafe.simulation.n14_images import save_rgb_frame
from occluactisafe.common.schemas import SourceTrack,ObservationPacket

ROOT=Path(__file__).resolve().parents[2]


class SuiteTests(unittest.TestCase):
    def test_native_bgra_png_preserves_channels_and_unicode_path(self):
        from PIL import Image
        native=SimpleNamespace(width=2,height=1,raw_data=bytes((0,0,255,255,0,255,0,255)))
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'中文相机目录';folder.mkdir();path=folder/'实际原始帧.png'
            proof=save_rgb_frame(native,path)
            with Image.open(path) as decoded:
                self.assertEqual(list(decoded.getdata()),[(255,0,0),(0,255,0)])
                self.assertEqual(hashlib.sha256(decoded.tobytes()).hexdigest(),proof['rgb_pixels_sha256'])
            self.assertEqual(proof['native_size'],[2,1])

    def configs(self):
        directory=ROOT/'configs/scenarios'
        return yaml.safe_load((directory/'occluded_junction.yaml').read_text('utf-8')),yaml.safe_load((directory/'n14_suite.yaml').read_text('utf-8'))

    def test_frozen_cases_keep_original_pose_and_separate_category_motion(self):
        base,suite=self.configs();cases=expand_cases(base,suite)
        self.assertEqual(len(cases),4)
        self.assertEqual(cases[0][1]['ego_spawn'],base['ego_spawn'])
        self.assertEqual(cases[0][1]['occluder_motion'],base['occluder_motion'])
        self.assertEqual(len(participant_specs(cases[1][1])),2)
        ped=participant_specs(cases[2][1])[0]
        self.assertEqual(ped['category'],'pedestrian');self.assertTrue(ped['blueprint'].startswith('walker.'))
        self.assertNotIn('occluder_motion',cases[2][1]);self.assertEqual(cases[2][1]['visibility_size'],[1280,720])
        self.assertNotEqual(cases[2][1]['target_start_s'],cases[3][1]['target_start_s'])
        bad=deepcopy(suite);bad['rgb_size']=[1280,720]
        with self.assertRaises(ValueError):expand_cases(base,bad)

    def test_participant_ids_do_not_alias_and_walker_wait_is_external_only(self):
        base,suite=self.configs();config=expand_cases(base,suite)[3][1];spec=participant_specs(config)[0]
        self.assertEqual(walker_action(spec,1.14)['speed'],0.)
        self.assertEqual(walker_action(spec,1.15)['speed'],2.)
        self.assertEqual(walker_action(spec,3.,held=True)['speed'],0.)
        config['additional_targets']=[dict(spec)]
        with self.assertRaises(ValueError):participant_specs(config)

    def test_pedestrian_threat_survives_vehicle_distractor(self):
        ego=dict(x=-8.,y=0.,vx=8.,vy=0.,yaw=0.,length=4.,width=2.)
        ped=SourceTrack('ped-1','vehicle-A','session-1','vehicle-A','pedestrian','target','carla-n14','map-right-handed',
            'geometric_center',3.,3.,3.,3.,0.,-2.5,0.,2.5,1.5708,.6,.5,'annotation',0.,((0.,0.),(0.,0.)),True,True,True,True,True,1.)
        far=replace(ped,source_track_id='car-2',category='vehicle',x=100.,y=100.,length=4.,width=2.)
        def inputs(tracks):return {'time_s':3.,'ego':ego,'local':ObservationPacket('map-A','1','map-right-handed',60,'carla-n14',3.,3.,tracks).to_dict(),'received':[]}
        self.assertFalse(TrajectoryAEB('local_aeb').step(inputs((far,)))['warning'])
        decision=TrajectoryAEB('local_aeb').step(inputs((far,ped)))
        self.assertTrue(decision['brake']);self.assertEqual({t['category'] for t in decision['known_targets']},{'vehicle','pedestrian'})
        self.assertEqual({t['track_id'] for t in decision['known_targets']},{'car-2','ped-1'})

    def test_walker_batch_uses_native_control_and_parent_tick_only(self):
        calls=[]
        client=SimpleNamespace(apply_batch_sync=lambda commands,tick:(calls.append((commands,tick)) or [SimpleNamespace(error=None)]))
        carla=SimpleNamespace(Vector3D=lambda **v:v,WalkerControl=lambda **v:v,
            command=SimpleNamespace(ApplyWalkerControl=lambda identity,control:(identity,control)))
        request=dict(kind='walker_control',actor_id=4,value=dict(direction=dict(x=0.,y=-1.,z=0.),speed=1.5,jump=False))
        self.assertEqual(execute_batch(client,carla,[request],{4}),1)
        self.assertFalse(calls[0][1]);self.assertEqual(calls[0][0][0][1]['speed'],1.5)

    def test_hd_schematic_is_drawn_natively_without_bitmap_upscaling(self):
        ego=dict(x=-20.,y=0.,vx=8.,vy=0.,yaw=0.,length=4.,width=2.)
        observation={'time_s':0.,'ego':ego,'local':None,'received':[]}
        config={'conflict_xy':[0.,0.],'ego_route':[dict(x=-40.,y=0.),dict(x=30.,y=0.)],
                'target_route':[dict(x=0.,y=-40.),dict(x=0.,y=30.)],'overview':{'output_scale':1.5}}
        decision=TrajectoryAEB('cooperative_aeb').step(observation)
        with patch('PIL.Image.Image.resize',side_effect=AssertionError('不允许放大旧位图')):
            image=render_overview(observation,decision,config,'cooperative_aeb')
        self.assertEqual(image.size,(1920,1080))


if __name__=='__main__':unittest.main()
