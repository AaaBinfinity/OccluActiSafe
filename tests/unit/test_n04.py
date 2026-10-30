"""N04机制与入口测试；构造夹具不冒充真实轨迹实验。"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from occluactisafe.common.coordinates import Pose2D
from occluactisafe.data.split import build_group_split, audit_group_split
from occluactisafe.data.windows import build_causal_window, input_hash, FEATURES
from occluactisafe.data.labels import build_future_labels
from occluactisafe.prediction.dataset import MaskedScaler, save_sample, load_sample, PredictionDataset

ROOT=Path(__file__).resolve().parents[2]


def observation(t, identity='a', **changes):
    row={'track_id':identity,'clock_domain':'sim','observation_time':t,'source_available_time':t,
         'arrival_time':t,'usable_time':t,'x':10*t,'y':0.,'source_row':round(t*10)+10,
         'category':'VEHICLE','tag':'TARGET_AGENT'}
    row.update(changes);return row


class GroupTests(unittest.TestCase):
    def records(self):
        rows=[{'source_file':f'{i}.csv','location_id':f'road{i}','original_split':'train','base_episode_id':f'ep{i}'} for i in range(8)]
        rows.extend([dict(rows[0],source_file='other_view.csv'),dict(rows[0],source_file='braking_branch.csv'),
                     dict(rows[0],source_file='steering_branch.csv'),dict(rows[0],source_file='keep_branch.csv'),
                     dict(rows[0],source_file='overlap_window.csv'),dict(rows[0],source_file='official_val.csv',original_split='val')])
        return rows

    def test_d03_views_branches_overlap_and_recording_cannot_cross_roles(self):
        rows=self.records();rows[1]['recording_id']='raw-1';rows[2]['recording_id']='raw-1'
        split=build_group_split(rows);self.assertEqual(audit_group_split(split)['cross_split_shared_keys'],0)
        by_file={r['source_file']:r for r in split['files']}
        for name in ('other_view.csv','braking_branch.csv','steering_branch.csv','keep_branch.csv','overlap_window.csv','official_val.csv'):
            self.assertEqual(by_file[name]['group_id'],by_file['0.csv']['group_id'])
        self.assertEqual(by_file['1.csv']['project_role'],by_file['2.csv']['project_role'])
        self.assertFalse(by_file['official_val.csv']['project_input_allowed'])
        sets={role:{r['group_id'] for r in split['files'] if r['project_role']==role} for role in ('train','model_val','risk_calibration','test')}
        for a in sets:
            for b in sets:
                if a!=b:self.assertFalse(sets[a]&sets[b])

    def test_deterministic_order_seed_and_insufficient_groups_fail(self):
        rows=self.records();self.assertEqual(build_group_split(rows),build_group_split(list(reversed(rows))))
        with self.assertRaises(ValueError):build_group_split(rows[:3])
        with self.assertRaises(ValueError):build_group_split([{'source_file':'none','original_split':'train'}])
        with self.assertRaises(ValueError):build_group_split(rows,ratios=[1,0,0,0])


class WindowTests(unittest.TestCase):
    def build(self,rows,now=1.9,**kwargs):
        return build_causal_window(rows,now,Pose2D(0,0,0),clock_domain='sim',**kwargs)

    def test_d02_two_ids_backward_difference_and_real_interval(self):
        rows=[observation(i/10) for i in range(-2,20)]
        rows.extend(observation(i/10,'b',x=0.,y=5*i/10) for i in range(-2,20))
        w=self.build(rows)
        np.testing.assert_allclose(w['features'][0,:,2],10,atol=1e-10)
        np.testing.assert_allclose(w['features'][1,:,3],5,atol=1e-10)
        np.testing.assert_allclose(w['features'][:,:,4:6],0,atol=1e-10)
        self.assertTrue(w['feature_mask'][:,:,2:6].all())
        self.assertAlmostEqual(w['history_times'][-1]-w['history_times'][0],1.9)
        missing=self.build([r for r in rows if not (r['track_id']=='a' and r['observation_time']==0.4)])
        self.assertFalse(missing['observed_mask'][0,4]);self.assertFalse(missing['feature_mask'][0,4].any())
        self.assertAlmostEqual(missing['actual_dt'][0,5],0.2)
        self.assertAlmostEqual(missing['features'][0,5,2],10)

    def test_m04_future_replacement_and_future_only_targets_do_not_change_inputs(self):
        history=[observation(i/10) for i in range(20)]
        future=[observation(i/10) for i in range(20,30)]
        before=self.build(history+future)
        changed=[dict(r,x=1e15,y=-1e15,category='PEDESTRIAN') for r in future]
        after=self.build(history+changed+[observation(2.1,'future_only')])
        self.assertEqual(before['input_hash'],after['input_hash'])
        np.testing.assert_array_equal(before['features'],after['features'])
        labels=build_future_labels(future,before,future_steps=10)
        changed_labels=build_future_labels(changed,after,future_steps=10)
        self.assertFalse(np.array_equal(labels['future_positions'],changed_labels['future_positions']))
        self.assertNotIn('future_only',after['track_ids'])

    def test_arrival_is_not_usable_and_wrong_domains_rejected(self):
        row=observation(1.0,arrival_time=1.0,usable_time=1.2)
        self.assertEqual(self.build([row],1.1)['track_ids'],[])
        self.assertEqual(self.build([row],1.2)['track_ids'],['a'])
        with self.assertRaises(ValueError):self.build([dict(row,clock_domain='wall')],1.2)
        with self.assertRaises(ValueError):self.build([dict(row,arrival_time=1.3)],1.4)

    def test_missing_future_is_masked_and_true_zero_is_valid(self):
        w=self.build([observation(1.9)])
        labels=build_future_labels([observation(2.,x=0.,y=0.),observation(2.2)],w,future_steps=4)
        self.assertEqual(labels['future_mask'].tolist(),[[True,False,True,False]])
        np.testing.assert_array_equal(labels['future_positions'][0,0],[0,0])
        self.assertFalse(w['feature_mask'][0,-1,2])
        stationary=self.build([observation(1.9,vx=0.,vy=0.,velocity_reliable=True,heading=1.)])
        self.assertTrue(stationary['feature_mask'][0,-1,2]);self.assertFalse(stationary['feature_mask'][0,-1,6])

    def test_reference_rotation_and_ignored_hidden_fields(self):
        row=observation(1.9,x=12.,y=23.,vx=1.,vy=0.,velocity_reliable=True,future_gt=[1e20])
        w=build_causal_window([row],1.9,Pose2D(10,20,np.pi/2),clock_domain='sim')
        np.testing.assert_allclose(w['features'][0,-1,:4],[3,-2,0,-1],atol=1e-10)
        self.assertNotIn('future_gt',w)


class ScalerDatasetTests(unittest.TestCase):
    def test_d04_only_train_valid_features_affect_frozen_scaler(self):
        x=np.zeros((2,20,len(FEATURES)));m=np.ones_like(x,dtype=bool);m[:,:,9]=False;x[:,:,9]=1e30
        a=MaskedScaler();a.update(x,m,role='train');first=a.freeze()
        for role in ('model_val','risk_calibration','test'):
            with self.assertRaises(ValueError):a.update(np.full_like(x,1e20),m,role=role)
        self.assertEqual(a.freeze(),first);self.assertEqual(first['count'][9],0)
        b=MaskedScaler();b.update(x[:1],m[:1],role='train');b.update(x[1:],m[1:],role='train')
        self.assertEqual(b.freeze(),first)

    def test_labels_separate_input_reader_hash_and_lazy_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'samples/0';folder.mkdir(parents=True)
            w=build_causal_window([observation(i/10) for i in range(20)],1.9,Pose2D(0,0,0),clock_domain='sim')
            labels=build_future_labels([observation(2.)],w)
            path=folder/'one.npz';save_sample(path,w,labels)
            (folder/'index.jsonl').write_text(json.dumps({'project_role':'train','status':'written','sample_path':'samples/0/one.npz'})+'\n',encoding='utf-8')
            scaler=MaskedScaler();scaler.update(w['features'],w['feature_mask'],role='train')
            (root/'scaler.json').write_text(json.dumps(scaler.freeze()),encoding='utf-8')
            dataset=PredictionDataset(root,'train');self.assertEqual(len(dataset),1)
            before=dataset.inputs(0);self.assertNotIn('future_positions',before)
            self.assertEqual(load_sample(path)['input_hash'],w['input_hash'])
            path.with_suffix('.labels.npz').unlink()
            np.testing.assert_array_equal(before['features'],dataset.inputs(0)['features'])
            with self.assertRaises(FileNotFoundError):dataset.labels(0)


if __name__=='__main__':unittest.main()
