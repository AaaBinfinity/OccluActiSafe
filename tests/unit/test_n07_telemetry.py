"""逐步指标、图表来源与中断恢复语义。"""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from test_n07 import fixture
from occluactisafe.prediction.telemetry import StepLog,FIELDS
from occluactisafe.prediction.training_plots import render_training_plots
from occluactisafe.prediction.train import train_seed,load_training_config

ROOT=Path(__file__).resolve().parents[2]


class TelemetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_strict_step_log_prefix_rejects_duplicates_missing_and_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);log=StepLog(root/'steps.csv')
            row={k:1 for k in FIELDS};row['batch_target_indices']='[0]'
            log.append(row);prefix=log.sha256
            with self.assertRaises(ValueError):log.append(row)
            row={**row,'global_step':2};log.append(row);log.close()
            new=StepLog(root/'resume.csv',source=root/'steps.csv',steps=1,prefix_sha256=prefix)
            self.assertEqual(new.rows,1);new.append(row);new.close()
            self.assertEqual((root/'resume.csv').read_bytes(),(root/'steps.csv').read_bytes())
            with self.assertRaises(ValueError):StepLog(root/'bad.csv',source=root/'steps.csv',steps=1,prefix_sha256='wrong')
            with self.assertRaises(ValueError):StepLog(root/'missing.csv',source=root/'none.csv',steps=1,prefix_sha256=prefix)

    def test_plot_manifest_hashes_and_csv_regeneration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);log=StepLog(root/'step_metrics.csv')
            for step in (1,2,3):
                row={k:float(step) for k in FIELDS};row.update(global_step=step,epoch=1,batch_index=step,batch_target_indices='[0]',gpu_allocated_mib=None,gpu_reserved_mib=None)
                log.append(row)
            log.close();report=render_training_plots(root)
            self.assertEqual(report['step_rows'],3);self.assertEqual(report['epoch_rows'],0);self.assertEqual(len(report['images']),6)
            for p in report['images']:self.assertEqual(hashlib.sha256((root/p['file']).read_bytes()).hexdigest(),p['sha256'])
            self.assertEqual(report['sources']['step_metrics.csv'],hashlib.sha256((root/'step_metrics.csv').read_bytes()).hexdigest())
            self.assertEqual(render_training_plots(root,output_dir=root/'regenerated')['sources'],report['sources'])
            self.assertTrue((root/'regenerated/step_training.png').exists())

    def test_mid_epoch_failure_resume_preserves_every_update_and_epoch_statistics(self):
        from occluactisafe.prediction import train
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);batch,y,mask=fixture(count=4,history=4,future=3)
            arrays={k:v.cpu().numpy() for k,v in batch.items()}
            arrays['history_times']=np.broadcast_to(arrays['history_times'],(4,4)).copy()
            arrays.update(truth=y.numpy(),future_mask=mask.numpy(),keys=[{'source_file':f'{i}.csv','group_id':'fixture','track_id':str(i),'category':0} for i in range(4)])
            data={'train':arrays,'model_val':arrays};identity={'task':{'history_steps':4,'future_steps':3,'dt':.1},'fixture':True}
            config=load_training_config(ROOT/'configs/experiments/tfd_gru.yaml',overrides={'device':'cpu','epochs':2,'batch_size':2,'seeds':[17]})
            config['model']={'hidden_size':8,'modes':6,'dropout':.1}
            base=root/'baseline';base.mkdir();expected=train_seed(config,data,identity,base,17,torch.device('cpu'))
            broken=root/'interrupted';broken.mkdir();actual_guard=train.enforce_budget;counter=[0]
            def interrupt(*args,**kwargs):
                counter[0]+=1
                if counter[0]==2:raise RuntimeError('injected interruption before second optimizer update')
                return actual_guard(*args,**kwargs)
            with patch.object(train,'enforce_budget',side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'injected interruption'):train_seed(config,data,identity,broken,17,torch.device('cpu'))
            self.assertTrue((broken/'seed_17/step_training.png').exists())
            saved=torch.load(broken/'seed_17/interrupted.pt',weights_only=True)
            self.assertEqual(saved['progress']['steps'],1);self.assertEqual(saved['progress']['epoch_batches'],1)
            recovered=root/'recovered';recovered.mkdir()
            result=train_seed({**config,'resume':str(broken/'seed_17/interrupted.pt')},data,identity,recovered,17,torch.device('cpu'))
            self.assertEqual(expected['metrics'],result['metrics'])
            self.assertEqual(json.loads((base/'seed_17/curve.json').read_text()),json.loads((recovered/'seed_17/curve.json').read_text()))
            with (recovered/'seed_17/step_metrics.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
            self.assertEqual([int(r['global_step']) for r in rows],[1,2,3,4])
            self.assertEqual(len(rows),result['steps'])


if __name__=='__main__':unittest.main()
