"""N07真实GPU上的匀速过拟合诊断；不替代TFD训练。"""
import csv
import json
import hashlib
import os
from pathlib import Path
import sys
import time
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'unit'))
from test_n07 import fixture
from occluactisafe.common.device import seed_everything,select_device
from occluactisafe.prediction.baselines.gru import MultiModalGRU
from occluactisafe.prediction.losses import multimodal_loss
from occluactisafe.prediction.train import forward


class GRUGPUDiagnostics(unittest.TestCase):
    def test_eight_constant_velocity_targets_fit_below_point_one_meter(self):
        seed_everything(17,torch);device=select_device('cuda:0',torch)
        model=MultiModalGRU(future_steps=40,hidden_size=128,dropout=0).to(device)
        batch,truth,mask=fixture(device=device);optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=0)
        directory=Path(os.environ.get('N07_EVIDENCE_DIR','H:/OccluActiSafeDataset/artifacts/n07_20261030'))
        directory.mkdir(parents=True,exist_ok=True);curve=[];raw=[];started=time.perf_counter()
        with (directory/'overfit_step_metrics.csv').open('w',encoding='utf-8',newline='') as f:
            fields=('step','target_count','loss','regression_loss','classification_loss','train_minADE_m','gradient_norm_before_clip','learning_rate','step_seconds','elapsed_seconds')
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for step in range(1,1001):
                torch.cuda.synchronize(device);before=time.perf_counter()
                model.train();optimizer.zero_grad();out=forward(model,batch);loss,record=multimodal_loss(out,truth,mask)
                self.assertTrue(torch.isfinite(loss));loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                self.assertTrue(torch.isfinite(norm));optimizer.step();torch.cuda.synchronize(device)
                row={'step':step,'target_count':8,'loss':float(loss.detach()),'regression_loss':record['reg'],
                     'classification_loss':record['mode'],'train_minADE_m':record['train_minADE'],
                     'gradient_norm_before_clip':float(norm),'learning_rate':optimizer.param_groups[0]['lr'],
                     'step_seconds':time.perf_counter()-before,'elapsed_seconds':time.perf_counter()-started}
                raw.append(row);writer.writerow(row);f.flush()
                if step%25==0:
                    model.eval()
                    with torch.no_grad(): score=float(torch.linalg.vector_norm(forward(model,batch)['pred']-truth[:,None],dim=-1).mean(-1).min(-1).values.mean())
                    curve.append({'step':step,'loss':float(loss.detach()),'minADE_m':score})
                    if score<.1:break
        with (directory/'overfit_curve.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(curve[0]));w.writeheader();w.writerows(curve)
        (directory/'overfit_result.json').write_text(json.dumps({'targets':8,'history':20,'future':40,'device':str(device),'steps':step,'minADE_m':score,'threshold_m':.1},indent=2),encoding='utf-8')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for ax,key,label in zip(axes,('loss','train_minADE_m'),('Batch objective','Training minADE before update (m)')):
            ax.plot([r['step'] for r in raw],[r[key] for r in raw],linewidth=.8)
            ax.set(xlabel='Completed optimizer update',ylabel=label);ax.grid(alpha=.25)
        fig.suptitle('Constructed GPU diagnostic: 8 constant-velocity targets, H20/F40; not TFD')
        for ext in ('png','svg'):fig.savefig(directory/f'overfit_step_curves.{ext}',dpi=160)
        plt.close(fig)
        names=('overfit_step_metrics.csv','overfit_curve.csv','overfit_step_curves.png','overfit_step_curves.svg')
        (directory/'overfit_plot_manifest.json').write_text(json.dumps({'rows':len(raw),'smoothing':False,
            'post_update_evaluation_interval':25,'hashes':{n:hashlib.sha256((directory/n).read_bytes()).hexdigest() for n in names}},indent=2),encoding='utf-8')
        self.assertEqual(len(raw),step)
        self.assertLess(score,.1)


if __name__=='__main__':unittest.main()
