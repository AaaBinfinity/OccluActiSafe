"""N07真实GPU上的匀速过拟合诊断；不替代TFD训练。"""
import csv
import json
import os
from pathlib import Path
import sys
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
        directory.mkdir(parents=True,exist_ok=True);curve=[]
        for step in range(1,1001):
            model.train();optimizer.zero_grad();out=forward(model,batch);loss,record=multimodal_loss(out,truth,mask)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            if step%25==0:
                model.eval()
                with torch.no_grad(): score=float(torch.linalg.vector_norm(forward(model,batch)['pred']-truth[:,None],dim=-1).mean(-1).min(-1).values.mean())
                curve.append({'step':step,'loss':float(loss.detach()),'minADE_m':score})
                if score<.1:break
        with (directory/'overfit_curve.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(curve[0]));w.writeheader();w.writerows(curve)
        (directory/'overfit_result.json').write_text(json.dumps({'targets':8,'history':20,'future':40,'device':str(device),'steps':step,'minADE_m':score,'threshold_m':.1},indent=2),encoding='utf-8')
        self.assertLess(score,.1)


if __name__=='__main__':unittest.main()
