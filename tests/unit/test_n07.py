"""N07 M07/M08、历史掩码与恢复状态的可手算机制测试。"""
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
from occluactisafe.common.device import seed_everything
from occluactisafe.prediction.baselines.gru import MultiModalGRU
from occluactisafe.prediction.losses import multimodal_loss
from occluactisafe.prediction.train import save_checkpoint,load_checkpoint,forward,load_training_config,working_set_bytes

ROOT=Path(__file__).resolve().parents[2]


def fixture(count=8, history=20, future=40, device='cpu'):
    t=torch.arange(-(history-1),1,device=device)*.1
    speed=torch.linspace(2,4,count,device=device)
    x=torch.zeros(count,history,10,device=device);x[:,:,0]=speed[:,None]*t;x[:,:,2]=speed[:,None]
    batch={'features':x,'feature_mask':torch.ones_like(x,dtype=torch.bool),
        'observed_mask':torch.ones(count,history,dtype=torch.bool,device=device),'actual_dt':torch.full((count,history),.1,device=device),
        'history_times':t,'categories':torch.zeros(count,dtype=torch.long,device=device),
        'anchor_position':torch.zeros(count,2,dtype=torch.float64,device=device)}
    y=torch.zeros(count,future,2,dtype=torch.float64,device=device);y[:,:,0]=speed[:,None]*torch.arange(1,future+1,device=device)*.1
    return batch,y,torch.ones(count,future,dtype=torch.bool,device=device)


class GRUMechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_m07_winner_masks_and_no_labels(self):
        truth=torch.zeros(2,4,2,dtype=torch.float64);truth[0,1:3]=float('nan');truth[1]=float('nan')
        mask=torch.tensor([[1,0,0,1],[0,0,0,0]],dtype=torch.bool)
        pred=torch.full((2,6,4,2),10.,dtype=torch.float64,requires_grad=True)
        with torch.no_grad():pred[0,2]=0
        logits=torch.full((2,6),-20.);logits[0,2]=20;logits.requires_grad_()
        out={'pred':pred,'mode_logits':logits,'prediction_valid':torch.ones(2,dtype=torch.bool)}
        loss,record=multimodal_loss(out,truth,mask)
        self.assertEqual(record['winner'][0],2);self.assertLess(float(loss),1e-8)
        changed=truth.clone();changed[~mask]=9999
        self.assertEqual(float(multimodal_loss(out,changed,mask)[0]),float(loss))
        bad=truth.clone();bad[0,0]=float('nan')
        with self.assertRaises(ValueError):multimodal_loss(out,bad,mask)
        self.assertIsNone(multimodal_loss(out,truth,torch.zeros_like(mask))[0])

    def test_forward_masks_short_empty_permutation_and_no_future(self):
        seed_everything(17,torch);model=MultiModalGRU(future_steps=4,hidden_size=16,dropout=0).eval()
        batch,truth,mask=fixture(count=3,history=5,future=4)
        batch['observed_mask'][0]=False;batch['feature_mask'][0]=False
        batch['observed_mask'][1,2]=False;batch['feature_mask'][1,2]=False
        a=forward(model,batch)
        changed={k:v.clone() for k,v in batch.items()};changed['features'][~changed['feature_mask']]=float('nan');changed['actual_dt'][~changed['observed_mask']]=float('nan')
        b=forward(model,changed);torch.testing.assert_close(a['pred'],b['pred'])
        self.assertEqual(a['prediction_valid'].tolist(),[False,True,True]);self.assertTrue(torch.isfinite(a['pred']).all())
        self.assertEqual(a['pred'].shape,(3,6,4,2));self.assertEqual(a['weights'][0].sum(),0)
        torch.testing.assert_close(a['weights'][1:].sum(-1),torch.ones(2))
        batch['truth']=truth*1e8;batch['future_mask']=mask
        torch.testing.assert_close(forward(model,batch)['pred'],a['pred'])
        order=torch.tensor([2,0,1]);permuted={k:(v[order] if v.ndim and v.shape[0]==3 else v) for k,v in batch.items()}
        torch.testing.assert_close(forward(model,permuted)['pred'][torch.argsort(order)],a['pred'])

    def test_m08_full_resume_next_update_and_rng_reproduction(self):
        seed_everything(29,torch);batch,y,mask=fixture(count=2,history=4,future=3)
        model=MultiModalGRU(future_steps=3,hidden_size=16,dropout=.1)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
        def step(m,o):
            o.zero_grad();loss,_=multimodal_loss(forward(m,batch),y,mask);loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters()));o.step()
        before=[p.detach().clone() for p in model.parameters()];step(model,optimizer)
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,model.parameters())))
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'latest.pt';contract={'scaler':'x','modes':6,'optimizer':'AdamW'}
            save_checkpoint(path,model,optimizer,contract,{'epoch':0,'next_batch':1,'steps':1})
            expected_random=(random.random(),float(np.random.rand()),float(torch.rand(())))
            step(model,optimizer);expected=[p.detach().clone() for p in model.parameters()]
            copy=MultiModalGRU(future_steps=3,hidden_size=16,dropout=.1);opt=torch.optim.AdamW(copy.parameters(),lr=.003)
            progress=load_checkpoint(path,copy,opt,contract)
            self.assertEqual(progress['steps'],1);self.assertEqual(progress['next_batch'],1)
            self.assertEqual(expected_random,(random.random(),float(np.random.rand()),float(torch.rand(()))))
            step(copy,opt)
            for a,b in zip(expected,copy.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
            model.eval();copy.eval();torch.testing.assert_close(forward(model,batch)['pred'],forward(copy,batch)['pred'],rtol=0,atol=0)
            for incompatible in ({**contract,'scaler':'other'},{**contract,'modes':1},{**contract,'optimizer':'SGD'}):
                with self.assertRaises(ValueError):load_checkpoint(path,copy,opt,incompatible)

    def test_config_import_and_resource_readout(self):
        config=load_training_config(ROOT/'configs/experiments/tfd_gru.yaml',overrides={'preflight_only':True},cli_overrides={'preflight_only':False})
        self.assertFalse(config['preflight_only']);self.assertGreater(working_set_bytes(),0)
        with self.assertRaises(ValueError):load_training_config(ROOT/'configs/experiments/tfd_gru.yaml',overrides={'epochs':61})
        result=subprocess.run([sys.executable,'-B','-c',f"import runpy; runpy.run_path({str(ROOT/'scripts/train_prediction.py')!r},run_name='import_only')"],capture_output=True)
        self.assertEqual(result.returncode,0);self.assertEqual(result.stdout,b'')


if __name__=='__main__':unittest.main()
