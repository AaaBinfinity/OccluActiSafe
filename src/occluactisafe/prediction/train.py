"""N07完整声明范围的GRU训练/验证；未来仅由loss/离线评分读取。"""
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
import yaml
from occluactisafe.common.config import PROJECT_ROOT, _UniqueLoader, load_config
from occluactisafe.common.device import seed_everything, select_device, inventory
from occluactisafe.common.device_smoke import load_smoke_config
from occluactisafe.common.run_manifest import create_run_manifest
from occluactisafe.data.split import canonical_hash
from occluactisafe.prediction.dataset import PredictionDataset, MaskedScaler
from occluactisafe.prediction.baselines.gru import MultiModalGRU
from occluactisafe.prediction.losses import multimodal_loss
from occluactisafe.prediction.evaluate import sha256, write_json
from occluactisafe.evaluation.prediction_metrics import score_target, MetricAccumulator
from occluactisafe.prediction.telemetry import StepLog
from occluactisafe.prediction.training_plots import render_training_plots

INPUTS = ('features','feature_mask','observed_mask','actual_dt','history_times','categories','anchor_position')


def load_training_config(path, *, overrides=None, cli_overrides=None):
    path = Path(path); path = path if path.is_absolute() else PROJECT_ROOT / path
    with path.open(encoding='utf-8') as f: config = yaml.load(f, Loader=_UniqueLoader)
    fields = {'windows_dir','output_root','model_config','budget_config','device','seeds','epochs','batch_size',
              'learning_rate','weight_decay','patience','gradient_clip','resume','preflight_only'}
    if not isinstance(config, dict) or set(config) != fields: raise ValueError('N07配置字段不完整/未知')
    for layer in (overrides, cli_overrides):
        if layer is not None:
            if not isinstance(layer, dict) or set(layer) - fields: raise ValueError('未知N07配置覆盖')
            config.update(layer)
    for key in ('windows_dir','output_root','model_config','budget_config'):
        if not isinstance(config[key], str) or not config[key]: raise ValueError('路径非法')
        p = Path(config[key]); config[key] = str((p if p.is_absolute() else PROJECT_ROOT / p).resolve())
    if config['resume'] is not None:
        p = Path(config['resume']); config['resume'] = str((p if p.is_absolute() else PROJECT_ROOT / p).resolve())
    if not isinstance(config['seeds'], list) or not config['seeds'] or len(set(config['seeds'])) != len(config['seeds']) or any(type(s) is not int or not 0 <= s < 2**32 for s in config['seeds']): raise ValueError('种子非法')
    for key in ('epochs','batch_size','patience'):
        if type(config[key]) is not int or config[key] <= 0: raise ValueError(f'{key}非法')
    for key in ('learning_rate','weight_decay','gradient_clip'):
        if isinstance(config[key], bool) or not isinstance(config[key], (int,float)) or not np.isfinite(config[key]) or config[key] < (0 if key == 'weight_decay' else 1e-12): raise ValueError(f'{key}非法')
    if type(config['preflight_only']) is not bool: raise ValueError('preflight_only非法')
    if config['resume'] and len(config['seeds']) != 1: raise ValueError('恢复任务须显式选择单个原seed')
    if config['device'] != 'cpu' and not str(config['device']).startswith('cuda:'): raise ValueError('设备非法')
    with Path(config['model_config']).open(encoding='utf-8') as f: model = yaml.load(f, Loader=_UniqueLoader)
    if set(model) != {'hidden_size','modes','dropout'} or type(model['hidden_size']) is not int or type(model['modes']) is not int or model['hidden_size'] < 1 or model['modes'] < 1 or not 0 <= model['dropout'] < 1: raise ValueError('GRU模型配置非法')
    budget = load_smoke_config(config['budget_config'])['training_budget']
    if config['epochs'] > budget['max_epochs']: raise ValueError('epoch超过N06冻结预算')
    return {**config, 'model': model, 'budget': budget}


def working_set_bytes():
    if __import__('sys').platform == 'win32':
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)] + [(k,ctypes.c_size_t) for k in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        c = Counters(); c.cb = ctypes.sizeof(c)
        kernel = ctypes.WinDLL('kernel32'); kernel.GetCurrentProcess.restype = wintypes.HANDLE
        fn = ctypes.WinDLL('psapi').GetProcessMemoryInfo
        fn.argtypes = [wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        if not fn(kernel.GetCurrentProcess(),ctypes.byref(c),c.cb): raise OSError('无法读取进程工作集')
        return c.WorkingSetSize
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if __import__('sys').platform == 'darwin' else 1024)


def enforce_budget(budget, started, prior_seconds=0, output_dir=PROJECT_ROOT):
    if working_set_bytes() > budget['max_host_working_set_mib'] * 1024**2: raise RuntimeError('主机工作集超过N06预算')
    if time.perf_counter() - started + prior_seconds >= budget['max_wall_seconds']: raise RuntimeError('达到N06墙钟预算，停止并保留latest检查点')
    if shutil.disk_usage(output_dir).free < 1024**3: raise OSError('产物盘剩余空间不足1GiB')


def prepare_data(config, directory):
    root = Path(config['windows_dir']); out = Path(config['output_root'])
    if root == out or root in out.parents or out in root.parents: raise ValueError('训练输出不能覆盖N04输入')
    state = json.loads((root/'n04_status.json').read_text(encoding='utf-8'))
    task = json.loads((root/'task_config.json').read_text(encoding='utf-8'))
    split = json.loads((root/'split_manifest.json').read_text(encoding='utf-8'))
    if state['status'] != 'completed' or canonical_hash({k:v for k,v in split.items() if k!='sha256'}) != state['split_sha256']: raise ValueError('N04未完成或划分哈希不符')
    scaler = json.loads((root/'scaler.json').read_text(encoding='utf-8'))
    if scaler['sha256'] != state['scaler_sha256'] or canonical_hash({k:v for k,v in scaler.items() if k!='sha256'}) != scaler['sha256']: raise ValueError('训练归一化身份不符')
    by_source = {r['source_file']:r for r in split['files']}
    data = {}; coverage = {}; started = time.perf_counter()
    with (directory/'input_inventory.jsonl').open('w',encoding='utf-8') as log:
        for role in ('train','model_val'):
            dataset = PredictionDataset(root,role,normalized=False)
            if not len(dataset): raise ValueError(f'{role}为空')
            chunks = {k:[] for k in (*INPUTS,'truth','future_mask')}; keys=[]; excluded=0
            for n,row in enumerate(dataset.index):
                if n % 500 == 0: enforce_budget(config['budget'],started,output_dir=directory)
                original = by_source[row['source_file']]
                if original['original_split']!='train' or original['project_role']!=role: raise ValueError('索引角色与冻结划分冲突')
                x = dataset.inputs(n); y = dataset.labels(n)
                allowed = x['prediction_target_mask'] & x['observed_mask'].all(1) & y['future_mask'].all(1)
                if not np.array_equal(allowed,y['target_eligible']): raise ValueError('TARGET_AGENT完整监督定义变化')
                if x['features'].shape[1]!=task['history_steps'] or not np.allclose(y['future_times'],np.arange(1,task['future_steps']+1)*task['dt']): raise ValueError('窗口时域不符')
                selected = np.flatnonzero(allowed)
                excluded += int(x['prediction_target_mask'].sum()) - len(selected)
                normalized = MaskedScaler.transform(x,scaler)
                for key in ('features','feature_mask','observed_mask','actual_dt','categories'):
                    chunks[key].append((normalized if key=='features' else np.asarray(x[key]))[selected])
                chunks['history_times'].append(np.broadcast_to(x['history_times'],(len(selected),task['history_steps'])))
                chunks['anchor_position'].append(x['features'][selected,-1,:2])
                chunks['truth'].append(y['future_positions'][selected]);chunks['future_mask'].append(y['future_mask'][selected])
                keys.extend({'source_file':row['source_file'],'group_id':row['group_id'],'track_id':x['track_ids'][i],
                             'category':int(x['categories'][i])} for i in selected)
                p = root/row['sample_path']
                log.write(json.dumps({'role':role,'source_file':row['source_file'],'input_hash':x['input_hash'],
                    'input_sha256':sha256(p),'labels_sha256':sha256(p.with_suffix('.labels.npz'))},sort_keys=True)+'\n')
                if (n+1)%5000==0: print(f'N07准备 {role}: {n+1}/{len(dataset)}',flush=True)
            if not keys: raise ValueError(f'{role}无完整监督')
            data[role] = {k:np.concatenate(v,axis=0) for k,v in chunks.items()}
            data[role]['keys'] = keys
            coverage[role] = {'windows':len(dataset),'eligible_targets':len(keys),'incomplete_TARGET_AGENT':excluded}
    identity = {'split_sha256':split['sha256'],'scaler_sha256':scaler['sha256'],
                'data':json.loads((root/'manifest.json').read_text(encoding='utf-8'))['data'],
                'inventory_sha256':sha256(directory/'input_inventory.jsonl'),'task':task,'coverage':coverage}
    write_json(directory/'data_identity.json',identity)
    write_json(directory/'normalization.json',scaler)
    for role in data:
        write_json(directory/f'{role}_targets.json',data[role]['keys'])
        np.savez_compressed(directory/f'{role}_cache.npz',**{k:v for k,v in data[role].items() if k!='keys'})
    return data, identity


def batch_to_device(data, indices, device):
    result = {}
    for key in (*INPUTS,'truth','future_mask'):
        a = data[key][indices]
        dtype = torch.float64 if key in ('anchor_position','truth') else torch.bool if key in ('feature_mask','observed_mask','future_mask') else torch.long if key=='categories' else torch.float32
        result[key] = torch.as_tensor(np.ascontiguousarray(a),dtype=dtype,device=device)
    return result


def forward(model, batch):
    return model(**{k:batch[k] for k in INPUTS})


def rng_state():
    n = np.random.get_state()
    return {'python':random.getstate(),'numpy':(n[0],n[1].tolist(),n[2],n[3],n[4]),
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python']);n=state['numpy'];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']: torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def save_checkpoint(path, model, optimizer, contract, progress):
    payload = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'contract':contract,
               'progress':progress,'rng':rng_state(),'format_version':1}
    temporary = path.with_suffix('.tmp');torch.save(payload,temporary);temporary.replace(path)


def load_checkpoint(path, model, optimizer, contract):
    payload = torch.load(path,map_location='cpu',weights_only=True)
    if payload['contract'] != contract or payload['format_version']!=1: raise ValueError('检查点数据/配置/优化器/归一化/模式身份不兼容')
    model.load_state_dict(payload['model'])
    if optimizer is not None: optimizer.load_state_dict(payload['optimizer'])
    restore_rng(payload['rng'])
    return payload['progress']


def evaluate(model, data, device, batch_size, *, save_dir=None, guard=None):
    model.eval(); overall = MetricAccumulator(); groups = {}; predictions=[]; logits=[]; rows=[]
    with torch.no_grad():
        for offset in range(0,len(data['features']),batch_size):
            if guard: guard()
            batch = batch_to_device(data,slice(offset,offset+batch_size),device)
            output = forward(model,batch)
            p = output['pred'].cpu().numpy(); l = output['mode_logits'].cpu().numpy()
            valid = output['prediction_valid'].cpu().numpy()
            for i in range(len(p)):
                row = score_target(p[i],data['truth'][offset+i],data['future_mask'][offset+i],prediction_valid=bool(valid[i]),top_k=model.modes,logits=l[i])
                overall.add(row);key=data['keys'][offset+i]
                name = str(key['category']);groups.setdefault(name,MetricAccumulator()).add(row)
                if save_dir: rows.append({**key,**row})
            if save_dir: predictions.append(p); logits.append(l)
    if save_dir:
        np.savez_compressed(save_dir/'validation_predictions.npz',predictions=np.concatenate(predictions),logits=np.concatenate(logits))
        with (save_dir/'per_agent_metrics.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return {'overall':overall.summary(),'categories':{k:v.summary() for k,v in groups.items()}}


def training_contract(config, identity, seed):
    return {'telemetry_schema':1,'data':identity,'model':config['model'],'seed':seed,'optimizer':{'name':'AdamW','lr':config['learning_rate'],'weight_decay':config['weight_decay']},
            'batch_size':config['batch_size'],'epochs':config['epochs'],'patience':config['patience'],'gradient_clip':config['gradient_clip'],
            'budget':config['budget'],'selection':'model_val minADE_m','history_steps':identity['task']['history_steps'],'future_steps':identity['task']['future_steps']}


def render_preserving_rng(directory):
    state=rng_state()
    try:return render_training_plots(directory)
    finally:restore_rng(state)


def train_seed(config, data, identity, group_dir, seed, device):
    seed_everything(seed,torch)
    directory=group_dir/f'seed_{seed}';directory.mkdir()
    model=MultiModalGRU(future_steps=identity['task']['future_steps'],**config['model']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    contract=training_contract(config,identity,seed)
    progress={'epoch':0,'next_batch':0,'steps':0,'best_score':None,'bad_epochs':0,'elapsed_seconds':0.,'curve':[],
              'epoch_loss_sum':0.,'epoch_batches':0,'epoch_grad_max':0.,'step_log_prefix_sha256':None}
    if config['resume']:
        progress=load_checkpoint(config['resume'],model,optimizer,contract)
        shutil.copy2(config['resume'],directory/'latest.pt')
        old_best=Path(config['resume']).parent/'best.pt'
        if old_best.exists(): shutil.copy2(old_best,directory/'best.pt')
    started=time.perf_counter(); prior=progress['elapsed_seconds']
    guard=lambda:enforce_budget(config['budget'],started,prior,directory)
    n=len(data['train']['features']); batch_size=config['batch_size']; stop_reason='epochs_completed'
    log=None
    try:
        log=StepLog(directory/'step_metrics.csv',source=Path(config['resume']).parent/'step_metrics.csv' if config['resume'] else None,
                    steps=progress['steps'],prefix_sha256=progress['step_log_prefix_sha256'])
        if progress['curve']:
            write_json(directory/'curve.json',progress['curve'])
            with (directory/'epoch_metrics.csv').open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(progress['curve'][0]));writer.writeheader();writer.writerows(progress['curve'])
        for epoch in range(progress['epoch'],config['epochs']):
            model.train(); order=np.random.default_rng(seed+epoch).permutation(n)
            start_batch=progress['next_batch'] if epoch==progress['epoch'] else 0
            for bi,offset in enumerate(range(0,n,batch_size)):
                if bi < start_batch: continue
                guard()
                if progress['steps']>=config['budget']['max_optimizer_steps']: raise RuntimeError('达到N06优化步数预算')
                if device.type=='cuda':torch.cuda.synchronize(device)
                step_started=time.perf_counter();indices=order[offset:offset+batch_size]
                batch=batch_to_device(data['train'],indices,device)
                optimizer.zero_grad(set_to_none=True); output=forward(model,batch)
                loss,record=multimodal_loss(output,batch['truth'],batch['future_mask'])
                if loss is None: raise ValueError('声明的完整监督批次没有有效目标')
                if not torch.isfinite(loss): raise RuntimeError('训练loss非有限')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'])
                if not torch.isfinite(norm): raise RuntimeError('梯度非有限')
                optimizer.step()
                if device.type=='cuda':torch.cuda.synchronize(device)
                step_seconds=time.perf_counter()-step_started;loss_value=float(loss.detach());grad_value=float(norm)
                progress.update(epoch=epoch,next_batch=bi+1,steps=progress['steps']+1)
                progress['epoch_loss_sum']+=loss_value;progress['epoch_batches']+=1
                progress['epoch_grad_max']=max(progress['epoch_grad_max'],grad_value)
                log.append({'global_step':progress['steps'],'epoch':epoch+1,'batch_index':bi+1,'target_count':record['valid_targets'],
                    'loss':loss_value,'regression_loss':record['reg'],'classification_loss':record['mode'],
                    'train_minADE_m':record['train_minADE'],'gradient_norm_before_clip':grad_value,
                    'learning_rate':optimizer.param_groups[0]['lr'],'step_seconds':step_seconds,
                    'elapsed_seconds':prior+time.perf_counter()-started,
                    'gpu_allocated_mib':torch.cuda.memory_allocated(device)/1024**2 if device.type=='cuda' else None,
                    'gpu_reserved_mib':torch.cuda.memory_reserved(device)/1024**2 if device.type=='cuda' else None,
                    'host_working_set_mib':working_set_bytes()/1024**2,'batch_target_indices':json.dumps(indices.tolist(),separators=(',',':'))})
                progress['step_log_prefix_sha256']=log.sha256
                if progress['steps']%200==0:
                    progress['elapsed_seconds']=prior+time.perf_counter()-started
                    save_checkpoint(directory/'latest.pt',model,optimizer,contract,progress)
                    print(f'N07 seed={seed} epoch={epoch+1} step={progress["steps"]} loss={loss_value:.5f}',flush=True)
            validation=evaluate(model,data['model_val'],device,batch_size,guard=guard)
            score=validation['overall']['minADE_m']
            if score is None: raise ValueError('验证集没有可评分对象')
            improved=progress['best_score'] is None or score<progress['best_score']
            progress['bad_epochs']=0 if improved else progress['bad_epochs']+1
            if improved: progress['best_score']=score
            progress['curve'].append({'epoch':epoch+1,'steps':progress['steps'],
                'train_loss':progress['epoch_loss_sum']/progress['epoch_batches'],
                'gradient_norm_max':progress['epoch_grad_max'],**validation['overall']})
            progress.update(epoch=epoch+1,next_batch=0,epoch_loss_sum=0.,epoch_batches=0,epoch_grad_max=0.)
            write_json(directory/'curve.json',progress['curve'])
            with (directory/'epoch_metrics.csv').open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(progress['curve'][0]));writer.writeheader();writer.writerows(progress['curve'])
            render_preserving_rng(directory)
            progress['elapsed_seconds']=prior+time.perf_counter()-started
            save_checkpoint(directory/'latest.pt',model,optimizer,contract,progress)
            if improved: save_checkpoint(directory/'best.pt',model,optimizer,contract,progress)
            print(f'N07 seed={seed} epoch={epoch+1} validation={validation["overall"]}',flush=True)
            if progress['bad_epochs']>=config['patience']: stop_reason='early_stopping';break
        # 检验best重载输出，随后完整保存best验证预测。
        before=batch_to_device(data['model_val'],slice(0,min(8,len(data['model_val']['features']))),device)
        load_checkpoint(directory/'best.pt',model,None,contract);model.eval()
        with torch.no_grad(): expected=forward(model,before)['pred'].clone()
        clone=MultiModalGRU(future_steps=identity['task']['future_steps'],**config['model']).to(device)
        load_checkpoint(directory/'best.pt',clone,None,contract);clone.eval()
        with torch.no_grad(): torch.testing.assert_close(expected,forward(clone,before)['pred'],rtol=0,atol=0)
        final=evaluate(model,data['model_val'],device,batch_size,save_dir=directory,guard=guard)
        plot_manifest=render_preserving_rng(directory)
        if device.type=='cuda': torch.cuda.synchronize(device)
        result={'seed':seed,'status':'completed','stop_reason':stop_reason,'steps':progress['steps'],'epochs':progress['epoch'],
            'wall_seconds':prior+time.perf_counter()-started,'metrics':final,'checkpoint_reload_equal':True,
            'step_log_rows':log.rows,'step_log_prefix_sha256':log.sha256,'plot_images':len(plot_manifest['images']),
            'best_sha256':sha256(directory/'best.pt'),'latest_sha256':sha256(directory/'latest.pt'),
            'host_working_set_end_bytes':working_set_bytes(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None}
        write_json(directory/'result.json',result);return result
    except BaseException as exc:
        progress['elapsed_seconds']=prior+time.perf_counter()-started
        save_checkpoint(directory/'interrupted.pt',model,optimizer,contract,progress)
        write_json(directory/'failure.json',{'error_type':type(exc).__name__,'error':str(exc),'progress':progress})
        if log is not None:
            try:render_preserving_rng(directory)
            except Exception as plot_error:write_json(directory/'plot_failure.json',{'error_type':type(plot_error).__name__,'error':str(plot_error)})
        raise
    finally:
        if log is not None:log.close()


def run(config):
    if config['device']=='cpu': mode='cpu_mechanism_diagnostic'
    else: mode='tfd_real_gru_training'
    root=Path(config['windows_dir'])
    task=json.loads((root/'task_config.json').read_text(encoding='utf-8'))
    provenance=json.loads((root/'manifest.json').read_text(encoding='utf-8'))['data']
    base=load_config(overrides={'runtime':{'seed':config['seeds'][0]},
        'prediction':{'history_steps':task['history_steps'],'future_steps':task['future_steps'],'future_dt':task['dt']},
        'planning':{'horizon_s':min(3.,task['future_steps']*task['dt'])}})
    manifest=create_run_manifest(config=base,input_mode=mode,data=provenance,output_root=config['output_root'])
    directory=Path(config['output_root'])/manifest['run_id']
    write_json(directory/'task_config.json',config);write_json(directory/'status.json',{'status':'running'})
    try:
        device=select_device(config['device'],torch);seed_everything(config['seeds'][0],torch)
        info=inventory(torch,device);write_json(directory/'environment.json',info)
        if device.type=='cuda':
            torch.cuda.set_per_process_memory_fraction(config['budget']['max_gpu_allocator_mib']*1024**2/info['total_bytes'],device)
            torch.cuda.reset_peak_memory_stats(device)
        data,identity=prepare_data(config,directory)
        if config['preflight_only']:
            result={'status':'preflight_completed','run_id':manifest['run_id'],'identity':identity}
        else:
            results=[train_seed(config,data,identity,directory,s,device) for s in config['seeds']]
            result={'status':'completed','run_id':manifest['run_id'],'results':results,'identity':identity,
                    'limitations':['N04 coordinate consistency unresolved; annotation-history diagnostic only','no test/calibration labels opened; no collision claim']}
        write_json(directory/'status.json',{**result,'completed_utc':datetime.now(timezone.utc).isoformat()})
        write_json(directory/'evidence_index.json',{p.relative_to(directory).as_posix():sha256(p) for p in directory.rglob('*') if p.is_file()})
        return {**result,'run_dir':str(directory)}
    except BaseException as exc:
        write_json(directory/'status.json',{'status':'failed','run_id':manifest['run_id'],'error_type':type(exc).__name__,'error':str(exc)})
        raise
