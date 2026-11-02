"""N17本机原生YOLO训练：全面留证、两权重与epoch边界恢复。"""
import copy
import csv
import io
import json
import math
from pathlib import Path
import random
import subprocess
import tempfile
import time

import numpy as np

from .detector_training import trainer_class
from .detector_local_checkpoints import publish_weights
from .detector_local_visuals import label_diagnostics,capture_validation
from .run_store import append_json,write_json,sha256


def append_csv(path,row):
    path=Path(path);exists=path.exists()
    with path.open('a',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(row))
        if not exists:writer.writeheader()
        writer.writerow(row);f.flush()


def cpu_tree(value):
    import torch
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def rich_trainer_class(expected_batch,policy,options):
    import torch
    import psutil
    from ultralytics.utils.torch_utils import unwrap_model
    base=trainer_class(expected_batch,policy,True)

    class LocalTrainer(base):
        def __init__(self,*args,**kwargs):
            self._local_options=options;self._local_epoch_stats={};self._local_restored=None
            super().__init__(*args,**kwargs)
            self.add_callback('on_fit_epoch_end',self._local_validation_finished)

        def check_resume(self,overrides):
            super().check_resume(overrides)
            # 新目录保存续训，旧失败/中断记录保持字节不变。
            if options.get('save_dir'):self.args.save_dir=options['save_dir']

        def _load_checkpoint_state(self,ckpt):
            super()._load_checkpoint_state(ckpt)
            if self.resume:
                meta=ckpt.get('n17_local')
                if not meta or meta['recipe_sha256']!=options['recipe_sha256'] or meta['model_id']!=options['model_id'] or meta['seed']!=options['seed']:
                    raise ValueError('本机恢复检查点的模型/seed/配方不匹配')
                unwrap_model(self.model).load_state_dict(meta['model_state'],strict=True)
                self.ema.ema.load_state_dict(meta['ema_state'],strict=True)
                self.scheduler.load_state_dict(meta['scheduler_state'])
                self._local_restored=meta

        def _setup_train(self,*args,**kwargs):
            result=super()._setup_train(*args,**kwargs)
            self._n17_step=0;self._local_attempt=0;self._local_elapsed_offset=0.;self._n17_elapsed=time.perf_counter()
            if self._local_restored:
                meta=self._local_restored
                self._n17_step=meta['applied_updates'];self._local_attempt=meta['batch_attempts'];self._local_elapsed_offset=meta['elapsed_seconds']
                trace=self.save_dir/'早停判定.jsonl';self.stopper.trace=None
                for line in trace.read_text(encoding='utf-8').splitlines():
                    record=json.loads(line);self.stopper(record['epoch'],record['fitness'])
                    if self.stopper.last_decision!=record:raise ValueError('恢复的早停原始记录与规则不同')
                self.stopper.trace=trace
                if self.stopper.last_epoch!=self.start_epoch:raise ValueError('早停记录和检查点epoch不一致')
                if self.stopper.last_decision['stop']:raise ValueError('已触发早停，不能继续增加训练epoch')
                if meta.get('loader_generator_state') is not None and self.train_loader.generator is not None:
                    self.train_loader.generator.set_state(meta['loader_generator_state'])
                    self.train_loader.reset()
                    self.train_loader.generator.set_state(meta['loader_generator_state'])
                random.setstate(meta['python_rng']);np.random.set_state(meta['numpy_rng']);torch.set_rng_state(meta['torch_rng'])
                torch.cuda.set_rng_state_all(meta['cuda_rng'])
                write_json(self.save_dir/'检查点恢复核验.json',{'status':'passed','start_epoch':self.start_epoch+1,
                    'applied_updates_restored':self._n17_step,'batch_attempts_restored':self._local_attempt,
                    'optimizer_restored':True,'scheduler_restored':True,'amp_scaler_restored':True,'model_and_ema_fp32_restored':True,
                    'rng_restored':True,'bitwise_data_stream_equivalence_claimed':False,
                    'source_checkpoint_sha256':options['resume_checkpoint_sha256']})
                self._local_restored=None
            # 首次训练或续训各保存实际开始的前三批，不用旧图冒充新批次。
            offset=self.start_epoch*len(self.train_loader)
            self.plot_idx=list(set(self.plot_idx+[offset,offset+1,offset+2]))
            self.args.save_period=self.save_period=-1
            import yaml
            (self.save_dir/'运行选项.yaml').write_text(yaml.safe_dump(vars(self.args),allow_unicode=True,sort_keys=False),encoding='utf-8')
            keys=('epochs','batch','nbs','imgsz','optimizer','lr0','lrf','momentum','weight_decay','warmup_epochs','patience',
                  'amp','deterministic','mosaic','mixup','copy_paste','hsv_h','hsv_s','hsv_v','degrees','translate','scale','shear','perspective','flipud','fliplr','save_period')
            (self.save_dir/'超参数.yaml').write_text(yaml.safe_dump({k:getattr(self.args,k) for k in keys},allow_unicode=True,sort_keys=False),encoding='utf-8')
            return result

        def get_validator(self):
            validator=super().get_validator()
            class RecordedValidator(type(validator)):
                def plot_val_samples(self,batch,ni):
                    append_json(self.save_dir/'验证批次图片来源.jsonl',{'native_file':f'val_batch{ni}_labels.jpg',
                        'phase':getattr(self,'_n17_phase','epoch'),'epoch':getattr(self,'_n17_epoch',None),
                        'input_files':list(batch['im_file']),'kind':'真实标签'})
                    return super().plot_val_samples(batch,ni)
                def plot_predictions(self,batch,preds,ni,max_det=None):
                    append_json(self.save_dir/'验证批次图片来源.jsonl',{'native_file':f'val_batch{ni}_pred.jpg',
                        'phase':getattr(self,'_n17_phase','epoch'),'epoch':getattr(self,'_n17_epoch',None),
                        'input_files':list(batch['im_file']),'kind':'实际模型预测'})
                    return super().plot_predictions(batch,preds,ni,max_det)
            validator.__class__=RecordedValidator
            return validator

        def validate(self):
            self.validator._n17_phase='epoch';self.validator._n17_epoch=self.epoch+1
            return super().validate()

        def plot_training_labels(self):
            super().plot_training_labels()
            labels=self.train_loader.dataset.labels
            boxes=np.concatenate([r['bboxes'] for r in labels],axis=0)
            cls=np.concatenate([r['cls'] for r in labels],axis=0)
            label_diagnostics(self.save_dir,boxes,cls,self.data['names'])

        def plot_training_samples(self,batch,ni):
            append_json(self.save_dir/'训练批次图片来源.jsonl',{'native_file':f'train_batch{ni}.jpg','epoch':self.epoch+1,
                        'batch_index':ni,'input_files':list(batch['im_file']),'augmentation':'见args.yaml，真实训练预处理结果'})
            return super().plot_training_samples(batch,ni)

        def optimizer_step(self):
            # 与锁定原生顺序相同；AMP跳步写尝试表，不伪造一次参数更新。
            self.scaler.unscale_(self.optimizer)
            norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=10.,error_if_nonfinite=False)
            loss=float(self.loss.detach());components={k:float(v.detach()) for k,v in self.loss_items.items()}
            if not math.isfinite(loss) or not all(math.isfinite(v) for v in components.values()):
                append_json(self.save_dir/'非有限损失失败.jsonl',{'epoch':self.epoch+1,'loss':str(loss),'components':{k:str(v) for k,v in components.items()}})
                raise FloatingPointError('损失非有限，保留失败而不把它记成早停')
            before=float(self.scaler.get_scale());lr=[float(g['lr']) for g in self.optimizer.param_groups]
            if not bool(self.amp) and not torch.isfinite(norm):raise FloatingPointError('FP32梯度非有限')
            self.scaler.step(self.optimizer);self.scaler.update();after=float(self.scaler.get_scale())
            applied=not bool(self.amp) or after>=before
            self.optimizer.zero_grad()
            if self.ema:self.ema.update(self.model)
            torch.cuda.synchronize();self._local_attempt+=1
            if applied:self._n17_step+=1
            row=dict(step=self._n17_step,epoch=self.epoch+1,attempt=self._local_attempt,
                     image_count=self._n17_images,target_count=self._n17_targets,loss=loss,
                     loss_components=json.dumps(components,sort_keys=True),training_error_definition='box_and_class_objective_losses',
                     gradient_norm=float(norm),learning_rate=lr[0],learning_rates=json.dumps(lr),
                     step_seconds=time.perf_counter()-self._n17_started,
                     elapsed_seconds=self._local_elapsed_offset+time.perf_counter()-self._n17_elapsed,
                     gpu_allocated_mib=torch.cuda.memory_allocated()/2**20,gpu_reserved_mib=torch.cuda.memory_reserved()/2**20,
                     host_rss_mib=psutil.Process().memory_info().rss/2**20,image_files=json.dumps(self._n17_files,ensure_ascii=False),
                     update_applied=applied,amp_scale_before=before,amp_scale_after=after)
            append_csv(self.save_dir/'逐批次优化尝试.csv',row)
            if applied:
                if not math.isfinite(float(norm)):raise FloatingPointError('实际更新含非有限梯度')
                append_csv(self.save_dir/'逐步优化指标.csv',row)
            stats=self._local_epoch_stats.setdefault(self.epoch+1,{'batches':0,'applied_updates':0,'amp_skips':0,'image_exposures':0,'target_exposures':0})
            stats['batches']+=1;stats['applied_updates']+=int(applied);stats['amp_skips']+=int(not applied)
            stats['image_exposures']+=self._n17_images;stats['target_exposures']+=self._n17_targets

        def save_metrics(self,metrics):
            super().save_metrics(metrics)
            capture_validation(self.validator,self.save_dir,self.epoch+1,'epoch')
            stats=self._local_epoch_stats[self.epoch+1]
            append_json(self.save_dir/'逐轮优化与资源.jsonl',{'epoch':self.epoch+1,**stats,
                        'gpu_peak_allocated_cumulative_mib':torch.cuda.max_memory_allocated()/2**20,
                        'host_rss_mib':psutil.Process().memory_info().rss/2**20})
            if stats['applied_updates']==0:raise FloatingPointError('整轮无实际参数更新，不能当作有效训练')

        def save_model(self):
            # 保留FP32优化器/原训练模型/EMA以及随机状态，供原生epoch边界恢复。
            old=(self.wdir,self.last,self.best)
            with tempfile.TemporaryDirectory(prefix='.检查点序列化-',dir=self.save_dir) as temporary:
                stage=Path(temporary);self.wdir=stage;self.last=stage/'last.pt';self.best=stage/'best.pt'
                try:written=super().save_model()
                finally:self.wdir,self.last,self.best=old
                if not written:return written
                checkpoint=torch.load(stage/'last.pt',map_location='cpu',weights_only=False)
                checkpoint['optimizer']=cpu_tree(self.optimizer.state_dict())
                checkpoint['n17_local']={'format':'n17-local-checkpoint-1','epoch':self.epoch+1,
                    'model_id':options['model_id'],'seed':options['seed'],'recipe_sha256':options['recipe_sha256'],
                    'source_fingerprint_sha256':options['source_fingerprint_sha256'],
                    'applied_updates':self._n17_step,'batch_attempts':self._local_attempt,
                    'elapsed_seconds':self._local_elapsed_offset+time.perf_counter()-self._n17_elapsed,
                    'model_state':cpu_tree(unwrap_model(self.model).state_dict()),'ema_state':cpu_tree(self.ema.ema.state_dict()),
                    'scheduler_state':cpu_tree(self.scheduler.state_dict()),'python_rng':random.getstate(),
                    'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                    'loader_generator_state':self.train_loader.generator.get_state() if self.train_loader.generator is not None else None}
                buffer=io.BytesIO();torch.save(checkpoint,buffer)
                publish_weights(self.save_dir,buffer.getvalue(),(stage/'best.pt').is_file(),
                                {'epoch':self.epoch+1,'applied_updates':self._n17_step,'batch_attempts':self._local_attempt})
            return True

        def _local_validation_finished(self,trainer):
            final=getattr(self,'_n17_final_eval',False)
            epoch=len((self.save_dir/'逐轮验证指标.jsonl').read_text(encoding='utf-8').splitlines()) if final else self.epoch+1
            if final:capture_validation(self.validator,self.save_dir,epoch,'best_checkpoint')
            if not final:
                uuid=str(getattr(torch.cuda.get_device_properties(0),'uuid',''))
                try:
                    result=subprocess.run(['nvidia-smi','--id=GPU-'+uuid,'--query-gpu=temperature.gpu,utilization.gpu,power.draw,memory.used',
                                           '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True)
                    append_json(self.save_dir/'逐轮显卡监测.jsonl',{'epoch':epoch,'raw':result.stdout.strip(),'units':['摄氏度','百分比','瓦','MiB'],'status':'read'})
                except (OSError,subprocess.SubprocessError) as exc:
                    append_json(self.save_dir/'逐轮显卡监测.jsonl',{'epoch':epoch,'status':'unavailable','reason':str(exc)})
                if options.get('pause_after_epoch')==epoch:raise KeyboardInterrupt('按显式设置在已保存epoch边界暂停')

        def final_eval(self):
            # last保留优化器/AMP/随机状态；两个文件均可直接YOLO加载。
            self._n17_final_eval=True
            try:
                if self.best.exists():
                    self.validator.args.plots=True;self.validator.args.compile=False
                    self.validator._n17_phase='best_checkpoint';self.validator._n17_epoch=self.epoch+1
                    self.metrics=self.validator(model=str(self.best));self.metrics.pop('fitness',None)
                    self.run_callbacks('on_fit_epoch_end')
            finally:self._n17_final_eval=False

    return LocalTrainer
