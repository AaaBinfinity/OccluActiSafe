"""锁定Ultralytics 8.4.197训练器并逐optimizer更新留证。"""
import csv
import json
import time
from pathlib import Path

from .run_store import append_json, write_json, sha256


def training_plots(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .visual_fonts import configure_plot_fonts
    configure_plot_fonts()
    directory = Path(directory)
    step_path = directory/'逐步优化指标.csv'
    if not step_path.exists():
        return
    with step_path.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    figs = []
    for filename, keys in [('逐步训练损失', ['loss']), ('逐步梯度和耗时', ['gradient_norm', 'step_seconds']),
                           ('逐步显存和学习率', ['gpu_allocated_mib', 'learning_rate'])]:
        fig, axes = plt.subplots(len(keys), 1, figsize=(10, 3*len(keys)), squeeze=False)
        for ax, key in zip(axes[:, 0], keys):
            ax.plot([int(r['step']) for r in rows], [float(r[key]) for r in rows], linewidth=.7)
            ax.set(xlabel='优化更新步', ylabel=key, title='原始值（未平滑）')
            ax.grid(alpha=.2)
        fig.tight_layout()
        for ext in ('png', 'svg'):
            path = directory/(filename+'.'+ext)
            if not path.exists():
                fig.savefig(path, dpi=160)
            figs.append({'file': path.name, 'sha256': sha256(path)})
        plt.close(fig)
    components=[json.loads(row['loss_components']) for row in rows]
    keys=sorted({k for row in components for k in row})
    fig,axes=plt.subplots(len(keys),1,figsize=(10,3*len(keys)),squeeze=False)
    for ax,key in zip(axes[:,0],keys):
        ax.plot([int(r['step']) for r in rows],[r.get(key,float('nan')) for r in components],linewidth=.7)
        ax.set(xlabel='优化更新步',ylabel=key,title='实际损失分项（原始值）');ax.grid(alpha=.2)
    fig.tight_layout()
    for ext in ('png','svg'):
        path=directory/('逐步全部损失分项.'+ext)
        if not path.exists():fig.savefig(path,dpi=160)
        figs.append({'file':path.name,'sha256':sha256(path)})
    plt.close(fig)
    epoch_path=directory/'逐轮验证指标.jsonl'
    epochs=[json.loads(line) for line in epoch_path.read_text(encoding='utf-8').splitlines()] if epoch_path.exists() else []
    if epochs:
        names=[k for k in epochs[0]['metrics'] if 'mAP' in k]
        fig,ax=plt.subplots(figsize=(10,4))
        for key in names:
            ax.plot([r['epoch'] for r in epochs],[r['metrics'].get(key,float('nan')) for r in epochs],marker='o',label=key)
        ax.set(xlabel='epoch',ylabel='原生验证指标',title='逐轮真实验证（非逐步验证）');ax.legend();ax.grid(alpha=.2);fig.tight_layout()
        for ext in ('png','svg'):
            path=directory/('逐轮验证变化.'+ext)
            if not path.exists():fig.savefig(path,dpi=160)
            figs.append({'file':path.name,'sha256':sha256(path)})
        plt.close(fig)
    source={'step_csv_sha256':sha256(step_path),'points':len(rows),'smoothing':False,'figures':figs,
            'epoch_jsonl_sha256':sha256(epoch_path) if epochs else None,'validation_epochs':len(epochs)}
    if not (directory/'训练图表来源.json').exists():write_json(directory/'训练图表来源.json',source)
    write_json(directory/'训练图表来源_完整分项.json',source)


def trainer_class(expected_batch):
    import torch
    import psutil
    from ultralytics.models.yolo.detect import DetectionTrainer

    class LoggedDetectionTrainer(DetectionTrainer):
        def _build_train_pipeline(self):
            if int(self.args.batch) != expected_batch:
                raise RuntimeError('固定公平batch发生改变（含框架OOM自动缩批），拒绝继续')
            return super()._build_train_pipeline()

        def preprocess_batch(self, batch):
            torch.cuda.synchronize()
            self._n17_started = time.perf_counter()
            out = super().preprocess_batch(batch)
            self._n17_images = len(out['img'])
            self._n17_targets = len(out['cls'])
            self._n17_files = list(out['im_file'])
            if not hasattr(self, '_n17_step'):
                self._n17_step = 0
                self._n17_elapsed = time.perf_counter()
            return out

        def optimizer_step(self):
            # 与锁定版本原实现同次序：unscale、clip10、step、update、zero、EMA。
            self.scaler.unscale_(self.optimizer)
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0,
                                                 error_if_nonfinite=True)
            components = {k: float(v.detach()) for k, v in self.loss_items.items()}
            loss = float(self.loss.detach())
            lr = [float(g['lr']) for g in self.optimizer.param_groups]
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update(self.model)
            torch.cuda.synchronize()
            self._n17_step += 1
            row = dict(step=self._n17_step, epoch=self.epoch+1,
                       image_count=self._n17_images, target_count=self._n17_targets,
                       loss=loss, loss_components=json.dumps(components, sort_keys=True),
                       training_error_definition='box_and_class_objective_losses',
                       gradient_norm=float(norm), learning_rate=lr[0], learning_rates=json.dumps(lr),
                       step_seconds=time.perf_counter()-self._n17_started,
                       elapsed_seconds=time.perf_counter()-self._n17_elapsed,
                       gpu_allocated_mib=torch.cuda.memory_allocated()/2**20,
                       gpu_reserved_mib=torch.cuda.memory_reserved()/2**20,
                       host_rss_mib=psutil.Process().memory_info().rss/2**20,
                       image_files=json.dumps(self._n17_files, ensure_ascii=False))
            path = self.save_dir/'逐步优化指标.csv'
            exists = path.exists()
            with path.open('a', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
                f.flush()

        def save_metrics(self, metrics):
            super().save_metrics(metrics)
            append_json(self.save_dir/'逐轮验证指标.jsonl',
                        {'epoch': self.epoch+1, 'metrics': {k: float(v) for k, v in metrics.items()},
                         'source': 'native_epoch_validation_and_train_objective'})

    return LoggedDetectionTrainer
