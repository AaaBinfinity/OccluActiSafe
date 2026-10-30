"""从保留的原始CSV自动导出静态图；惰性加载Matplotlib，不接触模型。"""
import csv
import hashlib
import json
from pathlib import Path


def _sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path):
    if not path.exists():return []
    with path.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def render_training_plots(directory, *, output_dir=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['path.simplify']=False
    directory=Path(directory);steps=_rows(directory/'step_metrics.csv');epochs=_rows(directory/'epoch_metrics.csv')
    output=Path(output_dir) if output_dir is not None else directory;output.mkdir(parents=True,exist_ok=True)
    images=[];sources={p.name:_sha(p) for p in (directory/'step_metrics.csv',directory/'epoch_metrics.csv') if p.exists()}
    def finish(fig,name,title):
        fig.suptitle(f'{directory.name} | {title}\nrun: {directory.parent.name}',fontsize=12)
        fig.supxlabel('Raw observations; no smoothing. Annotation-history pipeline; coordinate consistency unresolved.',fontsize=8)
        for ax in fig.axes:
            ax.grid(alpha=.22);ax.spines[['top','right']].set_visible(False)
        for ext in ('png','svg'):
            path=output/f'{name}.{ext}';fig.savefig(path,dpi=160)
            images.append({'file':path.name,'sha256':_sha(path),'title':title})
        plt.close(fig)
    if steps:
        x=[int(r['global_step']) for r in steps]
        fig,axes=plt.subplots(2,1,figsize=(10,7),layout='constrained',sharex=True)
        for key,label in (('loss','total'),('regression_loss','SmoothL1'),('classification_loss','mode CE')):
            axes[0].plot(x,[float(r[key]) for r in steps],label=label,linewidth=.8)
        axes[0].set(ylabel='Batch objective value');axes[0].legend()
        axes[1].plot(x,[float(r['train_minADE_m']) for r in steps],color='#287b51',linewidth=.8)
        axes[1].set(ylabel='Training batch minADE (m)',xlabel='Completed optimizer update (global_step)')
        finish(fig,'step_training','Every optimizer update: loss terms and training error')
        fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
        for ax,key,label in zip(axes.flat,('gradient_norm_before_clip','learning_rate','step_seconds','host_working_set_mib'),
                                ('Gradient norm before clipping','Learning rate','Synchronized update duration (s)','Host working set (MiB)')):
            ax.plot(x,[float(r[key]) for r in steps],linewidth=.8);ax.set(xlabel='global_step',ylabel=label)
        finish(fig,'step_optimization','Every optimizer update: optimization and resource trace')
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        axes[0].plot(x,[float(r['elapsed_seconds']) for r in steps]);axes[0].set(xlabel='global_step',ylabel='Cumulative wall time (s)')
        for key,label in (('gpu_allocated_mib','allocated'),('gpu_reserved_mib','reserved')):
            if all(r[key]!='' for r in steps):axes[1].plot(x,[float(r[key]) for r in steps],label=label)
        axes[1].set(xlabel='global_step',ylabel='GPU allocator memory (MiB)')
        if axes[1].lines:axes[1].legend()
        else:axes[1].text(.5,.5,'GPU memory not applicable (CPU diagnostic)',ha='center',transform=axes[1].transAxes)
        finish(fig,'step_resources','Every optimizer update: elapsed time and GPU memory')
    if epochs:
        x=[int(r['epoch']) for r in epochs]
        fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
        for key,label in (('ADE_m','top1 ADE'),('FDE_m','top1 FDE'),('minADE_m','minADE (K=6)'),('minFDE_m','minFDE (K=6)')):
            axes[0].plot(x,[float(r[key]) for r in epochs],marker='o',label=label)
        axes[0].set(xlabel='Completed epoch',ylabel='Full validation error (m)',xticks=x);axes[0].legend()
        axes[1].plot(x,[float(r['MissRate']) for r in epochs],marker='o',color='#b15b26')
        axes[1].set(xlabel='Completed epoch',ylabel='MissRate (FDE > 2 m)',ylim=(0,1),xticks=x)
        finish(fig,'epoch_validation','Full validation after each epoch (not per-update validation)')
    result={'schema_version':1,'renderer':'Matplotlib','version':matplotlib.__version__,
        'step_rows':len(steps),'epoch_rows':len(epochs),'sources':sources,'images':images,
        'smoothing':False,'downsampling':False,'step_axis':'completed optimizer updates',
        'validation_axis':'completed epochs','alternative_text':'Raw per-update loss/error, optimization/resource traces and epoch validation; inspect corresponding CSV for exact values.'}
    (output/'plot_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result
