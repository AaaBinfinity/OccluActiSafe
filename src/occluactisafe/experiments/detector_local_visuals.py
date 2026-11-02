"""N17本机训练图集：原生图中文归位、原值分组绘图和可复核原始来源。"""
import csv
import json
import math
from pathlib import Path
import re

import numpy as np

from .run_store import write_json, append_json, sha256


def pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .visual_fonts import configure_plot_fonts
    configure_plot_fonts()
    return plt


def save_figure(fig, path, figures):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ('png', 'svg'):
        target = path.with_suffix('.'+ext); fig.savefig(target, dpi=180)
        figures.append({'file': str(target), 'sha256': sha256(target)})


def native_image_name(stem):
    names = {'confusion_matrix':'混淆矩阵_原生', 'confusion_matrix_normalized':'归一化混淆矩阵_原生',
             'labels':'标签分布_原生', 'labels_correlogram':'标签相关图_原生', 'results':'训练结果总览_原生'}
    if stem in names: return names[stem]
    for prefix in ('Box', ''):
        for code, label in (('F1','F1曲线'),('P','精确率曲线'),('R','召回率曲线'),('PR','精确率召回率曲线')):
            if stem == prefix+code+'_curve': return label+'_原生'
    match = re.fullmatch(r'train_batch(\d+)', stem)
    if match: return '训练批次示例_'+match[1].zfill(6)
    match = re.fullmatch(r'val_batch(\d+)_(labels|pred)', stem)
    if match: return '验证批次'+match[1]+'_'+('真实标签' if match[2]=='labels' else '模型预测')
    if any('\u4e00' <= c <= '\u9fff' for c in stem): return stem
    raise ValueError('尚未声明中文名称的原生图：'+stem)


def normalize_native_images(directory):
    """只处理本次任务直接输出，不复制旧实验、不改已有归档；JPG原件另保留PNG。"""
    from PIL import Image
    directory = Path(directory); records=[]
    for source in sorted(directory.iterdir()):
        if source.suffix.lower() not in ('.png','.jpg','.jpeg','.svg'): continue
        target = source.with_stem(native_image_name(source.stem))
        if target != source:
            if target.exists(): raise FileExistsError(target)
            source.rename(target)
        item={'file':target.name,'original_file':source.name,'sha256':sha256(target)}
        if target.suffix.lower() in ('.jpg','.jpeg'):
            png=target.with_suffix('.png')
            if not png.exists():
                with Image.open(target) as im: im.convert('RGB').save(png)
            item['png_file']=png.name;item['png_sha256']=sha256(png)
        records.append(item)
    write_json(directory/'原生图片中文名称与哈希.json',{'images':records,'jpeg_originals_retained':True})
    return records


def label_diagnostics(directory, boxes, classes, names):
    directory=Path(directory);boxes=np.asarray(boxes,dtype=float).reshape(-1,4);classes=np.asarray(classes).reshape(-1)
    if len(boxes)!=len(classes) or not np.isfinite(boxes).all():raise ValueError('标签统计输入不一致或非有限')
    counts=[int(np.sum(classes==i)) for i in range(len(names))]
    raw=directory/'训练标签几何原值.npz';np.savez_compressed(raw,xywh_normalized=boxes,class_ids=classes.astype(int))
    write_json(directory/'训练标签分类计数.json',{'names':names,'counts':counts,'instances':len(boxes),
               'zero_gt_classes':[names[i] for i,v in enumerate(counts) if not v],
               'units':'原始标签归一化xywh，非米制','raw_sha256':sha256(raw)})
    plt=pyplot();figures=[]
    fig,ax=plt.subplots(figsize=(9,4));ax.bar(list(names.values()),counts);ax.set(ylabel='标注实例数',title='训练标签类别分布（原始计数）')
    save_figure(fig,directory/'标签类别分布',figures);plt.close(fig)
    labels=['中心x/图宽','中心y/图高','框宽/图宽','框高/图高']
    fig,axes=plt.subplots(4,4,figsize=(12,11))
    for i in range(4):
        for j in range(4):
            ax=axes[i,j]
            if i==j:ax.hist(boxes[:,j],bins=40)
            elif len(boxes):ax.hist2d(boxes[:,j],boxes[:,i],bins=40,cmap='Blues')
            if i==3:ax.set_xlabel(labels[j])
            if j==0:ax.set_ylabel(labels[i])
    fig.suptitle('训练标签几何相关图（全部标注的二维直方图）')
    save_figure(fig,directory/'标签几何相关图',figures);plt.close(fig)
    for item in figures:item['file']=Path(item['file']).name
    write_json(directory/'标签图表来源.json',{'raw_file':raw.name,'raw_sha256':sha256(raw),'instances':len(boxes),'figures':figures})


def capture_validation(validator, directory, epoch, phase):
    """只在真实epoch验证/最佳权重验证结束时捕获；无GT类用null，不填假召回。"""
    directory=Path(directory);metrics=validator.metrics;box=metrics.box
    names={int(k):v for k,v in metrics.names.items()};ids=np.asarray(box.ap_class_index,dtype=int).tolist()
    curves=[]
    for x,y,xlabel,ylabel in metrics.curves_results:
        a,b=np.asarray(x),np.asarray(y)
        if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('原生验证曲线含非有限值')
        curves.append({'x':a.tolist(),'y':b.tolist(),'xlabel':xlabel,'ylabel':ylabel,
                       'valid':bool(a.size and b.size),'class_ids':ids})
    counts=np.asarray(getattr(metrics,'nt_per_class',[])).tolist()
    table=[]
    for index,name in names.items():
        if index in ids:
            j=ids.index(index)
            item={'class_id':index,'name':name,'gt':int(counts[index]) if index<len(counts) else None,
                  'precision':float(box.p[j]),'recall':float(box.r[j]),'f1':float(box.f1[j]),
                  'ap50':float(box.ap50[j]),'ap50_95':float(box.ap[j]),'valid':True}
        else:item={'class_id':index,'name':name,'gt':0,'precision':None,'recall':None,'f1':None,'ap50':None,'ap50_95':None,'valid':False,'reason':'此验证集无可计算GT类别'}
        table.append(item)
    cm=getattr(validator,'confusion_matrix',None)
    result={'epoch':epoch,'phase':phase,'names':names,'class_ids_with_gt':ids,'classes':table,'curves':curves,
            'confusion_matrix':np.asarray(cm.matrix).tolist() if cm is not None and validator.args.plots else None,
            'native_confusion_confidence':getattr(cm,'conf',None),'native_confusion_iou':getattr(cm,'iou_thres',None),
            'category_id_map':list(validator.class_map),'native_metrics':metrics.results_dict,
            'note':'原生验证定义；不是N16固定0.25操作点评分，不是逐步验证'}
    target=directory/'逐轮验证原值'/('最佳权重验证.json' if phase=='best_checkpoint' else f'第{epoch:04d}轮.json')
    write_json(target,result)
    if phase=='epoch':append_json(directory/'逐轮分类验证指标.jsonl',{'epoch':epoch,'classes':table,'source':str(target),'sha256':sha256(target)})
    return result


def validation_figures(directory, raw, figures):
    plt=pyplot();directory=Path(directory);names={int(k):v for k,v in raw['names'].items()}
    chinese={'Recall':'召回率','Precision':'精确率','Confidence':'置信度','F1':'F1'}
    for curve in raw['curves']:
        fig,ax=plt.subplots(figsize=(8,5));x=np.asarray(curve['x']);y=np.asarray(curve['y'])
        if curve['valid']:
            y=np.atleast_2d(y)
            for i,values in enumerate(y):
                if len(values)!=len(x):raise ValueError('验证曲线长度与原始横轴不一致')
                ax.plot(x,values,label=names.get(curve['class_ids'][i],str(i)) if i<len(curve['class_ids']) else str(i))
            ax.legend();ax.set(xlim=(0,1),ylim=(0,1))
        else:ax.text(.5,.5,'当前验证曲线不可估计；保留原始空值',ha='center',transform=ax.transAxes)
        xlabel,ylabel=chinese[curve['xlabel']],chinese[curve['ylabel']]
        ax.set(xlabel=xlabel,ylabel=ylabel,title='最佳权重原生验证（真实原值）');ax.grid(alpha=.2)
        save_figure(fig,directory/(ylabel+'_'+xlabel+'曲线'),figures);plt.close(fig)
    if raw['confusion_matrix'] is not None:
        cm=np.asarray(raw['confusion_matrix'],dtype=float)
        for normalize in (False,True):
            display=cm/np.maximum(cm.sum(axis=0,keepdims=True),1) if normalize else cm
            fig,ax=plt.subplots(figsize=(8,7));im=ax.imshow(display,cmap='Blues');fig.colorbar(im,ax=ax)
            ticks=list(names.values())+['背景'];ax.set(xticks=range(len(ticks)),yticks=range(len(ticks)),xticklabels=ticks,yticklabels=ticks,xlabel='真实类别',ylabel='预测类别')
            ax.tick_params(axis='x',rotation=35)
            for i in range(len(cm)):
                for j in range(len(cm)):ax.text(j,i,f'{display[i,j]:.2f}' if normalize else str(int(cm[i,j])),ha='center',va='center',fontsize=8)
            ax.set_title('原生验证混淆矩阵'+('（按真实类别归一化）' if normalize else '（计数）'))
            save_figure(fig,directory/('混淆矩阵_归一化' if normalize else '混淆矩阵_计数'),figures);plt.close(fig)


def comprehensive_plots(directory, points_per_page=20000):
    """逐CSV流式分页，全部原值均入图，不将数百万行转成Python大列表。"""
    directory=Path(directory);source=directory/'逐步优化指标.csv';figures=[];points=0
    plt=pyplot()
    if source.exists():
        with source.open(encoding='utf-8',newline='') as handle:
            reader=csv.DictReader(handle);page=[];number=0
            def draw(rows,number):
                x=[int(r['step']) for r in rows]
                specs=[('逐步训练损失',['loss']),('逐步梯度和耗时',['gradient_norm','step_seconds']),
                       ('逐步显存和学习率',['gpu_allocated_mib','gpu_reserved_mib','host_rss_mib','learning_rate'])]
                components=[json.loads(r['loss_components']) for r in rows]
                keys=sorted({k for c in components for k in c})
                specs.append(('逐步全部损失分项',keys))
                for title,fields in specs:
                    fig,axes=plt.subplots(len(fields),1,figsize=(11,3*len(fields)),squeeze=False)
                    for ax,key in zip(axes[:,0],fields):
                        values=[c.get(key,math.nan) for c in components] if title=='逐步全部损失分项' else [float(r[key]) for r in rows]
                        ax.plot(x,values,linewidth=.65);ax.set(xlabel='已执行的优化更新步',ylabel=key,title=f'原值未平滑：第{x[0]}～{x[-1]}步');ax.grid(alpha=.2)
                    save_figure(fig,directory/'逐步原值分页'/f'第{number:04d}组'/title,figures);plt.close(fig)
            for row in reader:
                page.append(row);points+=1
                if len(page)>=points_per_page:number+=1;draw(page,number);page=[]
            if page:number+=1;draw(page,number)
    epoch_path=directory/'逐轮验证指标.jsonl'
    epochs=[json.loads(line) for line in epoch_path.read_text(encoding='utf-8').splitlines()] if epoch_path.exists() else []
    if epochs:
        for title,keys in [('逐轮训练与验证损失',sorted({k for e in epochs for k in e['metrics'] if 'loss' in k})),
                           ('逐轮精确率召回率和平均精度',sorted({k for e in epochs for k in e['metrics'] if 'metrics/' in k})),
                           ('逐轮学习率变化',sorted({k for e in epochs for k in e['metrics'] if 'lr/' in k}))]:
            if not keys:continue
            fig,ax=plt.subplots(figsize=(11,5))
            for key in keys:ax.plot([e['epoch'] for e in epochs],[e['metrics'].get(key,math.nan) for e in epochs],label=key)
            ax.set(xlabel='已完成epoch',ylabel='原生指标原值',title=title);ax.legend(fontsize=8);ax.grid(alpha=.2)
            save_figure(fig,directory/title,figures);plt.close(fig)
    trace=directory/'早停判定.jsonl'
    if trace.exists():
        rows=[json.loads(s) for s in trace.read_text(encoding='utf-8').splitlines()]
        if rows:
            fig,axes=plt.subplots(2,1,figsize=(11,7))
            for key in ('fitness','best_fitness'):axes[0].plot([r['epoch'] for r in rows],[r[key] for r in rows],label=key)
            axes[0].legend();axes[0].set(ylabel='原生fitness')
            axes[1].plot([r['epoch'] for r in rows],[r['epochs_without_improvement'] for r in rows]);axes[1].axhline(rows[-1]['patience'],color='red',label='patience')
            axes[1].set(xlabel='epoch',ylabel='连续未改善轮数');axes[1].legend();save_figure(fig,directory/'早停依据与最佳指标',figures);plt.close(fig)
    best=directory/'逐轮验证原值/最佳权重验证.json'
    if best.exists():validation_figures(directory,read_json_local(best),figures)
    for item in figures:item['file']=Path(item['file']).relative_to(directory).as_posix()
    result={'points':points,'points_per_page':points_per_page,'all_step_values_plotted':True,'smoothing':False,
            'validation_epochs':len(epochs),'step_csv_sha256':sha256(source) if source.exists() else None,
            'epoch_jsonl_sha256':sha256(epoch_path) if epoch_path.exists() else None,'figures':figures}
    write_json(directory/'训练图表来源_完整分项.json',result)
    return result


def read_json_local(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
