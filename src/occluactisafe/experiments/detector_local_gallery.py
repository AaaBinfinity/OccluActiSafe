"""最佳权重的固定操作点错误样例；只读model_val，不把原生曲线当统一评分。"""
from collections import Counter,defaultdict
import csv
import math
from pathlib import Path

from .run_store import read_json,write_json,sha256


def gallery_predictions(predictions,mapping,known):
    """零面积框保留原值；操作点内按IoU=0计FP，不传入严格Detection契约。"""
    from occluactisafe.perception.types import Detection
    detections=defaultdict(list);zero=[];zero_fp=Counter()
    for index,item in enumerate(predictions):
        key=str(item['image_id'])
        if key not in known:raise ValueError('预测JSON含验证清单外帧')
        if item['category_id'] not in mapping:raise ValueError('原生预测类别映射未知')
        score=item['score'];box=item['bbox']
        if type(score) not in (float,int) or not math.isfinite(score) or not 0<=score<=1:
            raise ValueError(f'预测{index}置信度必须是[0,1]有限数')
        if (not isinstance(box,(list,tuple)) or len(box)!=4
                or any(type(v) not in (float,int) or not math.isfinite(v) for v in box)):
            raise ValueError(f'预测{index}框必须是四个有限数')
        x,y,w,h=box
        if w<0 or h<0 or not math.isfinite(x+w) or not math.isfinite(y+h):
            raise ValueError(f'预测{index}框尺寸为负或坐标溢出')
        if w==0 or h==0 or x+w==x or y+h==y:
            counted=score>=.25
            zero.append({'prediction_index':index,'image_id':key,'category_id':item['category_id'],
                         'class_id':mapping[item['category_id']],'bbox_xywh':list(box),'score':score,
                         'counted_as_fp':counted,'reason':'zero_area_after_export',
                         'treatment':'IoU=0，阈值内计FP；绘制位置线，不扩框、不改原预测'})
            zero_fp[key]+=int(counted)
        elif score>=.25:
            detections[key].append(Detection((x,y,x+w,y+h),score,mapping[item['category_id']]))
    return detections,zero,zero_fp


def detection_gallery(cfg,directory,maximum=12):
    from .detector_data import load_index
    from .detector_evaluation import overlay
    from occluactisafe.evaluation.detection_metrics import GroundTruthObject,MetricFrame,score_detections
    import numpy as np
    from PIL import Image,ImageOps,ImageDraw,ImageFont
    from .visual_fonts import chinese_font
    directory=Path(directory);prediction=directory/'predictions.json';raw=directory/'逐轮验证原值/最佳权重验证.json'
    if not prediction.exists() or not raw.exists():raise FileNotFoundError('缺少实际最佳权重预测，不能构造错误样例')
    metadata=read_json(raw);mapping={value:cfg['classes'][i] for i,value in enumerate(metadata['category_id_map'])}
    rows,_=load_index(cfg);rows=[r for r in rows if r['role']=='model_val'];known={r['key'] for r in rows}
    if type(maximum) is not int or maximum<1:raise ValueError('样例上限必须为正整数')
    if not rows or len(known)!=len(rows):raise ValueError('model_val清单为空或含重复帧')
    detections,zero,zero_fp=gallery_predictions(read_json(prediction),mapping,known)
    scores=[]
    for row in rows:
        gt=tuple(GroundTruthObject(tuple(g['xyxy']),g['class_id'],None) for g in row['gt'])
        result=score_detections([MetricFrame(row['key'],gt,tuple(detections[row['key']]))])
        scores.append({'key':row['key'],'domain':row['domain'],'camera_side':row['camera_side'],
                       'gt':len(gt),'tp':result['tp'],'fp':result['fp']+zero_fp[row['key']],'fn':result['fn'],
                       'fp_renderable':result['fp'],'fp_zero_area':zero_fp[row['key']]})
    output=directory/'误检漏检与检测样例';output.mkdir(exist_ok=False)
    write_json(output/'零面积预测原始记录.json',{'source_predictions_sha256':sha256(prediction),'records':zero,
                                              'score_threshold':.25,'iou_for_zero_area':0})
    with (output/'逐帧固定操作点诊断.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(scores[0]));writer.writeheader();writer.writerows(scores)
    selected=sorted(scores,key=lambda r:(-r['fn']-r['fp'],r['key']))[:maximum]
    indexed={r['key']:r for r in rows};images=[]
    for i,item in enumerate(selected):
        row=indexed[item['key']]
        if sha256(row['image'])!=row['image_sha256']:raise ValueError('诊断原图SHA改变')
        with Image.open(row['image']) as im:rgb=np.asarray(im.convert('RGB'))
        overlay(rgb,row,detections[row['key']],output,i)
        target=output/f'检测叠加_{row["domain"]}_{i:04d}.png'
        with Image.open(target) as original:canvas=ImageOps.expand(original.convert('RGB'),border=(0,0,0,75),fill='black')
        draw=ImageDraw.Draw(canvas);font=ImageFont.truetype(str(chinese_font()),20)
        local=cfg['local_profile'];scope='短程联调，非正式泛化结果' if local['development_only'] else '本次单模型验证诊断'
        draw.text((10,canvas.height-68),f"{local['model_id']} | seed={local['seed']} | {scope} | TP={item['tp']} FP={item['fp']} FN={item['fn']}",font=font,fill='white')
        draw.text((10,canvas.height-35),f"原图未缩放；置信度≥0.25，IoU=0.5；FP中含零面积框{item['fp_zero_area']}，详见原始记录",font=font,fill='white')
        canvas.save(target)
        images.append({**item,'file':target.name,'sha256':sha256(target),'input_image_sha256':row['image_sha256'],
                       'input_file':row['image'],'pixel_size':[row['width'],row['height']],'render_pixel_size':list(canvas.size),
                       'classification':'误检漏检样例' if item['fp']+item['fn'] else '无误检漏检的实际示例'})
    zero_images=[];affected=sorted({r['image_id'] for r in zero if r['counted_as_fp']})
    for i,key in enumerate(affected[:maximum]):
        row=indexed[key]
        if sha256(row['image'])!=row['image_sha256']:raise ValueError('诊断原图SHA改变')
        items=[r for r in zero if r['image_id']==key and r['counted_as_fp']]
        with Image.open(row['image']) as im:canvas=ImageOps.expand(im.convert('RGB'),border=(0,0,0,100),fill='black')
        draw=ImageDraw.Draw(canvas);font=ImageFont.truetype(str(chinese_font()),20)
        for record in items:
            x,y,w,h=record['bbox_xywh'];draw.line([(x,y),(x+w,y+h)],fill=(255,170,0),width=4)
        draw.text((10,canvas.height-93),'零面积预测位置复核 | 橙色为原始退化框的位置线，不扩为有效矩形',font=font,fill='white')
        draw.text((10,canvas.height-62),f"帧 {key} | 阈值内零面积框{len(items)}个，均按IoU=0计FP",font=font,fill='white')
        draw.text((10,canvas.height-31),f"首框xywh={items[0]['bbox_xywh']} score={items[0]['score']}",font=font,fill='white')
        target=output/f'零面积框位置复核_{i:04d}.png';canvas.save(target)
        zero_images.append({'key':key,'file':target.name,'sha256':sha256(target),'input_file':row['image'],
                            'input_image_sha256':row['image_sha256'],'pixel_size':[row['width'],row['height']],
                            'render_pixel_size':list(canvas.size),'prediction_indices':[r['prediction_index'] for r in items]})
    result={'scope':'仅固定0.25/IoU0.5离线诊断，非完整选型/跟踪或闭环收益','validation_frames':len(rows),
            'gt':sum(r['gt'] for r in scores),'tp':sum(r['tp'] for r in scores),'fp':sum(r['fp'] for r in scores),'fn':sum(r['fn'] for r in scores),
            'source_predictions_sha256':sha256(prediction),'best_validation_sha256':sha256(raw),
            'best_checkpoint_sha256':sha256(directory/'weights/best.pt'),
            'configuration_sha256':cfg['_config_sha256'],'source_fingerprint_sha256':cfg['local_profile']['source_fingerprint_sha256'],
            'score_threshold':.25,'iou_threshold':.5,'images':images,
            'zero_area_total':len(zero),'zero_area_at_threshold':sum(zero_fp.values()),
            'renderable_at_threshold':sum(len(v) for v in detections.values()),
            'zero_area_policy':'保留全部原值，阈值内按IoU=0计FP；阈值外不计分，不改变GT分母',
            'zero_area_records_file':'零面积预测原始记录.json',
            'zero_area_records_sha256':sha256(output/'零面积预测原始记录.json'),
            'zero_area_affected_frames':len(affected),'zero_area_images':zero_images,
            'gallery_implementation_sha256':sha256(__file__)}
    write_json(output/'诊断图来源与指标.json',result)
    return result
