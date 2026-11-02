"""最佳权重的固定操作点错误样例；只读model_val，不把原生曲线当统一评分。"""
from collections import defaultdict
import csv
from pathlib import Path

from .run_store import read_json,write_json,sha256


def detection_gallery(cfg,directory,maximum=12):
    from .detector_data import load_index
    from .detector_evaluation import overlay
    from occluactisafe.evaluation.detection_metrics import GroundTruthObject,MetricFrame,score_detections
    from occluactisafe.perception.types import Detection
    import numpy as np
    from PIL import Image,ImageOps,ImageDraw,ImageFont
    from .visual_fonts import chinese_font
    directory=Path(directory);prediction=directory/'predictions.json';raw=directory/'逐轮验证原值/最佳权重验证.json'
    if not prediction.exists() or not raw.exists():raise FileNotFoundError('缺少实际最佳权重预测，不能构造错误样例')
    metadata=read_json(raw);mapping={value:cfg['classes'][i] for i,value in enumerate(metadata['category_id_map'])}
    rows,_=load_index(cfg);rows=[r for r in rows if r['role']=='model_val'];known={r['key'] for r in rows}
    detections=defaultdict(list)
    for item in read_json(prediction):
        key=str(item['image_id'])
        if key not in known:raise ValueError('预测JSON含验证清单外帧')
        if item['category_id'] not in mapping:raise ValueError('原生预测类别映射未知')
        if item['score']<.25:continue
        x,y,w,h=item['bbox'];detections[key].append(Detection((x,y,x+w,y+h),item['score'],mapping[item['category_id']]))
    scores=[]
    for row in rows:
        gt=tuple(GroundTruthObject(tuple(g['xyxy']),g['class_id'],None) for g in row['gt'])
        result=score_detections([MetricFrame(row['key'],gt,tuple(detections[row['key']]))])
        scores.append({'key':row['key'],'domain':row['domain'],'camera_side':row['camera_side'],
                       'gt':len(gt),'tp':result['tp'],'fp':result['fp'],'fn':result['fn']})
    output=directory/'误检漏检与检测样例';output.mkdir(exist_ok=True)
    with (output/'逐帧固定操作点诊断.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(scores[0]));writer.writeheader();writer.writerows(scores)
    selected=sorted(scores,key=lambda r:(-r['fn']-r['fp'],r['key']))[:maximum]
    indexed={r['key']:r for r in rows};images=[]
    for i,item in enumerate(selected):
        row=indexed[item['key']]
        with Image.open(row['image']) as im:rgb=np.asarray(im.convert('RGB'))
        overlay(rgb,row,detections[row['key']],output,i)
        target=output/f'检测叠加_{row["domain"]}_{i:04d}.png'
        with Image.open(target) as original:canvas=ImageOps.expand(original.convert('RGB'),border=(0,0,0,75),fill='black')
        draw=ImageDraw.Draw(canvas);font=ImageFont.truetype(str(chinese_font()),20)
        local=cfg['local_profile'];scope='短程联调，非正式泛化结果' if local['development_only'] else '本次单模型验证诊断'
        draw.text((10,canvas.height-68),f"{local['model_id']} | seed={local['seed']} | {scope} | TP={item['tp']} FP={item['fp']} FN={item['fn']}",font=font,fill='white')
        draw.text((10,canvas.height-35),'原图像素未缩放；置信度≥0.25，IoU=0.5；完整来源与哈希见同目录JSON',font=font,fill='white')
        canvas.save(target)
        images.append({**item,'file':target.name,'sha256':sha256(target),'input_image_sha256':row['image_sha256'],
                       'input_file':row['image'],'pixel_size':[row['width'],row['height']],'render_pixel_size':list(canvas.size),
                       'classification':'误检漏检样例' if item['fp']+item['fn'] else '无误检漏检的实际示例'})
    result={'scope':'仅固定0.25/IoU0.5离线诊断，非完整选型/跟踪或闭环收益','validation_frames':len(rows),
            'gt':sum(r['gt'] for r in scores),'tp':sum(r['tp'] for r in scores),'fp':sum(r['fp'] for r in scores),'fn':sum(r['fn'] for r in scores),
            'source_predictions_sha256':sha256(prediction),'best_validation_sha256':sha256(raw),
            'score_threshold':.25,'iou_threshold':.5,'images':images}
    write_json(output/'诊断图来源与指标.json',result)
    return result
