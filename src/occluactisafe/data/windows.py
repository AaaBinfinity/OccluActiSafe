"""只使用决策时已可用历史的窗口与特征；不接收监督未来对象。"""
from collections import defaultdict
import hashlib
import json
import math
import numpy as np
from occluactisafe.common.coordinates import Pose2D, world_to_local

FEATURES=("x","y","vx","vy","ax","ay","sin_heading","cos_heading","length","width")


def number(value):
    if value is None or isinstance(value,bool): return None
    try: value=float(value)
    except (TypeError,ValueError): return None
    return value if math.isfinite(value) else None


def available(row, now, domain):
    if row.get("clock_domain") != domain:
        raise ValueError("历史记录时钟域不一致")
    times={k:number(row.get(k)) for k in ("observation_time","source_available_time","arrival_time","usable_time")}
    if any(v is None for v in times.values()): raise ValueError("缺少有限的采集/可用时刻")
    if not times['observation_time'] <= times['source_available_time'] <= times['arrival_time'] <= times['usable_time']:
        raise ValueError("采集、源处理、到达、可用顺序非法")
    if any(t>now for t in times.values()): return False
    state=number(row.get('state_time',times['observation_time']))
    if state is None or state>now: return False
    if 'valid_until' in row and (number(row['valid_until']) is None or number(row['valid_until'])<now): return False
    return True


def input_hash(window):
    digest=hashlib.sha256()
    for key in ('features','feature_mask','observed_mask','history_times','actual_dt','categories','prediction_target_mask'):
        a=np.ascontiguousarray(window[key]);digest.update(key.encode()+a.dtype.str.encode()+str(a.shape).encode()+a.tobytes())
    meta={k:window[k] for k in ('track_ids','anchor_time','anchor_pose','clock_domain','selected_records')}
    digest.update(json.dumps(meta,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
    return digest.hexdigest()


def build_causal_window(records, decision_time, anchor_pose, *, clock_domain,
                        history_steps=20, dt=0.1):
    """按真实采集时刻填格点，无插值；缺失槽为零占位且mask=False。

    record必须显式携带四种时间和clock_domain；仅位置齐全的合法历史可产生目标。
    速度优先用可靠原字段，否则用相邻过去观测差分；加速度也只用过去速度。
    actual_dt记录相邻真实观测间隔，缺格点不伪造观测。anchor_pose由调用方的合法
    本车定位提供，标签与模型forward不在本节点实现。
    """
    if type(history_steps) is not int or history_steps<1 or not isinstance(anchor_pose,Pose2D): raise ValueError('窗口或锚点非法')
    now=number(decision_time); dt=number(dt)
    if now is None or dt is None or dt<=0: raise ValueError('时间或步长非法')
    past=defaultdict(list)
    for row in records:
        if not available(row,now,clock_domain): continue
        if number(row.get('x')) is None or number(row.get('y')) is None: continue
        identity=row.get('track_id')
        if not isinstance(identity,str) or not identity: raise ValueError('track_id必须明确')
        past[identity].append(row)
    start=now-(history_steps-1)*dt
    # 不让只有早于整个历史窗口的目标进入当前输入。
    ids=sorted(k for k,rows in past.items() if any(r['observation_time']>=start-1e-7 for r in rows))
    shape=(len(ids),history_steps,len(FEATURES)); x=np.zeros(shape,dtype=np.float64);mask=np.zeros(shape,dtype=bool)
    obs=np.zeros(shape[:2],dtype=bool);delta=np.zeros(shape[:2],dtype=np.float64)
    categories=np.full(len(ids),3,dtype=np.int64);targets=np.zeros(len(ids),dtype=bool);selected=[]
    c,s=math.cos(anchor_pose.yaw),math.sin(anchor_pose.yaw)
    for i,identity in enumerate(ids):
        previous=None;previous_velocity=None;seen=set()
        rows=sorted(past[identity],key=lambda r:(r['observation_time'],r.get('source_row',0)))
        for row in rows:
            t=float(row['observation_time'])
            if t in seen: continue
            seen.add(t); px,py=float(row['x']),float(row['y'])
            gap=None if previous is None else t-previous[0]
            vx,vy=number(row.get('vx')),number(row.get('vy'))
            if not row.get('velocity_reliable',False) or vx is None or vy is None:
                vx=vy=None
                if previous is not None and gap>0: vx,vy=(px-previous[1])/gap,(py-previous[2])/gap
            ax=ay=None
            if vx is not None and previous_velocity is not None and t>previous_velocity[0]:
                ax=(vx-previous_velocity[1])/(t-previous_velocity[0]);ay=(vy-previous_velocity[2])/(t-previous_velocity[0])
            j=round((t-start)/dt)
            if 0<=j<history_steps and abs(t-(start+j*dt))<=1e-6:
                lx,ly=world_to_local(px,py,anchor_pose);values=[lx,ly,None,None,None,None,None,None,None,None]
                if vx is not None: values[2:4]=[c*vx+s*vy,-s*vx+c*vy]
                if ax is not None: values[4:6]=[c*ax+s*ay,-s*ax+c*ay]
                yaw=number(row.get('heading'))
                if yaw is not None and (vx is None or math.hypot(vx,vy)>=0.2): values[6:8]=[math.sin(yaw-anchor_pose.yaw),math.cos(yaw-anchor_pose.yaw)]
                for k,name in ((8,'length'),(9,'width')):
                    v=number(row.get(name))
                    if v is not None and v>0: values[k]=v
                for k,v in enumerate(values):
                    if v is not None: x[i,j,k]=v;mask[i,j,k]=True
                obs[i,j]=True;delta[i,j]=0 if gap is None else gap
                categories[i]={'VEHICLE':0,'PEDESTRIAN':1,'BICYCLE':2}.get(str(row.get('category','')).upper(),3)
                targets[i]=row.get('tag')=='TARGET_AGENT'
                selected.append((identity,row.get('source_row',0),t))
            previous=(t,px,py)
            previous_velocity=(t,vx,vy) if vx is not None else None
    window={'features':x,'feature_mask':mask,'observed_mask':obs,'actual_dt':delta,
            'history_times':np.arange(-(history_steps-1),1,dtype=np.float64)*dt,
            'categories':categories,'prediction_target_mask':targets,'track_ids':ids,
            'anchor_time':now,'anchor_pose':[anchor_pose.x,anchor_pose.y,anchor_pose.yaw],
            'clock_domain':clock_domain,'selected_records':selected}
    window['input_hash']=input_hash(window)
    return window
