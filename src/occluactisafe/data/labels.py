"""N04 仅建立离线未来位置监督；不建立碰撞/近冲突/反事实标签。"""
import numpy as np
from occluactisafe.common.coordinates import Pose2D, world_to_local
from .windows import number


def build_future_labels(records, window, *, future_steps=40, dt=0.1):
    if type(future_steps) is not int or future_steps<1 or number(dt) is None or dt<=0: raise ValueError('未来时域非法')
    ids={identity:i for i,identity in enumerate(window['track_ids'])};now=window['anchor_time']
    pose=Pose2D(*window['anchor_pose']);values=np.zeros((len(ids),future_steps,2));mask=np.zeros((len(ids),future_steps),dtype=bool)
    for row in records:
        identity=row.get('track_id');t=number(row.get('observation_time'))
        if identity not in ids or t is None or t<=now: continue
        if row.get('clock_domain')!=window['clock_domain']:raise ValueError('监督时钟域错误')
        j=round((t-now)/dt)-1
        if 0<=j<future_steps and abs(t-(now+(j+1)*dt))<=1e-6:
            px,py=number(row.get('x')),number(row.get('y'))
            if px is not None and py is not None and not mask[ids[identity],j]:
                values[ids[identity],j]=world_to_local(px,py,pose);mask[ids[identity],j]=True
    return {'future_positions':values,'future_mask':mask,'future_times':np.arange(1,future_steps+1)*dt}
