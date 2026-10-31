"""N14在线侧：严格观测白名单与当前轨迹CV-AEB，不导入CARLA。"""
import math
from occluactisafe.common.schemas import ObservationPacket, ReceivedMessage


class TrajectoryAEB:
    def __init__(self,strategy,*,prefix_s=3.,horizon_s=3.):
        if strategy not in ('reference','local_aeb','cooperative_aeb'):raise ValueError('未知策略')
        self.strategy=strategy;self.prefix=prefix_s;self.horizon=horizon_s;self.latched=False

    def step(self,observation):
        if set(observation)!={'time_s','ego','local','received'}:raise ValueError('在线输入含未知字段')
        now=observation['time_s'];ego=observation['ego'];targets=[];used=[]
        if set(ego)!={'x','y','vx','vy','yaw','length','width'}:raise ValueError('自车字段非法')
        if observation['local'] is not None:
            packet=ObservationPacket.from_dict(observation['local'])
            if (packet.clock_domain,packet.coordinate_frame_id,packet.map_id)!=('carla-n14','map-right-handed','map-A'):raise ValueError('本地身份不匹配')
            if packet.source_available_time>now:raise ValueError('本地观测尚不可用')
            targets.extend(packet.tracks);used.append('vehicle-A')
        if self.strategy=='cooperative_aeb':
            for raw in observation['received']:
                record=ReceivedMessage.from_dict(raw)
                if (record.clock_domain,record.payload.source_id,record.payload.coordinate_frame_id)!=('carla-n14','roadside-1','map-right-handed'):raise ValueError('消息身份不匹配')
                if record.arrival_time>now or record.usable_time>now:raise ValueError('消息尚未可用')
                if record.payload.valid_until<now:continue
                targets.extend(t.track for t in record.payload.targets);used.append(record.payload.source_id)
        conflicts=[]
        for t in targets:
            if t.state_time>now or not t.velocity_valid:raise ValueError('状态时刻或速度非法')
            age=now-t.state_time;dx=t.x+t.vx*age-ego['x'];dy=t.y+t.vy*age-ego['y']
            vx=t.vx-ego['vx'];vy=t.vy-ego['vy'];den=vx*vx+vy*vy
            tau=max(0.,min(self.horizon,-(dx*vx+dy*vy)/den)) if den>1e-9 else 0.
            # 保守圆包络，仅用于TTC触发；实际OBB事件与collision sensor在评测旁路。
            radius=(math.hypot(ego['length'],ego['width'])+math.hypot(t.length,t.width))/2
            if math.hypot(dx+vx*tau,dy+vy*tau)<=radius:conflicts.append(tau)
        if self.strategy!='reference' and now>=self.prefix and conflicts:self.latched=True
        return {'brake':self.latched,'used_sources':sorted(set(used)),
                'predicted_conflict_s':min(conflicts) if conflicts else None,'input_mode':'trajectory_input'}
