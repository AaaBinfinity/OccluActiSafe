"""N14在线侧：严格观测白名单与当前轨迹CV-AEB，不导入CARLA。"""
import math
from occluactisafe.common.schemas import ObservationPacket, ReceivedMessage


class TrajectoryAEB:
    def __init__(self,strategy,*,prefix_s=3.,horizon_s=3.,braking=None):
        if strategy not in ('reference','local_aeb','cooperative_aeb'):raise ValueError('未知策略')
        self.strategy=strategy;self.prefix=prefix_s;self.horizon=horizon_s;self.latched=False
        # IDE/机制夹具默认值对应N13既有实测；实际入口从已核验哈希的标定文件传入。
        self.braking=dict(braking or {'deceleration_mps2':7.378285614969219,
            'response_delay_s':.15,'control_period_s':.05,'margin_m':5.,'max_track_age_s':.5})
        if set(self.braking)!={'deceleration_mps2','response_delay_s','control_period_s','margin_m','max_track_age_s'}:
            raise ValueError('制动预算字段不完整')
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in self.braking.values()):
            raise ValueError('制动预算参数非法')
        if min(self.braking[k] for k in ('deceleration_mps2','control_period_s','max_track_age_s'))<=0:raise ValueError('预算参数必须为正')
        self.cached={};self.first_warning=None;self.first_emergency=None;self.first_target_available=None

    def step(self,observation):
        if set(observation)!={'time_s','ego','local','received'}:raise ValueError('在线输入含未知字段')
        now=observation['time_s'];ego=observation['ego'];targets=[];used=[]
        if set(ego)!={'x','y','vx','vy','yaw','length','width'}:raise ValueError('自车字段非法')
        if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in (now,*ego.values())):
            raise ValueError('时刻或自车状态非有限数值')
        if now<0 or min(ego['length'],ego['width'])<=0:raise ValueError('时刻或自车尺度非法')
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
                for target in record.payload.targets:
                    t=target.track;key=(t.source_id,t.session_id,t.source_track_id)
                    if key not in self.cached or t.state_time>=self.cached[key][0].state_time:
                        self.cached[key]=(t,record.payload.valid_until)
            self.cached={k:(t,until) for k,(t,until) in self.cached.items()
                         if until>=now and now-t.state_time<=self.braking['max_track_age_s']}
            targets.extend(t for t,_ in self.cached.values())
            if self.cached:used.append('roadside-1')
        conflicts=[];known=[];speed=math.hypot(ego['vx'],ego['vy'])
        if targets and self.first_target_available is None:self.first_target_available=now
        budget=speed*(self.braking['response_delay_s']+self.braking['control_period_s'])+speed*speed/(2*self.braking['deceleration_mps2'])+self.braking['margin_m']
        for t in targets:
            if t.state_time>now or not t.velocity_valid:raise ValueError('状态时刻或速度非法')
            age=now-t.state_time;dx=t.x+t.vx*age-ego['x'];dy=t.y+t.vy*age-ego['y']
            vx=t.vx-ego['vx'];vy=t.vy-ego['vy'];den=vx*vx+vy*vy
            tau=max(0.,min(self.horizon,-(dx*vx+dy*vy)/den)) if den>1e-9 else 0.
            # 保守圆包络的首次接触时间；实际OBB与collision sensor仍在评测旁路。
            radius=(math.hypot(ego['length'],ego['width'])+math.hypot(t.length,t.width))/2
            entry=None
            if math.hypot(dx+vx*tau,dy+vy*tau)<=radius:
                dot=dx*vx+dy*vy;c=dx*dx+dy*dy-radius*radius
                entry=0. if c<=0 else max(0.,(-dot-math.sqrt(max(0.,dot*dot-den*c)))/den)
                conflicts.append({'entry_s':entry,'distance_m':speed*entry,'source_id':t.source_id})
            known.append({'source_id':t.source_id,'track_id':t.source_track_id,'state_time':t.state_time,'age_s':age,
                'x':ego['x']+dx,'y':ego['y']+dy,'vx':t.vx,'vy':t.vy,'yaw':t.yaw,'length':t.length,'width':t.width,
                'predicted_entry_s':entry,'prediction_xy':[[t.x+t.vx*(age+i*.1),t.y+t.vy*(age+i*.1)] for i in range(round(self.horizon/.1)+1)]})
        warning=bool(conflicts)
        if warning and self.first_warning is None:self.first_warning=now
        distance=min((c['distance_m'] for c in conflicts),default=None)
        if self.strategy!='reference' and now>=self.prefix and distance is not None and distance<=budget:
            self.latched=True
            if self.first_emergency is None:self.first_emergency=now
        return {'brake':self.latched,'used_sources':sorted(set(used)),
                'warning':warning,'mode':('停车保持' if speed<.1 else '紧急制动') if self.latched else ('风险预警' if warning else '正常行驶'),
                'first_warning_s':self.first_warning,'first_emergency_s':self.first_emergency,
                'first_target_available_s':self.first_target_available,
                'predicted_conflict_s':min(c['entry_s'] for c in conflicts) if conflicts else None,
                'distance_to_conflict_envelope_m':distance,'stopping_budget_m':budget,
                'stopping_margin_m':distance-budget if distance is not None else None,
                'braking_parameters':self.braking,'known_targets':known,'input_mode':'trajectory_input'}
