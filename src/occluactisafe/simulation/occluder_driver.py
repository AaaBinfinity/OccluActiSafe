"""N14同向遮挡车脚本：仅使用自身状态及冻结停车线。"""
import math


class OccluderDriver:
    def __init__(self,spawn,conflict_xy,*,speed_mps,brake_front_m):
        self.spawn=spawn;self.conflict=conflict_xy;self.speed=speed_mps;self.brake_front=brake_front_m
        self.integral=0.;self.stopping=False

    def step(self,state,*,initializing=False):
        yaw=-math.radians(self.spawn['yaw']);dx=state['x']-self.spawn['x'];dy=state['y']+self.spawn['y']
        cross=-math.sin(yaw)*dx+math.cos(yaw)*dy
        error=(yaw-state['yaw']+math.pi)%(2*math.pi)-math.pi
        ahead=(state['x']-self.conflict[0])*math.cos(yaw)+(state['y']+self.conflict[1])*math.sin(yaw)
        front_distance=-ahead-state['length']/2
        if not initializing and front_distance<=self.brake_front:self.stopping=True
        if self.stopping:return {'throttle':0.,'brake':1.,'steer':0.}
        speed=math.hypot(state['vx'],state['vy']);e=self.speed-speed
        candidate=self.integral+e*.05;raw=.5+.25*e+.05*candidate
        if 0<=raw<=1:self.integral=candidate
        pedal=max(-1.,min(1.,.5+.25*e+.05*self.integral))
        # 自身直线车道反馈；这是外生场景车辆驾驶脚本，不套用Audi制动标定。
        steer=-max(-.15,min(.15,.5*error-.06*cross))
        return {'throttle':max(0.,pedal),'brake':max(0.,-pedal),'steer':steer}
