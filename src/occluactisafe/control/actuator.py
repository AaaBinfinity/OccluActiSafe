"""N13 标定表驱动执行器：范围外拒绝，积分抗饱和。"""
import math


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} 必须为有限数值')
    return float(value)


def _table(points, name):
    if not isinstance(points, (list, tuple)) or len(points) < 2:
        raise ValueError(f'{name} 至少需要两个标定点')
    table = [(_number(p[0],name),_number(p[1],name)) for p in points]
    if any(a[0] >= b[0] or a[1] > b[1] for a,b in zip(table,table[1:])):
        raise ValueError(f'{name} 标定点必须按输入严格递增且输出单调')
    return table


def _interpolate(points, value):
    if not points[0][0] <= value <= points[-1][0]: raise ValueError('超出已标定范围')
    for left,right in zip(points,points[1:]):
        if left[0] <= value <= right[0]:
            return left[1]+(right[1]-left[1])*(value-left[0])/(right[0]-left[0])
    return points[-1][1]


class CalibratedActuator:
    """输入加速度m/s²、转角rad和速度m/s；仅接受提供的实测表。"""
    def __init__(self, calibration, *, kp, ki, steering_rate_per_s):
        self.accel=_table(calibration['acceleration_to_pedal'],'加速度表')
        self.steering=_table(calibration['angle_to_steer'],'转向表')
        self.speed_min=_number(calibration['speed_min_mps'],'最小速度')
        self.speed_max=_number(calibration['speed_max_mps'],'最大速度')
        self.kp=_number(kp,'kp');self.ki=_number(ki,'ki');self.rate=_number(steering_rate_per_s,'转向速率')
        if self.speed_min<0 or self.speed_max<=self.speed_min or min(self.kp,self.ki)<0 or self.rate<=0:
            raise ValueError('速度范围或控制器参数非法')
        if self.accel[0][1]<-1 or self.accel[-1][1]>1 or self.steering[0][1]<-1 or self.steering[-1][1]>1:
            raise ValueError('标定输出超出归一化范围')
        self.integral=0.;self.steer=0.

    def step(self, *, speed_mps, target_speed_mps, acceleration_mps2, angle_rad, dt_s):
        speed=_number(speed_mps,'速度');target=_number(target_speed_mps,'目标速度')
        acceleration=_number(acceleration_mps2,'加速度');angle=_number(angle_rad,'转角');dt=_number(dt_s,'dt')
        if dt<=0 or not self.speed_min<=speed<=self.speed_max or not self.speed_min<=target<=self.speed_max:
            raise ValueError('速度或时间超出标定范围')
        feedforward=_interpolate(self.accel,acceleration)
        requested_steer=_interpolate(self.steering,angle)
        error=target-speed
        candidate=self.integral+error*dt
        unconstrained=feedforward+self.kp*error+self.ki*candidate
        if -1<=unconstrained<=1 or (unconstrained>1 and error<0) or (unconstrained< -1 and error>0):
            self.integral=candidate
        raw=feedforward+self.kp*error+self.ki*self.integral
        pedal=max(-1.,min(1.,raw));change=self.rate*dt
        self.steer=max(self.steer-change,min(self.steer+change,requested_steer))
        return {'throttle':max(0.,pedal),'brake':max(0.,-pedal),'steer':self.steer,
                'saturated':abs(raw)>1.,'requested_acceleration_mps2':acceleration,'requested_angle_rad':angle}
