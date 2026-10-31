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


def _slices(value, name):
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(f'{name} 至少需要两个速度切片')
    result=[(_number(entry['speed_mps'],'切片速度'),_table(entry['points'],name)) for entry in value]
    if any(a[0]>=b[0] for a,b in zip(result,result[1:])):
        raise ValueError('速度切片必须严格递增')
    return result


def _speed_map(slices, speed, request):
    if not slices[0][0] <= speed <= slices[-1][0]: raise ValueError('速度超出标定切片')
    for (v0,t0),(v1,t1) in zip(slices,slices[1:]):
        if v0 <= speed <= v1:
            # 两端均须覆盖请求，不对任一速度切片外推。
            if speed == v0: return _interpolate(t0,request)
            if speed == v1: return _interpolate(t1,request)
            y0=_interpolate(t0,request);y1=_interpolate(t1,request)
            return y0+(y1-y0)*(speed-v0)/(v1-v0)
    raise ValueError('速度不在有效切片')


class CalibratedActuator:
    """输入加速度m/s²、转角rad和速度m/s；仅接受提供的实测表。"""
    def __init__(self, calibration, *, kp, ki, steering_rate_per_s):
        self.vehicle_blueprint=calibration.get('vehicle_blueprint')
        self.surface_id=calibration.get('surface_id')
        if not all(isinstance(x,str) and x.strip() for x in (self.vehicle_blueprint,self.surface_id)):
            raise ValueError('必须声明车辆蓝图与已标定附着条件')
        self.accel_slices=_slices(calibration['acceleration_by_speed'],'加速度表') if 'acceleration_by_speed' in calibration else None
        self.steering_slices=_slices(calibration['steering_by_speed'],'转向表') if 'steering_by_speed' in calibration else None
        self.accel=None if self.accel_slices else _table(calibration['acceleration_to_pedal'],'加速度表')
        self.steering=None if self.steering_slices else _table(calibration['angle_to_steer'],'转向表')
        self.speed_min=_number(calibration['speed_min_mps'],'最小速度')
        self.speed_max=_number(calibration['speed_max_mps'],'最大速度')
        self.kp=_number(kp,'kp');self.ki=_number(ki,'ki');self.rate=_number(steering_rate_per_s,'转向速率')
        if self.speed_min<0 or self.speed_max<=self.speed_min or min(self.kp,self.ki)<0 or self.rate<=0:
            raise ValueError('速度范围或控制器参数非法')
        tables=([x[1] for x in self.accel_slices] if self.accel_slices else [self.accel])+([x[1] for x in self.steering_slices] if self.steering_slices else [self.steering])
        if any(table[0][1]<-1 or table[-1][1]>1 for table in tables):
            raise ValueError('标定输出超出归一化范围')
        for slices in (self.accel_slices,self.steering_slices):
            if slices and (self.speed_min<slices[0][0] or self.speed_max>slices[-1][0]):
                raise ValueError('声明速度范围超出标定切片')
        self.integral=0.;self.steer=0.

    def step(self, *, speed_mps, target_speed_mps, acceleration_mps2, angle_rad, dt_s, vehicle_blueprint, surface_id):
        if vehicle_blueprint != self.vehicle_blueprint or surface_id != self.surface_id:
            raise ValueError('车辆或附着条件超出标定范围')
        speed=_number(speed_mps,'速度');target=_number(target_speed_mps,'目标速度')
        acceleration=_number(acceleration_mps2,'加速度');angle=_number(angle_rad,'转角');dt=_number(dt_s,'dt')
        if dt<=0 or not self.speed_min<=speed<=self.speed_max or not self.speed_min<=target<=self.speed_max:
            raise ValueError('速度或时间超出标定范围')
        feedforward=_speed_map(self.accel_slices,speed,acceleration) if self.accel_slices else _interpolate(self.accel,acceleration)
        requested_steer=_speed_map(self.steering_slices,speed,angle) if self.steering_slices else _interpolate(self.steering,angle)
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
