"""米制float64 CV Kalman；位置测量、solve和Joseph协方差更新。"""

from dataclasses import dataclass
import numpy as np

from occluactisafe.communication.protocol import finite
from occluactisafe.common.schemas import SourceTrack


def vector(value):
    a = np.array(value, dtype=np.float64, copy=True)
    if a.shape != (4,) or not np.isfinite(a).all():
        raise ValueError("状态必须为有限[x,y,vx,vy]")
    a.flags.writeable = False
    return a


def covariance(value, size=4):
    a = np.array(value, dtype=np.float64, copy=True)
    if a.shape != (size, size) or not np.isfinite(a).all():
        raise ValueError("协方差形状或有限性错误")
    if not np.allclose(a, a.T, atol=1e-8, rtol=0):
        raise ValueError("协方差不对称")
    a = (a + a.T) / 2
    if np.linalg.eigvalsh(a)[0] < -1e-8:
        raise ValueError("协方差不是半正定矩阵")
    a.flags.writeable = False
    return a


@dataclass(frozen=True)
class FilterConfig:
    sigma_ax: float = 1.0
    sigma_ay: float = 1.0
    default_position_variance: float = 1.0
    min_position_variance: float = 1e-6
    known_velocity_variance: float = 4.0
    unknown_velocity_variance: float = 25.0
    low_speed_mps: float = .1

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            finite(getattr(self, name), name, minimum=1e-12)
        if self.default_position_variance < self.min_position_variance:
            raise ValueError("未知R不能小于最小方差")


def cv_predict(z, p, dt, sigma_a=(1., 1.)):
    dt = finite(dt, "dt")
    z, p = vector(z), covariance(p)
    if len(sigma_a) != 2:
        raise ValueError("sigma_a需要两个分量")
    sigma = [finite(v, "sigma_a") for v in sigma_a]
    f = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    g = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]], dtype=np.float64)
    q = g @ np.diag(np.square(sigma)) @ g.T
    return vector(f @ z), covariance(f @ p @ f.T + q)


def position_update(z, p, measurement, r):
    z, p, r = vector(z), covariance(p), covariance(r, 2)
    y = np.asarray(measurement, dtype=np.float64)
    if y.shape != (2,) or not np.isfinite(y).all() or np.linalg.eigvalsh(r)[0] <= 0:
        raise ValueError("位置测量须有限，R须正定且非零")
    h = np.array([[1., 0, 0, 0], [0, 1., 0, 0]])
    s = h @ p @ h.T + r
    k = np.linalg.solve(s, (p @ h.T).T).T
    residual = y - h @ z
    a = np.eye(4) - k @ h
    return vector(z + k @ residual), covariance(a @ p @ a.T + k @ r @ k.T)


def measurement_covariance(track: SourceTrack, config: FilterConfig):
    if track.covariance_xy is None:
        r = np.eye(2) * config.default_position_variance
        reason = "unknown_R_configured"
    else:
        r = covariance(track.covariance_xy, 2)
        values, axes = np.linalg.eigh(r)
        reason = "provided_R"
        if values[0] < config.min_position_variance:
            values = np.maximum(values, config.min_position_variance)
            r = (axes * values) @ axes.T
            reason = "provided_R_regularized"
    # 外部已外推状态不是在last_seen时刻的新位置；保留年龄并增加不确定性。
    age = track.state_time - track.last_seen_time
    if age > 0:
        r = r + np.diag([config.sigma_ax**2, config.sigma_ay**2]) * age**4 / 4
        reason += "+derived_state_age"
    return covariance(r, 2), reason


@dataclass(frozen=True)
class KalmanState:
    z: np.ndarray
    p: np.ndarray
    state_time: float
    last_seen_time: float
    updates: int
    velocity_valid: bool

    def __post_init__(self):
        object.__setattr__(self, "z", vector(self.z))
        object.__setattr__(self, "p", covariance(self.p))
        finite(self.state_time, "state_time"); finite(self.last_seen_time, "last_seen_time")
        if self.state_time < self.last_seen_time or type(self.updates) is not int or self.updates < 1:
            raise ValueError("后验时序或测量计数错误")
        if type(self.velocity_valid) is not bool:
            raise ValueError("velocity_valid须为布尔值")

    def snapshot(self, at, config):
        return cv_predict(self.z, self.p, at - self.state_time, (config.sigma_ax, config.sigma_ay))


def initialize(track: SourceTrack, config=FilterConfig()):
    if not track.state_valid_mask:
        raise ValueError("不能用无效位置初始化")
    r, reason = measurement_covariance(track, config)
    z = [track.x, track.y, track.vx if track.velocity_valid else 0., track.vy if track.velocity_valid else 0.]
    p = np.zeros((4, 4)); p[:2, :2] = r
    p[2:, 2:] = np.eye(2) * (config.known_velocity_variance if track.velocity_valid else config.unknown_velocity_variance)
    return KalmanState(z, p, track.state_time, track.last_seen_time, 1, track.velocity_valid), reason


def correct(previous: KalmanState, track: SourceTrack, config=FilterConfig()):
    if not track.state_valid_mask or not track.observed_mask:
        raise ValueError("纯外推或无效状态不能充当测量更新")
    if track.last_seen_time <= previous.last_seen_time:
        raise ValueError("重复/倒序真实观测不能重复更新")
    z, p = previous.snapshot(track.state_time, config)
    r, reason = measurement_covariance(track, config)
    z, p = position_update(z, p, [track.x, track.y], r)
    velocity_valid = previous.velocity_valid or track.state_time > previous.state_time
    return KalmanState(z, p, track.state_time, track.last_seen_time, previous.updates + 1, velocity_valid), reason
