"""右手二维平面坐标：x 向前、y 向左；位置米，航向逆时针弧度。"""

from dataclasses import dataclass
import math
from typing import Tuple


Point2D = Tuple[float, float]


def _finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限数值")


def normalize_yaw(yaw: float) -> float:
    """将弧度规范到 [-π, π)。"""
    _finite(yaw, "yaw")
    return (yaw + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class Pose2D:
    """已知参考点的地图位置（米）和朝向（弧度）。"""

    x: float
    y: float
    yaw: float

    def __post_init__(self) -> None:
        _finite(self.x, "x")
        _finite(self.y, "y")
        _finite(self.yaw, "yaw")


def local_to_world(x: float, y: float, origin: Pose2D) -> Point2D:
    """把局部 x 向前、y 向左的点变为地图坐标。"""
    _finite(x, "x")
    _finite(y, "y")
    c, s = math.cos(origin.yaw), math.sin(origin.yaw)
    return origin.x + c * x - s * y, origin.y + s * x + c * y


def world_to_local(x: float, y: float, origin: Pose2D) -> Point2D:
    """地图点相对原点的逆变换。"""
    _finite(x, "x")
    _finite(y, "y")
    dx, dy = x - origin.x, y - origin.y
    c, s = math.cos(origin.yaw), math.sin(origin.yaw)
    return c * dx + s * dy, -s * dx + c * dy


def shift_reference_point(pose: Pose2D, forward_m: float, left_m: float = 0.0) -> Pose2D:
    """按车辆局部坐标偏移参考点；例如后轴到几何中心用正向偏移。"""
    _finite(forward_m, "forward_m")
    _finite(left_m, "left_m")
    x, y = local_to_world(forward_m, left_m, pose)
    return Pose2D(x, y, pose.yaw)


def carla_to_project(x: float, y: float, yaw_degrees: float) -> Pose2D:
    """CARLA 平面/角度转统一右手系：X=x、Y=-y、yaw=-yaw_CARLA。"""
    _finite(x, "x")
    _finite(y, "y")
    _finite(yaw_degrees, "yaw_degrees")
    return Pose2D(x, -y, normalize_yaw(-math.radians(yaw_degrees)))


def project_to_carla(pose: Pose2D) -> Tuple[float, float, float]:
    """统一右手位置与弧度转 CARLA 平面及度，角度按模 360 等价。"""
    return pose.x, -pose.y, math.degrees(normalize_yaw(-pose.yaw))


def carla_vector_to_project(x: float, y: float) -> Point2D:
    """转换速度或位移向量，不对其施加平移。反变换相同。"""
    _finite(x, "x")
    _finite(y, "y")
    return x, -y
