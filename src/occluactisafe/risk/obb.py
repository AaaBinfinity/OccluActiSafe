"""中心参考的OBB/圆；边接触按风险口径相交。"""

from dataclasses import dataclass, replace
import math
from numbers import Real

EPSILON_M = 1e-6


def real(value, name, *, nonnegative=False, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name}须为有限实数")
    if (nonnegative and value < 0) or (positive and value <= 0):
        raise ValueError(f"{name}范围非法")
    return float(value)


def point(value):
    if len(value) != 2: raise ValueError("平面坐标需要两个分量")
    return tuple(real(v, "coordinate") for v in value)


@dataclass(frozen=True)
class OBB:
    center: tuple[float, float]
    heading: float
    length: float
    width: float
    margin: float = 0.

    def __post_init__(self):
        object.__setattr__(self, "center", point(self.center))
        real(self.heading, "heading"); real(self.length, "length", positive=True)
        real(self.width, "width", positive=True); real(self.margin, "margin", nonnegative=True)

    @property
    def radius(self): return math.hypot(self.length / 2 + self.margin, self.width / 2 + self.margin)


@dataclass(frozen=True)
class Circle:
    center: tuple[float, float]
    radius: float
    margin: float = 0.

    def __post_init__(self):
        object.__setattr__(self, "center", point(self.center))
        real(self.radius, "radius", positive=True); real(self.margin, "margin", nonnegative=True)


def _axes(box):
    c, s = math.cos(box.heading), math.sin(box.heading)
    return (c, s), (-s, c)


def obb_corners(box: OBB):
    a, b = _axes(box); hx, hy = box.length / 2 + box.margin, box.width / 2 + box.margin
    return tuple((box.center[0] + x * hx * a[0] + y * hy * b[0],
                  box.center[1] + x * hx * a[1] + y * hy * b[1]) for x, y in ((1, 1), (-1, 1), (-1, -1), (1, -1)))


def obb_intersects(a: OBB, b: OBB, eps=EPSILON_M):
    real(eps, "eps", nonnegative=True)
    axes_a, axes_b = _axes(a), _axes(b)
    delta = (b.center[0] - a.center[0], b.center[1] - a.center[1])
    for n in (*axes_a, *axes_b):
        ra = sum(h * abs(axis[0] * n[0] + axis[1] * n[1]) for h, axis in zip((a.length / 2 + a.margin, a.width / 2 + a.margin), axes_a))
        rb = sum(h * abs(axis[0] * n[0] + axis[1] * n[1]) for h, axis in zip((b.length / 2 + b.margin, b.width / 2 + b.margin), axes_b))
        if abs(delta[0] * n[0] + delta[1] * n[1]) > ra + rb + eps: return False
    return True


def circle_obb_intersects(circle: Circle, box: OBB, eps=EPSILON_M):
    real(eps, "eps", nonnegative=True)
    delta = (circle.center[0] - box.center[0], circle.center[1] - box.center[1])
    a, b = _axes(box)
    x, y = (sum(v * n for v, n in zip(delta, axis)) for axis in (a, b))
    dx, dy = max(0., abs(x) - box.length / 2 - box.margin), max(0., abs(y) - box.width / 2 - box.margin)
    return math.hypot(dx, dy) <= circle.radius + circle.margin + eps


def intersects(a, b, eps=EPSILON_M):
    if isinstance(a, OBB) and isinstance(b, OBB): return obb_intersects(a, b, eps)
    if isinstance(a, Circle) and isinstance(b, OBB): return circle_obb_intersects(a, b, eps)
    if isinstance(a, OBB) and isinstance(b, Circle): return circle_obb_intersects(b, a, eps)
    if isinstance(a, Circle) and isinstance(b, Circle):
        real(eps, "eps", nonnegative=True)
        return math.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1]) <= a.radius + a.margin + b.radius + b.margin + eps
    raise ValueError("只接受OBB/圆几何")


def expanded(shape, margin):
    real(margin, "motion_margin", nonnegative=True)
    return replace(shape, margin=shape.margin + margin)


def segment_obb_interval(start, end, box: OBB):
    """解析线段/盒的slab区间，供已知几何遮挡射线使用。"""
    start, end = point(start), point(end)
    lo, hi = 0., 1.
    for axis, half in zip(_axes(box), (box.length / 2 + box.margin, box.width / 2 + box.margin)):
        p = sum((v - c) * n for v, c, n in zip(start, box.center, axis))
        d = sum((v - w) * n for v, w, n in zip(end, start, axis))
        if abs(d) < 1e-15:
            if abs(p) > half + EPSILON_M: return None
        else:
            t0, t1 = sorted(((-half - p) / d, (half - p) / d))
            lo, hi = max(lo, t0), min(hi, t1)
            if lo > hi: return None
    return lo, hi
