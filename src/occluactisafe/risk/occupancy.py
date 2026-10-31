"""声明插值模型下的保守运动区间检查；不记录真值碰撞。"""

from bisect import bisect_left
from dataclasses import dataclass
import math
from occluactisafe.common.coordinates import Pose2D, normalize_yaw
from .obb import OBB, Circle, EPSILON_M, real, expanded, intersects, obb_corners


@dataclass(frozen=True)
class PoseTrajectory:
    times: tuple
    poses: tuple
    clock_domain: str
    coordinate_frame_id: str
    valid_mask: tuple | None = None

    def __post_init__(self):
        times = tuple(real(t, "time") for t in self.times)
        poses = tuple(p if isinstance(p, Pose2D) else Pose2D(*p) for p in self.poses)
        mask = (True,) * len(times) if self.valid_mask is None else tuple(self.valid_mask)
        if len(times) < 2 or len(times) != len(poses) or len(times) != len(mask): raise ValueError("轨迹需要同长度的至少两个采样")
        if any(a >= b for a, b in zip(times, times[1:])): raise ValueError("轨迹时刻必须严格递增")
        if any(type(v) is not bool for v in mask): raise ValueError("轨迹掩码必须为bool")
        for value in (self.clock_domain, self.coordinate_frame_id):
            if not isinstance(value, str) or not value.strip(): raise ValueError("轨迹时钟域/坐标不能为空")
        object.__setattr__(self, "times", times); object.__setattr__(self, "poses", poses); object.__setattr__(self, "valid_mask", mask)

    def at(self, when):
        real(when, "query_time")
        i = bisect_left(self.times, when)
        if i < len(self.times) and self.times[i] == when:
            if self.valid_mask[i]: return self.poses[i]
            raise ValueError("查询位置被掩码排除")
        if i == 0 or i == len(self.times) or not self.valid_mask[i - 1] or not self.valid_mask[i]:
            raise ValueError("不跨轨迹掩码洞插值或向时域外推")
        a, b = self.poses[i - 1], self.poses[i]
        f = (when - self.times[i - 1]) / (self.times[i] - self.times[i - 1])
        return Pose2D(a.x + (b.x - a.x) * f, a.y + (b.y - a.y) * f,
                      normalize_yaw(a.yaw + normalize_yaw(b.yaw - a.yaw) * f))

    def contiguous_end(self, start):
        try: self.at(start)
        except ValueError: return None
        i = bisect_left(self.times, start)
        while i < len(self.times) and self.valid_mask[i]: i += 1
        return self.times[i - 1]

    def knots(self, start, end):
        return sorted({start, end, *(t for t in self.times if start < t < end)})


@dataclass(frozen=True)
class IntervalConfig:
    min_interval_s: float = .01
    epsilon_m: float = EPSILON_M
    max_nodes: int = 65536

    def __post_init__(self):
        real(self.min_interval_s, "min_interval_s", positive=True); real(self.epsilon_m, "epsilon_m", nonnegative=True)
        if self.min_interval_s > .01: raise ValueError("N10区间分辨率不得大于.01s")
        if type(self.max_nodes) is not int or self.max_nodes <= 0: raise ValueError("max_nodes须为正整数")


@dataclass(frozen=True)
class IntervalResult:
    event: dict | None
    trace: tuple
    algorithm_complete: bool = True

    @property
    def potential_contact(self): return self.event is not None


def _motion_margin(path, geometry, lo, hi):
    if hi == lo: return 0.
    a, b = path.at(lo), path.at(hi)
    v = math.hypot(b.x - a.x, b.y - a.y) / (hi - lo)
    omega = abs(normalize_yaw(b.yaw - a.yaw)) / (hi - lo)
    return real((v + geometry.motion_radius * omega) * (hi - lo) / 2, "motion_bound", nonnegative=True)


def _require_interval(path, start, end):
    real(start, "start"); real(end, "end")
    if end < start: raise ValueError("区间不能倒序")
    stop = path.contiguous_end(start)
    if stop is None or stop < end - 1e-12: raise ValueError("轨迹未连续覆盖待检区间")


def first_conflict_time(a_path, a_geometry, b_path, b_geometry, start, end, config=IntervalConfig()):
    """最早不能排除的时间小区间；sampled_overlap仍仅是预测几何。"""
    if (a_path.clock_domain, a_path.coordinate_frame_id) != (b_path.clock_domain, b_path.coordinate_frame_id):
        raise ValueError("轨迹时钟域/坐标不一致")
    _require_interval(a_path, start, end); _require_interval(b_path, start, end)
    trace = []; exhausted = False

    def visit(lo, hi):
        nonlocal exhausted
        if len(trace) >= config.max_nodes:
            exhausted = True
            return {"kind": "potential_overlap", "uncertain_interval": True, "reason": "interval_budget_exhausted",
                    "interval_start": lo, "interval_end": hi, "sampled_overlap_time": None, "actual_collision": None}
        mid = (lo + hi) / 2
        a, b = a_geometry.at(a_path.at(mid)), b_geometry.at(b_path.at(mid))
        ma, mb = _motion_margin(a_path, a_geometry, lo, hi), _motion_margin(b_path, b_geometry, lo, hi)
        bound_overlap = intersects(expanded(a, ma), expanded(b, mb), config.epsilon_m)
        row = {"interval_start": lo, "interval_end": hi, "midpoint": mid,
               "motion_margin_a_m": ma, "motion_margin_b_m": mb, "bound_overlap": bound_overlap}
        trace.append(row)
        if not bound_overlap: row["status"] = "separation_certified"; return None
        if hi - lo <= config.min_interval_s:
            sampled = [t for t in sorted({lo, mid, hi}) if intersects(a_geometry.at(a_path.at(t)), b_geometry.at(b_path.at(t)), config.epsilon_m)]
            row["status"] = "sampled_geometry_overlap" if sampled else "uncertain_interval"
            return {"kind": "predicted_geometry_overlap" if sampled else "potential_overlap", "uncertain_interval": not bool(sampled),
                    "interval_start": lo, "interval_end": hi, "sampled_overlap_time": sampled[0] if sampled else None,
                    "position_a_midpoint": list(a.center), "position_b_midpoint": list(b.center), "actual_collision": None}
        row["status"] = "subdivided"
        return visit(lo, mid) or visit(mid, hi)

    knots = sorted(set(a_path.knots(start, end) + b_path.knots(start, end)))
    if start == end: event = visit(start, end)
    else:
        event = None
        for lo, hi in zip(knots, knots[1:]):
            event = visit(lo, hi)
            if event is not None: break
    return IntervalResult(event, tuple(trace), not exhausted)


def _inside(shape, bounds, eps):
    x0, y0, x1, y1 = bounds
    if isinstance(shape, OBB): points = obb_corners(shape)
    elif isinstance(shape, Circle):
        r = shape.radius + shape.margin; x, y = shape.center
        points = ((x - r, y), (x + r, y), (x, y - r), (x, y + r))
    else: raise ValueError("未知几何")
    return all(x0 - eps <= x <= x1 + eps and y0 - eps <= y <= y1 + eps for x, y in points)


def boundary_violation(path, geometry, start, end, bounds, config=IntervalConfig()):
    """对声明地图矩形的整车体/区间包含检查；不证明动力学可行。"""
    _require_interval(path, start, end)
    if len(bounds) != 4: raise ValueError("地图边界需要四个值")
    bounds = tuple(real(v, "bounds") for v in bounds)
    if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]: raise ValueError("地图矩形非法")
    trace = []; exhausted = False

    def visit(lo, hi):
        nonlocal exhausted
        if len(trace) >= config.max_nodes:
            exhausted = True
            return {"kind": "potential_boundary_violation", "interval_start": lo, "interval_end": hi, "reason": "interval_budget_exhausted"}
        mid = (lo + hi) / 2
        bound = expanded(geometry.at(path.at(mid)), _motion_margin(path, geometry, lo, hi))
        inside = _inside(bound, bounds, config.epsilon_m)
        trace.append({"interval_start": lo, "interval_end": hi, "inside_bound": inside})
        if inside: return None
        if hi - lo <= config.min_interval_s:
            outside = any(not _inside(geometry.at(path.at(t)), bounds, config.epsilon_m) for t in {lo, mid, hi})
            return {"kind": "predicted_boundary_violation" if outside else "potential_boundary_violation", "interval_start": lo, "interval_end": hi}
        return visit(lo, mid) or visit(mid, hi)

    knots = path.knots(start, end); event = None
    if start == end: event = visit(start, end)
    for lo, hi in zip(knots, knots[1:]):
        event = visit(lo, hi)
        if event is not None: break
    return IntervalResult(event, tuple(trace), not exhausted)
