"""已知平面几何的可视网格与未知可达区域；不接收隐藏目标真值。"""

from dataclasses import dataclass
import math
import numpy as np
from occluactisafe.common.coordinates import Pose2D, normalize_yaw
from occluactisafe.risk.obb import OBB, real, point, intersects, segment_obb_interval
from occluactisafe.risk.occupancy import PoseTrajectory, first_conflict_time, boundary_violation, IntervalConfig
from .participants import ParticipantGeometry


@dataclass(frozen=True)
class GridSpec:
    bounds: tuple = (-8., -8., 8., 8.)
    resolution_m: float = .5

    def __post_init__(self):
        if len(self.bounds) != 4: raise ValueError("网格边界需四个分量")
        b = tuple(real(v, "bounds") for v in self.bounds)
        real(self.resolution_m, "resolution", positive=True)
        if b[2] <= b[0] or b[3] <= b[1]: raise ValueError("网格边界倒序")
        sizes = ((b[3] - b[1]) / self.resolution_m, (b[2] - b[0]) / self.resolution_m)
        if any(abs(v - round(v)) > 1e-8 or v < 1 for v in sizes) or sizes[0] * sizes[1] > 40000:
            raise ValueError("边界必须整除分辨率，且不超过40000格")
        object.__setattr__(self, "bounds", b)

    @property
    def shape(self):
        b, r = self.bounds, self.resolution_m
        return round((b[3] - b[1]) / r), round((b[2] - b[0]) / r)

    def center(self, row, col):
        return self.bounds[0] + (col + .5) * self.resolution_m, self.bounds[1] + (row + .5) * self.resolution_m


@dataclass(frozen=True)
class KnownObstacle:
    obstacle_id: str
    box: OBB
    available_time: float
    clock_domain: str
    coordinate_frame_id: str

    def __post_init__(self):
        _identities(self.obstacle_id, self.clock_domain, self.coordinate_frame_id)
        if not isinstance(self.box, OBB): raise ValueError("已知障碍需OBB")
        real(self.available_time, "obstacle_available")


def _identities(*values):
    if any(not isinstance(s, str) or not s.strip() for s in values): raise ValueError("来源/时钟/坐标标识不能为空")


@dataclass(frozen=True)
class SensorView:
    source_id: str
    pose: Pose2D
    fov_rad: float
    range_m: float
    quality: float
    observation_time: float
    available_time: float
    valid_until: float
    clock_domain: str
    coordinate_frame_id: str

    def __post_init__(self):
        _identities(self.source_id, self.clock_domain, self.coordinate_frame_id)
        if not isinstance(self.pose, Pose2D): raise ValueError("视点需Pose2D")
        real(self.fov_rad, "FOV", positive=True); real(self.range_m, "range", positive=True)
        real(self.quality, "quality", nonnegative=True)
        if self.fov_rad > 2 * math.pi or self.quality > 1: raise ValueError("FOV/质量范围非法")
        for name in ("observation_time", "available_time", "valid_until"): real(getattr(self, name), name)
        if not self.observation_time <= self.available_time <= self.valid_until: raise ValueError("视点可用时间倒序")


def _mask(value, spec, name):
    result = np.asarray(value)
    if result.shape != spec.shape or result.dtype != np.bool_: raise ValueError(f"{name}需同网格bool数组")
    result = result.copy(); result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FreeSpaceEvidence:
    """显式的整格证空声明，不从空检测自动生成；来源可靠性仍待真实感知验证。"""
    source_id: str
    mask: object
    observation_time: float
    available_time: float
    valid_until: float
    clock_domain: str
    coordinate_frame_id: str

    def __post_init__(self):
        _identities(self.source_id, self.clock_domain, self.coordinate_frame_id)
        for name in ("observation_time", "available_time", "valid_until"): real(getattr(self, name), name)
        if not self.observation_time <= self.available_time <= self.valid_until: raise ValueError("证空时间倒序")
        a = np.asarray(self.mask)
        if a.dtype != np.bool_ or a.ndim != 2: raise ValueError("证空必须为二维bool声明")
        a = a.copy(); a.setflags(write=False); object.__setattr__(self, "mask", a)


@dataclass(frozen=True)
class OcclusionRiskRegion:
    region_id: str
    center: tuple
    cell_radius_m: float
    reference_time: float
    category: str
    speed_bound_mps: float
    acceleration_bound_mps2: float
    body_radius_m: float
    provenance: str

    def __post_init__(self):
        _identities(self.region_id, self.provenance)
        if self.category not in ("vehicle", "pedestrian", "cyclist"): raise ValueError("未知区域类别假设非法")
        object.__setattr__(self, "center", point(self.center))
        real(self.reference_time, "region_reference_time")
        for name in ("cell_radius_m", "speed_bound_mps", "acceleration_bound_mps2"):
            real(getattr(self, name), name, nonnegative=True)
        real(self.body_radius_m, "body_radius_m", positive=True)

    def reachable_radius(self, when):
        age = real(when, "reach_time") - self.reference_time
        if age < 0: raise ValueError("不能回溯未知区到参考时间以前")
        return self.cell_radius_m + self.body_radius_m + self.speed_bound_mps * age + .5 * self.acceleration_bound_mps2 * age * age


@dataclass(frozen=True)
class VisibilityGrid:
    spec: GridSpec
    now: float
    clock_domain: str
    coordinate_frame_id: str
    visibility: object  # 0未知/范围外，1几何可视，2遮挡
    occupied: object
    certified_free: object
    drivable: object
    obstacles: tuple
    regions: tuple
    accepted_sources: tuple
    closed_domain: bool


def build_visibility(spec, *, now, clock_domain, coordinate_frame_id, views=(), obstacles=(),
                     free_evidence=(), drivable=None, closed_domain=False, max_age_s=.5, min_quality=.5):
    real(now, "now"); real(max_age_s, "max_age_s", nonnegative=True); real(min_quality, "min_quality", nonnegative=True)
    _identities(clock_domain, coordinate_frame_id)
    if type(closed_domain) is not bool or min_quality > 1: raise ValueError("域声明/质量非法")
    drive = np.zeros(spec.shape, dtype=bool) if drivable is None else _mask(drivable, spec, "drivable")
    identity = (clock_domain, coordinate_frame_id)
    def legal(item):
        return ((item.clock_domain, item.coordinate_frame_id) == identity and item.available_time <= now <= item.valid_until
                and 0 <= now - item.observation_time <= max_age_s)
    obs = tuple(o for o in obstacles if (o.clock_domain, o.coordinate_frame_id) == identity and o.available_time <= now)
    if len({o.obstacle_id for o in obs}) != len(obs): raise ValueError("重复已知障碍ID")
    usable = tuple(v for v in views if legal(v) and v.quality >= min_quality)
    if len({v.source_id for v in usable}) != len(usable): raise ValueError("每个来源只接受一个当前视点")
    visible = np.zeros(spec.shape, bool); occluded = visible.copy(); occupied = visible.copy(); free = visible.copy()
    source_masks = {}
    for row, col in np.ndindex(spec.shape):
        center = spec.center(row, col)
        cell = OBB(center, 0., spec.resolution_m, spec.resolution_m)
        occupied[row, col] = any(intersects(cell, o.box) for o in obs)
    for view in usable:
        current = np.zeros(spec.shape, bool)
        for row, col in np.ndindex(spec.shape):
            center = spec.center(row, col); dx, dy = center[0] - view.pose.x, center[1] - view.pose.y
            if math.hypot(dx, dy) > view.range_m or abs(normalize_yaw(math.atan2(dy, dx) - view.pose.yaw)) > view.fov_rad / 2: continue
            blocked = any(segment_obb_interval((view.pose.x, view.pose.y), center, o.box) is not None for o in obs)
            if blocked: occluded[row, col] = True
            else: current[row, col] = True
        source_masks[view.source_id] = (current, view.observation_time); visible |= current
    for proof in free_evidence:
        if legal(proof) and proof.source_id in source_masks:
            current, observed_at = source_masks[proof.source_id]
            if proof.observation_time == observed_at:
                free |= _mask(proof.mask, spec, "free_evidence") & current
    free &= ~occupied
    grid = np.where(visible, 1, np.where(occluded, 2, 0)).astype(np.uint8)
    # 部分占用单元仍可包含未知空间，故不能将整格从未知区中删除。
    hypotheses = (("vehicle", 15., 3., math.hypot(2., 1.)), ("pedestrian", 2., 1., .3), ("cyclist", 8., 2., math.hypot(.9, .3)))
    regions = tuple(OcclusionRiskRegion(f"unknown-{r}-{c}-{category}", spec.center(r, c), spec.resolution_m / math.sqrt(2),
                                       now, category, v, a, radius, "unknown_or_occluded_or_missing_free_certificate")
                    for r, c in np.ndindex(spec.shape) if not free[r, c] for category, v, a, radius in hypotheses)
    for array in (grid, occupied, free, drive): array.setflags(write=False)
    return VisibilityGrid(spec, now, clock_domain, coordinate_frame_id, grid, occupied, free, drive,
                          obs, regions, tuple(sorted(source_masks)), closed_domain)


def _distance_to_segment(point, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]; length_sq = dx * dx + dy * dy
    f = 0. if length_sq == 0 else max(0., min(1., ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length_sq))
    return math.hypot(point[0] - a[0] - f * dx, point[1] - a[1] - f * dy)


def assess_constraints(path, geometry, knowledge, start, end, config=IntervalConfig()):
    if knowledge is None: return {"blocked": True, "reasons": ["knowledge_missing"], "actual_collision": None}
    if (path.clock_domain, path.coordinate_frame_id) != (knowledge.clock_domain, knowledge.coordinate_frame_id):
        raise ValueError("约束与轨迹时钟/坐标不一致")
    if start != knowledge.now: raise ValueError("约束快照必须在当前采用时刻重建，不能回填过去或沿用陈旧证空")
    reasons = []; details = []; trace = []
    if not knowledge.closed_domain: reasons.append("outside_domain_unknown")
    boundary = boundary_violation(path, geometry, start, end, knowledge.spec.bounds, config)
    trace.extend(boundary.trace)
    if boundary.event: reasons.append("map_boundary"); details.append(boundary.event)
    static = [(o.obstacle_id, o.box) for o in knowledge.obstacles]
    for row, col in np.ndindex(knowledge.spec.shape):
        if not knowledge.drivable[row, col]:
            static.append((f"non_drivable-{row}-{col}", OBB(knowledge.spec.center(row, col), 0., knowledge.spec.resolution_m, knowledge.spec.resolution_m)))
    for identity, box in static:
        times = (start, end) if end > start else (start, start + 1.)
        fixed = PoseTrajectory(times, (Pose2D(*box.center, box.heading),) * 2, path.clock_domain, path.coordinate_frame_id)
        body = ParticipantGeometry("vehicle", box.length, box.width, margin=box.margin)
        result = first_conflict_time(path, geometry, fixed, body, start, end, config)
        trace.extend(result.trace)
        if result.event:
            reasons.append("known_occupied_or_non_drivable"); details.append({"obstacle_id": identity, **result.event}); break
    tested = 0; unknown_hit = None
    knots = path.knots(start, end)
    segments = list(zip(knots, knots[1:])) if end > start else [(start, end)]
    for region in knowledge.regions:
        tested += 1
        for lo, hi in segments:
            a, b = path.at(lo), path.at(hi)
            bound = region.reachable_radius(hi) + geometry.motion_radius
            if _distance_to_segment(region.center, (a.x, a.y), (b.x, b.y)) <= bound + config.epsilon_m:
                unknown_hit = {"region_id": region.region_id, "category_hypothesis": region.category, "interval_start": lo,
                               "interval_end": hi, "reachable_radius_m": region.reachable_radius(hi), "kind": "unknown_reachable_constraint"}; break
        if unknown_hit: break
    if unknown_hit: reasons.append("unknown_reachable_region"); details.append(unknown_hit)
    return {"blocked": bool(reasons), "reasons": reasons, "details": details, "trace": trace,
            "regions_total": len(knowledge.regions), "regions_tested": tested,
            "regions_unexamined": len(knowledge.regions) - tested, "actual_collision": None}
