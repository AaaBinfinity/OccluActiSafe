"""共同参考时刻的门控与含未匹配虚拟列的线性分配。"""

from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

from occluactisafe.communication.protocol import finite
from .state_filter import vector, covariance


@dataclass(frozen=True)
class AssociationConfig:
    position_gate_m: float = 4.
    mahalanobis_gate_sq: float = 9.21
    unmatched_cost: float = 1.5
    velocity_scale_mps: float = 10.
    velocity_weight: float = .2
    heading_weight: float = .1
    ambiguity_margin: float = .05

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            finite(getattr(self, name), name, minimum=0 if name.endswith('weight') or name == 'ambiguity_margin' else 1e-12)


@dataclass(frozen=True)
class AssociationState:
    identity: tuple
    category: str
    role: str
    reference_point: str
    z: np.ndarray
    p: np.ndarray
    velocity_valid: bool
    yaw: float | None

    def __post_init__(self):
        object.__setattr__(self, "z", vector(self.z))
        object.__setattr__(self, "p", covariance(self.p))


@dataclass(frozen=True)
class Assignment:
    pairs: tuple[tuple[int, int], ...]
    unmatched_rows: tuple[int, ...]
    unmatched_columns: tuple[int, ...]
    objective: float


def minimum_cost_assignment(costs, allowed=None, *, unmatched_cost=5.):
    costs = np.asarray(costs, dtype=np.float64)
    finite(unmatched_cost, "unmatched_cost", minimum=1e-12)
    if costs.ndim != 2 or not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError("代价矩阵须为有限非负二维数组")
    if allowed is None:
        allowed = np.ones(costs.shape, dtype=bool)
    allowed = np.asarray(allowed)
    if allowed.shape != costs.shape or allowed.dtype != np.bool_:
        raise ValueError("门控矩阵须为同形状布尔数组")
    n, m = costs.shape
    if n == 0:
        return Assignment((), (), tuple(range(m)), 0.)
    augmented = np.full((n, m + n), np.inf)
    augmented[:, :m] = np.where(allowed & (costs < unmatched_cost), costs, np.inf)
    augmented[np.arange(n), m + np.arange(n)] = unmatched_cost
    rows, columns = linear_sum_assignment(augmented)
    pairs = tuple((int(i), int(j)) for i, j in zip(rows, columns) if j < m)
    matched_rows = {i for i, _ in pairs}; matched_columns = {j for _, j in pairs}
    return Assignment(pairs, tuple(i for i in range(n) if i not in matched_rows),
                      tuple(j for j in range(m) if j not in matched_columns), float(augmented[rows, columns].sum()))


def _pair_cost(a, b, config):
    if a.role != b.role or a.reference_point != b.reference_point:
        return "role_or_reference_mismatch", 0., None
    if a.category != b.category and "unknown" not in (a.category, b.category):
        return "category_mismatch", 0., None
    delta = a.z[:2] - b.z[:2]
    distance = float(np.linalg.norm(delta))
    if distance > config.position_gate_m:
        return "position_gate", 0., None
    s = a.p[:2, :2] + b.p[:2, :2] + np.eye(2) * 1e-12
    d2 = float(delta @ np.linalg.solve(s, delta))
    diagnostic = {"distance_m": distance, "mahalanobis_sq": d2}
    if d2 > config.mahalanobis_gate_sq:
        return "mahalanobis_gate", 0., diagnostic
    cost = math.sqrt(max(0., d2) / config.mahalanobis_gate_sq)
    if a.velocity_valid and b.velocity_valid:
        cost += config.velocity_weight * float(np.linalg.norm(a.z[2:] - b.z[2:])) / config.velocity_scale_mps
    if a.yaw is not None and b.yaw is not None:
        cost += config.heading_weight * abs(math.atan2(math.sin(a.yaw - b.yaw), math.cos(a.yaw - b.yaw))) / math.pi
    return ("allowed" if cost < config.unmatched_cost else "unmatched_cheaper"), cost, diagnostic


def associate(rows, columns, config=AssociationConfig(), *, column_members=None):
    """调用方规范排序且将各状态推进到相同state_time；不靠局部ID猜配对。"""
    costs = np.zeros((len(rows), len(columns)), dtype=np.float64)
    allowed = np.zeros(costs.shape, dtype=bool)
    reasons = [["" for _ in columns] for _ in rows]
    diagnostics = [[None for _ in columns] for _ in rows]
    for i, a in enumerate(rows):
        for j, b in enumerate(columns):
            reason, cost, diagnostic = _pair_cost(a, b, config)
            if reason == "allowed" and column_members is not None:
                for member in column_members[j]:
                    member_reason, _, _ = _pair_cost(a, member, config)
                    if member_reason != "allowed":
                        reason = "group_member_" + member_reason; break
            costs[i, j] = cost
            diagnostics[i][j] = diagnostic
            allowed[i, j] = reason == "allowed"
            reasons[i][j] = reason
    # 对称地拒绝近似并列候选，保留独有对象；不会把身份作为真值约束。
    ambiguous_rows = []
    ambiguous_columns = []
    for i in range(len(rows)):
        values = np.sort(costs[i, allowed[i]])
        if len(values) > 1 and values[1] - values[0] <= config.ambiguity_margin:
            ambiguous_rows.append(i)
    for j in range(len(columns)):
        values = np.sort(costs[allowed[:, j], j])
        if len(values) > 1 and values[1] - values[0] <= config.ambiguity_margin:
            ambiguous_columns.append(j)
    for i in ambiguous_rows:
        for j in range(len(columns)):
            if allowed[i, j]: reasons[i][j] = "ambiguous_row"
        allowed[i, :] = False
    for j in ambiguous_columns:
        for i in range(len(rows)):
            if allowed[i, j]: reasons[i][j] = "ambiguous_column"
        allowed[:, j] = False
    matched = minimum_cost_assignment(costs, allowed, unmatched_cost=config.unmatched_cost)
    report = {"row_identities": [list(v.identity) for v in rows], "column_identities": [list(v.identity) for v in columns],
              "costs": costs.tolist(), "allowed": allowed.tolist(), "reasons": reasons, "distances": diagnostics,
              "pairs": [list(pair) for pair in matched.pairs], "unmatched_rows": list(matched.unmatched_rows),
              "unmatched_columns": list(matched.unmatched_columns), "objective": matched.objective}
    return matched, report
