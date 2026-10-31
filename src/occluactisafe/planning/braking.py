"""分段常jerk精确积分；不通过裁剪负速度伪造剖面可行性。"""

from dataclasses import dataclass
import math
from occluactisafe.risk.obb import real


@dataclass(frozen=True)
class MotionState:
    s: float
    v: float
    a: float
    j: float = 0.

    def __post_init__(self):
        for name in ("s", "v", "a", "j"): real(getattr(self, name), name)
        if self.v < -1e-9: raise ValueError("非法负速度")


def integrate(state, jerk, dt):
    real(jerk, "jerk"); real(dt, "dt", nonnegative=True)
    return MotionState(state.s + state.v * dt + state.a * dt**2 / 2 + jerk * dt**3 / 6,
                       state.v + state.a * dt + jerk * dt**2 / 2, state.a + jerk * dt, jerk)


@dataclass(frozen=True)
class MotionProfile:
    initial: MotionState
    phases: tuple  # (jerk m/s³, duration s)

    def __post_init__(self):
        phases = tuple(tuple(p) for p in self.phases); state = self.initial
        for jerk, dt in phases:
            real(jerk, "jerk"); real(dt, "phase_duration", positive=True)
            if jerk > 0 and 0 < -state.a / jerk < dt: integrate(state, jerk, -state.a / jerk)
            state = integrate(state, jerk, dt)
        if abs(state.a) > 1e-7: raise ValueError("剖面结束加速度不为零")
        object.__setattr__(self, "phases", phases)

    @property
    def duration(self): return math.fsum(dt for _, dt in self.phases)

    def at(self, elapsed):
        real(elapsed, "elapsed", nonnegative=True)
        state = self.initial
        for jerk, duration in self.phases:
            if elapsed < duration: return integrate(state, jerk, elapsed)
            state = integrate(state, jerk, duration); elapsed -= duration
        # 仅消除精确停车构造的末端浮点残差；非停车剖面保持真实速度。
        if abs(state.v) < 1e-9: state = MotionState(state.s, 0., 0.)
        return MotionState(state.s + state.v * elapsed, state.v, 0., 0.)

    def prefix(self, start, duration):
        real(start, "start", nonnegative=True); real(duration, "duration", nonnegative=True)
        boundaries = [0.]; total = 0.
        for _, dt in self.phases: total += dt; boundaries.append(total)
        knots = sorted({start, start + duration, *(t for t in boundaries if start < t < start + duration)})
        return tuple((self.at((a + b) / 2).j, b - a) for a, b in zip(knots, knots[1:]) if b > a)


def braking_profile(v0, a0=0., *, B=4.5, J=8., delay=0., queued_jerk=0., s0=0.):
    real(v0, "v0", nonnegative=True); real(a0, "a0"); real(B, "B", positive=True); real(J, "J", positive=True)
    real(delay, "delay", nonnegative=True); real(queued_jerk, "queued_jerk")
    initial = MotionState(s0, v0, a0)
    start = integrate(initial, queued_jerk, delay)
    if start.a < -B or (start.a < 0 and start.v < start.a**2 / (2 * J) - 1e-12): raise ValueError("当前状态不满足S曲线停车边界")
    if start.v == 0 and start.a == 0: phases = []
    else:
        hold = (start.v + start.a**2 / (2 * J) - B**2 / J) / B
        peak = B if hold >= 0 else math.sqrt(J * start.v + start.a**2 / 2)
        down, up = (start.a + peak) / J, peak / J
        if down < 0: raise ValueError("负建立时间")
        phases = [(j, dt) for j, dt in ((-J, down), (0., max(0., hold)), (J, up)) if dt > 0]
    result = MotionProfile(initial, tuple(([(queued_jerk, delay)] if delay else []) + phases))
    stop = result.at(result.duration)
    if abs(stop.v) > 1e-7 or abs(stop.a) > 1e-7: raise ValueError("停车边界未满足")
    return result
