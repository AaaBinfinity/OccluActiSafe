"""固定已排队前缀，升级制动而不逐周期重置执行延迟。"""

from occluactisafe.planning.braking import MotionState, MotionProfile, braking_profile
from occluactisafe.risk.obb import real


class AnalyticExecutor:
    def __init__(self, x=0., speed=12., delay=.15):
        real(speed, "speed", nonnegative=True); real(delay, "delay", nonnegative=True)
        self.profile = MotionProfile(MotionState(x, speed, 0.), ())
        self.started = 0.; self.delay = delay; self.priority = 0; self.requests = []

    def at(self, now): return self.profile.at(now - self.started)

    def preview(self, now, B=None, J=None):
        state = self.at(now)
        if B is None: return self.profile, self.started
        requested_priority = {(1.5, 4.): 1, (4.5, 8.): 2}.get((B, J))
        if requested_priority is None: raise ValueError("当前解析执行器仅接受冻结制动档")
        if requested_priority == self.priority: return self.profile, self.started
        if requested_priority < self.priority: raise ValueError("不能削弱已锁存制动剖面")
        prefix = self.profile.prefix(now - self.started, self.delay)
        future = self.at(now + self.delay)
        tail = braking_profile(future.v, future.a, B=B, J=J, s0=future.s)
        return MotionProfile(state, (*prefix, *tail.phases)), now

    def request(self, now, action):
        priorities = {"NORMAL": 0, "SLOW": 1, "DEGRADED": 1, "AEB": 2}
        if action not in priorities: raise ValueError("未知纵向动作")
        priority = priorities[action]
        if priority <= self.priority: return False
        B, J = (4.5, 8.) if priority == 2 else (1.5, 4.)
        profile, started = self.preview(now, B, J)
        self.profile, self.started, self.priority = profile, started, priority
        self.requests.append({"requested": now, "effective": now + self.delay, "action": action, "B": B, "J": J})
        return True

    def control(self, now):
        state = self.at(now)
        return {"acceleration_mps2": state.a, "jerk_mps3": state.j,
                "throttle": min(1., max(0., state.a / 2.)), "brake": min(1., max(0., -state.a / 4.5)), "steer": 0.}
