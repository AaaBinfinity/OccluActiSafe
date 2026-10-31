"""N11唯一决策/执行调用点，状态、风险、输出分别留存。"""

from .vehicle_agent import VehicleAgent
from .safety_monitor import check_control
from occluactisafe.control.executor import AnalyticExecutor


class AnalyticPipeline:
    def __init__(self, strategy, *, speed=12., delay=.15):
        self.vehicle = VehicleAgent(strategy)
        self.executor = AnalyticExecutor(speed=speed, delay=delay)

    def step(self, now, ego, **observations):
        bundle, action = self.vehicle.step(now, ego, self.executor, **observations)
        self.executor.request(now, action)
        control = self.executor.control(now)
        check_control(control)
        return {"time": now, "state": self.executor.at(now).__dict__, "control": control,
                "audit": self.vehicle.last_audit, "request": bundle.to_dict()}
