"""解析纵向输出有限性/互斥检查，不读取环境对象或事件。"""

import math


def check_control(control):
    if any(not math.isfinite(v) for v in control.values()): raise ValueError("非有限控制")
    if not (0 <= control["throttle"] <= 1 and 0 <= control["brake"] <= 1 and control["steer"] == 0): raise ValueError("解析纵向控制范围错误")
    if control["throttle"] * control["brake"] != 0: raise ValueError("油门制动不互斥")
    return True
