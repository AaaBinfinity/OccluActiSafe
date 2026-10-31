"""独立解析世界/传感器/事后事件，不把隐藏位置传给VehicleAgent。"""

import hashlib
import json
from occluactisafe.common.clock import Timestamp
from occluactisafe.common.schemas import SourceTrack, EgoState
from occluactisafe.communication.channel import SimChannel
from occluactisafe.communication.protocol import ChannelConfig
from .pipeline import AnalyticPipeline
from .roadside_edge import make_prediction


def observed_track(x, now, source):
    """只有当前合法传感器可视时由环境调用，x不能从规划器反查。"""
    return SourceTrack("front-0" if source == "vehicle-A" else "edge-17", source, "session-1", "camera",
                       "vehicle", "target", "sim-n11", "map-right-handed", "geometric_center", now, now, now, now,
                       x, 0., 0., 0., 0., 4., 2., "declared_analytic_body", None, None, True, True, True, True, True, 1.)


def ego_state(state, now):
    return EgoState("vehicle-A", "sim-n11", "map-right-handed", "geometric_center", now, now, now,
                    state.s, 0., state.v, 0., 0., 4., 2., True, True)


def run_episode(target_x, strategy, information, *, duration=6., dt=.05, propagation=.2):
    if information not in ("local", "cooperative"): raise ValueError("未知信息来源轴")
    if strategy not in ("none", "ttc", "risk"): raise ValueError("未知控制轴")
    if dt != .05: raise ValueError("N11固定解析步长.05s")
    if target_x <= 4 or duration <= 0: raise ValueError("夹具需要正净间距/运行时长")
    pipeline = AnalyticPipeline(strategy)
    channel = SimChannel(ChannelConfig("I2V", "sim-n11", propagation_delay_s=propagation))
    initial = {"ego_x": 0., "ego_v": 12., "target_x": target_x, "body": [4.,2.], "dt": dt, "propagation": propagation}
    rows = []; events = []; first_contact = None
    for step in range(round(duration / dt) + 1):
        now = round(step * dt, 10); state = pipeline.executor.at(now)
        # 合成可见性规则冻结；感知输出仅当前状态，隐藏/未来状态留在环境内。
        local = (observed_track(target_x, now, "vehicle-A"),) if abs(target_x - state.s) <= 8. else ()
        if information == "cooperative" and step % 2 == 0:
            channel.send(make_prediction((observed_track(target_x, now, "roadside-1"),), now, step // 2), Timestamp(now,"sim-n11"))
        received = channel.deliver(Timestamp(now,"sim-n11"))
        row = pipeline.step(now, ego_state(state, now), local_tracks=local, received=received, unknown=not bool(local))
        row["sensor_local_visible"] = bool(local)
        row["received"] = [r.to_dict() for r in received]
        rows.append(row)
        if now >= duration: continue
        # 事后世界几何事件：静态目标整车净间距。独立于预测风险、不反馈决策。
        after = pipeline.executor.at(now + dt)
        if first_contact is None and state.s <= target_x + 4 and after.s >= target_x - 4:
            lo, hi = now, now + dt
            if state.s >= target_x - 4: hi = now
            else:
                for _ in range(45):
                    mid = (lo + hi) / 2
                    if pipeline.executor.at(mid).s >= target_x - 4: hi = mid
                    else: lo = mid
            impact = pipeline.executor.at(hi)
            first_contact = {"kind": "analytic_world_geometry_contact", "time": hi, "ego_speed_mps": impact.v,
                             "relative_speed_mps": impact.v, "carla_collision": None, "near_conflict_proxy": None}
            events.append(first_contact)
    digest = lambda value: hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    return {"initial": initial, "initial_sha256": digest(initial), "exogenous_schedule_sha256": digest({"roadside_period":.1,"local_range":8.,"propagation":propagation}),
            "strategy": strategy,"information": information,"rows": rows,"events": events,"channel_events": channel.events,
            "requests": pipeline.executor.requests,"final": pipeline.executor.at(duration).__dict__,
            "episode_replay_sha256": digest(rows),"actual_collision": None,"analytic_contact": bool(events)}
