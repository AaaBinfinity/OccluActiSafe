"""给定纵向候选及当前状态CV-TTC，复用N10连续区间几何。"""

import math
from occluactisafe.common.coordinates import Pose2D
from occluactisafe.perception.participants import ParticipantGeometry, geometry_from_track
from occluactisafe.risk.coverage import Candidate
from occluactisafe.risk.occupancy import PoseTrajectory, first_conflict_time


def longitudinal_candidates(executor, now, domain, frame, horizon=3.):
    result = []
    for name, limits in (("NORMAL", None), ("SLOW", (1.5, 4.)), ("AEB", (4.5, 8.))):
        feasible = True
        try: profile, started = executor.preview(now, *(limits or (None, None)))
        except ValueError:
            profile, started = executor.preview(now); feasible = False
        duration = horizon + executor.delay
        times = [now + i * .05 for i in range(math.ceil(duration / .05))]
        times.append(now + duration)
        poses = tuple(Pose2D(profile.at(t - started).s, 0., 0.) for t in times)
        result.append(Candidate(name, "queued_longitudinal_" + name.lower(), PoseTrajectory(tuple(times), poses, domain, frame),
                                ParticipantGeometry("vehicle", 4., 2.), feasible))
    return tuple(result)


def compute_ttc_cv(ego, tracks, *, now, horizon=3.):
    """只接当前合法SourceTrack。返回首次潜在区间的保守左端；无事件为None。"""
    if not ego.velocity_valid or not ego.yaw_valid: raise ValueError("TTC自车状态无效")
    if ego.state_time > now or ego.source_available_time > now: raise ValueError("TTC不能采用未来自车状态")
    if ego.reference_point != "geometric_center": raise ValueError("TTC自车需要显式中心参考")
    a = PoseTrajectory((now, now + horizon), tuple(Pose2D(ego.x + ego.vx * (now - ego.state_time + dt), ego.y + ego.vy * (now - ego.state_time + dt), ego.yaw) for dt in (0., horizon)), ego.clock_domain, ego.coordinate_frame_id)
    geometry = ParticipantGeometry("vehicle", ego.length, ego.width)
    events = []
    for track in tracks:
        if track.source_available_time > now or track.state_time > now or track.last_seen_time > now: raise ValueError("TTC不能采用未来目标")
        if (track.clock_domain, track.coordinate_frame_id) != (ego.clock_domain, ego.coordinate_frame_id): raise ValueError("TTC时钟/坐标不一致")
        if not track.velocity_valid: raise ValueError("TTC目标速度无效")
        body = geometry_from_track(track, pedestrian_radius=.3 if track.category == "pedestrian" else None)
        b = PoseTrajectory((now, now + horizon), tuple(Pose2D(track.x + track.vx * (now + dt - track.state_time),
                            track.y + track.vy * (now + dt - track.state_time), track.yaw or 0.) for dt in (0., horizon)), ego.clock_domain, ego.coordinate_frame_id)
        conflict = first_conflict_time(a, geometry, b, body, now, now + horizon)
        if conflict.event: events.append(conflict.event)
    event = min(events, key=lambda e: e["interval_start"]) if events else None
    return {"ttc_s": max(0., event["interval_start"] - now) if event else None, "event": event,
            "kind": "declared_cv_geometry", "horizon_s": horizon}
