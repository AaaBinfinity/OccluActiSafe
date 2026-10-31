"""N11合法状态→CV→N02预测消息，无真值或未来标签入口。"""

from occluactisafe.common.schemas import PredictionMessage, PredictionTarget
from occluactisafe.prediction.baselines.cv import predict_state


def make_prediction(tracks, now, sequence, *, source_id="roadside-1", domain="sim-n11", frame="map-right-handed"):
    relative = tuple((i + 1) / 10 for i in range(40)); targets = []
    for track in tracks:
        track.to_dict()
        if track.source_available_time > now or track.state_time > now: raise ValueError("不可使用尚未可用状态")
        if (track.clock_domain, track.coordinate_frame_id) != (domain, frame): raise ValueError("CV时钟/坐标不一致")
        if not track.velocity_valid: raise ValueError("CV需要有效速度")
        result = predict_state((track.x, track.y), (track.vx, track.vy), state_time=track.state_time,
                               last_seen_time=track.last_seen_time, anchor_time=now, future_times=relative)
        targets.append(PredictionTarget(track, (tuple(tuple(p) for p in result["positions"]),), (1.,), ((True,) * 40,)))
    return PredictionMessage("1.0", "prediction", source_id, "session-1", sequence, "vehicle-A", "map-A", "1", frame,
                             sequence, domain, "n11-declared-cv-no-weights", now, now, now, now, now, now + .5,
                             tuple(now + dt for dt in relative), now + 4., tuple(targets))
