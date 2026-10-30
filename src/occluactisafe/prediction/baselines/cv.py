"""米制单模式 CV：最后两次有效位置的真实时间差，无未来输入。"""
import numpy as np


def predict(history, timestamps, mask, *, future_times, anchor_time=0.0):
    """输入[N,H,2]、[H]、[N,H]；输出[N,1,F,2]及逐目标有效性。

    一点/全空历史无速度，明确invalid；缺测可跨格差分；转向沿最后线段。
    先把最后观测外推到anchor，再按相对anchor的future_times外推。
    """
    x = np.asarray(history, dtype=float)
    t = np.asarray(timestamps, dtype=float)
    m = np.asarray(mask)
    future = np.asarray(future_times, dtype=float)
    if x.ndim != 3 or x.shape[-1] != 2 or t.shape != (x.shape[1],) or m.shape != x.shape[:2] or m.dtype.kind != 'b':
        raise ValueError('CV位置、时间或掩码形状非法')
    if not np.isfinite(anchor_time) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0) or np.any(t > anchor_time):
        raise ValueError('CV历史时间必须递增且不晚于anchor')
    if future.ndim != 1 or not future.size or not np.isfinite(future).all() or np.any(future <= 0) or np.any(np.diff(future) <= 0):
        raise ValueError('未来相对时间必须为递增正数')
    if not np.isfinite(x[m]).all():
        raise ValueError('有效历史位置非有限')
    output = np.zeros((len(x), 1, len(future), 2))
    velocity = np.zeros((len(x), 2))
    valid = np.zeros(len(x), dtype=bool)
    last_seen = np.full(len(x), np.nan)
    for i in range(len(x)):
        indices = np.flatnonzero(m[i])
        if len(indices):
            last_seen[i] = t[indices[-1]]
        if len(indices) < 2:
            continue
        a, b = indices[-2:]
        velocity[i] = (x[i, b] - x[i, a]) / (t[b] - t[a])
        output[i, 0] = x[i, b] + (anchor_time - t[b] + future[:, None]) * velocity[i]
        valid[i] = True
    if not np.isfinite(output).all():
        raise ValueError('CV外推溢出')
    return {'positions': output, 'valid': valid, 'velocity': velocity,
            'last_seen_time': last_seen, 'K': 1}


def predict_state(position, velocity, *, state_time, last_seen_time, anchor_time, future_times):
    """已外推状态按state_time继续外推，不重新从last_seen_time起算。"""
    p, v, f = np.asarray(position, float), np.asarray(velocity, float), np.asarray(future_times, float)
    if p.shape != (2,) or v.shape != (2,) or f.ndim != 1 or not f.size:
        raise ValueError('状态形状非法')
    if not np.isfinite(np.r_[p, v, f, state_time, last_seen_time, anchor_time]).all() or not last_seen_time <= state_time <= anchor_time or np.any(f < 0) or np.any(np.diff(f) <= 0):
        raise ValueError('状态时间或有限性非法')
    return {'positions': p + (anchor_time - state_time + f[:, None]) * v,
            'last_seen_time': last_seen_time, 'state_time': state_time, 'anchor_time': anchor_time}
