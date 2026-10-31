"""仅桥接进程使用的评测旁路。"""
from occluactisafe.evaluation.truth_store import TruthStore


class TruthLogger:
    def __init__(self, path): self._store = TruthStore(path)
    def log_ego(self, frame, actor):
        transform = actor.get_transform()
        self._store.append({'frame': frame, 'ego_id': actor.id, 'x_m': transform.location.x,
                            'y_m': transform.location.y, 'z_m': transform.location.z})
    def close(self): self._store.close()
