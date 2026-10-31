"""有界、按帧严格配对；未来帧不用于填补当前帧。"""
from threading import Condition
import time


class FrameBuffer:
    def __init__(self, *, capacity=64):
        if type(capacity) is not int or capacity < 1: raise ValueError("capacity必须为正整数")
        self.capacity = capacity; self._frames = {}; self._condition = Condition(); self._closed = False

    def put(self, frame, value):
        if type(frame) is not int or frame < 0: raise ValueError("frame必须为非负整数")
        with self._condition:
            if self._closed: return False
            if frame in self._frames: return False
            if len(self._frames) >= self.capacity: self._frames.pop(min(self._frames))
            self._frames[frame] = value; self._condition.notify_all(); return True

    def get(self, frame, timeout_s):
        if type(frame) is not int or frame < 0 or not isinstance(timeout_s, (float,int)) or isinstance(timeout_s,bool) or not 0 <= timeout_s < float('inf'):
            raise ValueError("帧号或超时非法")
        deadline = time.monotonic() + timeout_s
        with self._condition:
            for old in [key for key in self._frames if key < frame]: del self._frames[old]
            while frame not in self._frames and not self._closed:
                left = deadline - time.monotonic()
                if left <= 0: break
                self._condition.wait(left)
            if frame in self._frames: return {'frame': frame, 'valid': True, 'value': self._frames.pop(frame)}
            return {'frame': frame, 'valid': False, 'value': None, 'reason': 'closed' if self._closed else 'timeout'}

    def close(self):
        with self._condition:
            self._closed = True; self._condition.notify_all()
