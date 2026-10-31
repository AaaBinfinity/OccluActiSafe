"""N14评测旁路：实际碰撞、OBB重叠和配对身份分别记录。"""
import hashlib
import json
from occluactisafe.risk.obb import OBB,intersects


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def overlap(ego,target):
    def box(s):return OBB((s['x'],s['y']),s['yaw'],s['length'],s['width'])
    return intersects(box(ego),box(target))
