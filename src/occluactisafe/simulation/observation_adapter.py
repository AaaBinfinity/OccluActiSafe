"""核心端的显式白名单：不会接收真值或未来帧。"""
import base64
from occluactisafe.common.schemas import ControlResult, TickEvent
from occluactisafe.core.safety_monitor import check_control


def decode_tick(message):
    if not isinstance(message, dict) or set(message) != {'event', 'camera'}:
        raise ValueError('桥接响应含未知字段或缺少字段')
    event = TickEvent.from_dict(message['event'])
    camera = message['camera']
    if not isinstance(camera, dict) or set(camera) != {'frame', 'valid', 'width', 'height', 'bgra', 'reason'}:
        raise ValueError('摄像头观测字段不符合白名单')
    if type(camera['frame']) is not int or camera['frame'] != event.frame_id or type(camera['valid']) is not bool:
        raise ValueError('摄像头帧或有效性错误')
    if camera['valid']:
        if camera['reason'] is not None or camera['width'] != 320 or camera['height'] != 180:
            raise ValueError('摄像头尺寸或状态错误')
        raw = base64.b64decode(camera['bgra'], validate=True)
        if len(raw) != 320 * 180 * 4: raise ValueError('摄像头图像长度错误')
    elif any(camera[k] is not None for k in ('width', 'height', 'bgra')) or camera['reason'] not in ('timeout', 'closed'):
        raise ValueError('无效帧必须明确原因且不携带图像')
    return event, camera


def stationary_control(event, index):
    """N12 静止同步测试的安全复核控制；不是碰撞规避策略。"""
    if not isinstance(event, TickEvent): raise TypeError('只接受已校验的合法 tick')
    action = {'throttle': 0.0, 'brake': 1.0, 'steer': 0.0}
    check_control(action)
    return ControlResult(str(index), 'stationary', 'success', True, event.clock_domain,
                         event.sim_time, event.sim_time + 1.0, **action, fallback_reason=None)
