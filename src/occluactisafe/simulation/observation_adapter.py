"""核心端的显式白名单：不会接收真值或未来帧。"""
import base64
import hashlib
import zlib
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


def decode_preview_tick(message):
    """可视诊断传输层解复用；核心观测仍经过原始严格白名单。"""
    if not isinstance(message,dict) or set(message)!={'observation','display'}:raise ValueError('可视传输字段非法')
    event,camera=decode_tick(message['observation'])
    display=message['display']
    if not isinstance(display,dict) or set(display)!={'frame','valid','width','height','encoding','data','reason'}:
        raise ValueError('显示相机字段非法')
    if type(display['frame']) is not int or display['frame']!=event.frame_id or type(display['valid']) is not bool or display['encoding']!='zlib_bgra':
        raise ValueError('显示相机帧或编码非法')
    raw=None;digest=None
    if display['valid']:
        if (display['width'],display['height'])!=(1280,720) or display['reason'] is not None:raise ValueError('显示相机尺寸非法')
        compressed=base64.b64decode(display['data'],validate=True);decoder=zlib.decompressobj()
        raw=decoder.decompress(compressed,1280*720*4+1)
        if len(raw)!=1280*720*4 or not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:raise ValueError('显示相机解压长度非法')
        digest=hashlib.sha256(display['data'].encode()).hexdigest()
    elif any(display[key] is not None for key in ('data','width','height')) or display['reason'] not in ('timeout','closed'):
        raise ValueError('显示无效帧携带数据')
    return event,camera,{**display,'raw_bgra':raw,'transport_sha256':digest}
