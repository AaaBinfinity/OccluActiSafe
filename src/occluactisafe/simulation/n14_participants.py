"""N14开发案例的外部参与者脚本；不向在线策略暴露计划或未来状态。"""
import math


def participant_specs(config):
    primary = dict(role='target', track_id=config.get('target_track_id','crossing-1'), category=config.get('target_category', 'vehicle'),
                   blueprint=config.get('target_blueprint', 'vehicle.audi.a2'), spawn=config['target_spawn'],
                   speed_mps=config.get('target_speed_mps', config['speed_mps']),
                   start_s=config.get('target_start_s', 0.))
    specs = [primary, *config.get('additional_targets', [])]
    roles = set(); identities = set()
    for spec in specs:
        if spec['role'] in roles or spec['role'] in ('ego', 'occluder') or spec['track_id'] in identities:
            raise ValueError('参与者语义身份重复')
        roles.add(spec['role']); identities.add(spec['track_id'])
        if spec['category'] not in ('vehicle', 'pedestrian'): raise ValueError('N14未支持的参与者类别')
        if any(not math.isfinite(spec[k]) or spec[k] < 0 for k in ('speed_mps', 'start_s')):
            raise ValueError('参与者速度/启动时间非法')
        if spec['category'] == 'pedestrian' and not spec['blueprint'].startswith('walker.pedestrian.'):
            raise ValueError('行人必须使用真实walker蓝图')
    return specs


def walker_action(spec, now, *, held=False):
    yaw = math.radians(spec['spawn']['yaw'])
    return {'direction': {'x': math.cos(yaw), 'y': math.sin(yaw), 'z': 0.},
            'speed': 0. if held or now < spec['start_s'] else spec['speed_mps'], 'jump': False}


def batch_control(role, actor_id, spec, value, now=0.):
    if spec['category'] == 'pedestrian':
        if 'direction' not in value: value = walker_action(spec, now, held=True)
        return {'kind': 'walker_control', 'actor_id': actor_id, 'value': value}
    return {'kind': 'control', 'actor_id': actor_id, 'value': value}
