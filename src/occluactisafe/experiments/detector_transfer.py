"""N17可检验的分类输出行迁移；只允许语义一一对应的人类类别。"""
import copy


def apply_class_aliases(model, aliases):
    aliases = dict(aliases or {})
    if aliases not in ({}, {'person': 'pedestrian'}):
        raise ValueError('本轮只允许person→pedestrian；禁止合并类重名覆盖')
    names = dict(model.model.names)
    if aliases:
        if list(names.values()).count('person') != 1 or 'pedestrian' in names.values():
            raise ValueError('源权重的person身份不唯一，拒绝猜测迁移')
        model.model.names = {k: aliases.get(v, v) for k, v in names.items()}
    return {'aliases': aliases, 'source_names': names, 'aliased_names': dict(model.model.names)}


def verify_person_transfer(model, classes):
    """在CPU构造目标头，验证框架确实逐张量复制person权重；不是精度实验。"""
    import torch
    from ultralytics.nn.tasks import DetectionModel
    src = model.model
    if list(src.names.values()).count('pedestrian') != 1 or 'pedestrian' not in classes:
        raise ValueError('必须先应用唯一person别名')
    source_row = next(k for k, v in src.names.items() if v == 'pedestrian')
    target_row = classes.index('pedestrian')
    target = DetectionModel(copy.deepcopy(src.yaml), nc=len(classes), ch=3, verbose=False)
    target.names = dict(enumerate(classes))
    target.load(src, verbose=False)
    src_state, dst_state = src.state_dict(), target.state_dict()
    checks = []
    src_head, dst_head = src.model[-1], target.model[-1]
    prefix = f'model.{len(target.model)-1}.'
    for branch in ('cv3', 'one2one_cv3'):
        layers = getattr(dst_head, branch, None)
        if layers is None:
            continue
        for i, seq in enumerate(layers):
            for parameter in ('weight', 'bias'):
                key = f'{prefix}{branch}.{i}.{len(seq)-1}.{parameter}'
                if key not in src_state or key not in dst_state:
                    raise ValueError(f'分类头参数结构不兼容：{key}')
                if not torch.equal(dst_state[key][target_row], src_state[key][source_row]):
                    raise ValueError(f'分类行未按预期复制：{key}')
                checks.append(key)
    if not checks:
        raise ValueError('没有找到可复核的分类输出行')
    return {'status': 'passed', 'verified_tensors': len(checks), 'tensor_keys': checks,
            'source_class_index': source_row, 'target_class_index': target_row,
            'scope': 'CPU张量迁移机制，不是训练或召回指标'}
