"""M07米制最优模式SmoothL1与模式分类；赢家仅供离线监督。"""
import torch
from torch.nn import functional as F


def multimodal_loss(out, truth, future_mask, target_mask=None, lambda_mode=1.0):
    prediction = out['pred']; logits = out['mode_logits']
    if truth.shape != prediction.shape[:1] + prediction.shape[2:] or future_mask.shape != truth.shape[:-1] or future_mask.dtype != torch.bool:
        raise ValueError('监督形状或掩码非法')
    if not torch.isfinite(truth[future_mask]).all():
        raise ValueError('有效未来GT非有限')
    valid = out['prediction_valid'] & future_mask.any(-1)
    if target_mask is not None:
        valid = valid & target_mask
    if not valid.any():
        return None, {'valid_targets': 0}
    safe = torch.where(future_mask[..., None], truth, torch.zeros_like(truth))
    distance = torch.linalg.vector_norm(prediction - safe[:, None], dim=-1)
    count = future_mask.sum(-1).clamp_min(1)
    ade = torch.where(future_mask[:, None], distance, torch.zeros_like(distance)).sum(-1) / count[:, None]
    winner = ade.detach().argmin(-1)
    selected = prediction[torch.arange(len(prediction), device=prediction.device), winner]
    element = F.smooth_l1_loss(selected, safe, reduction='none', beta=1.0).sum(-1)
    reg = (torch.where(future_mask, element, torch.zeros_like(element)).sum(-1) / (2 * count))[valid].mean()
    classification = F.cross_entropy(logits[valid], winner[valid])
    return reg + lambda_mode * classification, {'valid_targets': int(valid.sum()), 'reg': float(reg.detach()),
        'mode': float(classification.detach()), 'train_minADE': float(ade.min(-1).values[valid].mean().detach()),
        'winner': winner.detach().cpu().tolist()}
