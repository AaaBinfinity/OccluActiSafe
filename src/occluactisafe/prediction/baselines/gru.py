"""两层掩码GRUCell；独立目标时序基线，不接收未来或图/地图。"""
import torch
from torch import nn


class MultiModalGRU(nn.Module):
    def __init__(self, *, future_steps, hidden_size=128, modes=6, dropout=.1):
        super().__init__()
        self.future_steps, self.modes = future_steps, modes
        self.cell1 = nn.GRUCell(10 + 10 + 1 + 1, hidden_size)
        self.cell2 = nn.GRUCell(hidden_size, hidden_size)
        self.static = nn.Linear(hidden_size + 4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.delta = nn.Linear(hidden_size, modes * future_steps * 2)
        self.logits = nn.Linear(hidden_size, modes)

    def forward(self, *, features, feature_mask, observed_mask, actual_dt,
                history_times, categories, anchor_position):
        # 统一批次采用目标维[M,H,D]，同一目标的上下文不参与此普通GRU。
        if features.ndim != 3 or features.shape[-1] != 10 or feature_mask.shape != features.shape or observed_mask.shape != features.shape[:2] or actual_dt.shape != observed_mask.shape:
            raise ValueError('GRU历史/掩码形状非法')
        if feature_mask.dtype != torch.bool or observed_mask.dtype != torch.bool:
            raise ValueError('掩码必须为bool')
        if history_times.ndim == 1:
            history_times = history_times.expand(features.shape[0], -1)
        if history_times.shape != observed_mask.shape or anchor_position.shape != (features.shape[0], 2):
            raise ValueError('GRU时间或锚点形状非法')
        if not torch.isfinite(features[feature_mask]).all() or not torch.isfinite(actual_dt[observed_mask]).all() or not torch.isfinite(history_times).all() or not torch.isfinite(anchor_position).all():
            raise ValueError('有效输入非有限')
        valid_steps = observed_mask & feature_mask[..., :2].all(-1)
        clean = torch.where(feature_mask & valid_steps[..., None], features, torch.zeros_like(features))
        dt = torch.where(valid_steps, actual_dt, torch.zeros_like(actual_dt))
        if (dt < 0).any() or (history_times > 1e-7).any() or (history_times[:, 1:] <= history_times[:, :-1]).any():
            raise ValueError('历史时间或真实dt非法')
        h1 = features.new_zeros((len(features), self.cell1.hidden_size)); h2 = h1.clone()
        for t in range(features.shape[1]):
            inp = torch.cat((clean[:, t], feature_mask[:, t].to(features.dtype) * valid_steps[:, t, None],
                             dt[:, t, None], history_times[:, t, None].to(features.dtype)), -1)
            proposed1 = self.cell1(inp, h1)
            proposed2 = self.cell2(proposed1, h2)
            h1 = torch.where(valid_steps[:, t, None], proposed1, h1)
            h2 = torch.where(valid_steps[:, t, None], proposed2, h2)
        hidden = self.dropout(torch.relu(self.static(torch.cat((h2, torch.nn.functional.one_hot(categories, 4).to(features.dtype)), -1))))
        delta = self.delta(hidden).reshape(-1, self.modes, self.future_steps, 2)
        # 米制增量解码；锚位置保留float64，避免大地理偏移的float32相消。
        prediction = anchor_position[:, None, None, :] + delta.to(anchor_position.dtype).cumsum(2)
        valid = valid_steps.any(1)
        logits = self.logits(hidden)
        weights = torch.softmax(logits, -1) * valid[:, None]
        return {'pred': torch.where(valid[:, None, None, None], prediction, torch.zeros_like(prediction)),
                'mode_logits': logits, 'weights': weights, 'prediction_valid': valid}
