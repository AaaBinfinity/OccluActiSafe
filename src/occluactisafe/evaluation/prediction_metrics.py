"""逐目标等权的米制预测指标；min指标仅用于离线评分。"""
import numpy as np


def score_target(prediction, truth, mask, *, prediction_valid=True, top_k=1,
                 miss_threshold_m=2.0, logits=None):
    """[K,F,2]与[F,2]。FDE取最后有效监督点；无标签不计分。

    ADE/FDE使用最高logit模式；minADE和minFDE分别取top-K最小值。
    无预测计miss，误差为None，汇总必须同时报告缺预测分母。
    """
    p, y, m = np.asarray(prediction, float), np.asarray(truth, float), np.asarray(mask)
    if y.ndim != 2 or y.shape[-1] != 2 or m.shape != y.shape[:1] or m.dtype.kind != 'b' or p.ndim != 3 or p.shape[1:] != y.shape or p.shape[0] < 1:
        raise ValueError('指标张量形状非法')
    if type(top_k) is not int or not 1 <= top_k <= len(p) or not np.isfinite(miss_threshold_m) or miss_threshold_m <= 0:
        raise ValueError('top-K或Miss阈值非法')
    if not np.isfinite(y[m]).all():
        raise ValueError('有效未来真值非有限')
    base = {'valid_steps': int(m.sum()), 'ADE_m': None, 'FDE_m': None,
            'minADE_m': None, 'minFDE_m': None, 'miss': None}
    if not m.any():
        return {**base, 'status': 'no_labels'}
    if not prediction_valid:
        return {**base, 'status': 'missing_prediction', 'miss': True}
    if not np.isfinite(p[:, m]).all():
        raise ValueError('有效预测非有限')
    if logits is None:
        order = np.arange(len(p))
    else:
        logits = np.asarray(logits, float)
        if logits.shape != (len(p),) or not np.isfinite(logits).all():
            raise ValueError('模式logits非法')
        order = np.argsort(-logits, kind='stable')
    distance = np.linalg.norm(p[order[:top_k]][:, m] - y[m], axis=-1)
    ade, fde = distance.mean(axis=1), distance[:, -1]
    return {**base, 'status': 'scored', 'ADE_m': float(ade[0]), 'FDE_m': float(fde[0]),
            'minADE_m': float(ade.min()), 'minFDE_m': float(fde.min()),
            'miss': bool(fde.min() > miss_threshold_m)}


class MetricAccumulator:
    def __init__(self):
        self.total = self.scored = self.missing = self.no_labels = self.misses = 0
        self.sums = dict.fromkeys(('ADE_m', 'FDE_m', 'minADE_m', 'minFDE_m'), 0.0)

    def add(self, row):
        self.total += 1
        if row['status'] == 'no_labels':
            self.no_labels += 1
            return
        self.misses += int(row['miss'])
        if row['status'] == 'missing_prediction':
            self.missing += 1
            return
        if row['status'] != 'scored':
            raise ValueError('未知评分状态')
        self.scored += 1
        for key in self.sums:
            self.sums[key] += row[key]

    def summary(self):
        denominator = self.scored + self.missing
        return {'targets': self.total, 'scored': self.scored, 'missing_prediction': self.missing,
                'no_labels': self.no_labels, 'misses': self.misses, 'MissRate_denominator': denominator,
                'MissRate': self.misses / denominator if denominator else None,
                'error_denominator': self.scored,
                **{k: v / self.scored if self.scored else None for k, v in self.sums.items()}}
