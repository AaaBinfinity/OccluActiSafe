"""解析信息不足时的显式谨慎制动；不承诺盲区无残余冲突。"""


def fallback_reason(*, unknown, coverage_complete, prediction_missing):
    reasons = []
    if unknown: reasons.append("unknown_region")
    if not coverage_complete: reasons.append("coverage_incomplete")
    if prediction_missing: reasons.append("known_prediction_missing")
    return reasons
