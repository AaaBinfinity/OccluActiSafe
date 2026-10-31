"""纵向夹具门槛与AEB锁存，不实现AES或概率校准。"""

from dataclasses import dataclass
from occluactisafe.control.fallback import fallback_reason


@dataclass
class AEBDecision:
    strategy: str = "risk"
    latched: bool = False

    def __post_init__(self):
        if self.strategy not in ("none", "ttc", "risk"): raise ValueError("仅支持无干预/TTC-AEB/预测风险AEB")

    def choose(self, table, ttc, *, unknown, prediction_missing):
        rows = {r["candidate_id"]: r for r in table["candidates"]}
        normal = rows["NORMAL"]
        high = ((ttc["ttc_s"] is not None and ttc["ttc_s"] <= 2.5) if self.strategy == "ttc"
                else (normal["R_max"] is not None and normal["R_max"] >= .5))
        if self.strategy == "none": return {"action": "NORMAL", "reason": "reference_no_intervention", "no_safe_candidate_found": False}
        if high: self.latched = True
        no_safe = not any(not r["normal_blocked"] and r["R_max"] == 0 for r in rows.values())
        reasons = fallback_reason(unknown=unknown, coverage_complete=table["coverage"]["complete"], prediction_missing=prediction_missing)
        if self.latched: return {"action": "AEB", "reason": "emergency_latched" + ("_mitigation" if no_safe else ""), "no_safe_candidate_found": no_safe}
        if reasons or normal["normal_blocked"]:
            return {"action": "DEGRADED", "reason": "|".join(reasons or normal["block_reasons"]), "no_safe_candidate_found": no_safe}
        if normal["R_max"] and normal["R_max"] > .1:
            return {"action": "SLOW", "reason": "mechanism_caution_score", "no_safe_candidate_found": no_safe}
        return {"action": "NORMAL", "reason": "declared_clear_current_window", "no_safe_candidate_found": no_safe}
