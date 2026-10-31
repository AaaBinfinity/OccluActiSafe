"""未校准模式命中权重与多目标汇总；不产生控制动作。"""

import math
from occluactisafe.perception.visibility import assess_constraints
from .coverage import common_coverage
from .occupancy import first_conflict_time, IntervalConfig


def pairwise_mode_risk(candidate, target, start, end, config=IntervalConfig()):
    """同一模式整窗只计一次命中；不把逐时刻接触相加。"""
    modes = []; traces = []; hits = []
    for mode in target.modes:
        if mode.weight == 0: continue
        result = first_conflict_time(candidate.path, candidate.geometry, mode.path, target.geometry, start, end, config)
        modes.append({"target_id": target.target_id, "mode_id": mode.mode_id, "weight": mode.weight,
                      "heading_assumption": target.heading_assumption, "category": target.geometry.category,
                      "hit": result.potential_contact, "event": result.event, "interval_algorithm_complete": result.algorithm_complete})
        hits.append(mode.weight if result.potential_contact else 0.)
        traces.extend({"candidate_id": candidate.candidate_id, "target_id": target.target_id, "mode_id": mode.mode_id,
                       "window": "future", **t} for t in result.trace)
    return {"target_id": target.target_id, "raw_score": min(1., math.fsum(hits)), "modes": modes, "trace": traces}


def risk_table_for_candidates(candidates, targets, *, expected_target_ids, now, plan_start, horizon_s=3., knowledge=None, interval_config=IntervalConfig()):
    candidates, targets = tuple(candidates), tuple(targets)
    if not candidates or len({c.candidate_id for c in candidates}) != len(candidates): raise ValueError("候选不能为空或ID重复")
    domains = {(c.path.clock_domain, c.path.coordinate_frame_id) for c in candidates}
    domains.update((m.path.clock_domain, m.path.coordinate_frame_id) for t in targets for m in t.modes)
    if len(domains) != 1: raise ValueError("候选/目标必须同一时钟和坐标")
    coverage = common_coverage(targets, expected_target_ids=expected_target_ids, now=now, plan_start=plan_start, horizon_s=horizon_s)
    end = coverage["common_end"]; rows = []
    for candidate in candidates:
        reasons = []; prefix_rows = []; mode_rows = []; target_rows = []; traces = []
        stop = candidate.path.contiguous_end(now)
        candidate_complete = stop is not None and stop >= plan_start + horizon_s
        if not candidate_complete: reasons.append("candidate_path_incomplete")
        if not candidate.physically_feasible: reasons.append("declared_physically_infeasible")
        if not coverage["complete"]: reasons.append("target_coverage_incomplete")
        if not coverage["prefix_complete"]: reasons.append("prefix_coverage_incomplete")
        score_available = candidate_complete and candidate.physically_feasible and not coverage["gaps"] and end > plan_start
        if candidate_complete:
            constraints = assess_constraints(candidate.path, candidate.geometry, knowledge, now, plan_start + horizon_s, interval_config)
        else: constraints = {"blocked": True, "reasons": ["candidate_path_incomplete"], "actual_collision": None}
        reasons.extend(constraints["reasons"])
        for target in targets:
            for mode in target.modes:
                if mode.weight == 0: continue
                row = {"target_id": target.target_id, "mode_id": mode.mode_id, "weight": mode.weight,
                       "heading_assumption": target.heading_assumption, "category": target.geometry.category}
                prefix_stop = mode.path.contiguous_end(now)
                if candidate_complete and target.adoption_gap(now) is None and prefix_stop is not None and prefix_stop >= plan_start:
                    prefix = first_conflict_time(candidate.path, candidate.geometry, mode.path, target.geometry, now, plan_start, interval_config)
                    prefix_rows.append({**row, "event": prefix.event})
                    traces.extend({"candidate_id": candidate.candidate_id, "target_id": target.target_id, "mode_id": mode.mode_id, "window": "prefix", **t} for t in prefix.trace)
                    if prefix.potential_contact: reasons.append("prefix_potential_contact")
                if not score_available:
                    mode_rows.append({**row, "hit": None, "event": None, "interval_algorithm_complete": None})
            if score_available:
                pair = pairwise_mode_risk(candidate, target, plan_start, end, interval_config)
                mode_rows.extend(pair["modes"]); traces.extend(pair["trace"])
            target_rows.append({"target_id": target.target_id, "raw_score": pair["raw_score"] if score_available else None})
        values = [t["raw_score"] for t in target_rows]
        rows.append({"candidate_id": candidate.candidate_id, "reference_policy": candidate.reference_policy,
                     "window_start": plan_start, "window_end": end, "prefix_start": now, "prefix_end": plan_start,
                     "risk_kind": "uncalibrated_score", "R_max": max(values, default=0.) if score_available else None,
                     "R_sum": min(1., math.fsum(values)) if score_available else None, "targets": target_rows, "modes": mode_rows,
                     "prefix": prefix_rows, "normal_blocked": bool(reasons), "block_reasons": sorted(set(reasons)),
                     "constraints": constraints, "trace": traces, "calibrated_event_probability": None,
                     "actual_collision": None, "near_conflict_proxy_event": None, "simulation_counterfactual_event": None,
                     "action_approval": None})
    return {"coverage": coverage, "candidates": rows, "reference_policy_contract": "declared_input_paths_no_action_selection"}


evaluate_candidates = risk_table_for_candidates
