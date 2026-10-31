"""N10固定机制与开发图表；原始JSON/CSV逐区间保留，不执行控制。"""

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time
from uuid import uuid4
import numpy as np
from occluactisafe.common.config import PROJECT_ROOT, _yaml_file
from occluactisafe.common.coordinates import Pose2D
from occluactisafe.common.run_manifest import _source_identity
from occluactisafe.perception.participants import ParticipantGeometry
from occluactisafe.perception.visibility import GridSpec, KnownObstacle, SensorView, FreeSpaceEvidence, build_visibility
from .obb import OBB, Circle, intersects, obb_corners
from .occupancy import PoseTrajectory, IntervalConfig, first_conflict_time
from .coverage import ModeForecast, TargetForecast, Candidate
from .multimodal import evaluate_candidates


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def _grid_digest(grid):
    payload = {"visibility": grid.visibility.tolist(), "occupied": grid.occupied.tolist(), "free": grid.certified_free.tolist(),
               "regions": [asdict(r) for r in grid.regions], "sources": grid.accepted_sources}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def validate_config(config):
    config = deepcopy(config)
    if set(config) != {"schema_version", "episode_id", "clock_domain", "coordinate_frame_id", "output_root", "interval", "grid"}:
        raise ValueError("N10配置字段缺失或含未知字段")
    if config["schema_version"] != "n10-mechanism-1": raise ValueError("未知N10任务版本")
    for name in ("episode_id", "clock_domain", "coordinate_frame_id", "output_root"):
        if not isinstance(config[name], str) or not config[name].strip() or "://" in config[name]: raise ValueError(f"{name}须为公开标识/本地路径")
    for key, cls in (("interval", IntervalConfig), ("grid", GridSpec)):
        if not isinstance(config[key], dict) or set(config[key]) - set(cls.__dataclass_fields__): raise ValueError(f"{key}配置字段错误")
        config[key] = asdict(cls(**config[key]))
    root = Path(config["output_root"]).expanduser()
    config["output_root"] = str((root if root.is_absolute() else PROJECT_ROOT / root).resolve())
    return config


def load_diagnostic_config(config_file=None, *, overrides=None, cli_overrides=None):
    path = Path(config_file) if config_file is not None else PROJECT_ROOT / "configs/risk/n10.yaml"
    config = _yaml_file(path if path.is_absolute() else PROJECT_ROOT / path)
    for override in (overrides, cli_overrides):
        if override is not None:
            if not isinstance(override, dict) or set(override) - {"output_root"}: raise ValueError("只允许覆盖output_root；算法参数用显式YAML")
            config.update(override)
    return validate_config(config)


def _fixtures(config):
    domain, frame = config["clock_domain"], config["coordinate_frame_id"]
    car = ParticipantGeometry("vehicle", 4, 2)
    def path(x=0, y=0, end=4, start=0):
        return PoseTrajectory((start, end), (Pose2D(x, y, 0),) * 2, domain, frame)
    def target(identity="target", weight=.7, end=3):
        return TargetForecast(identity, car, (ModeForecast("hit", weight, path(end=end)), ModeForecast("miss", 1 - weight, path(y=10, end=end))), 0, 0, 0, 0, .5)
    def candidate(identity="candidate", x=0, y=0, feasible=True):
        return Candidate(identity, "declared_mechanism_path", path(x, y), car, feasible)
    spec = GridSpec(**config["grid"]); ones = np.ones(spec.shape, bool)
    view = SensorView("V", Pose2D(-6, 0, 0), math.pi, 30, 1, 0, 0, 1, domain, frame)
    wall = KnownObstacle("known-wall", OBB((3, 0), 0, .5, 8), 0, domain, frame)
    def knowledge(views=(), obstacles=(), free_evidence=(), closed_domain=False):
        return build_visibility(spec, now=0, clock_domain=domain, coordinate_frame_id=frame, views=views,
                                obstacles=obstacles, free_evidence=free_evidence, drivable=ones, closed_domain=closed_domain)
    full_view = SensorView("V", Pose2D(0, 0, 0), 2 * math.pi, 50, 1, 0, 0, 1, domain, frame)
    proof = FreeSpaceEvidence("V", ones, 0, 0, 1, domain, frame)
    grids = {"known_wall": knowledge((view,), (wall,)), "both_blind": knowledge(), "visible_no_free": knowledge((full_view,)),
             "closed_free_assumption": knowledge((full_view,), free_evidence=(proof,), closed_domain=True)}
    def case(name, cs, ts, key="known_wall", plan_start=0, expected=None):
        return {"case_id": name, "candidates": cs, "targets": ts, "knowledge_key": key, "plan_start": plan_start,
                "expected_target_ids": [t.target_id for t in ts] if expected is None else expected}
    prefix = PoseTrajectory((0., .2, .5, 4.), (Pose2D(0, 0, 0), Pose2D(0, 0, 0), Pose2D(10, 0, 0), Pose2D(10, 0, 0)), domain, frame)
    pt = TargetForecast("prefix-only", car, (ModeForecast("single", 1, prefix),), 0, 0, 0, 0, .5)
    hole = PoseTrajectory((0., 1., 2., 4.), (Pose2D(0, 0, 0),) * 4, domain, frame, (True, True, False, True))
    ht = TargetForecast("masked", car, (ModeForecast("single", 1, hole),), 0, 0, 0, 0, .5)
    stale = TargetForecast("stale", car, (ModeForecast("single", 1, path()),), -.5, 0, 0, 0, 1, max_observation_age_s=.2)
    cases = [case("mode_weight", (candidate(),), (target(),)), case("two_targets", (candidate(),), (target("a"), target("b", .4))),
             case("short_coverage", (candidate("normal"), candidate("offset", y=6)), (target("long", end=4), target("short", end=2.8))),
             case("missing_known", (candidate(),), (), expected=["known-retained"]),
             case("both_blind", (candidate(),), (), "both_blind"), case("visible_no_free", (candidate(),), (), "visible_no_free"),
             case("closed_free", (candidate(),), (), "closed_free_assumption"), case("infeasible", (candidate(feasible=False),), (target(),)),
             case("prefix_hazard", (candidate(),), (pt,), plan_start=.5), case("body_boundary", (candidate(x=7),), ()),
             case("known_wall", (candidate(x=3),), ()), case("mask_hole", (candidate(),), (ht,)), case("stale_known", (candidate(),), (stale,))]
    crossing_a = PoseTrajectory((0., .5), (Pose2D(0, 0, 0),) * 2, domain, frame)
    crossing_b = PoseTrajectory((0., .5), (Pose2D(-5, 0, 0), Pose2D(5, 0, 0)), domain, frame)
    near_b = PoseTrajectory((0., .5), (Pose2D(-5, 2.001, 0), Pose2D(5, 2.001, 0)), domain, frame)
    return cases, grids, crossing_a, crossing_b, near_b


def render_diagnostic_plots(run_dir, *, output_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(run_dir); out = Path(output_dir) if output_dir else directory
    out.mkdir(parents=True, exist_ok=True)
    sources = ("interval_subdivision.json", "risk_rows.csv", "grids.json")
    crossing = json.loads((directory / sources[0]).read_text(encoding="utf-8"))
    with (directory / sources[1]).open(encoding="utf-8", newline="") as stream: rows = list(csv.DictReader(stream))
    grids = json.loads((directory / sources[2]).read_text(encoding="utf-8")); figures = {}
    def save(fig, stem):
        try:
            fig.tight_layout()
            for suffix in ("png", "svg"):
                name = f"{stem}.{suffix}"
                fig.savefig(out / name, dpi=140, metadata={"Date": None} if suffix == "svg" else None)
                figures[name] = _sha(out / name)
        finally: plt.close(fig)
    # 只绘制原始细分步骤；并非均匀时间序列或训练步。
    with plt.rc_context({"svg.hashsalt": "n10-diagnostic-1", "font.family": "DejaVu Sans"}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        a = OBB((0, 0), 0, 2, 2)
        for shape, label in ((a, "stationary"), (OBB((-5, 0), 0, 2, 2), "start"), (OBB((5, 0), 0, 2, 2), "end")):
            corners = list(obb_corners(shape)); corners.append(corners[0]); axes[0].plot(*zip(*corners), label=label)
        axes[0].plot([-5, 5], [0, 0], "--", label="declared interpolation")
        event = crossing["crossing"]["event"]
        axes[0].set_title(f"Potential contact interval [{event['interval_start']:.4f}, {event['interval_end']:.4f}] s")
        axes[0].set(xlabel="x (m)", ylabel="y (m)", ylim=(-2, 3), aspect="equal"); axes[0].legend(fontsize=7, loc="upper center", ncol=2)
        for key, label in (("crossing", "crossing"), ("near_miss", "conservative near miss")):
            trace = crossing[key]["trace"]
            axes[1].plot(range(1, len(trace) + 1), [t["interval_end"] - t["interval_start"] for t in trace], "o-", label=label)
        axes[1].axhline(.01, color="black", linestyle=":", label="leaf limit 0.01 s")
        axes[1].set(xlabel="Actual DFS interval step", ylabel="interval width (s)", title="Unsmooth, no step aggregation")
        axes[1].legend(fontsize=8); save(fig, "geometry_intervals")
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        labels = [f"{r['case_id']}/{r['candidate_id']}" for r in rows]; x = np.arange(len(rows))
        for field, dx, color in (("R_max", -.18, "tab:blue"), ("R_sum", .18, "tab:orange")):
            positions = [i for i, row in enumerate(rows) if row[field] != ""]
            axes[0].bar(np.array(positions) + dx, [float(rows[i][field]) for i in positions], width=.35, label=field, color=color)
            zero = [i for i in positions if float(rows[i][field]) == 0]
            axes[0].plot(np.array(zero) + dx, [0.] * len(zero), "o", color=color, markersize=4)
        missing = [i for i, row in enumerate(rows) if row["R_max"] == ""]
        axes[0].plot(missing, [-.05] * len(missing), "x", color="gray", label="NA (not scored)")
        axes[0].set(ylabel="uncalibrated score", ylim=(-.1, 1.15), title="Zero = circle; NA = cross; no joint probability or action approval")
        axes[0].legend()
        axes[1].bar(x, [float(r["effective_horizon_s"]) for r in rows], label="common window")
        axes[1].axhline(3, linestyle=":", color="black", label="requested 3 s")
        axes[1].set(ylabel="future coverage (s)", xticks=x, xticklabels=labels)
        axes[1].tick_params(axis="x", labelrotation=65, labelsize=8); axes[1].legend()
        save(fig, "raw_scores_coverage")
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for ax, (name, grid) in zip(axes.flat, grids.items()):
            v = np.array(grid["visibility"], dtype=int); free = np.array(grid["certified_free"], bool)
            v[free] = 3; v[np.array(grid["occupied"], bool)] = 4
            x0, y0, x1, y1 = grid["bounds"]
            ax.imshow(v, origin="lower", extent=(x0, x1, y0, y1), vmin=0, vmax=4, cmap="viridis", interpolation="nearest")
            ax.set(title=f"{name}; free={int(free.sum())}/{free.size}", xlabel="x (m)", ylabel="y (m)")
        fig.suptitle("0 unknown; 1 geometric visibility; 2 occluded; 3 explicit free; 4 known occupied")
        save(fig, "visibility_constraints")
    manifest = {"sources": {name: _sha(directory / name) for name in sources}, "figures": figures,
                "smoothing": False, "downsampling": False, "input_mode": "mechanism_fixture", "training_steps": None}
    _json(out / "plot_manifest.json", manifest)
    return manifest


def run_diagnostics(config):
    config = validate_config(config); root = Path(config["output_root"])
    if not root.parent.is_dir(): raise FileNotFoundError(f"产物父目录未挂载/准备：{root.parent}")
    root.mkdir(exist_ok=True); directory = root / uuid4().hex; directory.mkdir()
    started = time.perf_counter(); checks = []; rows = []; evaluations = []
    _json(directory / "config.json", config)
    _json(directory / "manifest.json", {**_source_identity(PROJECT_ROOT), "started_utc": datetime.now(timezone.utc).isoformat(),
          "config_sha256": _sha(directory / "config.json"), "fixture_version": config["schema_version"], "data": None, "weights": None,
          "seed": None, "input_mode": "mechanism_fixture", "cuda_used": False,
          "environment": {"os": platform.platform(), "python": sys.version, "executable": sys.executable,
                          **{name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pyyaml", "matplotlib")}}})
    def check(identity, passed, actual):
        checks.append({"id": identity, "passed": bool(passed), "actual": actual})
        _json(directory / "checks.json", checks)
        if not passed: raise AssertionError(f"N10机制断言失败：{identity}")
    try:
        cases, grids, a, b, near_b = _fixtures(config)
        _json(directory / "input_scenarios.json", [{**{k: v for k, v in case.items() if k not in ("candidates", "targets")},
                                                  "candidates": [asdict(c) for c in case["candidates"]],
                                                  "targets": [asdict(t) for t in case["targets"]]} for case in cases])
        _json(directory / "grids.json", {name: {"bounds": k.spec.bounds, "resolution_m": k.spec.resolution_m,
              "now": k.now, "visibility": k.visibility.tolist(), "occupied": k.occupied.tolist(), "certified_free": k.certified_free.tolist(),
              "drivable": k.drivable.tolist(), "closed_domain": k.closed_domain, "accepted_sources": k.accepted_sources,
              "accepted_views": [asdict(v) for v in k.accepted_views],
              "accepted_free_evidence": [{**{key: value for key, value in asdict(p).items() if key != "mask"}, "mask": p.mask.tolist()}
                                         for p in k.accepted_free_evidence],
              "known_obstacles": [asdict(o) for o in k.obstacles]} for name, k in grids.items()})
        _json(directory / "unknown_regions.json", {name: [asdict(r) for r in k.regions] for name, k in grids.items()})
        baseline = grids["known_wall"]; causal = []
        for x in (-1e9, 1e9):
            future = SensorView("future", Pose2D(x, x, 0), 2 * math.pi, 30, 1, 2, 2, 3, config["clock_domain"], config["coordinate_frame_id"])
            altered = build_visibility(baseline.spec, now=0, clock_domain=config["clock_domain"], coordinate_frame_id=config["coordinate_frame_id"],
                                       views=(*baseline.accepted_views, future), obstacles=baseline.obstacles,
                                       drivable=baseline.drivable, closed_domain=baseline.closed_domain)
            causal.append({"unavailable_view": asdict(future), "current_input_output_sha256": _grid_digest(altered)})
        _json(directory / "visibility_audit.json", {"baseline_sha256": _grid_digest(baseline), "future_variants": causal,
                                                   "hidden_gt_input": "not_exposed_by_core_api", "model": "2d_grid_center_ray_not_pixel_perception"})
        check("B02_future_exclusion", all(c["current_input_output_sha256"] == _grid_digest(baseline) for c in causal), causal)
        bodies = {"vehicle": ParticipantGeometry("vehicle", 4, 2), "pedestrian": ParticipantGeometry("pedestrian", radius=.3),
                  "cyclist": ParticipantGeometry("cyclist", 1.8, .6),
                  "vehicle_rear_axle": ParticipantGeometry("vehicle", 4, 2, reference_point="rear_axle", center_offset_m=1.5)}
        _json(directory / "geometry_by_type.json", {name: {"declaration": asdict(g), "at_pose": {"x": 0, "y": 0, "yaw": math.pi / 2},
                                                           "shape": asdict(g.at(Pose2D(0, 0, math.pi / 2)))} for name, g in bodies.items()})
        geometry_cases = []
        box = OBB((0, 0), 0, 4, 2)
        for identity, other, expected in (("G01_same", box, True), ("G02_separate", OBB((4.1, 0), 0, 4, 2), False),
                   ("G02_touch", OBB((4, 0), 0, 4, 2), True), ("G03_rotate", OBB((0, 0), math.pi / 2, 4, 2), True),
                   ("G03_far", OBB((10, 10), math.pi / 2, 4, 2), False), ("B05_circle_touch", Circle((2.3, 1.4), .5), True),
                   ("B05_circle_clear", Circle((2.31, 1.4), .5), False)):
            actual = intersects(box, other); geometry_cases.append({"id": identity, "a": asdict(box), "b": asdict(other), "expected": expected, "actual": actual})
            check(identity, actual == expected, actual)
        _json(directory / "geometry_cases.json", geometry_cases)
        _json(directory / "rotated_cases.json", [c for c in geometry_cases if c["id"].startswith("G03")])
        invalid = []
        for args in (((0, 0), 0, 0, 2), ((0, 0), 0, 4, -2), ((0, 0), 0, 4, 2, -.1), ((0, 0), float("nan"), 4, 2)):
            try:
                OBB(*args)
                invalid.append({"input_repr": repr(args), "rejected": False})
            except ValueError as exc:
                invalid.append({"input_repr": repr(args), "rejected": True, "type": type(exc).__name__, "reason": str(exc)})
        _json(directory / "invalid_geometry.json", invalid)
        check("G04_invalid_rejected", all(r["rejected"] for r in invalid), invalid)
        small = ParticipantGeometry("vehicle", 2, 2); interval = IntervalConfig(**config["interval"])
        crossing = first_conflict_time(a, small, b, small, 0, .5, interval)
        near = first_conflict_time(a, small, near_b, small, 0, .5, interval)
        _json(directory / "interval_subdivision.json", {"paths": {"ego": asdict(a), "crossing": asdict(b), "near_miss": asdict(near_b)},
                                                 "geometry": asdict(small), "crossing": asdict(crossing), "near_miss": asdict(near)})
        check("G05_interval_crossing", crossing.potential_contact and crossing.event["actual_collision"] is None, crossing.event)
        check("conservative_near_miss", near.event is not None and near.event["uncertain_interval"], near.event)
        for case in cases:
            result = evaluate_candidates(case["candidates"], case["targets"], expected_target_ids=case["expected_target_ids"], now=0,
                                         plan_start=case["plan_start"], knowledge=grids[case["knowledge_key"]], interval_config=interval)
            evaluations.append({"case_id": case["case_id"], **result})
            _json(directory / "evaluations.json", evaluations)
            for row in result["candidates"]:
                rows.append({"case_id": case["case_id"], "candidate_id": row["candidate_id"], "R_max": row["R_max"], "R_sum": row["R_sum"],
                             "effective_horizon_s": result["coverage"]["effective_horizon_s"], "coverage_complete": result["coverage"]["complete"],
                             "normal_blocked": row["normal_blocked"], "block_reasons": "|".join(row["block_reasons"])})
            with (directory / "risk_rows.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        by_case = {e["case_id"]: e for e in evaluations}
        for name, identity in (("mode_risk_fixture.json", "mode_weight"), ("aggregate_risk.json", "two_targets"), ("horizon_cases.json", "short_coverage")):
            _json(directory / name, by_case[identity])
        row = lambda name: by_case[name]["candidates"][0]
        check("R01_weight", row("mode_weight")["R_max"] == .7, row("mode_weight")["R_max"])
        check("R02_aggregates", (row("two_targets")["R_max"], row("two_targets")["R_sum"]) == (.7, 1.), [.7, 1.])
        check("R03_common_2_8", all(r["window_end"] == 2.8 and r["normal_blocked"] for r in by_case["short_coverage"]["candidates"]), by_case["short_coverage"]["coverage"])
        for name, reason in (("both_blind", "unknown_reachable_region"), ("visible_no_free", "unknown_reachable_region"),
                             ("missing_known", "target_coverage_incomplete"), ("infeasible", "declared_physically_infeasible"),
                             ("prefix_hazard", "prefix_potential_contact"), ("body_boundary", "map_boundary"),
                             ("known_wall", "known_occupied_or_non_drivable"), ("stale_known", "target_coverage_incomplete")):
            check(name, reason in row(name)["block_reasons"], row(name)["block_reasons"])
        check("closed_free_no_approval", not row("closed_free")["normal_blocked"] and row("closed_free")["action_approval"] is None, row("closed_free")["block_reasons"])
        render_diagnostic_plots(directory)
        result = {"status": "passed", "run_id": directory.name, "run_dir": str(directory), "input_mode": "mechanism_fixture",
                  "checks_passed": len(checks), "checks_failed": 0, "checks_skipped": 0, "checks_not_run": 0,
                  "risk_scenarios": len(cases), "candidate_rows": len(rows), "geometric_cases": len(geometry_cases),
                  "crossing_event": crossing.event, "near_miss_event": near.event,
                  "grid_cells_per_snapshot": math.prod(next(iter(grids.values())).spec.shape),
                  "unknown_regions_by_snapshot": {name: len(k.regions) for name, k in grids.items()},
                  "wall_seconds": time.perf_counter() - started}
        _json(directory / "result.json", result)
        _json(directory / "status.json", {"status": "passed", "completed_utc": datetime.now(timezone.utc).isoformat()})
    except BaseException as exc:
        _json(directory / "status.json", {"status": "failed", "type": type(exc).__name__, "reason": str(exc), "checks_completed": len(checks)})
        if all((directory / name).is_file() for name in ("interval_subdivision.json", "risk_rows.csv", "grids.json")):
            try: render_diagnostic_plots(directory)
            except Exception as plot_exc: _json(directory / "plot_failure.json", {"type": type(plot_exc).__name__, "reason": str(plot_exc)})
        raise
    finally:
        _json(directory / "evidence_index.json", {p.name: _sha(p) for p in sorted(directory.iterdir()) if p.is_file() and p.name != "evidence_index.json"})
    return result
