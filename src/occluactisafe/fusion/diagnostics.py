"""N09固定机制、逐步状态/矩阵证据及自动诊断图；无真实数据入口。"""

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import sys
import time
from uuid import uuid4
import numpy as np

from occluactisafe.common.clock import Timestamp
from occluactisafe.common.config import PROJECT_ROOT, _yaml_file
from occluactisafe.common.run_manifest import _source_identity
from occluactisafe.common.schemas import ObservationPacket, SourceTrack
from occluactisafe.communication.protocol import integer
from .state_filter import FilterConfig, cv_predict, position_update
from .association import AssociationConfig, minimum_cost_assignment
from .history import FusionHistory, HistoryConfig


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def load_diagnostic_config(config_file=None, *, overrides=None, cli_overrides=None):
    path = Path(config_file) if config_file is not None else PROJECT_ROOT / "configs/fusion/n09.yaml"
    config = _yaml_file(path if path.is_absolute() else PROJECT_ROOT / path)
    for override in (overrides, cli_overrides):
        if override is not None:
            if not isinstance(override, dict) or set(override) - {"seed", "output_root"}:
                raise ValueError("脚本/CLI只覆盖seed/output_root；其他设置使用显式YAML")
            config.update(override)
    return validate_config(config)


def validate_config(config):
    config = deepcopy(config)
    if set(config) != {"schema_version", "seed", "episode_id", "clock_domain", "output_root", "map", "filter", "association", "history"}:
        raise ValueError("N09配置字段缺失或含未知字段")
    if config["schema_version"] != "n09-mechanism-1": raise ValueError("未知N09任务版本")
    integer(config["seed"], "seed", minimum=0)
    for name in ("episode_id", "clock_domain", "output_root"):
        if not isinstance(config[name], str) or not config[name].strip() or "://" in config[name]:
            raise ValueError(f"{name}须为非空公开标识或本地路径")
    if not isinstance(config["map"], dict) or set(config["map"]) != {"map_id", "map_version", "coordinate_frame_id"}:
        raise ValueError("地图配置字段错误")
    for name, cls in (("filter", FilterConfig), ("association", AssociationConfig), ("history", HistoryConfig)):
        if not isinstance(config[name], dict) or set(config[name]) - set(cls.__dataclass_fields__):
            raise ValueError(f"{name}配置字段错误")
        config[name] = asdict(cls(**config[name]))
    _engine(config)
    path = Path(config["output_root"]).expanduser()
    config["output_root"] = str((path if path.is_absolute() else PROJECT_ROOT / path).resolve())
    return config


def _engine(config):
    return FusionHistory(episode_id=config["episode_id"], clock_domain=config["clock_domain"],
                         filter_config=FilterConfig(**config["filter"]), association_config=AssociationConfig(**config["association"]),
                         history_config=HistoryConfig(**config["history"]), **config["map"])


def _track(config, source, local_id, at, x, y, vx, vy, category="vehicle"):
    length, width = {"vehicle": (4.2, 1.8), "pedestrian": (.6, .6), "cyclist": (1.8, .6)}[category]
    return SourceTrack(local_id, source, "session-1", "camera", category, "target", config["clock_domain"],
                       config["map"]["coordinate_frame_id"], "geometric_center", at, at, at + .02, at,
                       x, y, vx, vy, 0., length, width, "mechanism_fixture", None, None,
                       True, True, True, True, True, .9)


def _packet(config, tracks, at, *, available=None):
    return ObservationPacket(**config["map"], frame_id=int(round(at * 10)), clock_domain=config["clock_domain"],
                             observation_time=at, source_available_time=at + .02 if available is None else available, tracks=tuple(tracks))


def _write_csv(path, rows):
    if not rows: return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def render_diagnostic_plots(run_dir, *, output_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["path.simplify"] = False
    run_dir = Path(run_dir); out = run_dir if output_dir is None else Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    with (run_dir / "source_states.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    matrices = json.loads((run_dir / "association_matrices.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    for source in ("road", "vehicle"):
        selected = [r for r in rows if r["source"] == source and r["local_id"] in ("road-car", "local-car")]
        x = [float(r["state_time"]) for r in selected]
        for ax, name in zip(axes, ("x_m", "position_trace_m2", "age_s", "history_count")):
            ax.plot(x, [float(r[name]) for r in selected], "o-", label=source, markersize=3)
    for ax, label in zip(axes, ("Estimated x (m)", "Position trace(P) (m2)", "Observation age (s)", "History entries")):
        ax.set_ylabel(label); ax.grid(alpha=.2); ax.legend(loc="best")
    axes[0].set_title("N09 mechanism: separate source filters, three missing car frames")
    axes[-1].set_xlabel("Absolute software simulation time (s)")
    fig.text(.01, .01, f"Run {run_dir.name[:12]}; raw step records; no smoothing; not real tracking/avoidance", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    paths = []
    for extension in ("png", "svg"):
        path = out / f"source_filter_history.{extension}"; fig.savefig(path, dpi=150); paths.append(path)
    plt.close(fig)
    report = matrices[-1]["reports"][0]
    values = np.array(report["costs"]); allowed = np.array(report["allowed"], dtype=bool)
    fig, ax = plt.subplots(figsize=(8, 4))
    palette = plt.get_cmap("Blues").copy(); palette.set_bad("#eeeeee")
    ax.imshow(np.ma.masked_where(~allowed, values), cmap=palette, vmin=0, vmax=max(1., float(values.max())), aspect="auto")
    for i, j in np.ndindex(values.shape):
        ax.text(j, i, f"{values[i, j]:.3f}" if allowed[i, j] else "X", ha="center", va="center")
    ax.set_xticks(range(len(report["column_identities"])), [v[-1] for v in report["column_identities"]])
    ax.set_yticks(range(len(report["row_identities"])), [v[-1] for v in report["row_identities"]])
    ax.set(xlabel="Existing group representative", ylabel="Incoming source target", title="N09 mechanism: allowed costs and rejected pairs")
    fig.text(.01, .01, f"Run {run_dir.name[:12]}; X=forbidden; costs are not matching probabilities", fontsize=8)
    fig.tight_layout(rect=(0, .07, 1, 1))
    for extension in ("png", "svg"):
        path = out / f"association_gates.{extension}"; fig.savefig(path, dpi=150); paths.append(path)
    plt.close(fig)
    record = {"matplotlib_version": matplotlib.__version__, "input_mode": "mechanism_fixture", "smoothing": False, "downsampling": False,
              "sources": {n: _sha(run_dir / n) for n in ("source_states.csv", "association_matrices.json")},
              "figures": {p.name: _sha(p) for p in paths}}
    _json(out / "plot_manifest.json", record)
    return record


def run_diagnostics(config):
    config = validate_config(config)
    root = Path(config["output_root"])
    if not root.parent.is_dir(): raise FileNotFoundError(f"输出父目录不存在: {root.parent}")
    directory = root / uuid4().hex; directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {"run_id": directory.name, "created_utc": datetime.now(timezone.utc).isoformat(),
                "source": _source_identity(PROJECT_ROOT), "input_mode": "mechanism_fixture", "data": None, "weights": None,
                "config": config, "config_sha256": _canonical_sha(config),
                "environment": {"os": platform.platform(), "python": sys.version, "executable": sys.executable,
                                "packages": {n: importlib.metadata.version(n) for n in ("numpy", "PyYAML", "scipy", "matplotlib")}}}
    _json(directory / "manifest.json", manifest)
    checks, state_rows, snapshots, matrices, inputs = [], [], [], [], []

    def check(name, ok, actual):
        checks.append({"name": name, "passed": bool(ok), "actual": actual}); _json(directory / "checks.json", checks)
        if not ok: raise AssertionError(f"N09机制断言失败: {name}: {actual}")

    def stamp(at): return Timestamp(at, config["clock_domain"])

    try:
        z, p = cv_predict([0, 0, 10, 0], np.eye(4), .1)
        check("K01_cv_exact", np.allclose(z, [1, 0, 10, 0], atol=1e-12, rtol=0), {"state": z.tolist(), "dt_s": .1})
        z, p = np.zeros(4), np.eye(4); stable = []
        for i in range(200):
            z, p = cv_predict(z, p, .1)
            z, p = position_update(z, p, [i / 10, 2 * i / 10], np.eye(2) * 1e-6)
            stable.append({"step": i + 1, "min_eigenvalue": float(np.linalg.eigvalsh(p)[0]), "asymmetry": float(abs(p - p.T).max()),
                           **{f"z_{j}": float(z[j]) for j in range(4)}, **{f"p_{a}{b}": float(p[a, b]) for a, b in np.ndindex((4, 4))}})
        _write_csv(directory / "covariance_stability.csv", stable)
        check("K02_joseph_stability", all(r["min_eigenvalue"] >= -1e-8 and r["asymmetry"] <= 1e-12 for r in stable),
              {"updates": 200, "min_eigenvalue": min(r["min_eigenvalue"] for r in stable), "R_m2": 1e-6})
        cost = [[1., 8.], [7., 2.]]
        assigned = minimum_cost_assignment(cost, unmatched_cost=5)
        rejected = minimum_cost_assignment(cost, np.zeros((2, 2), bool), unmatched_cost=5)
        _json(directory / "assignment_cases.json", {"cost": cost, "unmatched_cost": 5, "allowed_pairs": assigned.pairs,
                                                    "allowed_objective": assigned.objective, "all_forbidden_pairs": rejected.pairs,
                                                    "all_forbidden_objective": rejected.objective})
        check("A01_assignment", assigned.pairs == ((0, 0), (1, 1)), list(assigned.pairs))
        check("A02_all_forbidden", not rejected.pairs and len(rejected.unmatched_rows) == 2 and len(rejected.unmatched_columns) == 2, asdict(rejected))
        rng = np.random.default_rng(config["seed"])
        engine = _engine(config)
        for i in range(20):
            t = 1 + i / 10; packets = []
            for source, car_id, ped_id in (("vehicle", "local-car", "local-person"), ("road", "road-car", "road-person")):
                tracks = [_track(config, source, ped_id, t, 20., .3 * (t - 1), 0., .3, "pedestrian")]
                if i not in (5, 6, 7):
                    tracks.append(_track(config, source, car_id, t, 10 * (t - 1) + float(rng.normal(0, .03)), 0., 10., 0.))
                if source == "road": tracks.append(_track(config, source, "road-only-cyclist", t, 50 + 2 * (t - 1), 5., 2., 0., "cyclist"))
                packets.append(_packet(config, tracks, t))
            inputs.append({"step": i + 1, "decision_time": t + .02, "packets": [p.to_dict() for p in packets]})
            _json(directory / "input_packets.json", inputs)
            result = engine.step(stamp(t + .02), packets)
            snapshots.append(result.to_dict()); matrices.append({"step": i + 1, "reports": list(result.associations)})
            with (directory / "lifecycle_history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")
            for agent in result.agents:
                for v in agent["sources"]:
                    p = np.asarray(v["covariance"])
                    state_rows.append({"step": i + 1, "state_time": result.state_time, "unified_id": agent["unified_id"],
                                       "source": v["key"][1], "local_id": v["key"][-1], "x_m": v["state"][0], "y_m": v["state"][1],
                                       "vx_mps": v["state"][2], "vy_mps": v["state"][3], "last_seen_time": v["last_seen_time"],
                                       "age_s": v["age_s"], "history_count": len(v["history"]), "measurement_updates": v["measurement_updates"],
                                       "position_trace_m2": float(np.trace(p[:2, :2])), "min_eigenvalue": float(np.linalg.eigvalsh(p)[0]),
                                       **{f"p_{a}{b}": float(p[a, b]) for a, b in np.ndindex((4, 4))}})
            _write_csv(directory / "source_states.csv", state_rows)
            _json(directory / "association_matrices.json", matrices)
        missing = [r for r in state_rows if r["source"] == "vehicle" and r["local_id"] == "local-car" and r["step"] in (6, 7, 8)]
        check("K03_three_missing_frames", len(missing) == 3 and all(r["last_seen_time"] == 1.4 for r in missing)
              and all(a["age_s"] < b["age_s"] and a["position_trace_m2"] < b["position_trace_m2"] for a, b in zip(missing, missing[1:])), missing)
        check("sources_not_double_updated", all(v["measurement_updates"] == 1 for a in snapshots[0]["agents"] for v in a["sources"]),
              {"first_source_count": sum(len(a["sources"]) for a in snapshots[0]["agents"]), "first_unified_count": len(snapshots[0]["agents"])})
        check("roadside_only_retained", all(any(s["key"][-1] == "road-only-cyclist" for a in snapshot["agents"] for s in a["sources"]) for snapshot in snapshots), {"steps": 20})
        check("history_honest_and_bounded", all(r["history_count"] <= config["history"]["max_points"] for r in state_rows)
              and all(len(snapshot["agents"]) == 3 for snapshot in snapshots), {"source_rows": len(state_rows), "accepted_source_updates": sum(len(p["tracks"]) for row in inputs for p in row["packets"])})
        base = [ObservationPacket.from_dict(p) for p in inputs[0]["packets"]]
        permutations = []
        for reverse_packets, reverse_tracks in itertools.product((False, True), repeat=2):
            packets = [ObservationPacket.from_dict({**p.to_dict(), "tracks": tuple(reversed(p.tracks)) if reverse_tracks else p.tracks}) for p in base]
            if reverse_packets: packets.reverse()
            result = _engine(config).step(stamp(1.02), packets)
            permutations.append({"feature_sha256": result.feature_sha256, "input_sha256": result.input_sha256})
        _json(directory / "permutation_cases.json", permutations)
        check("M05_fusion_permutation", len({r["feature_sha256"] for r in permutations}) == 1 and len({r["input_sha256"] for r in permutations}) == 1, permutations)
        baseline = _engine(config).step(stamp(1.02), base)
        future_cases = []
        for x in (1e9, -1e9):
            future = _packet(config, [_track(config, "future-source", "future-local", 10., x, 0., 0., 0.)], 10.)
            result = _engine(config).step(stamp(1.02), [future, *base])
            future_cases.append({"feature_sha256": result.feature_sha256, "input_sha256": result.input_sha256, "events": list(result.events)})
        _json(directory / "causality_cases.json", future_cases)
        check("M04_late_future_excluded", all(r["feature_sha256"] == baseline.feature_sha256 and r["input_sha256"] == baseline.input_sha256 for r in future_cases),
              {"baseline_features": baseline.feature_sha256, "variants": 2})
        repeat = _engine(config); repeat.step(stamp(1.02), base)
        duplicate = repeat.step(stamp(1.12), base)
        _json(directory / "duplicate_case.json", duplicate.to_dict())
        check("duplicates_no_new_history", all(v["measurement_updates"] == 1 and len(v["history"]) == 1 for a in duplicate.agents for v in a["sources"]),
              {"duplicate_events": sum(e["reason"] == "duplicate_or_out_of_order" for e in duplicate.events)})
        new_domain = config["clock_domain"] + "-new"
        repeat.begin_episode(config["episode_id"] + "-new", new_domain)
        reset_result = repeat.step(Timestamp(1.02, new_domain), base)
        check("episode_reset", not reset_result.agents and all(e["reason"] == "packet_clock_mismatch" for e in reset_result.events),
              {"new_episode_empty": not reset_result.agents, "old_packets_rejected": len(reset_result.events)})
        expired = engine.step(stamp(6.02)); _json(directory / "retention_expiry.json", expired.to_dict())
        check("memory_expiry_explicit", not expired.agents and any(e["reason"] == "retention_expired" for e in expired.events), "expired state is not a safety claim")
        stationary = _track(config, "vehicle", "low-speed", 1., 0., 0., 0., 0.)
        low = _engine(config).step(stamp(1.02), [_packet(config, [stationary], 1.)])
        check("low_speed_heading_invalid", not low.agents[0]["sources"][0]["yaw_valid"], low.agents[0]["sources"][0]["yaw"])
        render_diagnostic_plots(directory)
        result = {"status": "passed", "run_id": directory.name, "run_dir": str(directory), "input_mode": "mechanism_fixture",
                  "checks_passed": len(checks), "checks_failed": 0, "checks_skipped": 0, "checks_not_run": 0,
                  "steps": 20, "source_state_rows": len(state_rows),
                  "source_measurements": sum(len(p["tracks"]) for row in inputs for p in row["packets"]),
                  "final_unified_targets": len(snapshots[-1]["agents"]), "wall_seconds": time.perf_counter() - started}
        _json(directory / "result.json", result)
        _json(directory / "status.json", {"status": "passed", "completed_utc": datetime.now(timezone.utc).isoformat()})
    except BaseException as exc:
        _json(directory / "status.json", {"status": "failed", "type": type(exc).__name__, "reason": str(exc), "checks_completed": len(checks)})
        if (directory / "source_states.csv").is_file() and (directory / "association_matrices.json").is_file():
            try: render_diagnostic_plots(directory)
            except Exception as plot_exc: _json(directory / "plot_failure.json", {"type": type(plot_exc).__name__, "reason": str(plot_exc)})
        raise
    finally:
        _json(directory / "evidence_index.json", {p.name: _sha(p) for p in sorted(directory.iterdir()) if p.is_file() and p.name != "evidence_index.json"})
    return result
