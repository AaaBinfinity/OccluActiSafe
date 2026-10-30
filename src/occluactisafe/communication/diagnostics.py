"""保存逐消息原始证据及诊断图的N08机制入口；不接任何隐藏真值。"""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from uuid import uuid4

from occluactisafe.common.clock import Timestamp
from occluactisafe.common.config import PROJECT_ROOT, _yaml_file
from occluactisafe.common.run_manifest import _source_identity
from occluactisafe.common.schemas import ObservationMessage, PredictionMessage, ReceivedMessage
from .channel import SimChannel
from .protocol import ChannelConfig, Disturbance, encode_message, integer, service_seconds
from .validation import ReceiverPolicy, ReceiverState, query_prediction, validate_received


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def load_diagnostic_config(config_file=None, *, overrides=None, cli_overrides=None):
    selected = Path(config_file) if config_file is not None else PROJECT_ROOT / "configs/communication/n08.yaml"
    selected = selected if selected.is_absolute() else PROJECT_ROOT / selected
    config = _yaml_file(selected)
    if set(config) != {"schema_version", "seed", "clock_domain", "output_root", "channels", "receiver"}:
        raise ValueError("N08配置字段不完整或含未知字段")
    for override in (overrides, cli_overrides):
        if override is not None:
            if not isinstance(override, dict) or set(override) - {"seed", "output_root"}:
                raise ValueError("N08脚本/CLI只覆盖seed和output_root；其他配置请显式选择YAML")
            config.update(override)
    return validate_diagnostic_config(config)


def validate_diagnostic_config(config):
    config = deepcopy(config)
    if set(config) != {"schema_version", "seed", "clock_domain", "output_root", "channels", "receiver"}:
        raise ValueError("N08配置字段不完整或含未知字段")
    if config["schema_version"] != "n08-mechanism-1":
        raise ValueError("未知N08任务版本")
    integer(config["seed"], "seed", minimum=0)
    if not isinstance(config["clock_domain"], str) or not config["clock_domain"].strip():
        raise ValueError("clock_domain必须非空")
    if not isinstance(config["channels"], dict) or set(config["channels"]) != {"V2I", "I2V"}:
        raise ValueError("必须分别配置V2I/I2V")
    for direction, values in config["channels"].items():
        if not isinstance(values, dict) or set(values) - {
            "propagation_delay_s", "processing_delay_s", "capacity_bps", "max_queue_wait_s",
            "max_pending_messages", "max_message_bytes", "jitter_s", "loss_rate", "blackout_intervals",
        }:
            raise ValueError("未知信道配置")
        ChannelConfig(direction, config["clock_domain"], seed=config["seed"], **values)
    expected = {"max_observation_age_s", "max_state_age_s", "max_anchor_age_s", "planning_horizon_s"}
    if not isinstance(config["receiver"], dict) or set(config["receiver"]) != expected:
        raise ValueError("接收配置字段不完整或含未知字段")
    ReceiverPolicy("vehicle-A", "map-A", "1", "map-right-handed", config["clock_domain"],
                   {"roadside-1": "edge-1"}, **config["receiver"])
    output = config["output_root"]
    if not isinstance(output, str) or not output.strip() or "://" in output:
        raise ValueError("output_root必须为非空本地路径")
    p = Path(output).expanduser()
    config["output_root"] = str((p if p.is_absolute() else PROJECT_ROOT / p).resolve())
    return config


def _observation(domain, sequence=0, sent=12., observed=12.):
    """显式构造的匀速消息，无真实数据含义。"""
    track = dict(source_track_id="visible-1", source_id="vehicle-A", session_id="session-1", sensor_id="front",
                 category="vehicle", role="target", clock_domain=domain, coordinate_frame_id="map-right-handed",
                 reference_point="geometric_center", observation_time=observed, last_seen_time=observed,
                 source_available_time=observed, state_time=observed, x=0., y=0., vx=0., vy=10., yaw=None,
                 length=4.2, width=1.8, size_source="mechanism_fixture", size_uncertainty=None,
                 covariance_xy=None, observed_mask=True, state_valid_mask=True, size_valid=True,
                 velocity_valid=True, yaw_valid=False, quality=1.)
    return ObservationMessage("1.0", "observation", "vehicle-A", "session-1", sequence, "roadside-1",
                              "map-A", "1", "map-right-handed", sequence, domain, observed, observed,
                              sent, sent, 100., (track,))


def _prediction(domain, sequence=0, sent=12., anchor=12., track=None):
    times = tuple(anchor + i / 10 for i in range(1, 41))
    track = _observation(domain, observed=anchor, sent=anchor).tracks[0] if track is None else track
    target = dict(track=track.to_dict(), trajectories=(tuple((0., 10 * (t - anchor)) for t in times),),
                  weights=(1.,), valid_mask=((True,) * 40,))
    return PredictionMessage("1.0", "prediction", "roadside-1", "edge-1", sequence, "vehicle-A",
                             "map-A", "1", "map-right-handed", sequence, domain, "mechanism-not-trained-model",
                             anchor, anchor, sent, sent, sent, 100., times, times[-1], (target,))


def render_diagnostic_plots(run_dir, *, output_dir=None):
    """从CSV/JSON原值绘图；异目录重建不修改原run。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["path.simplify"] = False
    run_dir = Path(run_dir)
    output = run_dir if output_dir is None else Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (run_dir / "capacity_queue.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    duplex = json.loads((run_dir / "duplex_timing.json").read_text(encoding="utf-8"))
    graph_files = []
    fig, axes = plt.subplots(2, 1, figsize=(9, 6))
    if rows:
        x = [float(r["time"]) for r in rows]
        axes[0].plot(x, [float(r["queue_wait_s"]) for r in rows], "o-", label="Required wait (including rejected)")
        axes[0].axhline(.5, color="tab:red", linestyle="--", label="Admission wait limit 0.5 s")
        axes[1].step(x, [int(r["admitted"]) for r in rows], where="mid", label="Admitted=1; rejected=0")
        axes[1].set_yticks([0, 1])
    axes[0].set(ylabel="Queue wait (s)", title="N08 mechanism: 38,400 byte wire messages, 10 Hz, 1 Mbit/s")
    axes[1].set(xlabel="Simulation send time (s)", ylabel="Admission")
    for ax in axes: ax.grid(alpha=.2); ax.legend(loc="best")
    fig.text(.01, .01, f"Run {run_dir.name[:12]}; raw records; no smoothing; software channel only", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    for suffix in ("png", "svg"):
        path = output / f"capacity_queue.{suffix}"; fig.savefig(path, dpi=150); graph_files.append(path)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (name, row) in enumerate(duplex.items()):
        ax.plot([row["transmitted_time"], row["arrival_time"], row["usable_time"]], [i] * 3, "o-")
        for key, yshift in (("transmitted_time", .14), ("arrival_time", -.22), ("usable_time", .14)):
            ax.annotate(f"{key.split('_')[0]} {row[key]:.4f}", (row[key], i + yshift), ha="center", fontsize=8)
    ax.set(yticks=range(len(duplex)), yticklabels=list(duplex), xlabel="Absolute simulation time (s)",
           title="N08 mechanism: arrival and usable times remain separate", ylim=(-.5, 1.5))
    ax.grid(alpha=.2)
    ax.margins(x=.16)
    fig.text(.01, .01, f"Run {run_dir.name[:12]}; injected times; actual Python wall times logged separately", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    for suffix in ("png", "svg"):
        path = output / f"duplex_timing.{suffix}"; fig.savefig(path, dpi=150); graph_files.append(path)
    plt.close(fig)
    manifest = {"matplotlib_version": matplotlib.__version__, "smoothing": False, "downsampling": False,
                "input_mode": "mechanism_fixture", "run_id": run_dir.name,
                "sources": {name: _sha(run_dir / name) for name in ("capacity_queue.csv", "duplex_timing.json")},
                "figures": {p.name: _sha(p) for p in graph_files}}
    _json(output / "plot_manifest.json", manifest)
    return manifest


def run_diagnostics(config):
    config = validate_diagnostic_config(config)
    root = Path(config["output_root"])
    # 不自动创建不存在的盘符或数据集根；父目录错误必须清楚失败。
    if not root.parent.is_dir():
        raise FileNotFoundError(f"输出父目录不存在: {root.parent}")
    run_id = uuid4().hex
    directory = root / run_id
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    domain = config["clock_domain"]
    now = lambda t: Timestamp(t, domain)
    checks = []
    channels = {}
    validations = []
    inputs = []
    scheduled_rows = {}
    manifest = {"schema_version": "n08-mechanism-1", "run_id": run_id,
                "created_utc": datetime.now(timezone.utc).isoformat(), "input_mode": "mechanism_fixture",
                "data": None, "weights": None, "source": _source_identity(PROJECT_ROOT),
                "config": config, "config_sha256": _canonical_sha(config),
                "environment": {"os": platform.platform(), "python": sys.version, "executable": sys.executable},
                "not_implemented": ["V04 risk memory/actions", "UDP", "real devices", "closed-loop avoidance"]}
    _json(directory / "manifest.json", manifest)
    trace = (directory / "message_lifecycle.jsonl").open("w", encoding="utf-8")

    def sink(case):
        def write(row):
            if row["event"] == "scheduled":
                scheduled_rows[case] = row
            trace.write(json.dumps({"case": case, **row}, ensure_ascii=False, allow_nan=False) + "\n")
            trace.flush()
        return write

    def channel(name, cfg, **kwargs):
        c = SimChannel(cfg, event_sink=sink(name), **kwargs); channels[name] = c; return c

    def check(name, passed, actual):
        checks.append({"name": name, "passed": bool(passed), "actual": actual})
        _json(directory / "checks.json", checks)
        if not passed: raise AssertionError(f"N08断言失败: {name}: {actual}")

    def send(c, p, t):
        inputs.append({"case": next(k for k, v in channels.items() if v is c), "payload": p.to_dict()})
        _json(directory / "input_messages.json", inputs)
        return c.send(p, now(t))

    def receive_check(label, r, at, p, **kwargs):
        result = validate_received(r, now(at), p, **kwargs)
        validations.append({"case": label, "decision_time": at, "accepted": result.accepted,
                            "reason": result.reason, "sequence": r.payload.sequence,
                            "validation_wall_s": result.validation_wall_s, "coverage": result.coverage})
        _json(directory / "receive_checks.json", validations)
        return result

    try:
        fixed = ChannelConfig("I2V", domain, seed=config["seed"])
        c = channel("V01_fixed", fixed); send(c, _prediction(domain), 12)
        early = c.deliver(now(12.19)); due = c.deliver(now(12.2)); again = c.deliver(now(12.2))
        check("V01_boundary_once", not early and len(due) == 1 and not again,
              {"early": len(early), "due": len(due), "again": len(again)})
        p = ReceiverPolicy("vehicle-A", "map-A", "1", "map-right-handed", domain,
                           {"roadside-1": "edge-1"}, **config["receiver"])
        point = query_prediction(due[0], now=now(12.2), when=now(12.5), policy=p)
        check("V03_C06_absolute_time", point == (0., 5.), {"query": 12.5, "position_m": point, "anchor": 12.})
        late = _prediction(domain, 1)
        c = channel("C06_late_arrival", fixed, perturbations={
            ("roadside-1", "edge-1", 1): Disturbance(.4, False)})
        send(c, late, 12)
        before_late = c.deliver(now(12.3))
        late_check = receive_check("C06_not_arrived", ReceivedMessage(late, 12.4, 12.4, domain), 12.3, p)
        check("C06_future_arrival_excluded", not before_late and late_check.reason == "not_arrived",
              {"decision_time": 12.3, "future_arrival": 12.4, "delivered": len(before_late), "reason": late_check.reason})
        c.deliver(now(12.4))
        short = receive_check("TTL_not_coverage", due[0], 12.9, p, plan_start=now(13.1))
        check("V03_C06_coverage_rejection", short.reason == "coverage_insufficient", short.reason)
        delayed = ReceivedMessage(due[0].payload, 12.2, 12.4, domain)
        check("usable_gate", receive_check("before_usable", delayed, 12.3, p).reason == "not_usable", "not_usable")
        state = ReceiverState(p)
        accepted = receive_check("accepted_once", due[0], 12.2, p, state=state)
        duplicate = receive_check("duplicate", due[0], 12.2, p, state=state)
        check("duplicate_gate", accepted.accepted and duplicate.reason == "duplicate_or_out_of_order", duplicate.reason)
        c = channel("V02_loss", replace(fixed, loss_rate=1))
        send(c, _prediction(domain), 12)
        check("V02_total_loss", not c.deliver(now(13)), dict(c.stats))
        c = channel("V02_ties", fixed)
        for seq in (3, 1, 2): send(c, _prediction(domain, seq), 12)
        order = [r.payload.sequence for r in c.deliver(now(12.2))]
        check("V02_stable_ties", order == [3, 1, 2], order)
        table = {("roadside-1", "edge-1", 0): Disturbance(.4, False),
                 ("roadside-1", "edge-1", 1): Disturbance(.1, False)}
        c = channel("C06_reordering", fixed, perturbations=table)
        for seq in (0, 1): send(c, _prediction(domain, seq), 12)
        first = c.deliver(now(12.1))[0]; old = c.deliver(now(12.4))[0]
        state = ReceiverState(p)
        fresh = receive_check("reorder_new", first, 12.1, p, state=state)
        stale = receive_check("reorder_old", old, 12.4, p, state=state)
        check("C06_old_does_not_replace_new", fresh.accepted and stale.reason == "duplicate_or_out_of_order", stale.reason)
        c = channel("V04_blackout_subscope", replace(fixed, blackout_intervals=((12.1, 12.3),)))
        for seq in range(4): send(c, _prediction(domain, seq, 12 + seq / 10), 12 + seq / 10)
        surviving = c.deliver(now(13))
        seqs = [r.payload.sequence for r in surviving]
        check("V04_channel_only_blackout", seqs == [0, 3] and all(len(r.payload.targets) == 1 for r in surviving), seqs)
        # 因果双向：只在上行完成接收处理后读取消息，再注入非零推理耗时。
        up_cfg = ChannelConfig("V2I", domain, seed=config["seed"], **config["channels"]["V2I"])
        down_cfg = ChannelConfig("I2V", domain, seed=config["seed"], **config["channels"]["I2V"])
        up = channel("duplex_V2I", up_cfg); down = channel("duplex_I2V", down_cfg)
        obs = _observation(domain); send(up, obs, 12)
        arrival = scheduled_rows["duplex_V2I"]["arrival_time"]
        uploaded = up.deliver(now(arrival))[0]
        edge_policy = ReceiverPolicy("roadside-1", "map-A", "1", "map-right-handed", domain,
                                     {"vehicle-A": "session-1"}, **config["receiver"])
        upload_check = receive_check("V2I_usable", uploaded, uploaded.usable_time, edge_policy)
        check("duplex_upload_usable", upload_check.accepted, uploaded.usable_time)
        completed = uploaded.usable_time + .03
        pred = _prediction(domain, sent=completed, anchor=uploaded.usable_time, track=uploaded.payload.tracks[0])
        send(down, pred, completed)
        arrival = scheduled_rows["duplex_I2V"]["arrival_time"]
        downloaded = down.deliver(now(arrival))[0]
        out = receive_check("I2V_usable", downloaded, downloaded.usable_time, p)
        check("duplex_downlink_usable", out.accepted and completed > uploaded.usable_time, out.reason)
        duplex = {"V2I": {"transmitted_time": obs.transmitted_time, "arrival_time": uploaded.arrival_time,
                            "usable_time": uploaded.usable_time},
                  "I2V": {"transmitted_time": pred.transmitted_time, "arrival_time": downloaded.arrival_time,
                            "usable_time": downloaded.usable_time}}
        _json(directory / "duplex_timing.json", duplex)
        _json(directory / "duplex_stats.json", {"V2I": {**dict(up.stats), "validation_wall_seconds": upload_check.validation_wall_s},
             "I2V": {**dict(down.stats), "validation_wall_seconds": out.validation_wall_s},
             "injected_inference_s": .03, "edge_input_read_time": uploaded.usable_time})
        # 精确38400字节的合法JSON填充夹具，与20目标float32坐标的理论示例分开。
        cap_cfg = replace(fixed, capacity_bps=1e6, max_queue_wait_s=.5)
        cap_rows = []
        cap_sink = sink("capacity_10hz")
        def cap_log(row):
            cap_sink(row)
            if row["event"] in ("scheduled", "send_rejected"):
                cap_rows.append({k: row[k] for k in ("time", "sequence", "bytes", "queue_wait_s", "service_s", "arrival_time", "usable_time")})
                cap_rows[-1].update(admitted=int(row["event"] == "scheduled"), reason=row.get("reason", "admitted"))
                with (directory / "capacity_queue.csv").open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(cap_rows[-1])); writer.writeheader(); writer.writerows(cap_rows)
        cap = SimChannel(cap_cfg, event_sink=cap_log); channels["capacity_10hz"] = cap
        for seq in range(20):
            t = 12 + seq / 10; cap.deliver(now(t))
            padded = _prediction(domain, seq, t).to_dict()
            base_size = len(encode_message(PredictionMessage.from_dict(padded)))
            padded["model_hash"] += "X" * (38400 - base_size)
            send(cap, PredictionMessage.from_dict(padded), t)
        cap.deliver(now(30))
        check("capacity_service_38400", service_seconds(38400, 1e6) == .3072,
              {"bytes": 38400, "capacity_bit_s": 1e6, "service_s": .3072})
        check("capacity_10hz_queue_and_drop", any(r["queue_wait_s"] > 0 for r in cap_rows)
              and cap.stats["queue_timeout"] > 0 and all(r["bytes"] == 38400 for r in cap_rows), dict(cap.stats))
        _json(directory / "channel_stats.json", {k: dict(v.stats) for k, v in channels.items()})
        render_diagnostic_plots(directory)
        result = {"run_id": run_id, "run_dir": str(directory), "input_mode": "mechanism_fixture", "status": "passed",
                  "checks_passed": len(checks), "checks_failed": 0, "checks_skipped": 0, "checks_not_run": 0,
                  "wall_seconds": time.perf_counter() - started,
                  "V04_scope": "channel blackout/causality only; risk memory/actions not implemented"}
        _json(directory / "result.json", result)
        _json(directory / "status.json", {"status": "passed", "completed_utc": datetime.now(timezone.utc).isoformat()})
    except BaseException as exc:
        _json(directory / "status.json", {"status": "failed", "type": type(exc).__name__, "reason": str(exc),
                                         "checks_completed": len(checks), "wall_seconds": time.perf_counter() - started})
        if all((directory / name).is_file() for name in ("capacity_queue.csv", "duplex_timing.json")):
            try:
                render_diagnostic_plots(directory)
            except Exception as plot_exc:
                _json(directory / "plot_failure.json", {"type": type(plot_exc).__name__, "reason": str(plot_exc)})
        raise
    finally:
        trace.close()
        _json(directory / "evidence_index.json", {p.name: _sha(p) for p in sorted(directory.iterdir())
              if p.is_file() and p.name != "evidence_index.json"})
    return result
