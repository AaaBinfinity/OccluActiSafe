"""采集N15 CARLA同步双视角RGB/实例分割与可见框；IDE无必填参数。"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import uuid

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "data" / "carla_perception.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_scene_groups(config: dict) -> None:
    groups = {}
    for item in config["maps"]:
        map_name, role = item["map"], item["project_role"]
        if role not in {"train", "model_val", "test"}:
            raise ValueError(f"非法CARLA划分角色：{role}")
        if map_name in groups:
            raise ValueError(f"CARLA地图重复：{map_name}")
        groups[map_name] = role
    if set(groups.values()) != {"train", "model_val", "test"}:
        raise ValueError("CARLA地图级train/model_val/test必须各有至少一个组")
    if len(config["scenes"]) < 3 or len(set(config["scenes"])) != len(config["scenes"]):
        raise ValueError("场景清单须至少3项且不重复")
    if type(config["frames_per_scene"]) is not int or config["frames_per_scene"] < 1:
        raise ValueError("frames_per_scene必须为正整数")
    for key in ("moving_target_speed_mps", "pedestrian_speed_mps"):
        if float(config[key]) <= 0:
            raise ValueError(f"{key}必须为正数")
    if len(config["roadside_camera_local_xyz"]) != 3:
        raise ValueError("路侧相机相对位姿须为xyz三项")


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def summarize_scene_labels(scene: dict) -> dict:
    labels = [item for view in scene["labels"] for item in view["labels"]]
    distance_bins = {"0_10m": 0, "10_25m": 0, "over_25m": 0}
    size_bins = {"small_le_32px": 0, "medium_33_96px": 0, "large_over_96px": 0}
    for item in labels:
        distance = item.get("camera_distance_m")
        if distance is not None:
            distance_bins["0_10m" if distance <= 10 else "10_25m" if distance <= 25 else "over_25m"] += 1
        box = item["bbox_xyxy_visible_pixels"]
        extent = max(box[2] - box[0], box[3] - box[1])
        size_bins["small_le_32px" if extent <= 32 else "medium_33_96px" if extent <= 96 else "large_over_96px"] += 1
    return {
        "labels_by_category": {category: sum(item["category"] == category for item in labels)
                               for category in sorted({item["category"] for item in labels})},
        "camera_distance_bins_m": distance_bins,
        "bbox_max_dimension_bins_px": size_bins,
        "visible_occlusion_degree_unknown": sum(
            item["occluded_state"] == "visible_occlusion_degree_unknown" for item in labels),
        "truncated_by_image_edge": sum(item["truncated_state"] == "touches_image_edge" for item in labels),
        "negative_view_frames": sum(view["negative_frame"] for frame in scene["frames"]
                                     for view in frame["views"].values()),
        "hidden_actor_review_records": sum(len(view["hidden_actor_review"]) for view in scene["labels"]),
    }


def local_camera_location_to_world(transform, carla_module, xyz: list[float]):
    """CARLA Transform.transform 原位修改 Location 并返回 None，不能接收其返回值。"""
    location = carla_module.Location(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
    transform.transform(location)
    return location


def run(config: dict | None = None) -> dict:
    import carla
    import numpy as np
    from occluactisafe.data.adapters.carla import extract_visible_instances
    from occluactisafe.simulation.n14_images import save_rgb_frame
    from occluactisafe.simulation.sensors import FrameBuffer

    config = dict(config or yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8")))
    validate_scene_groups(config)
    output_root = Path(config["output_root"]).expanduser().resolve()
    dataset_root = Path(os.environ.get("OAS_DATASET_ROOT", "H:/OccluActiSafeDataset")).resolve()
    if not output_root.is_relative_to((dataset_root / "artifacts").resolve()):
        raise ValueError("CARLA图像、标注与清单必须保存在数据硬盘artifacts内")
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = "n15_carla_" + uuid.uuid4().hex
    run_dir = output_root / "02_运行产物" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    config_path = run_dir / "冻结采集配置.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    started = datetime.now().astimezone().isoformat()
    report = {"run_id": run_id, "status": "running", "started_at": started,
              "config_sha256": sha256(config_path), "carla_version": None,
              "map_roles": {item["map"]: item["project_role"] for item in config["maps"]},
              "groups": [], "failed_scenes": [], "source_commit": os.environ.get("OAS_SOURCE_COMMIT", "uncommitted"),
              "source_sha256": {relative: sha256(ROOT / relative) for relative in (
                  "scripts/capture_n15_carla.py", "src/occluactisafe/data/adapters/carla.py",
                  "src/occluactisafe/simulation/n14_images.py", "configs/data/carla_perception.yaml")},
              "environment": {"platform": platform.platform(), "python": sys.version,
                              "executable": sys.executable}}
    manifest_path = run_dir / "CARLA双视角数据清单.json"
    _write_json(manifest_path, report)
    client = carla.Client(config["host"], int(config["port"]))
    client.set_timeout(float(config["client_timeout_s"]))
    original_world = client.get_world()
    original_map = original_world.get_map().name
    original_settings = original_world.get_settings()
    original_weather = original_world.get_weather()
    np.random.seed(int(config["seed"]))
    try:
        report["carla_version"] = client.get_server_version()
        if report["carla_version"] != "0.9.16" or client.get_client_version() != "0.9.16":
            raise RuntimeError(f"CARLA版本不匹配：{client.get_client_version()}/{report['carla_version']}")
        for map_spec in config["maps"]:
            world = client.load_world(map_spec["map"])
            if map_spec["map"].split("_")[0] not in world.get_map().name:
                raise RuntimeError(f"地图加载不匹配：期望{map_spec['map']}，实际{world.get_map().name}")
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = float(config["fixed_delta_s"])
            settings.no_rendering_mode = False
            world.apply_settings(settings)
            world.set_pedestrians_seed(int(config["seed"]))
            world.set_weather(getattr(carla.WeatherParameters, map_spec["weather"]))
            map_hash = hashlib.sha256(world.get_map().to_opendrive().encode("utf-8")).hexdigest()
            for scene_index, scene_kind in enumerate(config["scenes"]):
                group_id = f"map:{map_spec['map']}"
                episode_id = f"{run_id}:{map_spec['map']}:{scene_kind}"
                scene_dir = run_dir / map_spec["project_role"] / map_spec["map"] / scene_kind
                scene_dir.mkdir(parents=True, exist_ok=False)
                actors, sensors, buffers = [], [], {}
                scene = {"episode_id": episode_id, "group_id": group_id,
                         "map": map_spec["map"], "map_opendrive_sha256": map_hash,
                         "project_role": map_spec["project_role"], "weather": map_spec["weather"],
                         "scene_kind": scene_kind, "status": "running", "frames": [], "labels": []}
                try:
                    spawns = world.get_map().get_spawn_points()
                    if len(spawns) < 2:
                        raise RuntimeError("地图可用车辆出生点不足")
                    ego_spawn = spawns[(scene_index * 37) % len(spawns)]
                    ego_bp = world.get_blueprint_library().find("vehicle.audi.a2")
                    ego_bp.set_attribute("role_name", "oas:n15:ego")
                    ego = world.try_spawn_actor(ego_bp, ego_spawn)
                    if ego is None:
                        raise RuntimeError("ego出生点被占用，保留失败场景")
                    actors.append(ego)
                    ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
                    waypoint = world.get_map().get_waypoint(ego_spawn.location, project_to_road=True)
                    ahead = waypoint.next(18.0)
                    moving_targets = []
                    moving_walker = None
                    if scene_kind in {"vehicle", "vehicle_occlusion"}:
                        if not ahead:
                            raise RuntimeError("ego路线前方无可用CARLA waypoint")
                        target_bp = world.get_blueprint_library().find("vehicle.audi.a2")
                        target_bp.set_attribute("role_name", "oas:n15:target:vehicle")
                        target = world.try_spawn_actor(target_bp, ahead[0].transform)
                        if target is None:
                            raise RuntimeError("车辆目标spawn失败")
                        actors.append(target)
                        moving_targets.append(target)
                        scene["actor_catalog"] = [{"actor_id": target.id, "track_id": f"{episode_id}:vehicle",
                                                   "category": "vehicle", "blueprint_id": target.type_id}]
                    elif scene_kind == "cyclist":
                        if not ahead:
                            raise RuntimeError("ego路线前方无可用CARLA waypoint")
                        bike_bp = world.get_blueprint_library().find("vehicle.bh.crossbike")
                        bike_bp.set_attribute("role_name", "oas:n15:target:cyclist")
                        bike = world.try_spawn_actor(bike_bp, ahead[0].transform)
                        if bike is None:
                            raise RuntimeError("骑行者目标spawn失败")
                        actors.append(bike)
                        moving_targets.append(bike)
                        scene["actor_catalog"] = [{"actor_id": bike.id, "track_id": f"{episode_id}:cyclist",
                                                   "category": "bicycle", "blueprint_id": bike.type_id}]
                    elif scene_kind == "pedestrian":
                        if not ahead:
                            raise RuntimeError("ego路线前方无可用CARLA waypoint")
                        right = ahead[0].transform.get_right_vector()
                        location = ahead[0].transform.location + right * 3.5
                        location.z += 0.2
                        walker_bp = world.get_blueprint_library().find("walker.pedestrian.0001")
                        walker_bp.set_attribute("role_name", "oas:n15:target:pedestrian")
                        walker = world.try_spawn_actor(walker_bp, carla.Transform(location, ahead[0].transform.rotation))
                        if walker is None:
                            raise RuntimeError("行人目标spawn失败")
                        actors.append(walker)
                        moving_walker = walker
                        scene["actor_catalog"] = [{"actor_id": walker.id, "track_id": f"{episode_id}:pedestrian",
                                                   "category": "pedestrian", "blueprint_id": walker.type_id}]
                    elif scene_kind == "negative":
                        scene["actor_catalog"] = []
                    for actor in scene["actor_catalog"]:
                        if actor["actor_id"] > 65535:
                            raise RuntimeError("CARLA实例相机16位actor ID溢出")

                    ego_transform = ego.get_transform()
                    roadside_xyz = config["roadside_camera_local_xyz"]
                    roadside_location = local_camera_location_to_world(ego_transform, carla, roadside_xyz)
                    roadside_transform = carla.Transform(roadside_location,
                        carla.Rotation(yaw=ego_transform.rotation.yaw,
                                       pitch=float(config["roadside_camera_pitch_deg"])))
                    views = (("车端", carla.Transform(carla.Location(x=1.5, z=1.6), carla.Rotation()), ego),
                             ("路侧", roadside_transform, None))
                    camera_sensors = {}
                    for view_name, transform, parent in views:
                        buffers[view_name + "_rgb"] = FrameBuffer(capacity=8)
                        buffers[view_name + "_mask"] = FrameBuffer(capacity=8)
                        for kind, sensor_type, suffix in (("rgb", "sensor.camera.rgb", "rgb"),
                                                          ("mask", "sensor.camera.instance_segmentation", "mask")):
                            bp = world.get_blueprint_library().find(sensor_type)
                            bp.set_attribute("image_size_x", str(config["image_size"][0]))
                            bp.set_attribute("image_size_y", str(config["image_size"][1]))
                            bp.set_attribute("fov", str(config["horizontal_fov_deg"]))
                            bp.set_attribute("sensor_tick", str(config["fixed_delta_s"]))
                            sensor = world.spawn_actor(bp, transform, attach_to=parent)
                            sensors.append(sensor)
                            if kind == "rgb":
                                camera_sensors[view_name] = sensor
                            sensor.listen(lambda image, key=view_name + "_" + suffix:
                                          buffers[key].put(image.frame, image))
                    for _ in range(5):
                        world.tick()
                    width, height = map(int, config["image_size"])
                    focal = width / 2 / __import__("math").tan(__import__("math").radians(float(config["horizontal_fov_deg"])) / 2)
                    calibration = {name: {"width": width, "height": height,
                        "horizontal_fov_deg": config["horizontal_fov_deg"],
                        "K": [[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]],
                        "camera_to_carla_world": sensor.get_transform().get_matrix()}
                        for name, sensor in (("车端", sensors[0]), ("路侧", sensors[2]))}
                    for local_index in range(int(config["frames_per_scene"])):
                        for actor in moving_targets:
                            forward = actor.get_transform().get_forward_vector()
                            actor.set_target_velocity(carla.Vector3D(
                                forward.x * float(config["moving_target_speed_mps"]),
                                forward.y * float(config["moving_target_speed_mps"]), 0.0))
                        if moving_walker is not None:
                            crossing = ego_transform.get_right_vector()
                            moving_walker.apply_control(carla.WalkerControl(
                                direction=carla.Vector3D(-crossing.x, -crossing.y, 0.0),
                                speed=float(config["pedestrian_speed_mps"]), jump=False))
                        frame_id = world.tick()
                        frame_record = {"carla_frame_id": frame_id, "source_time_s": None,
                                        "views": {}}
                        for view_name, _, _ in views:
                            rgb = buffers[view_name + "_rgb"].get(frame_id, 5.0)
                            mask = buffers[view_name + "_mask"].get(frame_id, 5.0)
                            if not rgb["valid"] or not mask["valid"]:
                                raise TimeoutError(f"{view_name}图像/实例帧缺失：frame={frame_id}")
                            rgb_image, mask_image = rgb["value"], mask["value"]
                            if (rgb_image.width, rgb_image.height) != (width, height) or (mask_image.width, mask_image.height) != (width, height):
                                raise ValueError("RGB与实例分割图像尺寸不一致")
                            if rgb_image.frame != mask_image.frame:
                                raise ValueError("RGB与实例分割frame不一致")
                            timestamp = float(rgb_image.timestamp)
                            if abs(timestamp - float(mask_image.timestamp)) > 1e-6:
                                raise ValueError("RGB与实例分割时间戳不一致")
                            if frame_record["source_time_s"] is None:
                                frame_record["source_time_s"] = timestamp
                            elif abs(frame_record["source_time_s"] - timestamp) > 1e-6:
                                raise ValueError("车端与路侧图像时间戳不一致")
                            stem = f"{view_name}_帧{local_index:04d}"
                            rgb_path, mask_path = scene_dir / f"{stem}_RGB.png", scene_dir / f"{stem}_实例掩码.png"
                            rgb_proof, mask_proof = save_rgb_frame(rgb_image, rgb_path), save_rgb_frame(mask_image, mask_path)
                            raw = np.frombuffer(mask_image.raw_data, dtype=np.uint8).reshape(height, width, 4)
                            mask_rgb = raw[:, :, :3][:, :, ::-1].copy()
                            label = extract_visible_instances(mask_rgb, scene["actor_catalog"],
                                frame_id=frame_id, episode_id=episode_id, source_time_s=timestamp,
                                rgb_path=rgb_path.name, instance_path=mask_path.name)
                            camera_location = camera_sensors[view_name].get_transform().location
                            actor_by_id = {actor.id: actor for actor in actors}
                            for visible_label in label["labels"]:
                                target_actor = actor_by_id[visible_label["actor_id"]]
                                target_location = target_actor.get_transform().location
                                visible_label["camera_distance_m"] = float(camera_location.distance(target_location))
                            label["view"] = view_name
                            label["rgb_sha256"] = rgb_proof["sha256"]
                            label["instance_sha256"] = mask_proof["sha256"]
                            label_path = scene_dir / f"{stem}_可见标注.json"
                            _write_json(label_path, label)
                            frame_record["views"][view_name] = {
                                "frame_id": int(rgb_image.frame), "rgb": rgb_path.name, "rgb_sha256": rgb_proof["sha256"],
                                "mask": mask_path.name, "mask_sha256": mask_proof["sha256"],
                                "label": label_path.name, "label_sha256": sha256(label_path),
                                "visible_objects": len(label["labels"]), "negative_frame": label["negative_frame"]}
                            scene["labels"].append(label)
                        scene["frames"].append(frame_record)
                    scene["camera_calibration"] = calibration
                    scene["status"] = "completed"
                except Exception as error:
                    scene.update(status="failed", error=f"{type(error).__name__}: {error}")
                    report["failed_scenes"].append({"episode_id": episode_id, "error": scene["error"]})
                finally:
                    for sensor in sensors:
                        try:
                            sensor.stop(); sensor.destroy()
                        except Exception:
                            pass
                    for actor in reversed(actors):
                        try:
                            actor.destroy()
                        except Exception:
                            pass
                    for buffer in buffers.values():
                        buffer.close()
                scene_path = scene_dir / "场景采集清单.json"
                _write_json(scene_path, scene)
                scene_summary = summarize_scene_labels(scene)
                report["groups"].append({"episode_id": episode_id, "group_id": group_id,
                    "map": map_spec["map"], "project_role": map_spec["project_role"],
                    "scene_kind": scene_kind, "status": scene["status"],
                    "frames": len(scene["frames"]), "manifest": str(scene_path),
                    "annotated_view_frames": sum(len(frame["views"]) for frame in scene["frames"]),
                    **scene_summary,
                    "manifest_sha256": sha256(scene_path)})
                _write_json(manifest_path, report)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            current = client.get_world()
            current.apply_settings(original_settings)
            current.set_weather(original_weather)
            if current.get_map().name != original_map:
                restored = client.load_world(original_map.rsplit("/", 1)[-1])
                restored.apply_settings(original_settings)
                restored.set_weather(original_weather)
            report["world_restored"] = True
        except Exception as error:
            report["world_restored"] = False
            report["restore_error"] = f"{type(error).__name__}: {error}"
        report["finished_at"] = datetime.now().astimezone().isoformat()
        if report.get("status") == "running":
            report["status"] = "completed" if not report["failed_scenes"] else "failed"
        _write_json(manifest_path, report)
    return {"run_id": run_id, "run_dir": str(run_dir), "status": report["status"], "report": report}


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    if args.output_root:
        config["output_root"] = str(args.output_root.expanduser().resolve())
    result = run(config)
    print(f"run_id={result['run_id']} status={result['status']} groups={len(result['report']['groups'])}")
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
