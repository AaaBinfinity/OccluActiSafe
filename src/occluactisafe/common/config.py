"""配置合并与校验：所有相对路径均基于明确的项目根目录。"""

import argparse
from copy import deepcopy
import math
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]

# 当前只定义 N02 用到的公共设置；模型和数据配置待对应节点授权后扩展。
DEFAULTS = {
    "clock": {"simulation_dt": 0.05, "sensor_dt": 0.1, "prediction_dt": 0.1,
              "planner_dt": 0.1, "actuator_dt": 0.05},
    "prediction": {"history_steps": 20, "future_steps": 40, "future_dt": 0.1},
    "planning": {"horizon_s": 3.0},
    "paths": {"data_root": "data", "output_root": "artifacts"},
    "runtime": {"seed": 0},
    "logging": {"keep_truth_separate": True, "save_failed_episodes": True},
}


class _UniqueLoader(yaml.SafeLoader):
    """Reject repeated keys rather than making YAML file ordering significant."""


def _construct_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValueError("配置键必须是字符串")
        if key in mapping:
            raise ValueError(f"YAML 含重复配置键: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def _merge(target, source, template, path=""):
    if not isinstance(source, dict):
        raise ValueError(f"配置 {path or '<root>'} 必须是映射")
    for key, value in source.items():
        if not isinstance(key, str) or key not in template:
            raise ValueError(f"未知配置字段: {path}{key}")
        field = f"{path}{key}"
        if isinstance(template[key], dict):
            _merge(target[key], value, template[key], field + ".")
        else:
            target[key] = value


def _yaml_file(path):
    try:
        with path.open("r", encoding="utf-8") as stream:
            result = yaml.load(stream, Loader=_UniqueLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML 解析失败: {path}") from exc
    if result is None:
        return {}
    if not isinstance(result, dict):
        raise ValueError(f"配置文件顶层必须是映射: {path}")
    return result


def _root_path(value, root):
    if not isinstance(value, str) or not value.strip() or "://" in value:
        raise ValueError("路径必须是非空本地路径")
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else root / path).resolve())


def _validate(config, root):
    clock = config["clock"]
    for section, names in (("clock", tuple(clock)), ("prediction", ("future_dt",)),
                           ("planning", ("horizon_s",))):
        for name in names:
            value = config[section][name]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{section}.{name} 必须是有限正数")
    for name in ("history_steps", "future_steps"):
        value = config["prediction"][name]
        if type(value) is not int or value <= 0:
            raise ValueError(f"prediction.{name} 必须是正整数")
    seed = config["runtime"]["seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("runtime.seed 必须是非负整数")
    for name, value in config["logging"].items():
        if type(value) is not bool:
            raise ValueError(f"logging.{name} 必须是布尔值")
    def multiple(value, base, label):
        ratio = value / base
        if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"{label} 必须是基础周期的整数倍")
    for name in ("sensor_dt", "prediction_dt", "planner_dt", "actuator_dt"):
        multiple(clock[name], clock["simulation_dt"], "clock." + name)
    multiple(config["prediction"]["future_dt"], clock["simulation_dt"], "prediction.future_dt")
    multiple(config["planning"]["horizon_s"], clock["planner_dt"], "planning.horizon_s")
    if config["prediction"]["future_steps"] * config["prediction"]["future_dt"] < config["planning"]["horizon_s"]:
        raise ValueError("预测时域不得短于规划时域")
    for name in config["paths"]:
        config["paths"][name] = _root_path(config["paths"][name], root)


def _config_path(path, root):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def validate_resolved_config(config):
    """校验完整、已解析的 N02 配置；不填充缺失字段，也不修改输入。"""
    def exact_fields(value, template, path="config"):
        if not isinstance(value, dict):
            raise ValueError(f"{path} 必须是映射")
        if set(value) != set(template):
            raise ValueError(f"{path} 字段不完整或含未知字段")
        for key, field_template in template.items():
            if isinstance(field_template, dict):
                exact_fields(value[key], field_template, f"{path}.{key}")

    exact_fields(config, DEFAULTS)
    for name, path in config["paths"].items():
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError(f"paths.{name} 必须是已解析的绝对路径")
    validated = deepcopy(config)
    _validate(validated, PROJECT_ROOT)
    if validated != config:
        raise ValueError("配置路径必须是规范化的绝对路径")
    return validated


def load_config(*, config_file=None, default_config_file=None, overrides=None,
                cli_overrides=None, project_root=None):
    """合并包默认→base.yaml→所选任务 YAML→脚本覆盖→显式 CLI。

    config_file 替换 default_config_file；相对配置文件、数据及输出路径均从项目根解析。
    显式覆盖中的 False 和 0 不会被视为缺省。返回深拷贝字典，路径为绝对路径。
    """
    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    merged = deepcopy(DEFAULTS)
    _merge(merged, _yaml_file(root / "configs" / "base.yaml"), DEFAULTS)
    selected = config_file if config_file is not None else default_config_file
    if selected is not None:
        _merge(merged, _yaml_file(_config_path(selected, root)), DEFAULTS)
    if overrides is not None:
        _merge(merged, overrides, DEFAULTS)
    if cli_overrides is not None:
        _merge(merged, cli_overrides, DEFAULTS)
    _validate(merged, root)
    return merged


def parse_config_cli(argv=None):
    """可供入口使用的可选 CLI；未传入的选项不加入覆盖项。"""
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument("--config")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--keep-truth-separate", action=argparse.BooleanOptionalAction)
    parser.add_argument("--data-root")
    args = vars(parser.parse_args(argv))
    result = {}
    if "seed" in args:
        result.setdefault("runtime", {})["seed"] = args["seed"]
    if "keep_truth_separate" in args:
        result.setdefault("logging", {})["keep_truth_separate"] = args["keep_truth_separate"]
    if "data_root" in args:
        result.setdefault("paths", {})["data_root"] = args["data_root"]
    return args.get("config"), result
