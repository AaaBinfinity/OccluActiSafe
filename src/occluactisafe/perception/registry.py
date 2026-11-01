"""按显式配置注册检测器构造器；注册不触发权重下载。"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


_FACTORIES: dict[str, Callable[..., Any]] = {}


def register_detector(name: str, factory: Callable[..., Any], *, replace: bool = False) -> None:
    key = name.strip().lower()
    if not key or not callable(factory):
        raise ValueError("检测器名称和构造器必须有效")
    if key in _FACTORIES and not replace:
        raise ValueError(f"检测器已注册：{key}")
    _FACTORIES[key] = factory


def create_detector(name: str, **config):
    key = name.strip().lower()
    if key not in _FACTORIES:
        raise KeyError(f"未注册检测器：{key}")
    return _FACTORIES[key](**config)


def registered_detectors() -> tuple[str, ...]:
    return tuple(sorted(_FACTORIES))
