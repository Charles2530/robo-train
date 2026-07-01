"""LightX2V-style layered YAML config loading."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_layered_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file with recursive `defaults` deep-merge support."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    return _load_with_stack(config_path.resolve(), stack=[])


def _load_with_stack(path: Path, stack: list[Path]) -> dict[str, Any]:
    if path in stack:
        cycle = " -> ".join(str(item) for item in [*stack, path])
        raise ValueError(f"cyclic config defaults: {cycle}")

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config root must be a mapping: {path}")

    merged: dict[str, Any] = {}
    for default_ref in payload.get("defaults", []) or []:
        default_path = _resolve_default(path.parent, default_ref)
        merged = _deep_merge(merged, _load_with_stack(default_path, [*stack, path]))

    local = {key: value for key, value in payload.items() if key != "defaults"}
    return _deep_merge(merged, local)


def _resolve_default(base_dir: Path, default_ref: str | dict[str, Any]) -> Path:
    if isinstance(default_ref, dict):
        if len(default_ref) != 1:
            raise ValueError(f"default mapping must contain one item: {default_ref}")
        default_ref = next(iter(default_ref.values()))
    if not isinstance(default_ref, str):
        raise TypeError(f"default reference must be a string: {default_ref!r}")
    return (base_dir / default_ref).resolve()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = _deep_merge(existing, value)
        else:
            result[key] = deepcopy(value)
    return result
