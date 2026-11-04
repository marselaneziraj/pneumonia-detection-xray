"""YAML config loading. Every hyperparameter lives in configs/*.yaml, not in argparse defaults."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REQUIRED_SECTIONS = (
    "experiment",
    "data",
    "augment",
    "model",
    "train",
    "selection",
    "eval",
    "paths",
)


class ConfigError(ValueError):
    pass


def _deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path, overrides: dict | None = None) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open() as f:
        cfg = yaml.safe_load(f) or {}
    if overrides:
        cfg = _deep_update(cfg, overrides)
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ConfigError(f"{path}: missing config sections {missing}")
    tr = cfg["selection"]["target_recall"]
    if not 0.0 < float(tr) <= 1.0:
        raise ConfigError(f"selection.target_recall must be in (0, 1], got {tr}")
    cfg["_config_path"] = str(path)
    return cfg


def run_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["runs_dir"]) / cfg["experiment"]["name"]
