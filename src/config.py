"""Загрузка конфигов: слияние нескольких YAML + переопределения вида key.sub=value."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_value(raw: str) -> Any:
    val = yaml.safe_load(raw)
    if isinstance(val, str):  # YAML 1.1 читает "2e-4" как строку
        try:
            return float(val)
        except ValueError:
            pass
    return val


def set_dotted(cfg: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def load_config(paths: list[str], overrides: list[str] | None = None) -> dict:
    cfg: dict = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Переопределение должно иметь вид key=value, получено: {item}")
        set_dotted(cfg, key.strip(), _parse_value(raw))
    return cfg


def make_run_name(cfg: dict) -> str:
    if cfg.get("run_name"):
        return cfg["run_name"]
    d = cfg["data"]
    parts = [
        cfg["method"],
        cfg["model"]["name"].split("/")[-1],
        d["train_dataset"],
        f"n{d['train_size'] or 'all'}",
    ]
    if cfg["method"] != "full":
        parts.append(f"r{cfg['lora']['r']}")
    parts += [f"lr{float(cfg['train']['lr']):g}", f"s{cfg['seed']}"]
    return "_".join(parts)


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p
