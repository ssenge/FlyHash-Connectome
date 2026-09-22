"""Config loading: config.example.yaml, with config.yaml merged over it."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    def __init__(self, d: dict):
        self._d = d

    def __getitem__(self, k: str) -> Any:
        return self._d[k]

    def get(self, k: str, default: Any = None) -> Any:
        return self._d.get(k, default)

    @property
    def raw(self) -> dict:
        return self._d

    def path(self, *parts: str) -> Path:
        """Resolve a config-relative path against the repo root."""
        p = Path(*parts)
        return p if p.is_absolute() else ROOT / p

    # Frequently used derived paths.
    @property
    def raw_dir(self) -> Path:
        return self.path(self._d["data"]["raw_dir"])

    @property
    def graph_dir(self) -> Path:
        return self.path(self._d["data"]["graph_dir"])

    @property
    def cache_dir(self) -> Path:
        return self.path(self._d["data"]["cache_dir"])


def load(path: str | Path | None = None) -> Config:
    example = ROOT / "config.example.yaml"
    d = yaml.safe_load(example.read_text())
    override = Path(path) if path else ROOT / "config.yaml"
    if override.exists():
        d = _merge(d, yaml.safe_load(override.read_text()) or {})
    return Config(d)
