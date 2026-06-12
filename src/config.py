"""Central configuration: paths, settings, the 2026 draw, and team-name normalization.

Everything else imports from here so there is a single source of truth. Loading is
cached, so repeated calls are cheap.

    from src.config import CONFIG
    CONFIG.teams              # the 48 qualified nations (canonical names)
    CONFIG.group_of("Spain")  # -> "H"
    CONFIG.normalize("Türkiye")  # -> "Turkey"
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml

try:  # optional dependency, never fatal
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(*_a: Any, **_k: Any) -> bool:
        return False


def _find_project_root(start: Path) -> Path:
    """Walk upward until we find the marker files, else fall back to two levels up."""
    for parent in [start, *start.parents]:
        if (parent / "config" / "settings.yaml").exists():
            return parent
    return start.parents[1]


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass(frozen=True)
class Config:
    """Immutable view over the YAML config + 2026 draw, with helper lookups."""

    root: Path = PROJECT_ROOT
    settings: dict[str, Any] = field(default_factory=dict)
    groups_raw: dict[str, Any] = field(default_factory=dict)
    confeds_raw: dict[str, Any] = field(default_factory=dict)

    # ---- construction -------------------------------------------------------
    @classmethod
    def load(cls) -> "Config":
        load_dotenv(PROJECT_ROOT / ".env")  # populate os.environ if .env present
        cfg_dir = PROJECT_ROOT / "config"
        return cls(
            root=PROJECT_ROOT,
            settings=_read_yaml(cfg_dir / "settings.yaml"),
            groups_raw=_read_yaml(cfg_dir / "groups_2026.yaml"),
            confeds_raw=_read_yaml(cfg_dir / "confederations.yaml"),
        )

    # ---- paths --------------------------------------------------------------
    def path(self, key: str) -> Path:
        """Resolve a logical path from settings.paths to an absolute Path (mkdir'd)."""
        rel = self.settings.get("paths", {}).get(key)
        if rel is None:
            raise KeyError(f"Unknown path key: {key!r}")
        p = (self.root / rel).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def raw(self) -> Path:
        return self.path("data_raw")

    @property
    def interim(self) -> Path:
        return self.path("data_interim")

    @property
    def processed(self) -> Path:
        return self.path("data_processed")

    @property
    def models_dir(self) -> Path:
        return self.path("models")

    @property
    def figures(self) -> Path:
        return self.path("figures")

    @property
    def vault_dir(self) -> Path:
        return self.path("vault")

    # ---- draw / groups ------------------------------------------------------
    @cached_property
    def groups(self) -> dict[str, list[str]]:
        return {g: list(teams) for g, teams in self.groups_raw["groups"].items()}

    @cached_property
    def teams(self) -> list[str]:
        return [t for teams in self.groups.values() for t in teams]

    @cached_property
    def hosts(self) -> list[str]:
        return list(self.groups_raw.get("hosts", []))

    @cached_property
    def _team_to_group(self) -> dict[str, str]:
        return {t: g for g, teams in self.groups.items() for t in teams}

    def group_of(self, team: str) -> str:
        return self._team_to_group[self.normalize(team)]

    def is_host(self, team: str) -> bool:
        return self.normalize(team) in set(self.hosts)

    @property
    def fmt(self) -> dict[str, Any]:
        return self.groups_raw["format"]

    # ---- confederations -----------------------------------------------------
    @cached_property
    def _team_to_confed(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for confed, teams in self.confeds_raw["confederations"].items():
            for t in teams:
                out[t] = confed
        return out

    def confederation_of(self, team: str) -> str:
        return self._team_to_confed.get(self.normalize(team), "UNKNOWN")

    @cached_property
    def confederation_prior(self) -> dict[str, float]:
        return dict(self.confeds_raw.get("confederation_strength_prior", {}))

    # ---- name normalization -------------------------------------------------
    @cached_property
    def _aliases(self) -> dict[str, str]:
        # case-insensitive, stripped keys
        return {k.strip().lower(): v for k, v in self.confeds_raw.get("aliases", {}).items()}

    def normalize(self, name: str) -> str:
        """Map any known spelling to the canonical martj42 name (idempotent)."""
        if name is None:
            return name
        key = str(name).strip()
        return self._aliases.get(key.lower(), key)

    # ---- secrets ------------------------------------------------------------
    @staticmethod
    def env(key: str, default: str | None = None) -> str | None:
        val = os.environ.get(key, default)
        return val or default


CONFIG = Config.load()
