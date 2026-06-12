"""Tiny cached-download helper shared by the data-source modules.

Everything lands in data/raw/ and is re-fetched only when older than
settings.data_sources.cache_max_age_days (or force=True). Keeps the scrapers polite
and makes the whole pipeline runnable offline once the cache is warm.
"""

from __future__ import annotations

import time
from pathlib import Path

import requests

from src.config import CONFIG

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) WorldCupPred/0.1 "
    "(+research; contact via project)"
)
_DEFAULT_MAX_AGE = float(CONFIG.settings.get("data_sources", {}).get("cache_max_age_days", 1))


def _age_days(path: Path) -> float:
    return (time.time() - path.stat().st_mtime) / 86400.0


def is_fresh(path: Path, max_age_days: float | None = None) -> bool:
    max_age = _DEFAULT_MAX_AGE if max_age_days is None else max_age_days
    return path.exists() and path.stat().st_size > 0 and _age_days(path) <= max_age


def cached_get(
    url: str,
    dest: Path,
    *,
    force: bool = False,
    max_age_days: float | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
    binary: bool = False,
) -> Path:
    """Download `url` to `dest` unless a fresh copy already exists. Returns `dest`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not force and is_fresh(dest, max_age_days):
        return dest

    hdrs = {"User-Agent": _USER_AGENT}
    if headers:
        hdrs.update(headers)
    resp = requests.get(url, headers=hdrs, timeout=timeout)
    resp.raise_for_status()

    if binary:
        dest.write_bytes(resp.content)
    else:
        dest.write_text(resp.text, encoding="utf-8")
    return dest
