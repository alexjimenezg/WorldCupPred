"""The training backbone: martj42/international_results.

Three CSVs, all internationals from 1872 to the present, refreshed continuously:
  - results.csv     date, home_team, away_team, home_score, away_score, tournament, city,
                    country, neutral
  - goalscorers.csv date, home_team, away_team, team, scorer, minute, own_goal, penalty
  - shootouts.csv   date, home_team, away_team, winner (+ first_shooter on newer versions)

We pull them straight from GitHub raw (confirmed reachable, no key) and normalize team
names to the canonical spelling so they line up with the 2026 draw.
"""

from __future__ import annotations

import pandas as pd

from src.config import CONFIG
from src.data.cache import cached_get

_DS = CONFIG.settings["data_sources"]
_BASE = _DS["results_base_url"]
_FILES = _DS["results_files"]


def _url(name: str) -> str:
    return f"{_BASE}/{_FILES[name]}"


def _fetch_csv(name: str, *, force: bool = False) -> pd.DataFrame:
    dest = CONFIG.raw / _FILES[name]
    cached_get(_url(name), dest, force=force)
    return pd.read_csv(dest)


def fetch_results(*, force: bool = False, normalize: bool = True) -> pd.DataFrame:
    """All international match results, typed and (optionally) name-normalized."""
    df = _fetch_csv("results", force=force)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_score", "away_score"]).reset_index(drop=True)
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(bool)
    if normalize:
        df["home_team"] = df["home_team"].map(CONFIG.normalize)
        df["away_team"] = df["away_team"].map(CONFIG.normalize)
    return df


def fetch_goalscorers(*, force: bool = False, normalize: bool = True) -> pd.DataFrame:
    df = _fetch_csv("goalscorers", force=force)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if normalize:
        for col in ("home_team", "away_team", "team"):
            if col in df.columns:
                df[col] = df[col].map(CONFIG.normalize)
    return df


def fetch_shootouts(*, force: bool = False, normalize: bool = True) -> pd.DataFrame:
    df = _fetch_csv("shootouts", force=force)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if normalize:
        for col in ("home_team", "away_team", "winner"):
            if col in df.columns:
                df[col] = df[col].map(CONFIG.normalize)
    return df


def load_all(*, force: bool = False) -> dict[str, pd.DataFrame]:
    """Convenience: every backbone table, name-normalized."""
    return {
        "results": fetch_results(force=force),
        "goalscorers": fetch_goalscorers(force=force),
        "shootouts": fetch_shootouts(force=force),
    }


if __name__ == "__main__":  # quick smoke test
    res = fetch_results()
    print(f"results: {len(res):,} matches, {res['date'].min().date()} -> {res['date'].max().date()}")
    print(res.tail(3).to_string(index=False))
