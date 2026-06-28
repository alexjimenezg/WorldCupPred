"""football-data.org client (optional, free tier) — 2026 fixtures, live scores, results.

Needs FOOTBALL_DATA_API_KEY in .env. Without it everything returns empty and manual entry
via the app/CLI still works. `import_finished_to_store` pulls finished World Cup matches
straight into the results store so the live loop can refit on real data automatically.
"""

from __future__ import annotations

import pandas as pd
import requests

from src.config import CONFIG

_BASE = CONFIG.settings["data_sources"]["football_data_base"]
_COMP = CONFIG.settings["data_sources"]["football_data_competition"]


def available() -> bool:
    return bool(CONFIG.env("FOOTBALL_DATA_API_KEY"))


def _get(path: str, params: dict | None = None) -> dict | None:
    key = CONFIG.env("FOOTBALL_DATA_API_KEY")
    if not key:
        return None
    r = requests.get(f"{_BASE}{path}", params=params or {},
                     headers={"X-Auth-Token": key}, timeout=20)
    if r.status_code != 200:
        print(f"[live_scores] HTTP {r.status_code}: {r.text[:120]}")
        return None
    return r.json()


def fetch_matches(status: str | None = None) -> pd.DataFrame:
    """All 2026 World Cup matches (optionally filtered by status: SCHEDULED/FINISHED/...)."""
    data = _get(f"/competitions/{_COMP}/matches",
                {"status": status} if status else None)
    if not data:
        return pd.DataFrame()
    rows = []
    for m in data.get("matches", []):
        score = m.get("score", {}).get("fullTime", {})
        rows.append({
            "utc_date": m.get("utcDate"), "status": m.get("status"),
            "stage": m.get("stage"), "group": m.get("group"),
            "home_team": CONFIG.normalize((m.get("homeTeam") or {}).get("name", "")),
            "away_team": CONFIG.normalize((m.get("awayTeam") or {}).get("name", "")),
            "home_score": score.get("home"), "away_score": score.get("away"),
        })
    return pd.DataFrame(rows)


_STAGE_MAP = {
    "GROUP_STAGE": "group", "LAST_32": "R32", "LAST_16": "R16",
    "QUARTER_FINALS": "QF", "SEMI_FINALS": "SF", "FINAL": "final",
}


def import_finished_to_store(*, store=None) -> int:
    """Pull finished WC matches into the results store. Returns number imported."""
    from src.results_store import ResultsStore
    store = store if store is not None else ResultsStore()
    df = fetch_matches(status="FINISHED")
    if df.empty:
        return 0
    count = 0
    for _, m in df.iterrows():
        if m["home_score"] is None or m["away_score"] is None:
            continue
        if m["home_team"] not in CONFIG.teams or m["away_team"] not in CONFIG.teams:
            continue
        stage = _STAGE_MAP.get(str(m["stage"]), "group")
        try:
            store.add(m["home_team"], m["away_team"], int(m["home_score"]),
                      int(m["away_score"]), stage=stage,
                      on=str(m["utc_date"])[:10] if m["utc_date"] else None)
            count += 1
        except ValueError:
            continue  # skip cross-group / non-field oddities
    return count


if __name__ == "__main__":
    if not available():
        print("FOOTBALL_DATA_API_KEY not set — add it to .env to enable live scores.")
    else:
        df = fetch_matches()
        print(f"{len(df)} WC matches; {int((df['status'] == 'FINISHED').sum())} finished")
        print(df.head(8).to_string(index=False))
