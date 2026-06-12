"""eloratings.net current international Elo ratings (cross-check / seed for our own Elo).

The site renders client-side from flat TSVs we can fetch directly (no key):
  - en.teams.tsv : "<2-letter code>\\t<Team name>"
  - World.tsv    : rank, _, <code>, <elo>, ... (only cols 0,2,3 are used; later cols
                   contain a mis-encoded U+2212 minus sign we ignore)

Our authoritative Elo is computed from match results in src/models/elo.py; this module is
a sanity/seed reference, so it degrades gracefully if the site format ever changes.
"""

from __future__ import annotations

import pandas as pd

from src.config import CONFIG
from src.data.cache import cached_get

_BASE = CONFIG.settings["data_sources"]["elo_url"].rstrip("/")


def _team_codes(*, force: bool = False) -> dict[str, str]:
    dest = CONFIG.raw / "eloratings_teams.tsv"
    cached_get(f"{_BASE}/en.teams.tsv", dest, force=force)
    codes: dict[str, str] = {}
    for line in dest.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            codes[parts[0].strip()] = parts[1].strip()
    return codes


def fetch_current_elo(*, force: bool = False) -> pd.DataFrame:
    """Return DataFrame[team, elo_external, world_rank_external] (best-effort, normalized)."""
    try:
        codes = _team_codes(force=force)
        dest = CONFIG.raw / "eloratings_world.tsv"
        cached_get(f"{_BASE}/World.tsv", dest, force=force)
        rows = []
        for line in dest.read_text(encoding="utf-8").splitlines():
            c = line.split("\t")
            if len(c) < 4:
                continue
            try:
                rank = int(c[0])
                rating = float(c[3])
            except ValueError:
                continue
            name = codes.get(c[2].strip())
            if name:
                rows.append((CONFIG.normalize(name), rating, rank))
        df = pd.DataFrame(rows, columns=["team", "elo_external", "world_rank_external"])
        return df.drop_duplicates("team").reset_index(drop=True)
    except Exception as exc:  # never break the pipeline on a cross-check source
        print(f"[elo_scraper] warning: could not fetch eloratings.net ({exc!r}); "
              "continuing without the external Elo seed.")
        return pd.DataFrame(columns=["team", "elo_external", "world_rank_external"])


def wc_teams_elo(*, force: bool = False) -> pd.DataFrame:
    """Current external Elo for just the 48 qualified teams (sorted strongest first)."""
    df = fetch_current_elo(force=force)
    wc = df[df["team"].isin(CONFIG.teams)].copy()
    return wc.sort_values("elo_external", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    wc = wc_teams_elo()
    print(f"matched {len(wc)}/48 WC teams to eloratings.net")
    print(wc.head(12).to_string(index=False))
    missing = set(CONFIG.teams) - set(wc["team"])
    if missing:
        print("unmatched:", sorted(missing))
