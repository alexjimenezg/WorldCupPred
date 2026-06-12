"""Team ranking feature.

A clean, free, *full* current FIFA ranking table is surprisingly hard to get (the FIFA
site is JS-rendered and Wikipedia only shows leaders/methodology). FIFA rank is also ~0.9
correlated with Elo, so we treat it as a secondary feature:

  1. If a maintained ranking file exists in data/raw (drop-in CSV: team,rank[,points]), use it.
  2. Otherwise fall back to the eloratings.net world rank as the rank proxy.

Either way callers get DataFrame[team, fifa_rank, ranking_source]. Swap in a real FIFA-rank
API later (e.g. football-data.org once keyed) without touching downstream code.
"""

from __future__ import annotations

import pandas as pd

from src.config import CONFIG
from src.data.elo_scraper import fetch_current_elo

_MANUAL_CSV = CONFIG.raw / "fifa_rankings_manual.csv"


def _from_manual_csv() -> pd.DataFrame | None:
    if not _MANUAL_CSV.exists():
        return None
    df = pd.read_csv(_MANUAL_CSV)
    cols = {c.lower(): c for c in df.columns}
    if "team" not in cols or "rank" not in cols:
        return None
    out = pd.DataFrame({
        "team": df[cols["team"]].map(CONFIG.normalize),
        "fifa_rank": df[cols["rank"]].astype(int),
    })
    out["ranking_source"] = "manual_csv"
    return out.drop_duplicates("team").reset_index(drop=True)


def _from_elo_proxy(*, force: bool = False) -> pd.DataFrame:
    elo = fetch_current_elo(force=force)
    if elo.empty:
        return pd.DataFrame(columns=["team", "fifa_rank", "ranking_source"])
    elo = elo.sort_values("elo_external", ascending=False).reset_index(drop=True)
    elo["fifa_rank"] = elo.index + 1
    elo["ranking_source"] = "elo_proxy"
    return elo[["team", "fifa_rank", "ranking_source"]]


def fetch_fifa_rankings(*, force: bool = False) -> pd.DataFrame:
    """Best-effort team ranking: manual CSV if present, else eloratings world rank."""
    manual = _from_manual_csv()
    if manual is not None and not manual.empty:
        return manual
    return _from_elo_proxy(force=force)


def wc_rankings(*, force: bool = False) -> pd.DataFrame:
    df = fetch_fifa_rankings(force=force)
    wc = df[df["team"].isin(CONFIG.teams)].copy()
    # Re-rank within the WC field so every qualified team has a dense rank too.
    wc = wc.sort_values("fifa_rank").reset_index(drop=True)
    wc["wc_seed_rank"] = wc.index + 1
    return wc


if __name__ == "__main__":
    wc = wc_rankings()
    src = wc["ranking_source"].iloc[0] if not wc.empty else "none"
    print(f"ranking source: {src}; matched {len(wc)}/48 WC teams")
    print(wc.head(10).to_string(index=False))
