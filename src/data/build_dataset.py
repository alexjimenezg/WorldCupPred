"""Merge the raw sources into the model-ready match-level table.

Output (data/processed/):
  matches.parquet         every international match, cleaned + enriched with:
                            outcome (H/D/A), result_home (1/.5/0), total_goals,
                            importance (category) and importance_weight (Elo K),
                            is_competitive, days_since_prev_* are added later in features.
  teams_reference.parquet the 48 qualified teams with group, confederation, external Elo
                            and rank (for the app / vault / priors).

This module owns the tournament -> importance mapping (World Football Elo weighting),
which both the Elo model and the goal model consume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.data.elo_scraper import fetch_current_elo
from src.data.fifa_rankings import fetch_fifa_rankings
from src.data.kaggle_results import fetch_results

# Tournament importance categories -> Elo K weight (World Football Elo convention:
# 60 World Cup, 50 continental finals, 45 Confederations Cup, 40 WC qualifiers,
# 35 Nations League, 30 other competitive, 20 friendly).
IMPORTANCE_WEIGHT: dict[str, float] = {
    "world_cup": 60.0,
    "continental_final": 50.0,
    "confederations_cup": 45.0,
    "world_cup_qual": 40.0,
    "nations_league": 35.0,
    "continental_qual": 30.0,
    "other_competitive": 30.0,
    "friendly": 20.0,
}

_CONTINENTAL_FINALS = (
    "uefa euro", "copa am", "african cup of nations", "afc asian cup",
    "gold cup", "concacaf championship", "ofc nations cup", "arab cup",
)


def classify_tournament(name: str) -> str:
    """Map a raw martj42 tournament string to an importance category."""
    n = str(name).strip().lower()
    if n == "friendly":
        return "friendly"
    if "confederations cup" in n:
        return "confederations_cup"
    if "world cup" in n:
        return "world_cup_qual" if "qualif" in n else "world_cup"
    if "nations league" in n:
        return "nations_league"
    is_final = any(key in n for key in _CONTINENTAL_FINALS)
    if is_final and "qualif" not in n:
        return "continental_final"
    if "qualif" in n:
        return "continental_qual"
    return "other_competitive"


def build_matches(*, force: bool = False, save: bool = True) -> pd.DataFrame:
    df = fetch_results(force=force).sort_values("date").reset_index(drop=True)

    df["outcome"] = np.where(
        df["home_score"] > df["away_score"], "H",
        np.where(df["home_score"] < df["away_score"], "A", "D"),
    )
    df["result_home"] = np.select(
        [df["outcome"] == "H", df["outcome"] == "D"], [1.0, 0.5], default=0.0
    )
    df["total_goals"] = df["home_score"] + df["away_score"]
    df["importance"] = df["tournament"].map(classify_tournament)
    df["importance_weight"] = df["importance"].map(IMPORTANCE_WEIGHT).astype(float)
    df["is_competitive"] = df["importance"] != "friendly"
    df["year"] = df["date"].dt.year

    if save:
        out = CONFIG.processed / "matches.parquet"
        df.to_parquet(out, index=False)
        print(f"[build_dataset] wrote {out}  ({len(df):,} matches)")
    return df


def build_team_reference(*, force: bool = False, save: bool = True) -> pd.DataFrame:
    """The 48 qualified teams with group, confederation, external Elo and rank."""
    rows = []
    for team in CONFIG.teams:
        rows.append({
            "team": team,
            "group": CONFIG.group_of(team),
            "confederation": CONFIG.confederation_of(team),
            "is_host": CONFIG.is_host(team),
        })
    ref = pd.DataFrame(rows)

    elo = fetch_current_elo(force=force)[["team", "elo_external"]]
    rank = fetch_fifa_rankings(force=force)[["team", "fifa_rank"]]
    ref = ref.merge(elo, on="team", how="left").merge(rank, on="team", how="left")

    # Fill any missing external Elo with the confederation prior (e.g. Curacao).
    pri = CONFIG.confederation_prior
    ref["elo_external"] = ref["elo_external"].fillna(
        ref["confederation"].map(pri)
    )
    if save:
        out = CONFIG.processed / "teams_reference.parquet"
        ref.to_parquet(out, index=False)
        print(f"[build_dataset] wrote {out}  ({len(ref)} teams)")
    return ref


def main(*, force: bool = False) -> None:
    matches = build_matches(force=force)
    ref = build_team_reference(force=force)
    print("\nimportance category counts:")
    print(matches["importance"].value_counts().to_string())
    print(f"\nteams_reference missing external Elo: "
          f"{int(ref['elo_external'].isna().sum())}")
    print(ref.sort_values('elo_external', ascending=False).head(8).to_string(index=False))


if __name__ == "__main__":
    main()
