"""Persistent store of played FIFA World Cup 2026 results.

Backs the live loop: every result entered in the app or CLI lands here (JSON), and is
exposed two ways:
  - `to_fixed()`   -> the simulator's `fixed` dict so simulate-from-now never re-randomizes
                      a match that already happened;
  - `to_match_rows()` -> rows shaped like matches.parquet so Elo/Dixon-Coles/ML can retrain
                      on the new results.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from src.config import CONFIG

KNOCKOUT_STAGES = {"R32", "R16", "QF", "SF", "final"}
_WEIGHT_WC = 60.0  # World Cup Elo importance weight


@dataclass
class Result:
    home: str
    away: str
    home_score: int
    away_score: int
    stage: str = "group"          # "group" or a knockout round
    date: str = ""


class ResultsStore:
    def __init__(self, path: Path | None = None):
        self.path = path or (CONFIG.processed / "wc2026_results.json")
        self.results: list[Result] = []
        self.load()

    # ---- persistence --------------------------------------------------------
    def load(self) -> "ResultsStore":
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.results = [Result(**r) for r in raw]
        return self

    def save(self) -> None:
        self.path.write_text(json.dumps([asdict(r) for r in self.results], indent=2),
                             encoding="utf-8")

    # ---- mutation -----------------------------------------------------------
    def add(self, home: str, away: str, home_score: int, away_score: int,
            stage: str = "group", on: str | None = None) -> Result:
        home, away = CONFIG.normalize(home), CONFIG.normalize(away)
        if home not in CONFIG.teams or away not in CONFIG.teams:
            raise ValueError(f"both teams must be in the 2026 field: {home}, {away}")
        if stage == "group" and CONFIG.group_of(home) != CONFIG.group_of(away):
            raise ValueError(f"{home} and {away} are not in the same group")
        r = Result(home, away, int(home_score), int(away_score), stage,
                   on or date.today().isoformat())
        # replace an existing entry for the same matchup+stage
        self.results = [x for x in self.results
                        if not (x.stage == stage and {x.home, x.away} == {home, away})]
        self.results.append(r)
        self.save()
        return r

    def undo_last(self) -> Result | None:
        if not self.results:
            return None
        r = self.results.pop()
        self.save()
        return r

    def clear(self) -> None:
        self.results = []
        self.save()

    # ---- views --------------------------------------------------------------
    def to_fixed(self) -> dict:
        groups: dict[tuple[str, str], tuple[int, int]] = {}
        knockouts: dict[tuple[str, str], tuple[int, int]] = {}
        for r in self.results:
            target = knockouts if r.stage in KNOCKOUT_STAGES else groups
            target[(r.home, r.away)] = (r.home_score, r.away_score)
        return {"groups": groups, "knockouts": knockouts}

    def to_match_rows(self) -> pd.DataFrame:
        """Rows shaped like matches.parquet, for retraining."""
        rows = []
        for r in self.results:
            outcome = ("H" if r.home_score > r.away_score
                       else "A" if r.home_score < r.away_score else "D")
            rows.append({
                "date": pd.Timestamp(r.date or "2026-06-15"),
                "home_team": r.home, "away_team": r.away,
                "home_score": r.home_score, "away_score": r.away_score,
                "tournament": "FIFA World Cup",
                "city": "", "country": "United States",
                "neutral": not CONFIG.is_host(r.home),
                "outcome": outcome,
                "result_home": 1.0 if outcome == "H" else 0.5 if outcome == "D" else 0.0,
                "total_goals": r.home_score + r.away_score,
                "importance": "world_cup", "importance_weight": _WEIGHT_WC,
                "is_competitive": True, "year": 2026,
            })
        return pd.DataFrame(rows)

    def __len__(self) -> int:
        return len(self.results)

    def summary(self) -> str:
        if not self.results:
            return "no results recorded yet"
        return "\n".join(
            f"  [{r.stage:5}] {r.home} {r.home_score}-{r.away_score} {r.away}"
            for r in self.results)


if __name__ == "__main__":
    s = ResultsStore()
    print(f"{len(s)} results in {s.path}")
    print(s.summary())
