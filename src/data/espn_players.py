"""Per-player tournament stats from ESPN match summaries (keyless).

ESPN's match `summary` endpoint exposes a `rosters` block: every player, their
position, and a vector of stats (goals, assists, saves, shots, cards, ...). We pull
that for each played match and aggregate into tournament leaderboards, a transparent
performance rating, and a Best XI / dream team (overall or per round).

Finished matches never change, so their rows are cached to disk (one JSON per event
id) and never re-fetched; only live/new matches hit the network.

Honest scope: roster stats cover scoring, goalkeeping, shooting and discipline for
*every* player. Passing accuracy and defensive interventions live only in the
per-match `leaders` block (top 1-3 per game) — handled separately as "match
standouts", not a full leaderboard. There is no official player rating in the free
feed; `rating` below is our own documented formula.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import requests

from src.config import CONFIG

_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/summary"

# roster stat names we keep (ESPN name -> our column)
_STAT_KEYS = {
    "appearances": "apps", "subIns": "sub_ins", "totalGoals": "goals",
    "goalAssists": "assists", "totalShots": "shots", "shotsOnTarget": "sot",
    "saves": "saves", "goalsConceded": "conceded", "shotsFaced": "shots_faced",
    "foulsCommitted": "fouls", "foulsSuffered": "fouled", "offsides": "offsides",
    "ownGoals": "own_goals", "yellowCards": "yellow", "redCards": "red",
}


def _role(position_name: str) -> str:
    p = (position_name or "").lower()
    if "goalkeeper" in p:
        return "GK"
    if "defender" in p or "back" in p:
        return "DEF"
    if "midfielder" in p:
        return "MID"
    if "forward" in p or "striker" in p:
        return "FWD"
    return ""  # substitutes / unknown — counted in leaderboards, not the XI


def _cache_path(event_id: str):
    d = CONFIG.interim / "player_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{event_id}.json"


def fetch_match_players(event_id: str, *, finished: bool, timeout: int = 20
                        ) -> list[dict]:
    """Per-player rows for one match. Finished matches are cached to disk forever."""
    cache = _cache_path(event_id)
    if finished and cache.exists():
        rows = json.loads(cache.read_text(encoding="utf-8"))
        if rows and "jersey" in rows[0]:  # schema guard: refetch older caches
            return rows

    r = requests.get(_SUMMARY, params={"event": event_id}, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    rows: list[dict] = []
    for block in data.get("rosters", []) or []:
        team = CONFIG.normalize(block.get("team", {}).get("displayName", ""))
        for p in block.get("roster", []) or []:
            ath = p.get("athlete", {})
            pos = p.get("position", {}).get("displayName", "")
            stats = {s.get("name"): s.get("value", 0) for s in p.get("stats", [])}
            jerseys = ath.get("jerseyImages") or []
            jersey = next((j.get("href") for j in jerseys
                           if "dark" in (j.get("rel") or [])),
                          jerseys[0].get("href") if jerseys else "")
            row = {"event_id": event_id, "team": team,
                   "player": ath.get("displayName", ""),
                   "player_id": str(ath.get("id", "")),
                   "jersey": jersey,
                   "position": pos, "role": _role(pos),
                   "starter": bool(p.get("starter"))}
            for espn_name, col in _STAT_KEYS.items():
                row[col] = float(stats.get(espn_name, 0) or 0)
            rows.append(row)
    if finished and rows:
        cache.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def build_player_frame(board: list[dict]) -> pd.DataFrame:
    """Concatenate per-player rows for every played match on the board, tagged by
    the round it belongs to (group / R32 / ...)."""
    from src.data.auto_results import infer_stage
    frames = []
    for m in board:
        if m["state"] != "post" or not m.get("id"):
            continue
        try:
            rows = fetch_match_players(m["id"], finished=True)
        except Exception:
            continue
        if not rows:
            continue
        df = pd.DataFrame(rows)
        stage = infer_stage(m["kickoff"], m["home"], m["away"]) or "group"
        df["stage"] = stage
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
_SUM_COLS = ["apps", "goals", "assists", "shots", "sot", "saves", "conceded",
             "shots_faced", "fouls", "fouled", "offsides", "own_goals",
             "yellow", "red"]


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """One row per player: summed stats + clean sheets + role + performance rating."""
    if df.empty:
        return df
    # clean sheet = a (goalkeeper) appearance with zero conceded
    df = df.copy()
    if "jersey" not in df.columns:  # tolerate rows from an older cache schema
        df["jersey"] = ""
    df["clean_sheet"] = ((df["apps"] > 0) & (df["conceded"] == 0)
                         & (df["role"] == "GK")).astype(int)
    grp = df.groupby(["player", "player_id", "team"], as_index=False)
    agg = grp[_SUM_COLS + ["clean_sheet"]].sum()

    # stable role = most frequent non-empty role across the player's matches
    role = (df[df["role"] != ""].groupby("player")["role"]
            .agg(lambda s: s.mode().iat[0] if len(s.mode()) else "")
            .rename("role"))
    agg = agg.merge(role, on="player", how="left").fillna({"role": ""})

    # most recent jersey image per player (kit + number; ESPN has no headshots)
    jersey = (df[df["jersey"] != ""].groupby("player")["jersey"].last()
              .rename("jersey"))
    agg = agg.merge(jersey, on="player", how="left").fillna({"jersey": ""})

    agg["conv"] = np.where(agg["shots"] > 0, agg["goals"] / agg["shots"], 0.0)
    agg["save_pct"] = np.where(agg["shots_faced"] > 0,
                               agg["saves"] / agg["shots_faced"], 0.0)
    agg["rating"] = rating(agg)
    return agg


def rating(a: pd.DataFrame) -> pd.Series:
    """Transparent composite performance score (ours, not an official metric).

    Rewards goals/assists/shots on target, plus saves & clean sheets for keepers;
    penalizes cards and own goals. Tuned only for plausibility, not calibrated.
    """
    return (a["goals"] * 4 + a["assists"] * 3 + a["sot"] * 0.5
            + a["saves"] * 0.5 + a["clean_sheet"] * 2 + a["fouled"] * 0.05
            - a["yellow"] * 1 - a["red"] * 3 - a["own_goals"] * 4
            - a["fouls"] * 0.1)


_FORMATION = [("GK", 1), ("DEF", 4), ("MID", 3), ("FWD", 3)]  # 4-3-3


def best_xi(agg: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pick a 4-3-3 dream team by rating within each role (starting positions)."""
    xi = {}
    for role, n in _FORMATION:
        pool = agg[agg["role"] == role].sort_values("rating", ascending=False)
        xi[role] = pool.head(n)
    return xi


if __name__ == "__main__":  # smoke test against the live board
    from src.data.espn_live import fetch_scoreboard
    board = fetch_scoreboard()
    df = build_player_frame(board)
    print(f"player-match rows: {len(df)} from "
          f"{df['event_id'].nunique() if len(df) else 0} matches")
    if len(df):
        agg = aggregate(df)
        top = agg.sort_values("goals", ascending=False).head(5)
        print(top[["player", "team", "goals", "assists", "rating"]].to_string(index=False))
