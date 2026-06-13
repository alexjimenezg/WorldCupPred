"""ESPN public scoreboard — keyless live scores, match minute, and in-game stats.

The undocumented-but-stable endpoint behind espn.com's World Cup scoreboard. No key,
no rate-limit drama at our polling cadence (the app refreshes about once a minute).
Three uses:
  - the app's Live tab: in-play score + minute + stats, joined to the model's
    conditional win probabilities (src.models.inplay);
  - played / upcoming match boards;
  - a sync source: finished matches land in the ResultsStore minutes after full time
    (martj42 lags by hours-to-days, football-data.org needs a key).
"""

from __future__ import annotations

import re

import pandas as pd
import requests

from src.config import CONFIG

_URL = CONFIG.settings["data_sources"]["espn_scoreboard"]

# in-game stats worth showing, in display order (ESPN name -> label)
STAT_LABELS = [
    ("possessionPct", "Possession %"),
    ("totalShots", "Shots"),
    ("shotsOnTarget", "On target"),
    ("wonCorners", "Corners"),
    ("foulsCommitted", "Fouls"),
    ("saves", "Saves"),
    ("yellowCards", "Yellow cards"),
]

_CLOCK_RE = re.compile(r"(\d+)")


def _tournament_range() -> str:
    d = CONFIG.groups_raw["dates"]
    a = str(d["opening_match"]).replace("-", "")
    b = str(d["final"]).replace("-", "")
    return f"{a}-{b}"


def _minute(display_clock: str, state: str) -> int:
    """'67'' -> 67, '90'+8'' -> 90; pre-match -> 0, finished -> 90."""
    if state == "post":
        return 90
    m = _CLOCK_RE.match(str(display_clock) or "")
    return min(int(m.group(1)), 90) if m else 0


def _parse_odds(comp: dict) -> dict | None:
    """Bookmaker 3-way moneyline that rides along in the scoreboard (DraftKings)."""
    from src.betting import american_to_decimal
    arr = comp.get("odds") or []
    o = next((x for x in arr if isinstance(x, dict)), None)
    if o is None:
        return None

    def side(name: str) -> float | None:
        node = (o.get("moneyline") or {}).get(name) or {}
        for k in ("close", "open"):
            d = american_to_decimal((node.get(k) or {}).get("odds"))
            if d:
                return d
        return None

    dh, da = side("home"), side("away")
    dd = american_to_decimal((o.get("drawOdds") or {}).get("moneyLine"))
    if not (dh and dd and da):
        return None
    return {"provider": (o.get("provider") or {}).get("name", "book"),
            "dec_home": dh, "dec_draw": dd, "dec_away": da,
            "over_under": o.get("overUnder")}


def fetch_scoreboard(dates: str | None = None, *, timeout: int = 20) -> list[dict]:
    """All WC matches in the date range (default: whole tournament), parsed flat.

    Each item: kickoff (pd.Timestamp, UTC), state ('pre'|'in'|'post'), detail,
    minute, home/away (canonical names), home_score/away_score,
    stats {label: (home_value, away_value)}.
    """
    r = requests.get(_URL, params={"dates": dates or _tournament_range(),
                                   "limit": 200}, timeout=timeout)
    r.raise_for_status()
    out: list[dict] = []
    for ev in r.json().get("events", []):
        comp = ev.get("competitions", [{}])[0]
        status = ev.get("status", {})
        state = status.get("type", {}).get("state", "pre")
        sides: dict[str, dict] = {}
        for c in comp.get("competitors", []):
            name = CONFIG.normalize(c.get("team", {}).get("displayName", ""))
            sides[c.get("homeAway", "home")] = {
                "team": name,
                "id": str(c.get("team", {}).get("id", "")),
                "score": int(c.get("score") or 0),
                "stats": {s.get("name"): s.get("displayValue")
                          for s in c.get("statistics", [])},
            }
        if "home" not in sides or "away" not in sides:
            continue
        h, a = sides["home"], sides["away"]
        stats = {}
        for key, label in STAT_LABELS:
            if key in h["stats"] or key in a["stats"]:
                stats[label] = (h["stats"].get(key, "0"), a["stats"].get(key, "0"))
        out.append({
            "kickoff": pd.Timestamp(ev.get("date")),
            "state": state,
            "detail": status.get("type", {}).get("detail", ""),
            "minute": _minute(status.get("displayClock", ""), state),
            "home": h["team"], "away": a["team"],
            "home_score": h["score"], "away_score": a["score"],
            "stats": stats,
            "odds": _parse_odds(comp),
            "events": _parse_events(comp, h["id"], a["id"]),
        })
    out.sort(key=lambda m: m["kickoff"])
    return out


def _parse_events(comp: dict, home_id: str, away_id: str) -> list[dict]:
    """Goals and cards with minute, side and player, sorted by minute."""
    out = []
    for d in comp.get("details", []) or []:
        text = (d.get("type") or {}).get("text", "")
        m = _CLOCK_RE.match(str((d.get("clock") or {}).get("displayValue", "")))
        minute = min(int(m.group(1)), 90) if m else 0
        tid = str((d.get("team") or {}).get("id", ""))
        side = "home" if tid == home_id else "away" if tid == away_id else None
        if side is None:
            continue
        if d.get("scoringPlay"):
            kind = "goal"
        elif d.get("redCard") or "Red" in text:
            kind = "red"
        elif d.get("yellowCard") or "Yellow" in text:
            kind = "yellow"
        else:
            continue
        players = d.get("athletesInvolved") or []
        out.append({
            "minute": minute, "type": kind, "side": side,
            "player": players[0].get("shortName", "") if players else "",
            "own_goal": bool(d.get("ownGoal")),
            "penalty": bool(d.get("penaltyKick")),
        })
    return sorted(out, key=lambda e: e["minute"])


def import_finished_to_store(store=None) -> int:
    """Upsert finished WC matches into the results store. Returns rows applied."""
    from src.data.auto_results import infer_stage
    from src.results_store import ResultsStore
    store = store or ResultsStore()
    count = 0
    for m in fetch_scoreboard():
        if m["state"] != "post":
            continue
        home, away = m["home"], m["away"]
        if home not in CONFIG.teams or away not in CONFIG.teams:
            continue
        stage = infer_stage(m["kickoff"], home, away)
        if stage is None:
            continue
        try:
            store.add(home, away, m["home_score"], m["away_score"], stage=stage,
                      on=str(m["kickoff"].date()))
            count += 1
        except ValueError:
            continue
    return count


if __name__ == "__main__":
    board = fetch_scoreboard()
    print(f"{len(board)} matches on the board")
    for m in board:
        if m["state"] != "pre":
            print(f"  [{m['state']:4}] {m['detail']:>14}  "
                  f"{m['home']} {m['home_score']}-{m['away_score']} {m['away']}  "
                  f"stats={list(m['stats'])[:3]}")
