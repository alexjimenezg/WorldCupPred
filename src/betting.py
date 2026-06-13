"""Betting analytics: price any market off the ensemble and find value vs the book.

Everything is derived from two model objects:
  - the ensemble's scoreline matrix for a fixture (src.models.ensemble.MatchPredictor)
    -> 1X2, double chance, over/under, both-teams-to-score, exact score;
  - the Monte-Carlo title-odds table -> outright markets (champion, reach final, ...).

Value definitions (decimal odds d, model probability p):
  implied   de-vigged bookmaker probability (proportional method)
  edge      p - implied                       (how much we disagree with the market)
  EV        p * d - 1                         (expected profit per unit staked)
  kelly     (d*p - 1) / (d - 1), floored at 0 (full Kelly; stake a fraction of it)

This is research tooling, not financial advice; the model can be confidently wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# (market code, human label) — fixture markets priced off the scoreline matrix
FIXTURE_MARKETS = [
    ("1", "Home win (90')"), ("X", "Draw (90')"), ("2", "Away win (90')"),
    ("1X", "Home or draw"), ("12", "Home or away"), ("X2", "Draw or away"),
    ("O", "Over total goals"), ("U", "Under total goals"),
    ("BTTS-Y", "Both teams score"), ("BTTS-N", "Not both score"),
    ("CS", "Exact score"),
]


def american_to_decimal(ml) -> float | None:
    """'+600' / -175 -> decimal odds. None when unparseable."""
    try:
        v = float(str(ml).replace("+", ""))
    except (TypeError, ValueError):
        return None
    if v == 0:
        return None
    return 1 + (v / 100 if v > 0 else 100 / abs(v))


def devig(dec_home: float, dec_draw: float, dec_away: float) -> np.ndarray:
    """Proportional de-vig of a 3-way market -> implied probabilities (sum 1)."""
    inv = np.array([1 / dec_home, 1 / dec_draw, 1 / dec_away])
    return inv / inv.sum()


def kelly_fraction(p: float, d: float) -> float:
    """Full-Kelly stake fraction; 0 when there is no edge or odds are broken."""
    if d <= 1:
        return 0.0
    return max(0.0, (d * p - 1) / (d - 1))


def market_prob(out: dict, market: str, *, line: float = 2.5,
                score: tuple[int, int] | None = None) -> float:
    """P(market) from a predictor output dict (needs scoreline + p_home/draw/away)."""
    m = out["scoreline"]
    i = np.arange(m.shape[0])
    big_i, big_j = np.meshgrid(i, np.arange(m.shape[1]), indexing="ij")
    if market == "1":
        return float(out["p_home"])
    if market == "X":
        return float(out["p_draw"])
    if market == "2":
        return float(out["p_away"])
    if market == "1X":
        return float(out["p_home"] + out["p_draw"])
    if market == "12":
        return float(out["p_home"] + out["p_away"])
    if market == "X2":
        return float(out["p_draw"] + out["p_away"])
    if market == "O":
        return float(m[(big_i + big_j) > line].sum())
    if market == "U":
        return float(m[(big_i + big_j) < line].sum())
    if market == "BTTS-Y":
        return float(m[(big_i > 0) & (big_j > 0)].sum())
    if market == "BTTS-N":
        return float(1 - m[(big_i > 0) & (big_j > 0)].sum())
    if market == "CS":
        h, a = score or (0, 0)
        if h < m.shape[0] and a < m.shape[1]:
            return float(m[h, a])
        return 0.0
    raise ValueError(f"unknown market: {market}")


def value_board(predictor, board: list[dict]) -> pd.DataFrame:
    """One row per (upcoming match with bookmaker odds) x (H/D/A outcome)."""
    from src.config import CONFIG
    rows = []
    for m in board:
        if m["state"] != "pre" or not m.get("odds"):
            continue
        home, away = m["home"], m["away"]
        if home not in CONFIG.teams or away not in CONFIG.teams:
            continue
        o = m["odds"]
        out = predictor.predict(home, away, not CONFIG.is_host(home))
        implied = devig(o["dec_home"], o["dec_draw"], o["dec_away"])
        legs = (("1", home, out["p_home"], o["dec_home"], implied[0]),
                ("X", "Draw", out["p_draw"], o["dec_draw"], implied[1]),
                ("2", away, out["p_away"], o["dec_away"], implied[2]))
        for code, label, p, d, imp in legs:
            rows.append({
                "kickoff": m["kickoff"], "match": f"{home} vs {away}",
                "pick": label, "code": code, "odds": round(d, 2),
                "implied": imp, "model": p, "edge": p - imp,
                "ev": p * d - 1, "kelly": kelly_fraction(p, d),
                "book": o.get("provider", "book"),
            })
    df = pd.DataFrame(rows)
    return df.sort_values("ev", ascending=False).reset_index(drop=True) if len(df) else df


if __name__ == "__main__":  # smoke test on a synthetic 60/25/15 market priced fair
    d = [1 / .55, 1 / .27, 1 / .18]  # book holds ~0% for simplicity
    print("devig:", devig(*d).round(3))
    print("kelly p=.30 d=4.0:", round(kelly_fraction(.30, 4.0), 4))
