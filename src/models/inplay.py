"""In-play conditional probabilities: P(final outcome | score at minute m).

Joins the live feed to the prediction model. The ensemble's pre-match expected goals
(lambda_home / lambda_away, per 90') are scaled by the fraction of the match left;
remaining goals for each side are Poisson, so the final score is the current score
plus an independent Poisson grid. That gives W/D/L conditional on the live state and
a projected final-score distribution — the live analogue of the pre-match scoreline
matrix in src.models.dixon_coles.

Deliberately simple (no red-card / momentum adjustments, independence instead of the
DC low-score correction — the rho term matters pre-match, far less mid-game).

    from src.models.inplay import conditional_outcome
    out = conditional_outcome(1.6, 0.9, home_score=1, away_score=0, minute=63)
    out["p_home"], out["p_draw"], out["p_away"], out["top_score"]
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

MAX_GOALS = 10  # cap on *additional* goals per side from now to full time


def conditional_outcome(lambda_home: float, lambda_away: float, *,
                        home_score: int, away_score: int, minute: int,
                        max_goals: int = MAX_GOALS) -> dict:
    """Final-result distribution given the current score at `minute` (0..90)."""
    frac = float(np.clip((90 - minute) / 90, 0.0, 1.0))
    k = np.arange(max_goals + 1)
    ph = poisson.pmf(k, lambda_home * frac)
    pa = poisson.pmf(k, lambda_away * frac)
    add = np.outer(ph, pa)
    add /= add.sum()  # renormalize the truncated grid

    diff = home_score - away_score + (k[:, None] - k[None, :])
    p_home = float(add[diff > 0].sum())
    p_draw = float(add[diff == 0].sum())
    p_away = float(add[diff < 0].sum())

    i, j = np.unravel_index(int(np.argmax(add)), add.shape)
    return {
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "top_score": (home_score + int(i), away_score + int(j)),
        "p_top_score": float(add[i, j]),
        "exp_home": home_score + lambda_home * frac,
        "exp_away": away_score + lambda_away * frac,
        "minutes_left": max(0, 90 - minute),
        "matrix": add,  # P(additional goals) — final score = current + index
    }


def total_goals_dist(out: dict, home_score: int, away_score: int,
                     max_total: int = 8) -> "pd.Series":
    """P(total final goals = k) from a conditional_outcome result."""
    import pandas as pd
    add = out["matrix"]
    base = home_score + away_score
    probs: dict[int, float] = {}
    for i in range(add.shape[0]):
        for j in range(add.shape[1]):
            k = base + i + j
            probs[min(k, max_total)] = probs.get(min(k, max_total), 0.0) + float(add[i, j])
    idx = list(range(0, max_total + 1))
    return pd.Series([probs.get(k, 0.0) for k in idx], index=idx)


def win_prob_timeline(lambda_home: float, lambda_away: float, events: list[dict], *,
                      upto: int, final_home: int, final_away: int) -> "pd.DataFrame | None":
    """Reconstruct W/D/L probability minute-by-minute from the goal events.

    Validates that the reconstructed score at `upto` matches the actual one
    (own-goal side conventions vary); returns None when it cannot be trusted.
    """
    import pandas as pd
    goals = [e for e in events if e["type"] == "goal"]

    def score_at(minute: int, flip_og: bool) -> tuple[int, int]:
        h = a = 0
        for g in goals:
            if g["minute"] > minute:
                continue
            side = g["side"]
            if flip_og and g.get("own_goal"):
                side = "away" if side == "home" else "home"
            h, a = (h + 1, a) if side == "home" else (h, a + 1)
        return h, a

    flip = False
    if score_at(upto, False) != (final_home, final_away):
        if score_at(upto, True) == (final_home, final_away):
            flip = True
        else:
            return None  # events don't reconcile with the score

    rows = []
    for m in range(0, upto + 1):
        h, a = score_at(m, flip)
        c = conditional_outcome(lambda_home, lambda_away,
                                home_score=h, away_score=a, minute=m)
        rows.append((m, c["p_home"], c["p_draw"], c["p_away"]))
    return pd.DataFrame(rows, columns=["minute", "home", "draw", "away"])


if __name__ == "__main__":  # smoke test: leading 1-0 at the hour
    out = conditional_outcome(1.5, 1.0, home_score=1, away_score=0, minute=60)
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in out.items()})
