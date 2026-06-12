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
    }


if __name__ == "__main__":  # smoke test: leading 1-0 at the hour
    out = conditional_outcome(1.5, 1.0, home_score=1, away_score=0, minute=60)
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in out.items()})
