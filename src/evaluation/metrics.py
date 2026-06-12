"""Probabilistic scoring metrics for 3-outcome (H/D/A) match forecasts.

Ranked Probability Score (RPS) is the standard for ordered soccer outcomes; lower is
better. Also Brier (multiclass) and log-loss. All take probabilities ordered
[P(home), P(draw), P(away)] and the realized outcome in {"H","D","A"}.
"""

from __future__ import annotations

import numpy as np

_ORDER = {"H": 0, "D": 1, "A": 2}


def _to_arrays(probs, outcomes):
    p = np.asarray(probs, dtype=float)
    y = np.array([_ORDER[o] for o in outcomes])
    return p, y


def ranked_probability_score(probs, outcomes) -> float:
    """Mean RPS over matches. probs: (n,3) ordered [H,D,A]; outcomes: list of 'H'/'D'/'A'."""
    p, y = _to_arrays(probs, outcomes)
    onehot = np.eye(3)[y]
    cum_p = np.cumsum(p, axis=1)
    cum_y = np.cumsum(onehot, axis=1)
    # RPS = sum over categories of (cumP - cumY)^2 / (categories - 1)
    return float(np.mean(np.sum((cum_p - cum_y) ** 2, axis=1) / 2.0))


def brier_score(probs, outcomes) -> float:
    p, y = _to_arrays(probs, outcomes)
    onehot = np.eye(3)[y]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def log_loss(probs, outcomes, eps: float = 1e-12) -> float:
    p, y = _to_arrays(probs, outcomes)
    pick = np.clip(p[np.arange(len(y)), y], eps, 1.0)
    return float(-np.mean(np.log(pick)))


def summary(probs, outcomes) -> dict[str, float]:
    return {
        "n": len(outcomes),
        "rps": ranked_probability_score(probs, outcomes),
        "brier": brier_score(probs, outcomes),
        "log_loss": log_loss(probs, outcomes),
        "accuracy": float(np.mean(
            np.argmax(np.asarray(probs), axis=1) == np.array([_ORDER[o] for o in outcomes])
        )),
    }
