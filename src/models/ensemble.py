"""The single match interface the simulator and app call.

Blends the engines' W/D/L probabilities with configurable weights (default favors
Dixon-Coles + Elo; ML/DL plug in at P4). The scoreline distribution comes from Dixon-Coles
but is rescaled so its win/draw/away marginals match the blended probabilities — that way
the simulator samples realistic scorelines (for goal-difference / goals-for tiebreakers)
while honoring the ensemble's outcome probabilities.

    mp = MatchPredictor.load_default()
    out = mp.predict("Spain", "Brazil", neutral=True)
    out["p_home"], out["p_draw"], out["p_away"], out["scoreline"]  # 2D np.ndarray
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from src.config import CONFIG
from src.models.dixon_coles import DixonColes
from src.models.elo import EloModel


class _Engine(Protocol):
    def predict_proba(self, home: str, away: str, neutral: bool = False) -> tuple[float, float, float]:
        ...


@dataclass
class MatchPredictor:
    elo: EloModel
    dc: DixonColes
    ml: Any | None = None          # set in P4 (HistGradientBoosting wrapper)
    dl: Any | None = None          # set in P4 (keras wrapper)
    weights: dict[str, float] | None = None

    def __post_init__(self) -> None:
        self.weights = dict(self.weights or CONFIG.settings["ensemble"]["weights"])

    # ---- helpers ------------------------------------------------------------
    def _active(self) -> dict[str, _Engine]:
        engines: dict[str, _Engine] = {"dixon_coles": self.dc, "elo": self.elo}
        if self.ml is not None:
            engines["ml"] = self.ml
        if self.dl is not None:
            engines["dl"] = self.dl
        return engines

    def _blended_wdl(self, home: str, away: str, neutral: bool) -> np.ndarray:
        engines = self._active()
        w = np.array([self.weights.get(k, 0.0) for k in engines])
        if w.sum() <= 0:
            w = np.ones(len(engines))
        w = w / w.sum()
        probs = np.array([list(e.predict_proba(home, away, neutral)) for e in engines.values()])
        blended = (w[:, None] * probs).sum(axis=0)
        return blended / blended.sum()

    @staticmethod
    def _rescale_scoreline(m: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Scale the home/draw/away regions of scoreline matrix m to target marginals."""
        ph = np.tril(m, -1).sum()
        pdr = np.trace(m)
        pa = np.triu(m, 1).sum()
        scale = np.ones_like(m)
        n = m.shape[0]
        tri_l = np.tril(np.ones((n, n), bool), -1)
        tri_u = np.triu(np.ones((n, n), bool), 1)
        diag = np.eye(n, dtype=bool)
        if ph > 0:
            scale[tri_l] = target[0] / ph
        if pdr > 0:
            scale[diag] = target[1] / pdr
        if pa > 0:
            scale[tri_u] = target[2] / pa
        out = m * scale
        return out / out.sum()

    # ---- main API -----------------------------------------------------------
    def predict(self, home: str, away: str, neutral: bool = False) -> dict[str, Any]:
        target = self._blended_wdl(home, away, neutral)
        scoreline = self._rescale_scoreline(self.dc.scoreline_matrix(home, away, neutral), target)
        lam, mu = self.dc.expected_goals(home, away, neutral)
        return {
            "home": CONFIG.normalize(home), "away": CONFIG.normalize(away), "neutral": neutral,
            "p_home": float(target[0]), "p_draw": float(target[1]), "p_away": float(target[2]),
            "lambda_home": lam, "lambda_away": mu, "scoreline": scoreline,
        }

    def predict_proba(self, home: str, away: str, neutral: bool = False) -> tuple[float, float, float]:
        t = self._blended_wdl(home, away, neutral)
        return float(t[0]), float(t[1]), float(t[2])

    # ---- construction -------------------------------------------------------
    @classmethod
    def load_default(cls, *, ml: Any | None = None, dl: Any | None = None) -> "MatchPredictor":
        return cls(elo=EloModel.load(), dc=DixonColes.load(), ml=ml, dl=dl)


if __name__ == "__main__":
    mp = MatchPredictor.load_default()
    print("active engines:", list(mp._active()), "weights:", mp.weights)
    for h, a, neu in [("Spain", "Brazil", True), ("England", "Panama", True),
                      ("United States", "Turkey", False)]:
        o = mp.predict(h, a, neu)
        sc = o["scoreline"]
        # most likely exact score
        ij = np.unravel_index(np.argmax(sc), sc.shape)
        print(f"{h} vs {a}: H={o['p_home']:.2f} D={o['p_draw']:.2f} A={o['p_away']:.2f} "
              f"| xg {o['lambda_home']:.2f}-{o['lambda_away']:.2f} | ML score {ij[0]}-{ij[1]}")
