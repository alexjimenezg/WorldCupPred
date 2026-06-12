"""International Elo: our authoritative team-strength rating, computed from match results.

World Football Elo conventions:
  expected_home = 1 / (1 + 10**(-(R_home + HFA - R_away)/400))
  R' = R + K * G * (score - expected),  K = tournament importance weight,
  G = goal-difference multiplier (1, 1.5, then (11+|d|)/8 for blowouts), HFA=0 if neutral.

On top of the ratings we fit a 3-parameter ordered-logit so Elo alone yields calibrated
W/D/L probabilities (Elo gives an expected score, not a draw probability). The class
supports incremental `update()` for the live tournament loop, and `save()`/`load()`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from src.config import CONFIG

_ELO = CONFIG.settings["elo"]
_START = float(_ELO["start_rating"])
_HFA = float(_ELO["home_advantage"])
_USE_GD = bool(_ELO.get("goal_difference_multiplier", True))


def goal_diff_multiplier(diff: int) -> float:
    d = abs(int(diff))
    if not _USE_GD or d <= 1:
        return 1.0
    if d == 2:
        return 1.5
    return (11.0 + d) / 8.0


@dataclass
class EloModel:
    ratings: dict[str, float] = field(default_factory=dict)
    # ordered-logit params for P(W/D/L) from adjusted Elo diff: theta0<theta1, beta
    _theta0: float = -0.5
    _theta1: float = 0.5
    _beta: float = 0.003
    fitted: bool = False

    # ---- rating access ------------------------------------------------------
    def rating(self, team: str) -> float:
        return self.ratings.get(CONFIG.normalize(team), _START)

    def adj_diff(self, home: str, away: str, neutral: bool = False) -> float:
        hfa = 0.0 if neutral else _HFA
        return self.rating(home) + hfa - self.rating(away)

    def expected_score(self, home: str, away: str, neutral: bool = False) -> float:
        return 1.0 / (1.0 + 10.0 ** (-self.adj_diff(home, away, neutral) / 400.0))

    # ---- single-match update (also used by the live loop) -------------------
    def update(self, home: str, away: str, hs: int, as_: int,
               weight: float, neutral: bool = False) -> None:
        home, away = CONFIG.normalize(home), CONFIG.normalize(away)
        exp_h = self.expected_score(home, away, neutral)
        score_h = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        g = goal_diff_multiplier(hs - as_)
        delta = weight * g * (score_h - exp_h)
        self.ratings[home] = self.rating(home) + delta
        self.ratings[away] = self.rating(away) - delta

    # ---- full-history pass --------------------------------------------------
    def fit(self, matches: pd.DataFrame, *, calibrate_since: int = 2002) -> "EloModel":
        """Process all matches in date order; record pre-match Elo; fit W/D/L calibration."""
        matches = matches.sort_values("date").reset_index(drop=True)
        home_pre = np.empty(len(matches))
        away_pre = np.empty(len(matches))
        h = matches["home_team"].to_numpy()
        a = matches["away_team"].to_numpy()
        hs = matches["home_score"].to_numpy()
        as_ = matches["away_score"].to_numpy()
        wt = matches["importance_weight"].to_numpy()
        neu = matches["neutral"].to_numpy()

        for i in range(len(matches)):
            home_pre[i] = self.rating(h[i])
            away_pre[i] = self.rating(a[i])
            self.update(h[i], a[i], hs[i], as_[i], wt[i], bool(neu[i]))

        matches = matches.copy()
        matches["home_elo_pre"] = home_pre
        matches["away_elo_pre"] = away_pre
        matches["elo_diff_pre"] = (
            home_pre - away_pre + np.where(neu, 0.0, _HFA)
        )
        self._fit_calibration(matches[matches["year"] >= calibrate_since])
        self.fitted = True
        self._history = matches  # retained for feature building / inspection
        return self

    @property
    def history(self) -> pd.DataFrame:
        return getattr(self, "_history", pd.DataFrame())

    # ---- ordered-logit W/D/L calibration ------------------------------------
    def _fit_calibration(self, df: pd.DataFrame) -> None:
        d = df["elo_diff_pre"].to_numpy()
        y = np.select([df["outcome"] == "A", df["outcome"] == "D"], [0, 1], default=2)

        def nll(params: np.ndarray) -> float:
            t0, gap, beta = params
            t1 = t0 + np.exp(gap)  # enforce t1 > t0
            c0 = expit(t0 - beta * d)            # P(y<=0)=P(away)
            c1 = expit(t1 - beta * d)            # P(y<=1)=P(away)+P(draw)
            p_away = np.clip(c0, 1e-9, 1)
            p_draw = np.clip(c1 - c0, 1e-9, 1)
            p_home = np.clip(1 - c1, 1e-9, 1)
            ll = (np.log(p_away) * (y == 0)
                  + np.log(p_draw) * (y == 1)
                  + np.log(p_home) * (y == 2))
            return -ll.sum()

        res = minimize(nll, x0=[-0.4, np.log(0.8), 0.003], method="Nelder-Mead",
                       options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6})
        t0, gap, beta = res.x
        self._theta0, self._theta1, self._beta = float(t0), float(t0 + np.exp(gap)), float(beta)

    def predict_proba(self, home: str, away: str, neutral: bool = False) -> tuple[float, float, float]:
        """Return (p_home, p_draw, p_away) from the fitted ordered logit."""
        d = self.adj_diff(home, away, neutral)
        c0 = expit(self._theta0 - self._beta * d)
        c1 = expit(self._theta1 - self._beta * d)
        p_away = c0
        p_draw = c1 - c0
        p_home = 1 - c1
        return float(p_home), float(p_draw), float(p_away)

    # ---- persistence --------------------------------------------------------
    def save(self, path: Path | None = None) -> Path:
        path = path or (CONFIG.models_dir / "elo_state.json")
        path.write_text(json.dumps({
            "ratings": self.ratings,
            "theta0": self._theta0, "theta1": self._theta1, "beta": self._beta,
            "fitted": self.fitted,
        }), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "EloModel":
        path = path or (CONFIG.models_dir / "elo_state.json")
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(ratings=d["ratings"], _theta0=d["theta0"], _theta1=d["theta1"],
                   _beta=d["beta"], fitted=d["fitted"])


def build_elo(*, save: bool = True) -> EloModel:
    """Fit Elo on the processed matches table and persist state + Elo-enriched matches."""
    matches = pd.read_parquet(CONFIG.processed / "matches.parquet")
    model = EloModel().fit(matches)
    if save:
        model.save()
        model.history.to_parquet(CONFIG.processed / "matches_elo.parquet", index=False)
        print(f"[elo] saved state ({len(model.ratings)} teams) + matches_elo.parquet")
    return model


if __name__ == "__main__":
    m = build_elo()
    ref = pd.read_parquet(CONFIG.processed / "teams_reference.parquet")
    cur = pd.DataFrame({"team": CONFIG.teams})
    cur["elo_computed"] = cur["team"].map(m.rating)
    cur = cur.merge(ref[["team", "elo_external"]], on="team")
    cur["delta"] = cur["elo_computed"] - cur["elo_external"]
    cur = cur.sort_values("elo_computed", ascending=False)
    print("\nTop 12 by our computed Elo (vs eloratings.net):")
    print(cur.head(12).to_string(index=False, float_format=lambda x: f"{x:.0f}"))
    corr = cur[["elo_computed", "elo_external"]].corr().iloc[0, 1]
    print(f"\ncorr(computed, eloratings.net) over 48 teams = {corr:.3f}")
    ph, pd_, pa = m.predict_proba("Spain", "Cape Verde", neutral=True)
    print(f"Spain vs Cape Verde (neutral): H={ph:.2f} D={pd_:.2f} A={pa:.2f}")
