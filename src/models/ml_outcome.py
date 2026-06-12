"""Gradient-boosted W/D/L classifier (sklearn HistGradientBoosting) — an ensemble member.

Captures non-linear interactions (Elo-gap curvature, host/confederation/draw effects) the
Elo ordered-logit and Poisson model can't. Trained with exponential time-decay sample
weights so recent matches matter more. Exposes the same `predict_proba(home, away, neutral)
-> (p_home, p_draw, p_away)` interface as the other engines, so it slots straight into the
ensemble. Self-contained on disk (classifier + form snapshot via joblib); Elo is injected.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.config import CONFIG
from src.features.build_features import (FEATURE_COLUMNS, build_features,
                                         make_feature_row, team_state)
from src.models.elo import EloModel

_CLASS_ORDER = ["H", "D", "A"]
_DECAY_XI = float(CONFIG.settings["dixon_coles"]["time_decay_xi"])


class MLOutcome:
    def __init__(self, clf: HistGradientBoostingClassifier | None = None,
                 state: dict | None = None, elo: EloModel | None = None):
        self.clf = clf
        self.state = state or {}
        self.elo = elo

    # ---- training -----------------------------------------------------------
    def fit(self, features: pd.DataFrame | None = None, *,
            elo: EloModel | None = None, ref_date: str = "2026-06-12") -> "MLOutcome":
        features = build_features(save=False) if features is None else features
        X = features[FEATURE_COLUMNS].to_numpy(float)
        y = features["outcome"].to_numpy()
        age = (pd.Timestamp(ref_date) - features["date"]).dt.days.to_numpy(float)
        w = np.exp(-_DECAY_XI * np.clip(age, 0, None))
        self.clf = HistGradientBoostingClassifier(
            loss="log_loss", learning_rate=0.05, max_iter=400, max_depth=None,
            max_leaf_nodes=31, l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.1, random_state=2026,
        ).fit(X, y, sample_weight=w)
        self.elo = elo or EloModel.load()
        self.state = team_state(ref_date=ref_date)
        return self

    # ---- prediction ---------------------------------------------------------
    def predict_proba(self, home: str, away: str, neutral: bool = False) -> tuple[float, float, float]:
        if self.elo is None:
            self.elo = EloModel.load()
        x = make_feature_row(home, away, neutral, self.elo, self.state).reshape(1, -1)
        proba = self.clf.predict_proba(x)[0]
        idx = {c: i for i, c in enumerate(self.clf.classes_)}
        return tuple(float(proba[idx[c]]) for c in _CLASS_ORDER)  # type: ignore[return-value]

    # ---- persistence --------------------------------------------------------
    def save(self, path: Path | None = None) -> Path:
        path = path or (CONFIG.models_dir / "ml_outcome.joblib")
        joblib.dump({"clf": self.clf, "state": self.state}, path)
        return path

    @classmethod
    def load(cls, path: Path | None = None, *, elo: EloModel | None = None) -> "MLOutcome":
        path = path or (CONFIG.models_dir / "ml_outcome.joblib")
        b = joblib.load(path)
        return cls(clf=b["clf"], state=b["state"], elo=elo)


def build_ml_outcome(*, save: bool = True) -> MLOutcome:
    model = MLOutcome().fit()
    if save:
        model.save()
        print("[ml_outcome] trained HistGradientBoosting + saved snapshot")
    return model


if __name__ == "__main__":
    m = build_ml_outcome()
    for h, a, neu in [("Spain", "Brazil", True), ("England", "Panama", True),
                      ("United States", "Turkey", False), ("Argentina", "Jordan", True)]:
        ph, pd_, pa = m.predict_proba(h, a, neu)
        print(f"{h:13} vs {a:12} H={ph:.2f} D={pd_:.2f} A={pa:.2f}")
