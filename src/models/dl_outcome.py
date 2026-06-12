"""Deep-learning W/D/L model: a keras MLP with learned per-nation embeddings.

Each team gets an embedding vector (so the net can learn team-specific style/quality beyond
Elo), concatenated with the standardized numeric features, then an MLP head -> softmax over
(home, draw, away). Trained with exponential time-decay sample weights and early stopping.
Mirrors the prior project's keras/joblib-preprocessor convention. Same
`predict_proba(home, away, neutral)` interface as the other engines.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quiet TF banner

import joblib
import numpy as np
import pandas as pd

from src.config import CONFIG
from src.features.build_features import (FEATURE_COLUMNS, build_features,
                                         make_feature_row, team_state)
from src.models.elo import EloModel

_CLASS_ORDER = ["H", "D", "A"]            # output column order
_Y = {"H": 0, "D": 1, "A": 2}
_DECAY_XI = float(CONFIG.settings["dixon_coles"]["time_decay_xi"])
_EMB_DIM = 8


class DLOutcome:
    def __init__(self, model=None, scaler=None, vocab: dict | None = None,
                 state: dict | None = None, elo: EloModel | None = None):
        self.model = model
        self.scaler = scaler
        self.vocab = vocab or {}
        self.state = state or {}
        self.elo = elo

    def _idx(self, team: str) -> int:
        return self.vocab.get(CONFIG.normalize(team), 0)  # 0 = UNK

    # ---- training -----------------------------------------------------------
    def fit(self, features: pd.DataFrame | None = None, *, elo: EloModel | None = None,
            ref_date: str = "2026-06-12", epochs: int = 40) -> "DLOutcome":
        import keras
        from sklearn.preprocessing import StandardScaler

        features = build_features(save=False) if features is None else features
        teams = sorted(set(features["home_team"]) | set(features["away_team"]))
        self.vocab = {t: i + 1 for i, t in enumerate(teams)}  # 0 reserved for UNK
        n_vocab = len(self.vocab) + 1

        Xnum = features[FEATURE_COLUMNS].to_numpy(float)
        self.scaler = StandardScaler().fit(Xnum)
        Xnum = self.scaler.transform(Xnum)
        hi = features["home_team"].map(self._idx).to_numpy()
        ai = features["away_team"].map(self._idx).to_numpy()
        y = features["outcome"].map(_Y).to_numpy()
        age = (pd.Timestamp(ref_date) - features["date"]).dt.days.to_numpy(float)
        w = np.exp(-_DECAY_XI * np.clip(age, 0, None))

        num_in = keras.Input(shape=(len(FEATURE_COLUMNS),), name="num")
        h_in = keras.Input(shape=(1,), name="home")
        a_in = keras.Input(shape=(1,), name="away")
        emb = keras.layers.Embedding(n_vocab, _EMB_DIM, name="team_emb")
        h_e = keras.layers.Flatten()(emb(h_in))
        a_e = keras.layers.Flatten()(emb(a_in))
        x = keras.layers.Concatenate()([num_in, h_e, a_e])
        x = keras.layers.Dense(64, activation="relu")(x)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(32, activation="relu")(x)
        out = keras.layers.Dense(3, activation="softmax")(x)
        self.model = keras.Model([num_in, h_in, a_in], out)
        self.model.compile(optimizer=keras.optimizers.Adam(1e-3),
                           loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        self.model.fit({"num": Xnum, "home": hi, "away": ai}, y, sample_weight=w,
                       validation_split=0.1, epochs=epochs, batch_size=512,
                       callbacks=[cb], verbose=0)
        self.elo = elo or EloModel.load()
        self.state = team_state(ref_date=ref_date)
        return self

    # ---- prediction ---------------------------------------------------------
    def predict_proba(self, home: str, away: str, neutral: bool = False) -> tuple[float, float, float]:
        if self.elo is None:
            self.elo = EloModel.load()
        x = make_feature_row(home, away, neutral, self.elo, self.state).reshape(1, -1)
        x = self.scaler.transform(x)
        hi = np.array([[self._idx(home)]]); ai = np.array([[self._idx(away)]])
        p = self.model.predict({"num": x, "home": hi, "away": ai}, verbose=0)[0]
        return float(p[0]), float(p[1]), float(p[2])

    # ---- persistence --------------------------------------------------------
    def save(self, dir_: Path | None = None) -> Path:
        dir_ = dir_ or CONFIG.models_dir
        self.model.save(dir_ / "dl_outcome.keras")
        joblib.dump({"scaler": self.scaler, "vocab": self.vocab, "state": self.state},
                    dir_ / "dl_outcome_meta.joblib")
        return dir_ / "dl_outcome.keras"

    @classmethod
    def load(cls, dir_: Path | None = None, *, elo: EloModel | None = None) -> "DLOutcome":
        import keras
        dir_ = dir_ or CONFIG.models_dir
        model = keras.models.load_model(dir_ / "dl_outcome.keras")
        b = joblib.load(dir_ / "dl_outcome_meta.joblib")
        return cls(model=model, scaler=b["scaler"], vocab=b["vocab"], state=b["state"], elo=elo)


def build_dl_outcome(*, save: bool = True) -> DLOutcome:
    model = DLOutcome().fit()
    if save:
        model.save()
        print("[dl_outcome] trained keras embedding MLP + saved")
    return model


if __name__ == "__main__":
    m = build_dl_outcome()
    for h, a, neu in [("Spain", "Brazil", True), ("England", "Panama", True),
                      ("United States", "Turkey", False), ("Argentina", "Jordan", True)]:
        ph, pd_, pa = m.predict_proba(h, a, neu)
        print(f"{h:13} vs {a:12} H={ph:.2f} D={pd_:.2f} A={pa:.2f}")
