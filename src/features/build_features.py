"""Leak-free match-level features for the ML/DL ensemble members.

The dominant signal is pre-match Elo (already leak-free in matches_elo.parquet). On top we
add recent-form levels computed with a shifted rolling window (so a match never sees its own
result), plus static context (neutral, host, confederation strength). The exact same feature
vector is reconstructable at prediction time for any future fixture via `make_feature_row`,
using current Elo ratings and a `team_state` snapshot — guaranteeing train/serve consistency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.models.elo import EloModel

FORM_WINDOW = int(CONFIG.settings["features"]["form_window_matches"])

# Order matters: models consume features in exactly this order.
FEATURE_COLUMNS = [
    "elo_home", "elo_away", "elo_diff",
    "home_form_points", "away_form_points",
    "home_form_gd", "away_form_gd",
    "neutral", "host_home", "host_away",
    "confed_str_home", "confed_str_away",
]


def _confed_strength(team: str) -> float:
    return CONFIG.confederation_prior.get(CONFIG.confederation_of(team), 1600.0)


def _team_perspective(matches: pd.DataFrame) -> pd.DataFrame:
    """Two rows per match (one per team) tagged with match_id + side, for rolling form.

    Keyed by match_id so form can be pivoted back exactly (no (date,team) fan-out on the
    rare same-day double-headers in old data)."""
    mid = matches.index.to_numpy()
    home = pd.DataFrame({
        "match_id": mid, "side": "home", "date": matches["date"].to_numpy(),
        "team": matches["home_team"].to_numpy(),
        "gf": matches["home_score"].to_numpy(), "ga": matches["away_score"].to_numpy(),
    })
    away = pd.DataFrame({
        "match_id": mid, "side": "away", "date": matches["date"].to_numpy(),
        "team": matches["away_team"].to_numpy(),
        "gf": matches["away_score"].to_numpy(), "ga": matches["home_score"].to_numpy(),
    })
    long = pd.concat([home, away], ignore_index=True)
    long["points"] = np.where(long.gf > long.ga, 3, np.where(long.gf == long.ga, 1, 0))
    long["gd"] = long.gf - long.ga
    return long.sort_values(["team", "date", "match_id"]).reset_index(drop=True)


def _rolling_form(long: pd.DataFrame, window: int) -> pd.DataFrame:
    g = long.groupby("team", group_keys=False)
    long = long.copy()
    long["form_points"] = g["points"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    long["form_gd"] = g["gd"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    return long


def build_features(matches_elo: pd.DataFrame | None = None, *,
                   window: int = FORM_WINDOW, save: bool = True) -> pd.DataFrame:
    """Build the training feature table from matches_elo (parquet, or a passed frame).

    Passing a train-only `matches_elo` (e.g. EloModel.fit(train).history) yields leak-free
    features for backtesting."""
    if matches_elo is None:
        m = pd.read_parquet(CONFIG.processed / "matches_elo.parquet").reset_index(drop=True)
    else:
        m = matches_elo.reset_index(drop=True)
    long = _rolling_form(_team_perspective(m), window)

    # pivot form back to match rows exactly via match_id + side
    home_form = (long[long.side == "home"].set_index("match_id")[["form_points", "form_gd"]]
                 .rename(columns={"form_points": "home_form_points", "form_gd": "home_form_gd"}))
    away_form = (long[long.side == "away"].set_index("match_id")[["form_points", "form_gd"]]
                 .rename(columns={"form_points": "away_form_points", "form_gd": "away_form_gd"}))
    feat = m.join(home_form).join(away_form)

    feat["elo_home"] = feat["home_elo_pre"]
    feat["elo_away"] = feat["away_elo_pre"]
    feat["elo_diff"] = feat["elo_diff_pre"]
    feat["host_home"] = feat["home_team"].map(CONFIG.is_host).astype(float)
    feat["host_away"] = feat["away_team"].map(CONFIG.is_host).astype(float)
    feat["confed_str_home"] = feat["home_team"].map(_confed_strength)
    feat["confed_str_away"] = feat["away_team"].map(_confed_strength)
    feat["neutral"] = feat["neutral"].astype(float)
    for c in ["home_form_points", "away_form_points", "home_form_gd", "away_form_gd"]:
        feat[c] = feat[c].fillna(feat[c].median())

    out = feat[["date", "home_team", "away_team", "outcome", "home_score",
                "away_score", *FEATURE_COLUMNS]].dropna(subset=["elo_diff"])
    if save:
        out.to_parquet(CONFIG.processed / "features.parquet", index=False)
        print(f"[features] wrote features.parquet ({len(out):,} rows, "
              f"{len(FEATURE_COLUMNS)} features)")
    return out


def team_state(*, window: int = FORM_WINDOW, ref_date: str | None = None) -> dict[str, dict]:
    """Latest form snapshot per team (mean over last `window` matches up to ref_date)."""
    m = pd.read_parquet(CONFIG.processed / "matches.parquet")
    if ref_date is not None:
        m = m[m["date"] <= ref_date]
    long = _team_perspective(m)
    last = long.groupby("team").tail(window)
    agg = last.groupby("team").agg(form_points=("points", "mean"),
                                   form_gd=("gd", "mean"))
    return agg.to_dict("index")


def make_feature_row(home: str, away: str, neutral: bool,
                     elo: EloModel, state: dict[str, dict]) -> np.ndarray:
    """Reconstruct the model feature vector for a future fixture (train/serve parity)."""
    home, away = CONFIG.normalize(home), CONFIG.normalize(away)
    eh, ea = elo.rating(home), elo.rating(away)
    hfa_diff = elo.adj_diff(home, away, neutral)
    sh = state.get(home, {"form_points": 1.3, "form_gd": 0.0})
    sa = state.get(away, {"form_points": 1.3, "form_gd": 0.0})
    row = {
        "elo_home": eh, "elo_away": ea, "elo_diff": hfa_diff,
        "home_form_points": sh["form_points"], "away_form_points": sa["form_points"],
        "home_form_gd": sh["form_gd"], "away_form_gd": sa["form_gd"],
        "neutral": float(neutral),
        "host_home": float(CONFIG.is_host(home)), "host_away": float(CONFIG.is_host(away)),
        "confed_str_home": _confed_strength(home), "confed_str_away": _confed_strength(away),
    }
    return np.array([row[c] for c in FEATURE_COLUMNS], dtype=float)


if __name__ == "__main__":
    feats = build_features()
    print(feats[FEATURE_COLUMNS].describe().T[["mean", "std", "min", "max"]].to_string())
