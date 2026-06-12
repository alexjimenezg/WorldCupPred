"""Walk-forward validation of the match engine (leak-free single split).

Fit Elo + Dixon-Coles only on matches up to `train_end`, then score probabilistic
forecasts on the held-out window. Compares each engine and the ensemble against a
base-rate baseline. RPS is the headline metric (lower is better).

    python -m src.evaluation.backtest                      # default 2023-01 -> 2025-12
    python -m src.evaluation.backtest --train-end 2022-11-01 --test-end 2022-12-20
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.evaluation import metrics
from src.models.dixon_coles import DixonColes
from src.models.ensemble import MatchPredictor
from src.models.elo import EloModel


def _collect(predict_proba, test: pd.DataFrame) -> list[tuple[float, float, float]]:
    return [predict_proba(h, a, bool(n)) for h, a, n in
            zip(test["home_team"], test["away_team"], test["neutral"])]


def backtest(train_end: str = "2023-01-01", test_end: str = "2025-12-31",
             *, competitive_only: bool = True, include_ml: bool = False,
             include_dl: bool = False, verbose: bool = True) -> pd.DataFrame:
    matches = pd.read_parquet(CONFIG.processed / "matches.parquet")
    train = matches[matches["date"] <= train_end]
    test = matches[(matches["date"] > train_end) & (matches["date"] <= test_end)].copy()
    if competitive_only:
        test = test[test["is_competitive"]]
    test = test.dropna(subset=["home_team", "away_team"]).reset_index(drop=True)

    elo = EloModel().fit(train)
    dc = DixonColes().fit(train, ref_date=train_end)

    # base-rate baseline from the training set
    rates = train["outcome"].value_counts(normalize=True)
    base = np.array([rates.get("H", 0.45), rates.get("D", 0.27), rates.get("A", 0.28)])
    base = base / base.sum()

    engines = {
        "baseline": [tuple(base)] * len(test),
        "elo": _collect(elo.predict_proba, test),
        "dixon_coles": _collect(dc.predict_proba, test),
    }
    ml = dl = None
    if include_ml or include_dl:
        from src.features.build_features import build_features
        feats = build_features(elo.history, save=False)
    if include_ml:
        from src.models.ml_outcome import MLOutcome
        ml = MLOutcome().fit(feats, elo=elo, ref_date=train_end)
        engines["ml"] = _collect(ml.predict_proba, test)
    if include_dl:
        from src.models.dl_outcome import DLOutcome
        dl = DLOutcome().fit(feats, elo=elo, ref_date=train_end)
        engines["dl"] = _collect(dl.predict_proba, test)

    ens = MatchPredictor(elo=elo, dc=dc, ml=ml, dl=dl)
    engines["ensemble"] = _collect(ens.predict_proba, test)

    outcomes = test["outcome"].tolist()
    rows = []
    for name, probs in engines.items():
        s = metrics.summary(probs, outcomes)
        s["model"] = name
        rows.append(s)
    out = pd.DataFrame(rows).set_index("model")[["n", "rps", "brier", "log_loss", "accuracy"]]

    if verbose:
        print(f"train<= {train_end}  test ({len(test)} competitive matches) "
              f"{test['date'].min().date()} -> {test['date'].max().date()}")
        print(out.to_string(float_format=lambda x: f"{x:.4f}"))
        best = out.drop("baseline")["rps"].idxmin()
        lift = 1 - out.loc[best, "rps"] / out.loc["baseline", "rps"]
        print(f"best: {best}  (RPS {out.loc[best, 'rps']:.4f}, "
              f"{lift:+.1%} vs baseline)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-end", default="2023-01-01")
    ap.add_argument("--test-end", default="2025-12-31")
    ap.add_argument("--ml", action="store_true", help="include HistGradientBoosting")
    ap.add_argument("--dl", action="store_true", help="include keras DL (slow)")
    args = ap.parse_args()
    backtest(args.train_end, args.test_end, include_ml=args.ml, include_dl=args.dl)
