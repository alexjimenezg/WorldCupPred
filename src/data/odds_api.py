"""The Odds API client (optional, free tier) — betting odds as a benchmark / sanity anchor.

Needs ODDS_API_KEY in .env. Without it every function returns empty and the pipeline runs
unaffected. The outright-winner market gives de-vigged implied champion probabilities to
compare against our simulation; the h2h market gives per-match implied W/D/L.
"""

from __future__ import annotations

import pandas as pd
import requests

from src.config import CONFIG

_BASE = CONFIG.settings["data_sources"]["odds_api_base"]
_SPORT = CONFIG.settings["data_sources"]["odds_sport_key"]


def available() -> bool:
    return bool(CONFIG.env("ODDS_API_KEY"))


def _get(path: str, params: dict) -> list | dict | None:
    key = CONFIG.env("ODDS_API_KEY")
    if not key:
        return None
    params = {**params, "apiKey": key}
    r = requests.get(f"{_BASE}{path}", params=params, timeout=20)
    if r.status_code != 200:
        print(f"[odds_api] HTTP {r.status_code}: {r.text[:120]}")
        return None
    return r.json()


def fetch_outright_odds(*, regions: str = "us,uk,eu") -> pd.DataFrame:
    """De-vigged implied champion probabilities per team (empty if no key)."""
    data = _get(f"/sports/{_SPORT}/odds",
                {"regions": regions, "oddsFormat": "decimal", "markets": "outrights"})
    if not data:
        return pd.DataFrame(columns=["team", "implied_prob", "decimal_odds"])
    prices: dict[str, list[float]] = {}
    for event in data:
        for bm in event.get("bookmakers", []):
            for mkt in bm.get("markets", []):
                for out in mkt.get("outcomes", []):
                    team = CONFIG.normalize(out["name"])
                    if out.get("price"):
                        prices.setdefault(team, []).append(float(out["price"]))
    if not prices:
        return pd.DataFrame(columns=["team", "implied_prob", "decimal_odds"])
    rows = [{"team": t, "decimal_odds": float(pd.Series(p).median()),
             "raw_prob": 1.0 / float(pd.Series(p).median())} for t, p in prices.items()]
    df = pd.DataFrame(rows)
    df["implied_prob"] = df["raw_prob"] / df["raw_prob"].sum()  # remove the vig
    return (df[["team", "implied_prob", "decimal_odds"]]
            .sort_values("implied_prob", ascending=False).reset_index(drop=True))


def benchmark_vs_simulation() -> pd.DataFrame:
    """Side-by-side of market implied champion prob vs our simulated champion prob."""
    market = fetch_outright_odds()
    if market.empty:
        return market
    sim_path = CONFIG.processed / "title_odds.parquet"
    if not sim_path.exists():
        return market
    sim = pd.read_parquet(sim_path)[["team", "champion"]]
    out = market.merge(sim, on="team", how="left").rename(columns={"champion": "model_prob"})
    out["edge"] = out["model_prob"] - out["implied_prob"]
    return out


if __name__ == "__main__":
    if not available():
        print("ODDS_API_KEY not set — add it to .env to enable betting-odds benchmarks.")
    else:
        print(benchmark_vs_simulation().head(15).to_string(index=False))
