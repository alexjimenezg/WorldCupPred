"""Monte Carlo tournament engine.

Wraps the ensemble in a caching `EnsembleSampler` (scoreline CDF per matchup is computed
once, then sampled cheaply), runs N full tournaments, and aggregates per-team stage
probabilities into a title-odds table. Supports "simulate from now" by passing already
played results in `fixed` (group scores and/or knockout results).

    from src.simulation.monte_carlo import run_simulation
    table = run_simulation(n=50000)          # uses saved models
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.models.ensemble import MatchPredictor
from src.simulation import tournament_2026 as T

_SIM = CONFIG.settings["simulation"]
_PEN_FLOOR = float(_SIM["penalty_floor"])

# cumulative stage columns (>= that round)
_STAGE_INDEX = {s: i for i, s in enumerate(T.STAGES)}
_REACH_COLUMNS = {
    "round32": "R32", "round16": "R16", "quarterfinal": "QF",
    "semifinal": "SF", "final": "final", "champion": "champion",
}


@dataclass
class EnsembleSampler:
    predictor: MatchPredictor
    rng: np.random.Generator
    _cache: dict = field(default_factory=dict)

    def _dist(self, home: str, away: str, neutral: bool):
        key = (home, away, neutral)
        hit = self._cache.get(key)
        if hit is None:
            out = self.predictor.predict(home, away, neutral)
            m = out["scoreline"]
            hit = (np.cumsum(m.ravel()), m.shape[1], out["p_home"], out["p_away"])
            self._cache[key] = hit
        return hit

    def sample_score(self, home: str, away: str, neutral: bool) -> tuple[int, int]:
        cdf, ncol, _ph, _pa = self._dist(home, away, neutral)
        idx = int(np.searchsorted(cdf, self.rng.random() * cdf[-1]))
        return divmod(idx, ncol)

    def conditional_home_win(self, home: str, away: str, neutral: bool) -> float:
        _cdf, _ncol, ph, pa = self._dist(home, away, neutral)
        p = ph / (ph + pa) if (ph + pa) > 0 else 0.5
        return float(np.clip(p, _PEN_FLOOR, 1 - _PEN_FLOOR))


def run_simulation(predictor: MatchPredictor | None = None, *, n: int | None = None,
                   seed: int | None = None, fixed: dict | None = None,
                   progress: bool = True, save: bool = True, sampler=None) -> pd.DataFrame:
    n = int(n or _SIM["n_iterations"])
    seed = _SIM["random_seed"] if seed is None else seed
    rng = np.random.default_rng(seed)
    if sampler is None:  # build the ensemble sampler unless one is injected (tests)
        predictor = predictor or MatchPredictor.load_default()
        sampler = EnsembleSampler(predictor, rng)
    else:
        sampler.rng = rng

    champ = Counter()
    won_group = Counter()
    reach = {t: Counter() for t in CONFIG.teams}

    t0 = time.time()
    for k in range(n):
        res = T.simulate_tournament(sampler, rng, fixed)
        champ[res.champion] += 1
        for t in res.group_winners:
            won_group[t] += 1
        for team, st in res.reached.items():
            reach[team][st] += 1
        if progress and (k + 1) % max(1, n // 10) == 0:
            print(f"  {k + 1:>6,}/{n:,} sims  ({time.time() - t0:4.1f}s)")

    rows = []
    for t in CONFIG.teams:
        cnt = reach[t]
        # P(reach >= stage): sum furthest-stage counts at or beyond the threshold
        def ge(stage: str) -> float:
            thr = _STAGE_INDEX[stage]
            return sum(c for s, c in cnt.items() if _STAGE_INDEX[s] >= thr) / n
        rows.append({
            "team": t, "group": CONFIG.group_of(t),
            "champion": champ[t] / n,
            "final": ge("final"),
            "semifinal": ge("SF"),
            "quarterfinal": ge("QF"),
            "round16": ge("R16"),
            "round32": ge("R32"),
            "win_group": won_group[t] / n,
        })
    table = (pd.DataFrame(rows)
             .sort_values("champion", ascending=False)
             .reset_index(drop=True))

    if save:
        table.to_parquet(CONFIG.processed / "title_odds.parquet", index=False)
        _append_history(table, n)
        _write_report(table, n, time.time() - t0)
    if progress:
        cached = len(getattr(sampler, "_cache", {}))
        print(f"done: {n:,} sims in {time.time() - t0:.1f}s ({cached} cached matchups)")
    return table


def _append_history(table: pd.DataFrame, n: int) -> None:
    """Append this run's table to odds_history.parquet so odds can be charted over time."""
    path = CONFIG.processed / "odds_history.parquet"
    snap = table.copy()
    snap.insert(0, "ts", pd.Timestamp.now().floor("s"))
    snap["n_sims"] = n
    if path.exists():
        snap = pd.concat([pd.read_parquet(path), snap], ignore_index=True)
    snap.to_parquet(path, index=False)


def _df_to_md(df: pd.DataFrame) -> str:
    """Minimal GitHub-flavored markdown table (avoids a tabulate dependency)."""
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def _write_report(table: pd.DataFrame, n: int, secs: float) -> None:
    path = CONFIG.path("reports") / "title_odds.md"
    pct = table.copy()
    for c in ["champion", "final", "semifinal", "quarterfinal", "round16", "round32", "win_group"]:
        pct[c] = (pct[c] * 100).map(lambda x: f"{x:4.1f}%")
    lines = [
        f"# FIFA World Cup 2026 — title odds ({n:,} Monte Carlo simulations)",
        "",
        f"_Generated in {secs:.1f}s. Champion probability, sorted._",
        "",
        _df_to_md(pct.head(24)),
        "",
        "Full table: `data/processed/title_odds.parquet`.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[monte_carlo] wrote {path}")


if __name__ == "__main__":
    tbl = run_simulation(n=20000)
    show = tbl.head(16).copy()
    for c in ["champion", "final", "semifinal", "quarterfinal", "round16", "win_group"]:
        show[c] = (show[c] * 100).round(1)
    print("\nFIFA World Cup 2026 — title odds (top 16):")
    print(show[["team", "group", "champion", "final", "semifinal",
                "quarterfinal", "win_group"]].to_string(index=False))
