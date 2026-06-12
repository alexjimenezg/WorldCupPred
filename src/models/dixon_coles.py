"""Dixon-Coles bivariate Poisson goal model — the primary match engine.

For home i vs away j:
    lambda = exp(base + att_i - def_j + home_adv)      # home expected goals
    mu     = exp(base + att_j - def_i)                  # away expected goals
Goals are Poisson with the Dixon-Coles low-score dependence correction tau(.,.;rho) on the
{0-0, 1-0, 0-1, 1-1} cells. Recent matches count more via exponential time decay
weight = exp(-xi * age_days). A ridge penalty on att/def gives identifiability AND shrinks
weak/low-sample teams toward average (the "hierarchical shrinkage" in the plan).

Fitting is split for speed and robustness:
  1. att/def/base/home_adv via a convex weighted-Poisson MLE (analytic gradient, L-BFGS-B);
  2. rho via a 1-D MLE of the DC correction holding the rest fixed.

predict() returns the full scoreline distribution and P(home/draw/away) for any fixture,
including neutral-site handling used by the tournament simulator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import lgamma
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from src.config import CONFIG

_DC = CONFIG.settings["dixon_coles"]
_XI = float(_DC["time_decay_xi"])
_MAXG = int(_DC["max_goals"])
_RECENT_YEARS = int(_DC["recent_years"])
_HFA_NEUTRAL = 0.0


def _poisson_pmf_vec(lam: float, kmax: int) -> np.ndarray:
    k = np.arange(kmax + 1)
    logp = -lam + k * np.log(max(lam, 1e-12)) - np.array([lgamma(i + 1) for i in k])
    return np.exp(logp)


@dataclass
class DixonColes:
    teams: list[str] = field(default_factory=list)
    attack: dict[str, float] = field(default_factory=dict)
    defense: dict[str, float] = field(default_factory=dict)
    base: float = 0.0
    home_adv: float = 0.25
    rho: float = -0.1
    ref_date: str = "2026-06-12"
    ridge: float = 0.01

    # ---- fitting ------------------------------------------------------------
    def fit(self, matches: pd.DataFrame, *, ref_date: str | None = None) -> "DixonColes":
        ref = pd.Timestamp(ref_date or self.ref_date)
        cutoff = ref - pd.Timedelta(days=365 * _RECENT_YEARS)
        df = matches[(matches["date"] >= cutoff) & (matches["date"] <= ref)].copy()

        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        idx = {t: k for k, t in enumerate(teams)}
        n = len(teams)
        i = df["home_team"].map(idx).to_numpy()
        j = df["away_team"].map(idx).to_numpy()
        x = df["home_score"].to_numpy(dtype=float)
        y = df["away_score"].to_numpy(dtype=float)
        age = (ref - df["date"]).dt.days.to_numpy(dtype=float)
        w = np.exp(-_XI * age)

        # param vector: [base, home_adv, att(n), def(n)]
        def unpack(p):
            return p[0], p[1], p[2:2 + n], p[2 + n:2 + 2 * n]

        def nll_grad(p):
            base, hadv, att, dfn = unpack(p)
            lam = np.exp(base + att[i] - dfn[j] + hadv)
            mu = np.exp(base + att[j] - dfn[i])
            # weighted Poisson NLL (drop constant log-factorials)
            nll = np.sum(w * (lam - x * np.log(lam) + mu - y * np.log(mu)))
            nll += 0.5 * self.ridge * (att @ att + dfn @ dfn)
            # gradients
            rh = w * (lam - x)   # d/d(loglam)
            ra = w * (mu - y)    # d/d(logmu)
            g_base = np.sum(rh + ra)
            g_h = np.sum(rh)
            g_att = np.zeros(n); g_def = np.zeros(n)
            np.add.at(g_att, i, rh);  np.add.at(g_att, j, ra)
            np.add.at(g_def, j, -rh); np.add.at(g_def, i, -ra)
            g_att += self.ridge * att
            g_def += self.ridge * dfn
            return nll, np.concatenate([[g_base, g_h], g_att, g_def])

        p0 = np.concatenate([[0.0, 0.25], np.zeros(2 * n)])
        res = minimize(nll_grad, p0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 500, "maxfun": 50000})
        base, hadv, att, dfn = unpack(res.x)
        # center attack for interpretability (absorb into base)
        shift = att.mean()
        att = att - shift
        dfn = dfn - dfn.mean()
        base = base + shift  # keep lambda unchanged on average

        self.teams = teams
        self.attack = {t: float(att[idx[t]]) for t in teams}
        self.defense = {t: float(dfn[idx[t]]) for t in teams}
        self.base = float(base)
        self.home_adv = float(hadv)
        self.ref_date = str(ref.date())
        self._fit_rho(i, j, x, y, w)
        return self

    def _fit_rho(self, i, j, x, y, w) -> None:
        att = np.array([self.attack[t] for t in self.teams])
        dfn = np.array([self.defense[t] for t in self.teams])
        lam = np.exp(self.base + att[i] - dfn[j] + self.home_adv)
        mu = np.exp(self.base + att[j] - dfn[i])
        m00 = (x == 0) & (y == 0); m10 = (x == 1) & (y == 0)
        m01 = (x == 0) & (y == 1); m11 = (x == 1) & (y == 1)

        def neg_ll(rho):
            tau = np.ones_like(lam)
            tau[m00] = 1 - lam[m00] * mu[m00] * rho
            tau[m10] = 1 + mu[m10] * rho
            tau[m01] = 1 + lam[m01] * rho
            tau[m11] = 1 - rho
            tau = np.clip(tau, 1e-9, None)
            return -np.sum(w * np.log(tau))

        res = minimize_scalar(neg_ll, bounds=(-0.2, 0.2), method="bounded")
        self.rho = float(res.x)

    # ---- prediction ---------------------------------------------------------
    def _lambda_mu(self, home: str, away: str, neutral: bool) -> tuple[float, float]:
        h, a = CONFIG.normalize(home), CONFIG.normalize(away)
        ah = self.attack.get(h, 0.0); dh = self.defense.get(h, 0.0)
        aa = self.attack.get(a, 0.0); da = self.defense.get(a, 0.0)
        hadv = _HFA_NEUTRAL if neutral else self.home_adv
        lam = np.exp(self.base + ah - da + hadv)
        mu = np.exp(self.base + aa - dh)
        return float(lam), float(mu)

    def scoreline_matrix(self, home: str, away: str, neutral: bool = False) -> np.ndarray:
        lam, mu = self._lambda_mu(home, away, neutral)
        ph = _poisson_pmf_vec(lam, _MAXG)
        pa = _poisson_pmf_vec(mu, _MAXG)
        m = np.outer(ph, pa)
        r = self.rho
        m[0, 0] *= 1 - lam * mu * r
        m[1, 0] *= 1 + mu * r
        m[0, 1] *= 1 + lam * r
        m[1, 1] *= 1 - r
        m = np.clip(m, 0, None)
        return m / m.sum()

    def predict_proba(self, home: str, away: str, neutral: bool = False) -> tuple[float, float, float]:
        m = self.scoreline_matrix(home, away, neutral)
        p_home = float(np.tril(m, -1).sum())
        p_draw = float(np.trace(m))
        p_away = float(np.triu(m, 1).sum())
        return p_home, p_draw, p_away

    def expected_goals(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        return self._lambda_mu(home, away, neutral)

    # ---- persistence --------------------------------------------------------
    def save(self, path: Path | None = None) -> Path:
        path = path or (CONFIG.models_dir / "dixon_coles.json")
        path.write_text(json.dumps({
            "teams": self.teams, "attack": self.attack, "defense": self.defense,
            "base": self.base, "home_adv": self.home_adv, "rho": self.rho,
            "ref_date": self.ref_date, "ridge": self.ridge,
        }), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "DixonColes":
        path = path or (CONFIG.models_dir / "dixon_coles.json")
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(teams=d["teams"], attack=d["attack"], defense=d["defense"],
                   base=d["base"], home_adv=d["home_adv"], rho=d["rho"],
                   ref_date=d["ref_date"], ridge=d.get("ridge", 0.01))


def build_dixon_coles(*, save: bool = True) -> DixonColes:
    matches = pd.read_parquet(CONFIG.processed / "matches.parquet")
    model = DixonColes().fit(matches)
    if save:
        model.save()
        print(f"[dixon_coles] fit on teams={len(model.teams)} "
              f"base={model.base:.3f} home_adv={model.home_adv:.3f} rho={model.rho:.3f}")
    return model


if __name__ == "__main__":
    m = build_dixon_coles()
    for h, a, neu in [("Spain", "Cape Verde", True), ("Brazil", "Scotland", True),
                      ("United States", "Paraguay", False), ("Argentina", "Jordan", True)]:
        lam, mu = m.expected_goals(h, a, neu)
        ph, pd_, pa = m.predict_proba(h, a, neu)
        print(f"{h:14}{lam:4.2f} - {mu:4.2f} {a:12}  "
              f"H={ph:.2f} D={pd_:.2f} A={pa:.2f}  (neutral={neu})")
