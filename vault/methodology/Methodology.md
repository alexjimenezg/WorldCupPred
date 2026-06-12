---
type: methodology
updated: 2026-06-12 01:36
---

# Methodology

A statistical core feeds a Monte Carlo tournament simulation.

1. **Match model** — [[Elo]] and [[Dixon-Coles]] (primary) produce W/D/L + a scoreline
   distribution; [[ML-HistGB]] and [[DL-Embedding-MLP]] add non-linear signal; the
   [[Ensemble]] blends them.
2. **Simulation** — the real 48-team 2026 format (12 groups, 8 best thirds, R32->Final
   bracket) is played 50,000 times, sampling each match from the ensemble. Aggregating gives
   per-team champion and stage probabilities.
3. **Live loop** — entering a result refits the base models and re-simulates only the
   remaining fixtures (simulate-from-now).

Validation by Ranked Probability Score on held-out matches and past tournaments.

[[00-Index|Index]]
