"""Monte Carlo aggregation invariants (injected fake sampler -> no trained models needed)."""

from __future__ import annotations

import numpy as np

from src.config import CONFIG
from src.simulation.monte_carlo import run_simulation


class FakeSampler:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def sample_score(self, home, away, neutral):
        return int(self.rng.integers(0, 4)), int(self.rng.integers(0, 3))

    def conditional_home_win(self, home, away, neutral):
        return 0.5


def _run(n=400, fixed=None):
    return run_simulation(sampler=FakeSampler(7), n=n, seed=7, fixed=fixed,
                          progress=False, save=False)


def test_probabilities_are_consistent():
    t = _run()
    assert len(t) == 48
    assert abs(t["champion"].sum() - 1.0) < 1e-9          # one champion
    assert abs(t["final"].sum() - 2.0) < 1e-6             # two finalists
    assert abs(t["round32"].sum() - 32.0) < 1e-6          # 32 reach the R32
    assert abs(t["win_group"].sum() - 12.0) < 1e-6        # 12 group winners


def test_reach_probabilities_monotonic():
    t = _run()
    for _, r in t.iterrows():
        assert r.champion <= r.final + 1e-9 <= r.semifinal + 1e-9
        assert r.semifinal <= r.quarterfinal + 1e-9 <= r.round16 + 1e-9 <= r.round32 + 1e-9


def test_simulate_from_now_fixes_a_result():
    # fix every Group H game so Spain wins big -> Spain always reaches at least R32
    teams = CONFIG.groups["H"]
    groups = {}
    for i, a in enumerate(teams):
        for b in teams[i + 1:]:
            groups[(a, b)] = (4, 0) if a == "Spain" else ((0, 4) if b == "Spain" else (0, 0))
    t = _run(fixed={"groups": groups}).set_index("team")
    assert t.loc["Spain", "round32"] == 1.0
    assert t.loc["Spain", "win_group"] == 1.0
