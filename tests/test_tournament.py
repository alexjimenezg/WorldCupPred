"""Unit tests for the 2026 tournament format and bracket (deterministic fake sampler)."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from src.config import CONFIG
from src.simulation import tournament_2026 as T


class FakeSampler:
    """Deterministic sampler: stronger (alphabetically-earlier) team tends to win."""
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def sample_score(self, home, away, neutral):
        return int(self.rng.integers(0, 4)), int(self.rng.integers(0, 3))

    def conditional_home_win(self, home, away, neutral):
        return 0.5


def test_bracket_shape():
    assert len(T.BRACKET_R32) == 16
    assert len(T.BRACKET_R16) == 8
    assert len(T.BRACKET_QF) == 4
    assert len(T.BRACKET_SF) == 2
    # exactly 8 third-place slots, each drawing from 5 eligible groups
    third_ties = [t for t in T.BRACKET_R32 if t[2] == "3"]
    assert len(third_ties) == 8
    assert set(t[0] for t in third_ties) == set(T.THIRD_SLOT_GROUPS)
    assert all(len(v) == 5 for v in T.THIRD_SLOT_GROUPS.values())
    # a winner-slot never draws a third from its own group
    for tie_id, grps in T.THIRD_SLOT_GROUPS.items():
        winner_group = dict((t[0], t[1]) for t in T.BRACKET_R32)[tie_id][1]
        assert winner_group not in grps


def test_group_ranking_points_then_gd_then_gf():
    rows = [T.TeamRow("W", pts=9, gf=7, ga=1), T.TeamRow("X", pts=4, gf=5, ga=4),
            T.TeamRow("Y", pts=4, gf=6, ga=5), T.TeamRow("Z", pts=0, gf=1, ga=9)]
    order = T._rank(rows, results={}, rng=np.random.default_rng(0))
    assert order[0] == "W" and order[3] == "Z"
    # X and Y tie on points (4); Y has more GF at equal GD -> Y above X
    assert order.index("Y") < order.index("X")


def test_head_to_head_breaks_exact_tie():
    # three teams identical on pts/gd/gf; head-to-head decides
    rows = [T.TeamRow("A", pts=3, gf=2, ga=2), T.TeamRow("B", pts=3, gf=2, ga=2),
            T.TeamRow("C", pts=3, gf=2, ga=2)]
    # A beat B, B beat C, C beat A is circular; make A beat both
    results = {("A", "B"): (1, 0), ("A", "C"): (1, 0), ("B", "C"): (1, 0)}
    order = T._rank(rows, results, rng=np.random.default_rng(0))
    assert order[0] == "A"  # A has best head-to-head record


def test_all_third_place_allocations_valid():
    for combo in combinations("ABCDEFGHIJKL", 8):
        alloc = T.allocate_thirds(list(combo))
        assert len(alloc) == 8
        assert set(alloc.values()) == set(combo)
        for slot, g in alloc.items():
            assert g in T.THIRD_SLOT_GROUPS[slot]


def test_simulate_tournament_is_well_formed():
    res = T.simulate_tournament(FakeSampler(1), np.random.default_rng(1))
    assert res.champion in CONFIG.teams
    assert len(res.finalists) == 2 and res.champion in res.finalists
    assert set(res.reached) == set(CONFIG.teams)        # every team has a furthest stage
    assert len(res.group_winners) == 12 and len(set(res.group_winners)) == 12
    assert all(s in T.STAGES for s in res.reached.values())


def test_fixed_group_result_is_respected():
    # force Spain to thrash its whole group; Spain must win Group H every time
    teams = CONFIG.groups["H"]
    fixed = {}
    for i, a in enumerate(teams):
        for b in teams[i + 1:]:
            fixed[(a, b)] = (5, 0) if a == "Spain" else ((0, 5) if b == "Spain" else (1, 1))
    for seed in range(5):
        order, _ = T.simulate_group(teams, FakeSampler(seed),
                                    np.random.default_rng(seed), fixed)
        assert order[0] == "Spain"
