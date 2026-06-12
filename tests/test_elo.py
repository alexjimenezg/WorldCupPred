"""Unit tests for the Elo model (no trained state needed)."""

from __future__ import annotations

import numpy as np

from src.models.elo import EloModel, goal_diff_multiplier


def test_goal_diff_multiplier():
    assert goal_diff_multiplier(0) == 1.0
    assert goal_diff_multiplier(1) == 1.0
    assert goal_diff_multiplier(-1) == 1.0
    assert goal_diff_multiplier(2) == 1.5
    assert goal_diff_multiplier(3) == (11 + 3) / 8
    assert goal_diff_multiplier(5) == (11 + 5) / 8


def test_update_is_zero_sum():
    m = EloModel(ratings={"A": 1600.0, "B": 1500.0})
    before = m.rating("A") + m.rating("B")
    m.update("A", "B", 2, 0, weight=40.0, neutral=True)
    after = m.rating("A") + m.rating("B")
    assert abs(before - after) < 1e-9  # rating mass conserved


def test_favorite_gains_less_when_winning_expected():
    m = EloModel(ratings={"A": 1800.0, "B": 1500.0})
    a0 = m.rating("A")
    m.update("A", "B", 1, 0, weight=40.0, neutral=True)
    fav_gain = m.rating("A") - a0
    m2 = EloModel(ratings={"A": 1500.0, "B": 1800.0})
    a0b = m2.rating("A")
    m2.update("A", "B", 1, 0, weight=40.0, neutral=True)
    underdog_gain = m2.rating("A") - a0b
    assert underdog_gain > fav_gain > 0  # upset win earns more


def test_expected_score_symmetry_and_home_advantage():
    m = EloModel(ratings={"A": 1500.0, "B": 1500.0})
    assert abs(m.expected_score("A", "B", neutral=True) - 0.5) < 1e-9
    # home advantage tilts the expectation above 0.5
    assert m.expected_score("A", "B", neutral=False) > 0.5


def test_predict_proba_is_a_distribution():
    m = EloModel(ratings={"A": 1700.0, "B": 1500.0})
    # give it a sane calibration
    m._theta0, m._theta1, m._beta = -0.4, 0.4, 0.003
    p = m.predict_proba("A", "B", neutral=True)
    assert len(p) == 3
    assert all(0 <= x <= 1 for x in p)
    assert abs(sum(p) - 1.0) < 1e-9
    # stronger team A should be favored
    assert p[0] > p[2]
