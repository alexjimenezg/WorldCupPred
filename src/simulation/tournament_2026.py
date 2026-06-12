"""The 2026 World Cup format: group standings (official tiebreakers), best-third
selection + allocation, and the knockout bracket — verified against the FIFA published
schedule (R32 ties 73-88, R16 89-96, QF 97-100, SF 101-102, Final 104).

This module is pure tournament logic. It takes a `sampler` (anything exposing
`sample_score` and `conditional_home_win`) so it can be unit-tested with a deterministic
fake and driven by the real ensemble in production. Venue/host-advantage rules live here;
match probabilities live in the sampler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.config import CONFIG

GROUP_LETTERS = list("ABCDEFGHIJKL")

# Round of 32: (tie_id, home_slot, away_slot). "1X"=winner X, "2X"=runner-up X,
# "3"=a best-third slot (its eligible groups are in THIRD_SLOT_GROUPS, keyed by tie_id).
BRACKET_R32: list[tuple[int, str, str]] = [
    (73, "2A", "2B"), (74, "1E", "3"), (75, "1F", "2C"), (76, "1C", "2F"),
    (77, "1I", "3"), (78, "2E", "2I"), (79, "1A", "3"), (80, "1L", "3"),
    (81, "1D", "3"), (82, "1G", "3"), (83, "2K", "2L"), (84, "1H", "2J"),
    (85, "1B", "3"), (86, "1J", "2H"), (87, "1K", "3"), (88, "2D", "2G"),
]
# Eligible group letters that can supply the best-third for each third-place slot.
THIRD_SLOT_GROUPS: dict[int, frozenset[str]] = {
    74: frozenset("ABCDF"), 77: frozenset("CDFGH"), 79: frozenset("CEFHI"),
    80: frozenset("EHIJK"), 81: frozenset("BEFIJ"), 82: frozenset("AEHIJ"),
    85: frozenset("EFGIJ"), 87: frozenset("DEIJL"),
}
BRACKET_R16 = [(89, 74, 77), (90, 73, 75), (91, 76, 78), (92, 79, 80),
               (93, 83, 84), (94, 81, 82), (95, 86, 88), (96, 85, 87)]
BRACKET_QF = [(97, 89, 90), (98, 93, 94), (99, 91, 92), (100, 95, 96)]
BRACKET_SF = [(101, 97, 98), (102, 99, 100)]
FINAL = (104, 101, 102)

STAGES = ["group", "R32", "R16", "QF", "SF", "final", "champion"]


class Sampler(Protocol):
    def sample_score(self, home: str, away: str, neutral: bool) -> tuple[int, int]: ...
    def conditional_home_win(self, home: str, away: str, neutral: bool) -> float: ...


# ---------------------------------------------------------------------------
# Group stage
# ---------------------------------------------------------------------------
_ROUND_ROBIN = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)]


def _is_host(team: str) -> bool:
    return CONFIG.is_host(team)


def _venue_home(team_a: str, team_b: str) -> tuple[str, str, bool]:
    """Order a group fixture so a host (if any) is the home side; return (home, away, neutral)."""
    a_host, b_host = _is_host(team_a), _is_host(team_b)
    if a_host and not b_host:
        return team_a, team_b, False
    if b_host and not a_host:
        return team_b, team_a, False
    return team_a, team_b, True  # neither or both hosts -> neutral


@dataclass
class TeamRow:
    team: str
    pts: int = 0
    gf: int = 0
    ga: int = 0
    @property
    def gd(self) -> int:
        return self.gf - self.ga


def simulate_group(teams: list[str], sampler: Sampler, rng: np.random.Generator,
                   fixed: dict | None = None) -> tuple[list[str], dict[str, TeamRow]]:
    """Play the 6 round-robin games; return (teams ranked 1st..4th, row per team)."""
    rows = {t: TeamRow(t) for t in teams}
    results: dict[tuple[str, str], tuple[int, int]] = {}
    for ii, jj in _ROUND_ROBIN:
        ta, tb = teams[ii], teams[jj]
        home, away, neutral = _venue_home(ta, tb)
        hg = ag = None
        if fixed is not None:
            got = fixed.get((home, away)) or (
                (lambda r: (r[1], r[0]) if r else None)(fixed.get((away, home))))
            if got is not None:
                hg, ag = got
        if hg is None:
            hg, ag = sampler.sample_score(home, away, neutral)
        results[(home, away)] = (hg, ag)
        rh, ra = rows[home], rows[away]
        rh.gf += hg; rh.ga += ag; ra.gf += ag; ra.ga += hg
        if hg > ag:
            rh.pts += 3
        elif hg < ag:
            ra.pts += 3
        else:
            rh.pts += 1; ra.pts += 1
    return _rank(list(rows.values()), results, rng), rows


def _h2h_key(team: str, members: set[str], results: dict) -> tuple[int, int, int]:
    pts = gf = ga = 0
    for (h, a), (hg, ag) in results.items():
        if h in members and a in members:
            if team == h:
                gf += hg; ga += ag; pts += 3 if hg > ag else (1 if hg == ag else 0)
            elif team == a:
                gf += ag; ga += hg; pts += 3 if ag > hg else (1 if hg == ag else 0)
    return (pts, gf - ga, gf)


def _rank(rows: list[TeamRow], results: dict, rng: np.random.Generator) -> list[str]:
    # primary: points, GD, GF
    rows = sorted(rows, key=lambda r: (r.pts, r.gd, r.gf), reverse=True)
    out: list[str] = []
    i = 0
    while i < len(rows):
        j = i + 1
        while j < len(rows) and (rows[j].pts, rows[j].gd, rows[j].gf) == \
                (rows[i].pts, rows[i].gd, rows[i].gf):
            j += 1
        tied = rows[i:j]
        if len(tied) == 1:
            out.append(tied[0].team)
        else:
            members = {r.team for r in tied}
            ranked = sorted(
                tied,
                key=lambda r: (*_h2h_key(r.team, members, results), rng.random()),
                reverse=True,
            )
            out.extend(r.team for r in ranked)
        i = j
    return out


# ---------------------------------------------------------------------------
# Best-third allocation (bipartite matching honoring each slot's eligible groups)
# ---------------------------------------------------------------------------
def allocate_thirds(qualifying_groups: list[str]) -> dict[int, str]:
    """Assign the 8 qualifying third-placed groups to the 8 third-place slots.

    A perfect matching always exists by FIFA's slot design; we pick the min-cost one with
    a deterministic tie-break (slot id, group letter) so the bracket is reproducible.
    """
    slots = sorted(THIRD_SLOT_GROUPS)  # 8 tie ids
    groups = list(qualifying_groups)
    big = 1000.0
    cost = np.full((len(slots), len(groups)), big)
    for si, s in enumerate(slots):
        for gi, g in enumerate(groups):
            if g in THIRD_SLOT_GROUPS[s]:
                cost[si, gi] = si * 0.001 + gi * 0.0001  # tiny deterministic preference
    r, c = linear_sum_assignment(cost)
    if any(cost[ri, ci] >= big for ri, ci in zip(r, c)):
        raise ValueError(f"no valid third-place allocation for {qualifying_groups}")
    return {slots[ri]: groups[ci] for ri, ci in zip(r, c)}


def select_best_thirds(thirds: dict[str, str], stats: dict[str, TeamRow]
                       ) -> tuple[list[str], dict[int, str]]:
    """thirds: group_letter -> team. Pick the best 8 by (pts, gd, gf); allocate to slots."""
    ranked = sorted(thirds, key=lambda g: (stats[g].pts, stats[g].gd, stats[g].gf),
                    reverse=True)
    qualifying = sorted(ranked[:8])
    slot_to_group = allocate_thirds(qualifying)
    return qualifying, slot_to_group


# ---------------------------------------------------------------------------
# Knockouts
# ---------------------------------------------------------------------------
def _play_knockout(home: str, away: str, sampler: Sampler, rng: np.random.Generator,
                   fixed: tuple[int, int] | None = None) -> str:
    neutral = True  # knockouts at neutral venues
    if fixed is not None:
        hg, ag = fixed
    else:
        hg, ag = sampler.sample_score(home, away, neutral)
    if hg > ag:
        return home
    if ag > hg:
        return away
    p = sampler.conditional_home_win(home, away, neutral)  # ET/penalties
    return home if rng.random() < p else away


@dataclass
class SimResult:
    champion: str = ""
    finalists: tuple[str, str] = ("", "")
    reached: dict[str, str] = field(default_factory=dict)  # team -> furthest stage
    group_winners: list[str] = field(default_factory=list)


def simulate_tournament(sampler: Sampler, rng: np.random.Generator,
                        fixed: dict | None = None) -> SimResult:
    """One full tournament. `fixed` may contain played group scores keyed by (home,away)
    and knockout results keyed by tie_id -> (home_goals, away_goals)."""
    fixed = fixed or {}
    fixed_groups = fixed.get("groups")
    fixed_ko = fixed.get("knockouts", {})

    reached: dict[str, str] = {t: "group" for t in CONFIG.teams}
    standings: dict[str, list[str]] = {}
    group_stats: dict[str, TeamRow] = {}   # group_letter -> 3rd-place TeamRow
    thirds: dict[str, str] = {}
    slot_team: dict[str, str] = {}

    for g, teams in CONFIG.groups.items():
        order, rows = simulate_group(teams, sampler, rng, fixed_groups)
        standings[g] = order
        slot_team[f"1{g}"] = order[0]
        slot_team[f"2{g}"] = order[1]
        thirds[g] = order[2]
        group_stats[g] = rows[order[2]]   # the 3rd-placed team's actual row
        for t in order[:2]:
            reached[t] = "R32"

    qualifying, slot_to_group = select_best_thirds(thirds, group_stats)
    for g in qualifying:
        reached[thirds[g]] = "R32"
    for tie_id, g in slot_to_group.items():
        slot_team[f"3@{tie_id}"] = thirds[g]

    # Round of 32
    winners: dict[int, str] = {}
    for tie_id, hs, as_ in BRACKET_R32:
        home = slot_team[hs]
        away = slot_team[as_] if as_ != "3" else slot_team[f"3@{tie_id}"]
        w = _play_knockout(home, away, sampler, rng, fixed_ko.get(tie_id))
        winners[tie_id] = w
        for t in (home, away):
            reached[t] = "R32"
        reached[w] = "R16"

    winners = _run_round(BRACKET_R16, winners, slot_team, sampler, rng, fixed_ko, reached, "QF")
    winners = _run_round(BRACKET_QF, winners, slot_team, sampler, rng, fixed_ko, reached, "SF")
    winners = _run_round(BRACKET_SF, winners, slot_team, sampler, rng, fixed_ko, reached, "final")

    f_id, s1, s2 = FINAL
    fa, fb = winners[s1], winners[s2]
    reached[fa] = reached[fb] = "final"
    champ = _play_knockout(fa, fb, sampler, rng, fixed_ko.get(f_id))
    reached[champ] = "champion"
    group_winners = [standings[g][0] for g in CONFIG.groups]
    return SimResult(champion=champ, finalists=(fa, fb), reached=reached,
                     group_winners=group_winners)


def _run_round(bracket, winners, slot_team, sampler, rng, fixed_ko, reached, next_stage):
    new = dict(winners)
    for tie_id, a_src, b_src in bracket:
        home, away = winners[a_src], winners[b_src]
        w = _play_knockout(home, away, sampler, rng, fixed_ko.get(tie_id))
        new[tie_id] = w
        reached[w] = next_stage
    return new
