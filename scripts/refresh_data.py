"""Refresh all data sources and rebuild the processed tables, then assert integrity.

Usage:
    python scripts/refresh_data.py            # use cache where fresh
    python scripts/refresh_data.py --force    # re-download everything
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script (`python scripts/refresh_data.py`).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CONFIG  # noqa: E402
from src.data import build_dataset  # noqa: E402
from src.data.kaggle_results import fetch_goalscorers, fetch_shootouts  # noqa: E402


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> int:
    ap = argparse.ArgumentParser(description="Refresh WorldCupPred data")
    ap.add_argument("--force", action="store_true", help="ignore cache, re-download")
    args = ap.parse_args()

    print("=" * 70)
    print("WorldCupPred — data refresh")
    print("=" * 70)

    matches = build_dataset.build_matches(force=args.force)
    ref = build_dataset.build_team_reference(force=args.force)
    # Pull the auxiliary tables too (cached), so they are on disk for later use.
    fetch_goalscorers(force=args.force)
    fetch_shootouts(force=args.force)

    # ---- integrity checks (plan verification step 1) -----------------------
    print("\nrunning integrity checks...")
    _assert(len(matches) > 45_000, f"too few matches: {len(matches)}")
    _assert(matches["date"].max().year >= 2026, "data does not reach 2026")
    _assert(matches[["home_score", "away_score"]].notna().all().all(), "null scores present")
    _assert(len(CONFIG.teams) == 48, "expected 48 qualified teams")
    _assert(len(set(CONFIG.teams)) == 48, "duplicate team in the draw")
    _assert(ref["elo_external"].notna().all(), "a qualified team has no Elo seed")
    _assert(set(ref["team"]) == set(CONFIG.teams), "team_reference != draw")

    print("  OK  matches:        {:>7,}".format(len(matches)))
    print("  OK  date range:     {} -> {}".format(
        matches["date"].min().date(), matches["date"].max().date()))
    print("  OK  qualified teams:{:>7}".format(len(CONFIG.teams)))
    print("  OK  all teams have an Elo seed and a group")
    print("\nprocessed artifacts in:", CONFIG.processed)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
