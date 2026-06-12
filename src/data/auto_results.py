"""Auto-import of played FIFA World Cup 2026 results — no API key required.

Two sources, unified by `sync_results`:
  1. football-data.org (src.data.live_scores) — preferred when FOOTBALL_DATA_API_KEY
     is set: official stage labels, available minutes after full time.
  2. martj42/international_results (the training backbone) — keyless fallback:
     re-download results.csv, keep "FIFA World Cup" matches on/after the opening match,
     infer the round from the official 2026 schedule windows, and upsert into the
     ResultsStore (store.add replaces same matchup+stage, so re-syncing is idempotent).

Known limitation: martj42 records the final score only, so a drawn knockout match decided
on penalties stays a draw in the store and the simulator re-randomizes the shootout
winner. football-data.org has the same fullTime-score shape; the third-place play-off is
skipped entirely (the simulated bracket has no bronze final).

CLI smoke test:  python -m src.data.auto_results
"""

from __future__ import annotations

import pandas as pd

from src.config import CONFIG
from src.results_store import ResultsStore

# Official 2026 knockout schedule windows (FIFA match calendar). Group stage runs
# 2026-06-11 .. 2026-06-27 (CONFIG.groups_raw["dates"]); knockout rounds follow.
_KO_WINDOWS: list[tuple[str, str, str | None]] = [
    ("2026-06-28", "2026-07-03", "R32"),
    ("2026-07-04", "2026-07-07", "R16"),
    ("2026-07-08", "2026-07-11", "QF"),
    ("2026-07-12", "2026-07-16", "SF"),
    ("2026-07-17", "2026-07-18", None),     # third-place play-off — not simulated
    ("2026-07-19", "2026-07-31", "final"),
]


def infer_stage(date, home: str, away: str) -> str | None:
    """Map a match date (+ the pairing) to the simulator's stage label, or None to skip."""
    d = pd.Timestamp(date)
    if d.tzinfo is not None:  # ESPN kickoffs are tz-aware UTC; windows are naive
        d = d.tz_convert("UTC").tz_localize(None)
    group_end = pd.Timestamp(CONFIG.groups_raw["dates"]["group_stage_end"])
    if d <= group_end:
        # same-group pairing is the defining property; anything else is noise
        return "group" if CONFIG.group_of(home) == CONFIG.group_of(away) else None
    for start, end, stage in _KO_WINDOWS:
        if pd.Timestamp(start) <= d <= pd.Timestamp(end):
            return stage
    return None


def fetch_wc2026_results(*, force: bool = True) -> pd.DataFrame:
    """Fresh martj42 results.csv filtered to played 2026 World Cup matches."""
    from src.data.kaggle_results import fetch_results
    df = fetch_results(force=force)
    opening = pd.Timestamp(CONFIG.groups_raw["dates"]["opening_match"])
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"] >= opening)]
    return wc.reset_index(drop=True)


def import_martj42_to_store(store: ResultsStore | None = None, *,
                            force: bool = True) -> int:
    """Upsert played WC 2026 matches from martj42 into the store. Returns rows applied."""
    store = store or ResultsStore()
    wc = fetch_wc2026_results(force=force)
    count = 0
    for _, m in wc.iterrows():
        home, away = m["home_team"], m["away_team"]
        if home not in CONFIG.teams or away not in CONFIG.teams:
            continue
        stage = infer_stage(m["date"], home, away)
        if stage is None:
            continue
        try:
            store.add(home, away, int(m["home_score"]), int(m["away_score"]),
                      stage=stage, on=str(pd.Timestamp(m["date"]).date()))
            count += 1
        except ValueError:
            continue  # cross-group oddity / name mismatch — leave for manual entry
    return count


def _snapshot(store: ResultsStore) -> dict[tuple, tuple]:
    """Order-agnostic {(matchup, stage): (scores aligned to sorted matchup)}."""
    snap = {}
    for r in store.results:
        a, b = sorted((r.home, r.away))
        scores = ((r.home_score, r.away_score) if r.home == a
                  else (r.away_score, r.home_score))
        snap[(a, b, r.stage)] = scores
    return snap


def sync_results(store: ResultsStore | None = None, *, force: bool = True) -> dict:
    """Pull results-to-date from every reachable source into the store.

    Returns {"n_changed": new-or-corrected results, "total": store size,
             "sources": which sources answered}.
    """
    store = store or ResultsStore()
    before = _snapshot(store)
    sources: list[str] = []

    from src.data import live_scores
    if live_scores.available():
        try:
            live_scores.import_finished_to_store(store=store)
            sources.append("football-data.org")
        except Exception as exc:  # network/key trouble — fall back, don't die
            print(f"[auto_results] football-data.org failed: {exc}")
    try:  # ESPN: keyless and updated within minutes of full time
        from src.data import espn_live
        espn_live.import_finished_to_store(store=store)
        sources.append("espn")
    except Exception as exc:
        print(f"[auto_results] espn failed: {exc}")
    try:
        import_martj42_to_store(store, force=force)
        sources.append("martj42")
    except Exception as exc:
        print(f"[auto_results] martj42 failed: {exc}")

    after = _snapshot(store)
    changed = sum(1 for k, v in after.items() if before.get(k) != v)
    return {"n_changed": changed, "total": len(store), "sources": sources}


if __name__ == "__main__":
    out = sync_results()
    print(f"synced from {', '.join(out['sources']) or 'no source reachable'}: "
          f"{out['n_changed']} new/changed, {out['total']} total in store")
