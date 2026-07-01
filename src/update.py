"""Live tournament loop: add a result -> refit -> simulate-from-now -> refreshed odds.

CLI:
    python -m src.update --sync                  # pull results to-date, refit, re-simulate
    python -m src.update --match "Spain 3-1 Cape Verde"
    python -m src.update --match "Brazil 2-0 Scotland" --retrain-ml -n 30000
    python -m src.update --list | --undo | --recompute | --reset

The same `apply_result` / `recompute` functions back the Streamlit app's Update tab.
Refitting Elo + Dixon-Coles on the augmented data is the "retrain the base"; ML is
retrained on request (slower). Already-played matches are fixed in the simulation so only
the remaining fixtures are randomized.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CONFIG  # noqa: E402
from src.models.dixon_coles import DixonColes  # noqa: E402
from src.models.elo import EloModel  # noqa: E402
from src.models.ensemble import MatchPredictor  # noqa: E402
from src.results_store import ResultsStore  # noqa: E402
from src.simulation.monte_carlo import run_simulation  # noqa: E402

_RESULT_RE = re.compile(r"^\s*(.+?)\s+(\d+)\s*[-:–]\s*(\d+)\s+(.+?)\s*$")


def parse_result(text: str) -> tuple[str, str, int, int]:
    m = _RESULT_RE.match(text)
    if not m:
        raise ValueError(f'could not parse "{text}" (expected e.g. "Spain 3-1 Cape Verde")')
    home, hg, ag, away = m.groups()
    return CONFIG.normalize(home), CONFIG.normalize(away), int(hg), int(ag)


def augmented_matches(store: ResultsStore) -> pd.DataFrame:
    """matches.parquet + the WC results recorded so far (for retraining)."""
    base = pd.read_parquet(CONFIG.processed / "matches.parquet")
    extra = store.to_match_rows()
    if extra.empty:
        return base
    cols = base.columns
    # keep="last" guards against double-counting once a refreshed matches.parquet
    # starts to contain the same WC 2026 matches the store holds
    return (pd.concat([base, extra[cols]], ignore_index=True)
              .drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
              .sort_values("date").reset_index(drop=True))


def refit_core(matches: pd.DataFrame) -> tuple[EloModel, DixonColes]:
    elo = EloModel().fit(matches)
    elo.save()
    elo.history.to_parquet(CONFIG.processed / "matches_elo.parquet", index=False)
    dc = DixonColes().fit(matches)
    dc.save()
    return elo, dc


def refit_ml(elo: EloModel):
    from src.features.build_features import build_features
    from src.models.ml_outcome import MLOutcome
    feats = build_features(elo.history, save=True)
    ml = MLOutcome().fit(feats, elo=elo)
    ml.save()
    return ml


def _regen_vault(table: pd.DataFrame, store: ResultsStore) -> None:
    try:
        from src.vault.generate_notes import generate_vault
        generate_vault(table, store)
    except ImportError:
        pass  # vault generator arrives in P6


def recompute(*, retrain_ml: bool = False, with_dl: bool = False,
              n_sims: int | None = None, seed: int | None = None,
              regen_vault: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Refit the base models on augmented data and re-simulate from now."""
    store = ResultsStore()
    # make sure penalty-shootout winners are recorded before we build `fixed`,
    # whichever source added the tie — otherwise the sim re-randomizes it.
    try:
        from src.data.auto_results import import_shootouts_to_store
        import_shootouts_to_store(store, force=False)
    except Exception as exc:  # network hiccup — proceed with what we have
        if verbose:
            print(f"  (shootout reconcile skipped: {exc})")
    matches = augmented_matches(store)
    if verbose:
        print(f"refitting on {len(matches):,} matches "
              f"(+{len(store)} WC results)...")
    elo, dc = refit_core(matches)
    ml = refit_ml(elo) if retrain_ml else None
    predictor = MatchPredictor.load_default(with_ml=True, with_dl=with_dl,
                                            ml=ml, dl=None)
    fixed = store.to_fixed()
    try:  # pin R32 pairings to the published draw so fixed results land right
        from src.data.auto_results import knockout_fixtures
        fixed["r32_draw"] = knockout_fixtures(force=False).get("R32", {})
    except Exception:
        fixed["r32_draw"] = {}
    table = run_simulation(predictor, n=n_sims, seed=seed, fixed=fixed,
                           progress=verbose)
    if regen_vault:
        _regen_vault(table, store)
    return table


def sync_and_recompute(*, retrain_ml: bool = False, n_sims: int | None = None,
                       verbose: bool = True) -> tuple[dict, pd.DataFrame | None]:
    """Pull results-to-date from all reachable sources; refit + re-simulate if anything
    new arrived. Returns (sync summary, new odds table or None if nothing changed)."""
    from src.data.auto_results import sync_results
    info = sync_results()
    if verbose:
        srcs = ", ".join(info["sources"]) or "no source reachable"
        print(f"sync [{srcs}]: {info['n_changed']} new/changed, "
              f"{info['total']} results total")
    if not info["n_changed"]:
        return info, None
    table = recompute(retrain_ml=retrain_ml, n_sims=n_sims, verbose=verbose)
    return info, table


def apply_result(text: str, *, stage: str = "group", retrain_ml: bool = False,
                 n_sims: int | None = None, verbose: bool = True) -> pd.DataFrame:
    home, away, hg, ag = parse_result(text)
    store = ResultsStore()
    r = store.add(home, away, hg, ag, stage=stage)
    if verbose:
        print(f"recorded: [{r.stage}] {r.home} {r.home_score}-{r.away_score} {r.away}")
    return recompute(retrain_ml=retrain_ml, n_sims=n_sims, verbose=verbose)


def _print_top(table: pd.DataFrame, k: int = 12) -> None:
    show = table.head(k).copy()
    for c in ["champion", "final", "quarterfinal"]:
        show[c] = (show[c] * 100).round(1)
    print(show[["team", "group", "champion", "final", "quarterfinal"]].to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser(description="WorldCupPred live update")
    ap.add_argument("--sync", action="store_true",
                    help="auto-import results to-date, then refit + re-simulate")
    ap.add_argument("--match", help='result string, e.g. "Spain 3-1 Cape Verde"')
    ap.add_argument("--stage", default="group",
                    help="group | R32 | R16 | QF | SF | final")
    ap.add_argument("--retrain-ml", action="store_true", help="also retrain the ML model")
    ap.add_argument("-n", "--iterations", type=int, default=None)
    ap.add_argument("--list", action="store_true", help="list recorded results")
    ap.add_argument("--undo", action="store_true", help="remove the last result")
    ap.add_argument("--recompute", action="store_true",
                    help="refit + resim without adding a result")
    ap.add_argument("--reset", action="store_true", help="clear all recorded results")
    args = ap.parse_args()

    store = ResultsStore()
    if args.list:
        print(f"{len(store)} results recorded:\n{store.summary()}")
        return 0
    if args.reset:
        store.clear()
        print("cleared all recorded results")
        return 0
    if args.undo:
        r = store.undo_last()
        print(f"removed: {r}" if r else "nothing to undo")
        if r:
            _print_top(recompute(n_sims=args.iterations))
        return 0
    if args.sync:
        info, table = sync_and_recompute(retrain_ml=args.retrain_ml,
                                         n_sims=args.iterations)
        if table is not None:
            print("\nupdated title odds (top 12):")
            _print_top(table)
        else:
            print("store already up to date — nothing to recompute")
        return 0
    if args.match:
        table = apply_result(args.match, stage=args.stage,
                             retrain_ml=args.retrain_ml, n_sims=args.iterations)
        print("\nupdated title odds (top 12):")
        _print_top(table)
        return 0
    if args.recompute:
        table = recompute(retrain_ml=args.retrain_ml, n_sims=args.iterations)
        print("\ntitle odds (top 12):")
        _print_top(table)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
