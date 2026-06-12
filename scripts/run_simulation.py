"""End-to-end: ensure data + models exist, fit if needed, run the Monte Carlo, report.

Usage:
    python scripts/run_simulation.py                 # default iterations (settings.yaml)
    python scripts/run_simulation.py -n 50000        # explicit iteration count
    python scripts/run_simulation.py --refit         # refit Elo + Dixon-Coles first
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CONFIG  # noqa: E402
from src.data import build_dataset  # noqa: E402
from src.models.dixon_coles import DixonColes, build_dixon_coles  # noqa: E402
from src.models.elo import EloModel, build_elo  # noqa: E402
from src.models.ensemble import MatchPredictor  # noqa: E402
from src.simulation.monte_carlo import run_simulation  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the WorldCupPred simulation")
    ap.add_argument("-n", "--iterations", type=int, default=None)
    ap.add_argument("--refit", action="store_true", help="refit all models")
    ap.add_argument("--with-dl", action="store_true", help="include keras DL in the ensemble")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if not (CONFIG.processed / "matches.parquet").exists():
        print("matches.parquet missing -> building dataset")
        build_dataset.main()

    elo_path = CONFIG.models_dir / "elo_state.json"
    dc_path = CONFIG.models_dir / "dixon_coles.json"
    ml_path = CONFIG.models_dir / "ml_outcome.joblib"
    if args.refit or not elo_path.exists():
        build_elo()
    if args.refit or not dc_path.exists():
        build_dixon_coles()
    if args.refit or not ml_path.exists():
        from src.models.ml_outcome import build_ml_outcome
        build_ml_outcome()
    if args.with_dl and (args.refit or not (CONFIG.models_dir / "dl_outcome.keras").exists()):
        from src.models.dl_outcome import build_dl_outcome
        build_dl_outcome()

    predictor = MatchPredictor.load_default(with_ml=True, with_dl=args.with_dl)
    print(f"\nrunning Monte Carlo ({args.iterations or CONFIG.settings['simulation']['n_iterations']:,} "
          f"iterations)...")
    table = run_simulation(predictor, n=args.iterations, seed=args.seed)

    print("\nFIFA World Cup 2026 — title odds (top 20):")
    show = table.head(20).copy()
    for c in ["champion", "final", "semifinal", "quarterfinal", "win_group"]:
        show[c] = (show[c] * 100).round(1)
    print(show[["team", "group", "champion", "final", "semifinal",
                "quarterfinal", "win_group"]].to_string(index=False))
    print(f"\nartifacts: {CONFIG.processed/'title_odds.parquet'} , {CONFIG.path('reports')/'title_odds.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
