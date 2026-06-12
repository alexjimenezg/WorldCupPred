"""Generate an Obsidian-compatible knowledge base from the latest data + simulation.

Emits plain-markdown notes with YAML frontmatter and [[wikilinks]] so Obsidian's graph view
renders team <-> group <-> confederation <-> model <-> simulation relationships. Re-run after
every update so the vault stays in sync (called automatically by src/update.recompute).

    from src.vault.generate_notes import generate_vault
    generate_vault()              # uses saved title_odds + models
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import CONFIG
from src.results_store import ResultsStore

_PCT = ["champion", "final", "semifinal", "quarterfinal", "round16", "round32", "win_group"]


def _fm(d: dict) -> str:
    lines = ["---"]
    for k, v in d.items():
        lines.append(f"{k}: {v}")
    lines.append("---")
    return "\n".join(lines)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def generate_vault(table: pd.DataFrame | None = None,
                   store: ResultsStore | None = None) -> Path:
    vault = CONFIG.vault_dir
    table = table if table is not None else pd.read_parquet(CONFIG.processed / "title_odds.parquet")
    ref = pd.read_parquet(CONFIG.processed / "teams_reference.parquet").set_index("team")
    store = store or ResultsStore()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    sim_name = f"Simulation {datetime.now().strftime('%Y-%m-%d %H%M')}"
    t = table.set_index("team")

    _team_notes(vault, table, ref, sim_name, stamp)
    _group_notes(vault, table, sim_name, stamp)
    _confederation_notes(vault, table, stamp)
    _model_notes(vault, stamp)
    _data_source_notes(vault, stamp)
    _methodology_note(vault, stamp)
    sim_path = _simulation_note(vault, table, store, sim_name, stamp)
    _index_note(vault, table, store, sim_name, stamp)
    return sim_path


def _team_notes(vault, table, ref, sim_name, stamp) -> None:
    t = table.set_index("team")
    for team in CONFIG.teams:
        g = CONFIG.group_of(team)
        confed = CONFIG.confederation_of(team)
        row = t.loc[team]
        meta = ref.loc[team] if team in ref.index else None
        elo = f"{meta['elo_external']:.0f}" if meta is not None else "n/a"
        rank = f"{int(meta['fifa_rank'])}" if meta is not None and pd.notna(meta["fifa_rank"]) else "n/a"
        rivals = ", ".join(f"[[{x}]]" for x in CONFIG.groups[g] if x != team)
        fm = _fm({
            "type": "team", "team": team, "group": g, "confederation": confed,
            "elo": elo, "fifa_rank": rank, "host": CONFIG.is_host(team),
            "champion_pct": round(float(row["champion"]) * 100, 2),
            "updated": stamp,
        })
        body = f"""# {team}

**Group [[Group {g}]]** · Confederation [[{confed}]] · Elo {elo} · World rank {rank}\
{' · 🏠 host nation' if CONFIG.is_host(team) else ''}

## Title odds (latest)
| Stage | Probability |
| --- | --- |
| Champion | {_pct(row['champion'])} |
| Reach final | {_pct(row['final'])} |
| Reach semi-final | {_pct(row['semifinal'])} |
| Reach quarter-final | {_pct(row['quarterfinal'])} |
| Reach round of 16 | {_pct(row['round16'])} |
| Reach round of 32 | {_pct(row['round32'])} |
| Win [[Group {g}]] | {_pct(row['win_group'])} |

## Group {g} rivals
{rivals}

## Links
Confederation [[{confed}]] · run [[{sim_name}]] · [[00-Index|Index]]
"""
        _write(vault / "teams" / f"{team}.md", _fm_join(fm, body))


def _group_notes(vault, table, sim_name, stamp) -> None:
    t = table.set_index("team")
    for g, teams in CONFIG.groups.items():
        rows = "\n".join(
            f"| [[{tm}]] | {_pct(t.loc[tm, 'win_group'])} | {_pct(t.loc[tm, 'round32'])} "
            f"| {_pct(t.loc[tm, 'champion'])} |"
            for tm in sorted(teams, key=lambda x: -t.loc[x, "champion"]))
        fm = _fm({"type": "group", "group": g, "updated": stamp})
        body = f"""# Group {g}

Teams: {', '.join(f'[[{x}]]' for x in teams)}

| Team | Win group | Reach R32 | Champion |
| --- | --- | --- | --- |
{rows}

Run [[{sim_name}]] · [[00-Index|Index]]
"""
        _write(vault / "groups" / f"Group {g}.md", _fm_join(fm, body))


def _confederation_notes(vault, table, stamp) -> None:
    t = table.set_index("team")
    confeds: dict[str, list[str]] = {}
    for tm in CONFIG.teams:
        confeds.setdefault(CONFIG.confederation_of(tm), []).append(tm)
    for confed, teams in confeds.items():
        teams = sorted(teams, key=lambda x: -t.loc[x, "champion"])
        total = sum(t.loc[x, "champion"] for x in teams)
        rows = "\n".join(f"| [[{tm}]] | [[Group {CONFIG.group_of(tm)}]] | "
                         f"{_pct(t.loc[tm, 'champion'])} |" for tm in teams)
        fm = _fm({"type": "confederation", "confederation": confed,
                  "n_teams": len(teams), "combined_champion_pct": round(total * 100, 1),
                  "updated": stamp})
        body = f"""# {confed}

{len(teams)} qualified teams · combined title probability {_pct(total)}

| Team | Group | Champion |
| --- | --- | --- |
{rows}

[[00-Index|Index]]
"""
        _write(vault / "confederations" / f"{confed}.md", _fm_join(fm, body))


def _model_notes(vault, stamp) -> None:
    import json
    cards = {
        "Elo": "International Elo computed from match results (World-Football-Elo weighting). "
               "Ordered-logit head turns the rating gap into calibrated W/D/L. "
               "See [[Dixon-Coles]], [[Ensemble]].",
        "Dixon-Coles": "Bivariate-Poisson goal model (attack/defense/home + low-score rho), "
                       "time-decay weighted. Primary match engine; yields the scoreline "
                       "distribution the [[Ensemble]] and simulator sample. See [[Elo]].",
        "ML-HistGB": "sklearn HistGradientBoosting W/D/L classifier on Elo + form + context "
                     "features. Non-linear ensemble member. See [[Ensemble]].",
        "DL-Embedding-MLP": "keras MLP with learned per-nation embeddings + numeric features "
                            "-> softmax W/D/L. See [[Ensemble]].",
        "Ensemble": "Weighted blend of [[Elo]], [[Dixon-Coles]], [[ML-HistGB]], "
                    "[[DL-Embedding-MLP]]; scoreline rescaled to the blended marginals and "
                    "fed to the [[Methodology|Monte Carlo]] simulator.",
    }
    # enrich the two stat cards with live params if available
    extra = {}
    try:
        dc = json.loads((CONFIG.models_dir / "dixon_coles.json").read_text())
        extra["Dixon-Coles"] = (f"\n\n**Params:** base={dc['base']:.3f}, "
                                f"home_adv={dc['home_adv']:.3f}, rho={dc['rho']:.3f}, "
                                f"teams={len(dc['teams'])}, ref={dc['ref_date']}.")
    except Exception:
        pass
    for name, desc in cards.items():
        fm = _fm({"type": "model", "model": name, "updated": stamp})
        _write(vault / "models" / f"{name}.md",
               _fm_join(fm, f"# {name}\n\n{desc}{extra.get(name, '')}\n\n[[00-Index|Index]]"))


def _data_source_notes(vault, stamp) -> None:
    sources = {
        "martj42-results": "All international matches 1872->present (the training backbone). "
                           "Free GitHub-raw CSVs. Feeds [[Elo]], [[Dixon-Coles]], [[ML-HistGB]].",
        "eloratings": "eloratings.net current international Elo (cross-check / seed).",
        "fifa-rankings": "Team ranking feature (Elo-proxy fallback when no full table).",
        "the-odds-api": "Betting odds via free-tier key (calibration anchor + benchmark). Optional.",
        "football-data-org": "2026 fixtures & live scores via free-tier key. Optional.",
    }
    for name, desc in sources.items():
        fm = _fm({"type": "data-source", "source": name, "updated": stamp})
        _write(vault / "data-sources" / f"{name}.md",
               _fm_join(fm, f"# {name}\n\n{desc}\n\n[[00-Index|Index]]"))


def _methodology_note(vault, stamp) -> None:
    fm = _fm({"type": "methodology", "updated": stamp})
    body = """# Methodology

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
"""
    _write(vault / "methodology" / "Methodology.md", _fm_join(fm, body))


def _simulation_note(vault, table, store, sim_name, stamp) -> Path:
    top = table.head(24)
    rows = "\n".join(
        f"| {i+1} | [[{r.team}]] | [[Group {r.group}]] | {_pct(r.champion)} "
        f"| {_pct(r.final)} | {_pct(r.quarterfinal)} |"
        for i, r in enumerate(top.itertuples(index=False)))
    played = (f"\n\n## Results fixed in this run ({len(store)})\n" +
              "\n".join(f"- {x.home} {x.home_score}-{x.away_score} {x.away} ({x.stage})"
                        for x in store.results)) if len(store) else ""
    fm = _fm({"type": "simulation", "generated": stamp, "results_fixed": len(store)})
    body = f"""# {sim_name}

Champion probability (top 24){' — ' + str(len(store)) + ' results fixed' if len(store) else ''}.

| # | Team | Group | Champion | Final | Quarter-final |
| --- | --- | --- | --- | --- | --- |
{rows}
{played}

[[00-Index|Index]]
"""
    path = vault / "simulations" / f"{sim_name}.md"
    _write(path, _fm_join(fm, body))
    _write(vault / "simulations" / "Latest.md",
           _fm_join(_fm({"type": "pointer", "updated": stamp}),
                    f"# Latest simulation\n\n-> [[{sim_name}]]"))
    return path


def _index_note(vault, table, store, sim_name, stamp) -> None:
    top5 = " · ".join(f"[[{r.team}]] {_pct(r.champion)}"
                      for r in table.head(5).itertuples(index=False))
    fm = _fm({"type": "index", "updated": stamp, "favorite": table.iloc[0]["team"]})
    body = f"""# WorldCupPred — Knowledge Base

_Updated {stamp}. {len(store)} World Cup results recorded._

**Current favorites:** {top5}

## Navigate
- [[Methodology]] · latest run [[{sim_name}]] (or [[Latest]])
- Teams: {', '.join(f'[[{x}]]' for x in table.head(8)['team'])} … (48 total in `teams/`)
- Groups: {' '.join(f'[[Group {g}]]' for g in CONFIG.groups)}
- Confederations: {' · '.join(f'[[{c}]]' for c in ['UEFA','CONMEBOL','CAF','AFC','CONCACAF','OFC'])}
- Models: [[Elo]] · [[Dixon-Coles]] · [[ML-HistGB]] · [[DL-Embedding-MLP]] · [[Ensemble]]
- Data: [[martj42-results]] · [[eloratings]] · [[fifa-rankings]] · [[the-odds-api]] · [[football-data-org]]

Open this folder as an Obsidian vault and use graph view to explore the links.
"""
    _write(vault / "00-Index.md", _fm_join(fm, body))


def _fm_join(frontmatter: str, body: str) -> str:
    return frontmatter + "\n\n" + body


if __name__ == "__main__":
    p = generate_vault()
    n = sum(1 for _ in CONFIG.vault_dir.rglob("*.md"))
    print(f"[vault] wrote {n} notes under {CONFIG.vault_dir}")
    print(f"latest simulation note: {p.name}")
