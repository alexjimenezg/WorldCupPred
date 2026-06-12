"""WorldCupPred — Streamlit operations console.

Tabs:
  Title odds     champion / stage probabilities for all 48 teams (bar chart + table)
  Update scores  enter a played result -> retrain the base -> simulate-from-now -> refresh
  Single match   any two teams (neutral toggle) -> ensemble W/D/L + scoreline heatmap
  Models         engine info, weights, and a quick walk-forward backtest
  Knowledge base browse the auto-generated Obsidian vault

Run:  streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.config import CONFIG

st.set_page_config(page_title="WorldCupPred 2026", page_icon="🏆", layout="wide")

_MODELS_READY = (CONFIG.models_dir / "elo_state.json").exists() and \
                (CONFIG.models_dir / "dixon_coles.json").exists()
_ODDS_PATH = CONFIG.processed / "title_odds.parquet"


@st.cache_resource(show_spinner="Loading models…")
def load_predictor(with_dl: bool):
    from src.models.ensemble import MatchPredictor
    return MatchPredictor.load_default(with_ml=True, with_dl=with_dl)


def load_table() -> pd.DataFrame | None:
    if _ODDS_PATH.exists():
        return pd.read_parquet(_ODDS_PATH)
    return None


def _store():
    from src.results_store import ResultsStore
    return ResultsStore()


def _results_count() -> int:
    return len(_store())


# ---------------------------------------------------------------------------
st.title("🏆 WorldCupPred — FIFA World Cup 2026")
st.caption("Statistical core (Elo + Dixon-Coles) + ML/DL ensemble → Monte Carlo tournament "
           "simulation. Enter results as they happen to refresh the title odds.")

if not _MODELS_READY:
    st.warning("Models not found. Run `python scripts/run_simulation.py` once to fit the "
               "models and produce the first title-odds table, then reload.")
    st.stop()

tab_odds, tab_update, tab_match, tab_models, tab_kb = st.tabs(
    ["Title odds", "Update scores", "Single match", "Models", "Knowledge base"])

# ---------------------------------------------------------------- Title odds
with tab_odds:
    table = load_table()
    if table is None:
        st.info("No simulation yet. Run `python scripts/run_simulation.py`.")
    else:
        store_n = _results_count()
        c1, c2, c3 = st.columns(3)
        c1.metric("Favorite", f"{table.iloc[0]['team']}",
                  f"{table.iloc[0]['champion']*100:.1f}% champion")
        c2.metric("Teams", "48")
        c3.metric("Results recorded", store_n)

        top = table.head(20)
        chart = top[["team", "champion"]].set_index("team") * 100
        st.subheader("Champion probability — top 20")
        st.bar_chart(chart, height=420, color="#1f9e63")

        st.subheader("Full stage probabilities")
        disp = table.copy()
        for col in ["champion", "final", "semifinal", "quarterfinal",
                    "round16", "round32", "win_group"]:
            disp[col] = (disp[col] * 100).round(1)
        st.dataframe(disp, width='stretch', hide_index=True,
                     column_config={c: st.column_config.NumberColumn(format="%.1f%%")
                                    for c in ["champion", "final", "semifinal",
                                              "quarterfinal", "round16", "round32", "win_group"]})

# ------------------------------------------------------------- Update scores
with tab_update:
    st.subheader("Record a result and refresh the odds")
    st.caption("Refits Elo + Dixon-Coles on the new result and re-simulates the remaining "
               "fixtures (already-played matches are held fixed).")

    mode = st.radio("Match type", ["Group stage", "Knockout"], horizontal=True)
    if mode == "Group stage":
        g = st.selectbox("Group", list(CONFIG.groups), key="upd_group")
        teams = CONFIG.groups[g]
        pairs = [(a, b) for i, a in enumerate(teams) for b in teams[i + 1:]]
        labels = [f"{a} vs {b}" for a, b in pairs]
        pick = st.selectbox("Fixture", labels, key="upd_fixture")
        home, away = pairs[labels.index(pick)]
        stage = "group"
    else:
        c1, c2 = st.columns(2)
        home = c1.selectbox("Home / Team A", CONFIG.teams, key="ko_home")
        away = c2.selectbox("Away / Team B", CONFIG.teams,
                            index=1, key="ko_away")
        stage = st.selectbox("Round", ["R32", "R16", "QF", "SF", "final"])

    c1, c2, c3 = st.columns([2, 1, 2])
    hg = c1.number_input(f"{home} goals", min_value=0, max_value=20, value=1, step=1)
    c2.markdown("<div style='text-align:center;padding-top:1.9rem'>—</div>",
                unsafe_allow_html=True)
    ag = c3.number_input(f"{away} goals", min_value=0, max_value=20, value=0, step=1)

    n_sims = st.select_slider("Simulations", options=[5000, 10000, 20000, 50000],
                              value=20000)
    retrain_ml = st.checkbox("Also retrain ML model (slower)", value=False)

    if st.button("Record result & re-simulate", type="primary", width='stretch'):
        if home == away:
            st.error("Pick two different teams.")
        else:
            from src import update as upd
            with st.spinner(f"Recording {home} {hg}-{ag} {away}, refitting and "
                            f"running {n_sims:,} simulations…"):
                store = upd.ResultsStore()
                store.add(home, away, int(hg), int(ag), stage=stage)
                new = upd.recompute(retrain_ml=retrain_ml, n_sims=int(n_sims), verbose=False)
            load_predictor.clear()
            st.success(f"Recorded {home} {hg}-{ag} {away} and refreshed odds.")
            show = new.head(10).copy()
            show["champion"] = (show["champion"] * 100).round(1)
            st.dataframe(show[["team", "group", "champion"]], hide_index=True,
                         width='stretch')

    st.divider()
    store = _store()
    st.write(f"**{len(store)} results recorded.**")
    if len(store):
        st.code(store.summary())
        cc1, cc2 = st.columns(2)
        if cc1.button("Undo last"):
            store.undo_last()
            st.rerun()
        if cc2.button("Reset all results"):
            store.clear()
            st.rerun()

# -------------------------------------------------------------- Single match
with tab_match:
    st.subheader("Head-to-head predictor")
    with_dl = st.toggle("Include deep-learning engine", value=False,
                        help="Adds the keras embedding model to the blend.")
    c1, c2, c3 = st.columns([3, 3, 2])
    home = c1.selectbox("Team A", CONFIG.teams, index=CONFIG.teams.index("Spain"))
    away = c2.selectbox("Team B", CONFIG.teams, index=CONFIG.teams.index("Brazil"))
    neutral = c3.toggle("Neutral venue", value=True)

    if home == away:
        st.info("Pick two different teams.")
    else:
        mp = load_predictor(with_dl)
        out = mp.predict(home, away, neutral)
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{home} win", f"{out['p_home']*100:.1f}%")
        c2.metric("Draw", f"{out['p_draw']*100:.1f}%")
        c3.metric(f"{away} win", f"{out['p_away']*100:.1f}%")
        c1, c2 = st.columns(2)
        c1.metric(f"{home} expected goals", f"{out['lambda_home']:.2f}")
        c2.metric(f"{away} expected goals", f"{out['lambda_away']:.2f}")

        sc = out["scoreline"][:7, :7]
        ij = np.unravel_index(np.argmax(out["scoreline"]), out["scoreline"].shape)
        st.caption(f"Most likely scoreline: **{home} {ij[0]}-{ij[1]} {away}**  "
                   f"(grid shows P(score), 0–6 goals)")
        heat = pd.DataFrame((sc * 100).round(1),
                            index=[f"{home} {i}" for i in range(7)],
                            columns=[f"{away} {j}" for j in range(7)])
        st.dataframe(heat.style.background_gradient(cmap="Greens", axis=None)
                     .format("{:.1f}"), width='stretch')

# ------------------------------------------------------------------- Models
with tab_models:
    st.subheader("Ensemble")
    weights = CONFIG.settings["ensemble"]["weights"]
    st.write("Blend weights (renormalized over available engines):")
    st.json(weights)
    import json
    try:
        dc = json.loads((CONFIG.models_dir / "dixon_coles.json").read_text())
        st.write(f"**Dixon-Coles:** base={dc['base']:.3f}, home_adv={dc['home_adv']:.3f}, "
                 f"rho={dc['rho']:.3f}, teams={len(dc['teams'])}, ref={dc['ref_date']}")
    except Exception:
        pass

    st.divider()
    st.write("**Walk-forward backtest** (train ≤ date, score held-out competitive matches; "
             "RPS lower is better):")
    train_end = st.text_input("Train cutoff", "2023-01-01")
    if st.button("Run backtest"):
        from src.evaluation.backtest import backtest
        with st.spinner("Training on the cutoff and scoring the holdout…"):
            res = backtest(train_end, "2025-12-31", include_ml=True, verbose=False)
        st.dataframe(res.round(4), width='stretch')

# ----------------------------------------------------------- Knowledge base
with tab_kb:
    st.subheader("Obsidian knowledge base")
    st.caption(f"Auto-generated under `{CONFIG.vault_dir}`. Open that folder as an Obsidian "
               "vault for the linked graph view.")
    notes = sorted(CONFIG.vault_dir.rglob("*.md"))
    if not notes:
        st.info("No vault yet — record a result or run the simulation to generate it.")
    else:
        rel = [str(p.relative_to(CONFIG.vault_dir)).replace("\\", "/") for p in notes]
        pick = st.selectbox(f"Browse {len(notes)} notes", rel,
                            index=rel.index("00-Index.md") if "00-Index.md" in rel else 0)
        st.markdown((CONFIG.vault_dir / pick).read_text(encoding="utf-8"))
