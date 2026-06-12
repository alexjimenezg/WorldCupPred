"""WorldCupPred — Streamlit operations console (FIFA World Cup 26 design).

Tabs:
  Live           ESPN live scoreboard joined to the model: in-play conditional W/D/L,
                 stats, upcoming + played boards (auto-refreshes every minute)
  Title odds     champion / stage probabilities for all 48 teams (plotly + table)
  Groups         live standings from recorded results + advance probabilities
  Bracket        most-likely Round-of-32 line-up + road-to-title funnel per team
  History        champion odds over time (one snapshot per saved simulation)
  Single match   any two teams (neutral toggle) -> ensemble W/D/L + scoreline heatmap
  Models         engine info, weights, and a quick walk-forward backtest
  Knowledge base browse the auto-generated Obsidian vault

Results sync (sidebar / on launch) pulls from ESPN + martj42 + football-data.org;
manual entry still exists via `python -m src.update --match ...`.

Run:  streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import CONFIG

st.set_page_config(page_title="WorldCupPred 2026", page_icon="🏆", layout="wide")

_MODELS_READY = (CONFIG.models_dir / "elo_state.json").exists() and \
                (CONFIG.models_dir / "dixon_coles.json").exists()
import importlib.util
_HAS_DL = (importlib.util.find_spec("keras") is not None
           and (CONFIG.models_dir / "dl_outcome.keras").exists())
_ODDS_PATH = CONFIG.processed / "title_odds.parquet"
_HISTORY_PATH = CONFIG.processed / "odds_history.parquet"

# WC26 tri-host palette
RED, GREEN, BLUE = "#e11d48", "#10b981", "#3b82f6"
_STAGE_COLS = ["champion", "final", "semifinal", "quarterfinal",
               "round16", "round32", "win_group"]

FLAGS = {
    "Mexico": "🇲🇽", "South Africa": "🇿🇦", "South Korea": "🇰🇷", "Czech Republic": "🇨🇿",
    "Canada": "🇨🇦", "Bosnia and Herzegovina": "🇧🇦", "Qatar": "🇶🇦", "Switzerland": "🇨🇭",
    "Brazil": "🇧🇷", "Morocco": "🇲🇦", "Haiti": "🇭🇹", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "United States": "🇺🇸", "Paraguay": "🇵🇾", "Australia": "🇦🇺", "Turkey": "🇹🇷",
    "Germany": "🇩🇪", "Curacao": "🇨🇼", "Ivory Coast": "🇨🇮", "Ecuador": "🇪🇨",
    "Netherlands": "🇳🇱", "Japan": "🇯🇵", "Sweden": "🇸🇪", "Tunisia": "🇹🇳",
    "Belgium": "🇧🇪", "Egypt": "🇪🇬", "Iran": "🇮🇷", "New Zealand": "🇳🇿",
    "Spain": "🇪🇸", "Cape Verde": "🇨🇻", "Saudi Arabia": "🇸🇦", "Uruguay": "🇺🇾",
    "France": "🇫🇷", "Senegal": "🇸🇳", "Iraq": "🇮🇶", "Norway": "🇳🇴",
    "Argentina": "🇦🇷", "Algeria": "🇩🇿", "Austria": "🇦🇹", "Jordan": "🇯🇴",
    "Portugal": "🇵🇹", "DR Congo": "🇨🇩", "Uzbekistan": "🇺🇿", "Colombia": "🇨🇴",
    "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Croatia": "🇭🇷", "Ghana": "🇬🇭", "Panama": "🇵🇦",
}


def flag(team: str) -> str:
    return f"{FLAGS.get(team, '🏳️')} {team}"


# ---------------------------------------------------------------- styling
st.markdown(f"""
<style>
.block-container {{ padding-top: 1.2rem; }}
.wc-hero {{
  border-radius: 14px; padding: 1.4rem 1.8rem 1.2rem;
  background: linear-gradient(135deg, #101a33 0%, #0d1426 60%, #131c30 100%);
  border: 1px solid #243153; margin-bottom: .6rem;
}}
.wc-stripe {{
  height: 6px; border-radius: 3px; margin-bottom: .9rem;
  background: linear-gradient(90deg, {RED} 0%, {RED} 33%, {GREEN} 33%, {GREEN} 66%, {BLUE} 66%, {BLUE} 100%);
}}
.wc-hero h1 {{ margin: 0; font-size: 2rem; letter-spacing: .04em; }}
.wc-hero p  {{ margin: .25rem 0 0; color: #93a4c8; }}
.wc-card {{
  border-radius: 12px; padding: .9rem 1rem; height: 100%;
  background: #141d33; border: 1px solid #243153;
}}
.wc-card .t {{ font-size: 1.05rem; font-weight: 600; }}
.wc-card .v {{ font-size: 1.7rem; font-weight: 700; margin-top: .15rem; }}
.wc-card .s {{ color: #93a4c8; font-size: .8rem; }}
.wc-tie {{
  border-radius: 10px; padding: .45rem .7rem; margin-bottom: .45rem;
  background: #141d33; border: 1px solid #243153; font-size: .92rem;
}}
.wc-tie .id {{ color: #93a4c8; font-size: .75rem; }}
.lv-card {{
  border-radius: 12px; padding: .8rem 1rem; margin-bottom: .8rem;
  background: #141d33; border: 1px solid #243153;
}}
.lv-card.live {{ border-color: {RED}; }}
.lv-head {{ display:flex; justify-content:space-between; color:#93a4c8;
            font-size:.78rem; margin-bottom:.35rem; }}
.lv-badge {{ color:#fff; background:{RED}; border-radius:6px; padding:.05rem .5rem;
             font-weight:700; animation: lvpulse 1.6s infinite; }}
@keyframes lvpulse {{ 50% {{ opacity:.55; }} }}
.lv-score {{ display:flex; justify-content:space-between; align-items:center;
             font-size:1.02rem; font-weight:600; margin-bottom:.45rem; }}
.lv-score .sc {{ font-size:1.45rem; font-weight:800; padding:0 .6rem; }}
.pstrip {{ display:flex; height: 22px; border-radius: 6px; overflow:hidden;
           font-size:.72rem; font-weight:700; color:#fff; margin:.25rem 0 .15rem; }}
.pstrip div {{ display:flex; align-items:center; justify-content:center;
               white-space:nowrap; overflow:hidden; }}
.pcap {{ color:#93a4c8; font-size:.74rem; margin-bottom:.4rem; }}
.lvrow {{ display:flex; align-items:center; gap:.45rem; font-size:.78rem;
          margin:.18rem 0; }}
.lvrow .val {{ width:2.4rem; text-align:center; }}
.lvrow .lab {{ width:6.8rem; text-align:center; color:#93a4c8; font-size:.72rem; }}
.lvrow .bar {{ flex:1; height:7px; background:#243153; border-radius:4px;
               position:relative; overflow:hidden; }}
.lvrow .bar div {{ position:absolute; top:0; bottom:0; }}
.lvrow .bar.h div {{ right:0; background:{GREEN}; }}
.lvrow .bar.a div {{ left:0; background:{BLUE}; }}
div[data-testid="stMetricValue"] {{ font-size: 1.55rem; }}
</style>
""", unsafe_allow_html=True)


def _plot(fig: go.Figure, h: int = 420) -> go.Figure:
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#e8ecf6", height=h,
                      margin=dict(l=10, r=10, t=36, b=10))
    fig.update_xaxes(gridcolor="#243153")
    fig.update_yaxes(gridcolor="#243153")
    return fig


# ---------------------------------------------------------------- data access
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


def _refresh_after_update():
    load_predictor.clear()
    st.rerun()


def run_sync(n_sims: int = 20000) -> None:
    """Pull results to-date; refit + re-simulate only if something new arrived."""
    from src import update as upd
    with st.spinner("Syncing results to-date (martj42 / football-data.org)…"):
        info, table = upd.sync_and_recompute(n_sims=n_sims, verbose=False)
    if table is not None:
        st.toast(f"✅ {info['n_changed']} new result(s) imported — odds refreshed.")
        _refresh_after_update()
    else:
        srcs = ", ".join(info["sources"]) or "no source reachable"
        st.toast(f"Already up to date ({srcs}) — {info['total']} results in store.")


def group_standings(store) -> dict[str, pd.DataFrame]:
    """Live group tables from the recorded results (official primary tiebreakers)."""
    rows = {t: {"team": t, "P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "Pts": 0}
            for t in CONFIG.teams}
    for r in store.results:
        if r.stage != "group":
            continue
        h, a = rows[r.home], rows[r.away]
        h["P"] += 1; a["P"] += 1
        h["GF"] += r.home_score; h["GA"] += r.away_score
        a["GF"] += r.away_score; a["GA"] += r.home_score
        if r.home_score > r.away_score:
            h["W"] += 1; a["L"] += 1; h["Pts"] += 3
        elif r.home_score < r.away_score:
            a["W"] += 1; h["L"] += 1; a["Pts"] += 3
        else:
            h["D"] += 1; a["D"] += 1; h["Pts"] += 1; a["Pts"] += 1
    out = {}
    for g, teams in CONFIG.groups.items():
        df = pd.DataFrame([rows[t] for t in teams])
        df["GD"] = df["GF"] - df["GA"]
        out[g] = (df.sort_values(["Pts", "GD", "GF"], ascending=False)
                    .reset_index(drop=True))
    return out


def most_likely_bracket(table: pd.DataFrame) -> list[tuple[int, str, str]]:
    """Illustrative R32 line-up: rank each group by win_group, allocate the 8 most
    probable thirds with the real FIFA slot table."""
    from src.simulation import tournament_2026 as T
    slot, thirds, third_p = {}, {}, {}
    for g, df in table.groupby("group"):
        df = df.sort_values("win_group", ascending=False).reset_index(drop=True)
        slot[f"1{g}"], slot[f"2{g}"] = df.loc[0, "team"], df.loc[1, "team"]
        thirds[g], third_p[g] = df.loc[2, "team"], df.loc[2, "round32"]
    qualifying = sorted(sorted(thirds, key=third_p.get, reverse=True)[:8])
    alloc = T.allocate_thirds(qualifying)
    ties = []
    for tie_id, hs, as_ in T.BRACKET_R32:
        away = thirds[alloc[tie_id]] if as_ == "3" else slot[as_]
        ties.append((tie_id, slot[hs], away))
    return ties


# ---------------------------------------------------------------- live board
@st.cache_data(ttl=45, show_spinner=False)
def fetch_board():
    from src.data import espn_live
    return espn_live.fetch_scoreboard()


def _pstrip(ph: float, pd_: float, pa: float) -> str:
    seg = ""
    for v, c in ((ph, GREEN), (pd_, "#64748b"), (pa, BLUE)):
        pct = v * 100
        label = f"{pct:.0f}%" if pct >= 8 else ""
        seg += f'<div style="width:{pct:.1f}%;background:{c}">{label}</div>'
    return f'<div class="pstrip">{seg}</div>'


def _stat_rows(stats: dict) -> str:
    rows = ""
    for label, (hv, av) in stats.items():
        try:
            hf, af = float(hv), float(av)
        except (TypeError, ValueError):
            continue
        tot = (hf + af) or 1.0
        rows += (f'<div class="lvrow"><span class="val">{hv}</span>'
                 f'<span class="bar h"><div style="width:{hf/tot*100:.0f}%"></div></span>'
                 f'<span class="lab">{label}</span>'
                 f'<span class="bar a"><div style="width:{af/tot*100:.0f}%"></div></span>'
                 f'<span class="val">{av}</span></div>')
    return rows


def _match_card(m: dict, mp) -> str:
    """One scoreboard card: header, score, model strip, stats."""
    from src.models.inplay import conditional_outcome
    home, away = m["home"], m["away"]
    known = home in CONFIG.teams and away in CONFIG.teams
    neutral = not (CONFIG.is_host(home) if known else False)
    pre = mp.predict(home, away, neutral) if known else None

    if m["state"] == "in":
        badge = f'<span class="lv-badge">LIVE {m["detail"]}</span>'
        cls, when = "lv-card live", ""
    elif m["state"] == "post":
        badge = "<span>FT</span>"
        cls, when = "lv-card", f"{m['kickoff']:%b %d}"
    else:
        badge = "<span>Upcoming</span>"
        cls, when = "lv-card", f"{m['kickoff']:%b %d · %H:%M} UTC"

    html = (f'<div class="{cls}"><div class="lv-head">{badge}<span>{when}</span></div>'
            f'<div class="lv-score"><span>{flag(home)}</span>'
            f'<span class="sc">{m["home_score"]} – {m["away_score"]}</span>'
            f'<span>{flag(away)}</span></div>')

    if pre is not None:
        if m["state"] == "in":
            c = conditional_outcome(pre["lambda_home"], pre["lambda_away"],
                                    home_score=m["home_score"],
                                    away_score=m["away_score"], minute=m["minute"])
            html += _pstrip(c["p_home"], c["p_draw"], c["p_away"])
            html += (f'<div class="pcap">live model · projected '
                     f'{c["top_score"][0]}–{c["top_score"][1]} · pre-match '
                     f'{pre["p_home"]*100:.0f}/{pre["p_draw"]*100:.0f}/'
                     f'{pre["p_away"]*100:.0f}</div>')
        elif m["state"] == "pre":
            html += _pstrip(pre["p_home"], pre["p_draw"], pre["p_away"])
            html += (f'<div class="pcap">model: {pre["p_home"]*100:.0f}% / '
                     f'{pre["p_draw"]*100:.0f}% / {pre["p_away"]*100:.0f}%</div>')
        else:
            html += (f'<div class="pcap">model had it '
                     f'{pre["p_home"]*100:.0f}% / {pre["p_draw"]*100:.0f}% / '
                     f'{pre["p_away"]*100:.0f}%</div>')

    if m["stats"] and m["state"] != "pre":
        html += _stat_rows(m["stats"])
    return html + "</div>"


@st.fragment(run_every=60)
def live_board():
    try:
        board = fetch_board()
    except Exception as exc:
        st.warning(f"Live feed unreachable right now: {exc}")
        return
    mp = load_predictor(False)
    live = [m for m in board if m["state"] == "in"]
    done = [m for m in board if m["state"] == "post"]
    pre = [m for m in board if m["state"] == "pre"]

    if live:
        st.markdown(f"#### 🔴 In play now ({len(live)})")
        cols = st.columns(min(2, len(live)))
        for i, m in enumerate(live):
            cols[i % len(cols)].markdown(_match_card(m, mp), unsafe_allow_html=True)
        st.caption("Win-probability strip is the model conditional on the current "
                   "score and minute (green = home, grey = draw, blue = away). "
                   "Auto-refreshes every minute.")
    else:
        st.info("No match in play right now — auto-refreshing every minute.")

    # nudge when a final result hasn't been folded into the odds yet
    store_keys = {(frozenset((r.home, r.away))) for r in _store().results}
    missing = [m for m in done
               if frozenset((m["home"], m["away"])) not in store_keys
               and m["home"] in CONFIG.teams and m["away"] in CONFIG.teams]
    if missing:
        st.warning(f"{len(missing)} finished match(es) not yet in the odds — "
                   "sync from the sidebar to refresh the simulation.")

    if pre:
        st.markdown("#### ⏭️ Next up")
        nxt = pre[:6]
        cols = st.columns(3)
        for i, m in enumerate(nxt):
            cols[i % 3].markdown(_match_card(m, mp), unsafe_allow_html=True)

    if done:
        st.markdown(f"#### ✅ Played ({len(done)})")
        for i, m in enumerate(reversed(done)):
            if i % 3 == 0:
                cols = st.columns(3)
            cols[i % 3].markdown(_match_card(m, mp), unsafe_allow_html=True)


# ---------------------------------------------------------------- header
st.markdown("""
<div class="wc-hero">
  <div class="wc-stripe"></div>
  <h1>🏆 WorldCupPred — FIFA World Cup 26™</h1>
  <p>United States · Canada · Mexico — 48 teams · 104 matches · Elo + Dixon-Coles +
  ML/DL ensemble → Monte-Carlo simulation</p>
</div>
""", unsafe_allow_html=True)

if not _MODELS_READY:
    st.warning("Models not found. Run `python scripts/run_simulation.py` once to fit the "
               "models and produce the first title-odds table, then reload.")
    st.stop()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### ⚙️ Live updates")
    auto = st.toggle("Auto-sync on launch", value=True,
                     help="Once per session: pull played results from martj42 / "
                          "football-data.org, refit and re-simulate if anything is new.")
    sync_sims = st.select_slider("Simulations on refresh",
                                 options=[5000, 10000, 20000, 50000], value=20000)
    if st.button("🔄 Sync results to-date", width='stretch', type="primary"):
        run_sync(int(sync_sims))
    store = _store()
    st.caption(f"**{len(store)}** result(s) recorded · odds table "
               f"{'✅ present' if _ODDS_PATH.exists() else '❌ missing'}")
    if _HISTORY_PATH.exists():
        hist = pd.read_parquet(_HISTORY_PATH)
        st.caption(f"Last simulation: {hist['ts'].max():%Y-%m-%d %H:%M}")
    st.divider()
    st.caption("Sources: martj42/international_results (keyless) · "
               "football-data.org + The Odds API via `.env` keys.")

if auto and not st.session_state.get("_auto_synced"):
    st.session_state["_auto_synced"] = True
    try:
        run_sync(int(sync_sims))
    except Exception as exc:
        st.toast(f"Auto-sync failed: {exc}")

# ---------------------------------------------------------------- tabs
(tab_live, tab_odds, tab_groups, tab_bracket, tab_history,
 tab_match, tab_models, tab_kb) = st.tabs(
    ["🔴 Live", "🏆 Title odds", "📊 Groups", "🛣️ Bracket", "📈 History",
     "⚔️ Single match", "🧠 Models", "📚 Knowledge base"])

table = load_table()

# ---------------------------------------------------------------- Live
with tab_live:
    live_board()

# ---------------------------------------------------------------- Title odds
with tab_odds:
    if table is None:
        st.info("No simulation yet. Run `python scripts/run_simulation.py`.")
    else:
        podium = table.head(3)
        medals = ["🥇", "🥈", "🥉"]
        cols = st.columns(4)
        for i, (col, (_, r)) in enumerate(zip(cols[:3], podium.iterrows())):
            col.markdown(
                f"""<div class="wc-card"><div class="t">{medals[i]} {flag(r['team'])}</div>
                <div class="v">{r['champion']*100:.1f}%</div>
                <div class="s">champion · {r['final']*100:.0f}% final · group {r['group']}</div>
                </div>""", unsafe_allow_html=True)
        cols[3].markdown(
            f"""<div class="wc-card"><div class="t">🎲 Tournament state</div>
            <div class="v">{len(_store())}</div>
            <div class="s">results recorded · 48 teams · 12 groups</div></div>""",
            unsafe_allow_html=True)

        st.markdown("")
        c1, c2 = st.columns([3, 2])
        with c1:
            st.subheader("Champion probability — top 20")
            top = table.head(20).iloc[::-1]
            fig = go.Figure(go.Bar(
                x=top["champion"] * 100,
                y=[flag(t) for t in top["team"]],
                orientation="h",
                marker=dict(color=top["champion"] * 100, colorscale=[[0, "#1d4ed8"], [.5, GREEN], [1, RED]]),
                text=[f"{v*100:.1f}%" for v in top["champion"]],
                textposition="outside",
            ))
            fig.update_layout(xaxis_title="P(champion) %", showlegend=False)
            st.plotly_chart(_plot(fig, 560), width='stretch')
        with c2:
            st.subheader("Champion share by confederation")
            conf = table.assign(conf=table["team"].map(CONFIG.confederation_of))
            agg = conf.groupby("conf", as_index=False)["champion"].sum()
            fig = px.pie(agg, names="conf", values="champion", hole=.45,
                         color_discrete_sequence=[BLUE, GREEN, RED, "#f59e0b",
                                                  "#8b5cf6", "#14b8a6"])
            fig.update_traces(textinfo="label+percent")
            st.plotly_chart(_plot(fig, 320), width='stretch')

            st.subheader("Stage odds — compare teams")
            picks = st.multiselect("Teams", table["team"].tolist(),
                                   default=table["team"].head(5).tolist(),
                                   label_visibility="collapsed")
            if picks:
                stages = ["round32", "round16", "quarterfinal", "semifinal",
                          "final", "champion"]
                fig = go.Figure()
                for t in picks:
                    r = table.set_index("team").loc[t]
                    fig.add_trace(go.Scatter(
                        x=["R32", "R16", "QF", "SF", "Final", "🏆"],
                        y=[r[s] * 100 for s in stages],
                        mode="lines+markers", name=flag(t)))
                fig.update_layout(yaxis_title="P(reach) %")
                st.plotly_chart(_plot(fig, 300), width='stretch')

        st.subheader("Full stage probabilities")
        disp = table.copy()
        disp["team"] = disp["team"].map(flag)
        for col in _STAGE_COLS:
            disp[col] = disp[col] * 100
        st.dataframe(
            disp, width='stretch', hide_index=True, height=520,
            column_config={
                "champion": st.column_config.ProgressColumn(
                    "champion", format="%.1f%%", min_value=0,
                    max_value=float(disp["champion"].max())),
                **{c: st.column_config.NumberColumn(format="%.1f%%")
                   for c in _STAGE_COLS if c != "champion"}})

# ---------------------------------------------------------------- Groups
with tab_groups:
    st.subheader("Group stage — live standings & advance probabilities")
    st.caption("Standings from recorded results; `advance` = P(reach Round of 32) and "
               "`win` = P(win group) from the latest simulation.")
    store = _store()
    tables = group_standings(store)
    odds_ix = table.set_index("team") if table is not None else None
    for row_start in range(0, 12, 3):
        cols = st.columns(3)
        for col, g in zip(cols, list(CONFIG.groups)[row_start:row_start + 3]):
            with col:
                st.markdown(f"##### Group {g}")
                df = tables[g].copy()
                if odds_ix is not None:
                    df["win"] = df["team"].map(odds_ix["win_group"]) * 100
                    df["advance"] = df["team"].map(odds_ix["round32"]) * 100
                df["team"] = df["team"].map(flag)
                show_cols = ["team", "P", "W", "D", "L", "GD", "Pts"]
                cfg = {}
                if odds_ix is not None:
                    show_cols += ["win", "advance"]
                    cfg = {"advance": st.column_config.ProgressColumn(
                               "advance", format="%.0f%%", min_value=0, max_value=100),
                           "win": st.column_config.NumberColumn(format="%.0f%%")}
                st.dataframe(df[show_cols], hide_index=True, width='stretch',
                             column_config=cfg)
    played = [r for r in store.results if r.stage == "group"]
    if played:
        st.markdown("##### Played group matches")
        st.markdown("  \n".join(
            f"`{r.date}` **{flag(r.home)} {r.home_score} – {r.away_score} {flag(r.away)}**"
            for r in played))

# ---------------------------------------------------------------- Bracket
with tab_bracket:
    if table is None:
        st.info("No simulation yet.")
    else:
        st.subheader("Most likely Round of 32")
        st.caption("Illustrative line-up: each group ranked by P(win group); the 8 most "
                   "probable third-placed sides allocated with FIFA's official slot table. "
                   "The simulation itself randomizes every group, this is just the modal view.")
        ties = most_likely_bracket(table)
        half1, half2 = ties[:8], ties[8:]
        c1, c2 = st.columns(2)
        for col, half, title in ((c1, half1, "Left half"), (c2, half2, "Right half")):
            with col:
                st.markdown(f"**{title}**")
                for tie_id, home, away in half:
                    col.markdown(
                        f"""<div class="wc-tie"><span class="id">Match {tie_id}</span><br>
                        {flag(home)} &nbsp;vs&nbsp; {flag(away)}</div>""",
                        unsafe_allow_html=True)

        st.divider()
        st.subheader("Road to the title")
        pick = st.selectbox("Team", table["team"].tolist(), index=0)
        r = table.set_index("team").loc[pick]
        fig = go.Figure(go.Funnel(
            y=["Round of 32", "Round of 16", "Quarter-final", "Semi-final",
               "Final", "Champion 🏆"],
            x=[r[s] * 100 for s in ["round32", "round16", "quarterfinal",
                                    "semifinal", "final", "champion"]],
            texttemplate="%{x:.1f}%",
            marker=dict(color=[BLUE, "#2f6fe0", GREEN, "#0ea36e", "#f59e0b", RED]),
        ))
        st.plotly_chart(_plot(fig, 420), width='stretch')

# ---------------------------------------------------------------- History
with tab_history:
    st.subheader("Champion odds over time")
    st.caption("One snapshot per saved simulation (manual entry, sync, or CLI run).")
    if not _HISTORY_PATH.exists():
        st.info("No history yet — it starts accumulating with the next simulation run.")
    else:
        hist = pd.read_parquet(_HISTORY_PATH)
        n_snaps = hist["ts"].nunique()
        latest = hist[hist["ts"] == hist["ts"].max()]
        default = latest.sort_values("champion", ascending=False)["team"].head(6).tolist()
        picks = st.multiselect("Teams", sorted(hist["team"].unique()), default=default)
        if picks:
            sub = hist[hist["team"].isin(picks)].copy()
            sub["champion"] *= 100
            fig = px.line(sub, x="ts", y="champion", color="team", markers=True,
                          labels={"ts": "", "champion": "P(champion) %", "team": ""})
            fig.for_each_trace(lambda tr: tr.update(name=flag(tr.name)))
            st.plotly_chart(_plot(fig, 440), width='stretch')
        if n_snaps < 2:
            st.caption("Only one snapshot so far — the lines appear once more "
                       "simulations are recorded.")

# ---------------------------------------------------------------- Single match
with tab_match:
    st.subheader("Head-to-head predictor")
    if _HAS_DL:
        with_dl = st.toggle("Include deep-learning engine", value=False,
                            help="Adds the keras embedding model to the blend.")
    else:
        with_dl = False
        st.caption("Deep-learning engine unavailable here (keras not installed) — "
                   "blend uses Elo + Dixon-Coles + ML.")
    c1, c2, c3 = st.columns([3, 3, 2])
    home = c1.selectbox("Team A", CONFIG.teams, index=CONFIG.teams.index("Spain"),
                        format_func=flag)
    away = c2.selectbox("Team B", CONFIG.teams, index=CONFIG.teams.index("Brazil"),
                        format_func=flag)
    neutral = c3.toggle("Neutral venue", value=True)

    if home == away:
        st.info("Pick two different teams.")
    else:
        mp = load_predictor(with_dl)
        out = mp.predict(home, away, neutral)

        fig = go.Figure()
        for name, val, color in ((flag(home), out["p_home"], GREEN),
                                 ("Draw", out["p_draw"], "#64748b"),
                                 (flag(away), out["p_away"], BLUE)):
            fig.add_trace(go.Bar(x=[val * 100], y=["outcome"], name=f"{name} "
                                 f"{val*100:.1f}%", orientation="h",
                                 marker_color=color,
                                 text=f"{name}<br>{val*100:.1f}%",
                                 textposition="inside"))
        fig.update_layout(barmode="stack", showlegend=False,
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(_plot(fig, 130), width='stretch')

        c1, c2 = st.columns(2)
        c1.metric(f"{flag(home)} expected goals", f"{out['lambda_home']:.2f}")
        c2.metric(f"{flag(away)} expected goals", f"{out['lambda_away']:.2f}")

        sc = out["scoreline"][:7, :7]
        ij = np.unravel_index(np.argmax(out["scoreline"]), out["scoreline"].shape)
        st.caption(f"Most likely scoreline: **{home} {ij[0]}-{ij[1]} {away}**  "
                   f"(grid shows P(score), 0–6 goals)")
        fig = px.imshow((sc * 100).round(1), text_auto=".1f",
                        color_continuous_scale=["#0b1220", GREEN],
                        labels=dict(x=f"{away} goals", y=f"{home} goals",
                                    color="P %"),
                        x=[str(j) for j in range(7)], y=[str(i) for i in range(7)])
        st.plotly_chart(_plot(fig, 480), width='stretch')

# ---------------------------------------------------------------- Models
with tab_models:
    st.subheader("Ensemble")
    weights = CONFIG.settings["ensemble"]["weights"]
    c1, c2 = st.columns([2, 3])
    with c1:
        st.write("Blend weights (renormalized over available engines):")
        st.json(weights)
    with c2:
        fig = px.pie(names=list(weights), values=list(weights.values()), hole=.5,
                     color_discrete_sequence=[GREEN, BLUE, "#f59e0b", RED])
        fig.update_traces(textinfo="label+percent")
        st.plotly_chart(_plot(fig, 260), width='stretch')
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

    st.divider()
    st.write("**Betting-odds benchmark** (de-vigged implied champion probability vs ours):")
    from src.data import odds_api
    if odds_api.available():
        bench = odds_api.benchmark_vs_simulation()
        if bench.empty:
            st.info("No outright market returned right now.")
        else:
            b = bench.copy()
            for c in ["implied_prob", "model_prob", "edge"]:
                if c in b:
                    b[c] = (b[c] * 100).round(1)
            st.dataframe(b, width='stretch', hide_index=True)
    else:
        st.caption("Set `ODDS_API_KEY` in .env to compare against bookmaker odds.")

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
