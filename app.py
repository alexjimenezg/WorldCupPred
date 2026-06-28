"""WorldCupPred — Streamlit operations console (FIFA World Cup 26 design).

Tabs:
  Live           ESPN live scoreboard joined to the model: in-play conditional W/D/L,
                 stats, match-detail visuals (win-prob timeline from the goal feed,
                 projected scoreline, total-goals dist), both squads on a pitch with
                 live per-player stats; refreshes every 30s
  Fixtures       all incoming matches grouped by day, filterable by team
  Odds           champion / stage probabilities for all 48 teams (plotly + table)
  Groups         live standings from recorded results + advance probabilities
  Bracket        full predicted wallchart R32 -> Final + road-to-title funnel
  Players        leaderboards (scorers/assists/keepers/shooting/rating/discipline)
                 + a 4-3-3 dream team, whole-tournament or per round (ESPN rosters)
  Value          model vs bookmaker odds (edge, EV, Kelly) + bet builder + parlay slip
  Trends         champion odds over time (one snapshot per saved simulation)
  Versus         any two teams (neutral toggle) -> ensemble W/D/L + scoreline heatmap
  Models         engine info, weights, and a quick walk-forward backtest
  Vault          browse the auto-generated Obsidian knowledge base

Results sync (sidebar / on launch) pulls from ESPN + martj42 + football-data.org;
manual entry still exists via `python -m src.update --match ...`.

The UI is a single responsive design system: CSS grid cards that reflow from a
phone's single column to a desktop's multi-column layout (no fixed-width columns),
pill tabs that scroll horizontally on small screens, and kickoff times in CDMX.

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


TZ_CDMX = "America/Mexico_City"


def cdmx(ts: pd.Timestamp) -> pd.Timestamp:
    """Display timezone: Mexico City (kickoffs arrive tz-aware UTC from ESPN)."""
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(TZ_CDMX)


# ---------------------------------------------------------------- design system
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&display=swap');

:root {{
  --bg:#0b1220; --card:#141d33; --card2:#1a2750; --line:#243153;
  --mut:#93a4c8; --txt:#e8ecf6; --red:{RED}; --green:{GREEN}; --blue:{BLUE};
  --r:16px;
}}
.stApp, .stApp p, .stApp span, .stApp div, .stApp label,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp button {{
  font-family:'Outfit', -apple-system, 'Segoe UI', sans-serif;
}}
/* keep Streamlit's icon fonts intact (sidebar arrows, expander chevrons, menu) */
.stApp [data-testid="stIconMaterial"],
.stApp .material-symbols-rounded, .stApp .material-symbols-outlined,
span[data-testid="stIconMaterial"] {{
  font-family:'Material Symbols Rounded','Material Symbols Outlined' !important;
}}
.block-container {{ padding: 1.0rem 1.4rem 4rem; max-width: 1500px; }}
@media (max-width: 640px) {{
  .block-container {{ padding: .55rem .55rem 4rem; }}
}}
header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}

/* ---- pill tabs (scroll horizontally on phones) ---- */
.stTabs [data-baseweb="tab-list"] {{
  gap:.35rem; flex-wrap:nowrap; overflow-x:auto; padding-bottom:.35rem;
  scrollbar-width:none;
}}
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {{ display:none; }}
.stTabs [data-baseweb="tab"] {{
  background:var(--card); border:1px solid var(--line); border-radius:999px;
  padding:.3rem 1.0rem; color:var(--mut); font-weight:600; white-space:nowrap;
  transition: all .15s;
}}
.stTabs [data-baseweb="tab"]:hover {{ color:#fff; border-color:var(--mut); }}
.stTabs [aria-selected="true"] {{
  background:linear-gradient(100deg, var(--red), #f43f5e);
  color:#fff !important; border-color:transparent;
}}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{
  display:none;
}}

/* ---- hero ---- */
.wc-hero {{
  border-radius:var(--r); padding:1.1rem 1.4rem 1rem; margin-bottom:.55rem;
  background:
    radial-gradient(1100px 320px at 8% -40%, rgba(225,29,72,.22), transparent),
    radial-gradient(900px 320px at 55% -50%, rgba(16,185,129,.16), transparent),
    radial-gradient(900px 320px at 100% -40%, rgba(59,130,246,.20), transparent),
    linear-gradient(135deg, #101a33 0%, #0d1426 70%);
  border:1px solid var(--line);
}}
.wc-stripe {{
  height:5px; border-radius:3px; margin-bottom:.8rem;
  background:linear-gradient(90deg, var(--red) 0 33%, var(--green) 33% 66%,
                             var(--blue) 66% 100%);
}}
.wc-hero h1 {{ margin:0; font-size:1.7rem; font-weight:800; letter-spacing:.02em; }}
.wc-hero p {{ margin:.3rem 0 0; color:var(--mut); font-size:.86rem; }}
.chiprow {{ display:flex; gap:.4rem; flex-wrap:wrap; margin-top:.55rem; }}
.chip {{
  font-size:.7rem; font-weight:700; letter-spacing:.05em; color:var(--txt);
  border:1px solid var(--line); border-radius:999px; padding:.14rem .6rem;
}}
.chip.r {{ border-color:var(--red); }} .chip.g {{ border-color:var(--green); }}
.chip.b {{ border-color:var(--blue); }}
@media (max-width:640px) {{
  .wc-hero {{ padding:.8rem .9rem .75rem; }}
  .wc-hero h1 {{ font-size:1.25rem; }}
  .wc-hero p {{ font-size:.78rem; }}
}}

/* ---- responsive card grids ---- */
.grid {{ display:grid; gap:.65rem; margin:.3rem 0 .8rem; }}
.g-pod  {{ grid-template-columns:repeat(auto-fit, minmax(185px, 1fr)); }}
.g-live {{ grid-template-columns:repeat(auto-fill, minmax(320px, 1fr)); }}
.g-mini {{ grid-template-columns:repeat(auto-fill, minmax(255px, 1fr)); }}
.g-grp  {{ grid-template-columns:repeat(auto-fill, minmax(295px, 1fr)); }}

.wc-card {{
  border-radius:var(--r); padding:.85rem 1rem;
  background:var(--card); border:1px solid var(--line);
  transition:transform .15s, box-shadow .15s;
}}
.wc-card:hover {{ transform:translateY(-2px); box-shadow:0 10px 26px rgba(0,0,0,.35); }}
.wc-card .t {{ font-size:1rem; font-weight:700; }}
.wc-card .v {{ font-size:1.65rem; font-weight:800; margin-top:.1rem; }}
.wc-card .s {{ color:var(--mut); font-size:.76rem; }}

/* ---- live cards ---- */
.lv-card {{
  border-radius:var(--r); padding:.75rem .95rem;
  background:var(--card); border:1px solid var(--line);
  transition:transform .15s, box-shadow .15s;
}}
.lv-card:hover {{ transform:translateY(-2px); box-shadow:0 10px 26px rgba(0,0,0,.35); }}
.lv-card.live {{ border-color:var(--red);
                 box-shadow:0 0 0 1px rgba(225,29,72,.35), 0 6px 18px rgba(225,29,72,.12); }}
.lv-head {{ display:flex; justify-content:space-between; color:var(--mut);
            font-size:.74rem; margin-bottom:.3rem; }}
.lv-badge {{ color:#fff; background:var(--red); border-radius:6px;
             padding:.05rem .5rem; font-weight:700; animation:lvpulse 1.6s infinite; }}
@keyframes lvpulse {{ 50% {{ opacity:.55; }} }}
.lv-score {{ display:flex; justify-content:space-between; align-items:center;
             font-size:.98rem; font-weight:700; margin-bottom:.4rem; gap:.3rem; }}
.lv-score .sc {{ font-size:1.4rem; font-weight:800; padding:0 .45rem;
                 font-variant-numeric:tabular-nums; }}
.pstrip {{ display:flex; height:21px; border-radius:7px; overflow:hidden;
           font-size:.7rem; font-weight:700; color:#fff; margin:.25rem 0 .12rem; }}
.pstrip div {{ display:flex; align-items:center; justify-content:center;
               white-space:nowrap; overflow:hidden; }}
.pcap {{ color:var(--mut); font-size:.72rem; margin-bottom:.35rem; }}
.lvrow {{ display:flex; align-items:center; gap:.45rem; font-size:.76rem;
          margin:.16rem 0; }}
.lvrow .val {{ width:2.3rem; text-align:center; }}
.lvrow .lab {{ width:6.6rem; text-align:center; color:var(--mut); font-size:.7rem; }}
.lvrow .bar {{ flex:1; height:6px; background:var(--line); border-radius:4px;
               position:relative; overflow:hidden; }}
.lvrow .bar div {{ position:absolute; top:0; bottom:0; }}
.lvrow .bar.h div {{ right:0; background:var(--green); }}
.lvrow .bar.a div {{ left:0; background:var(--blue); }}

/* ---- group cards ---- */
.grp-card {{
  background:var(--card); border:1px solid var(--line); border-radius:var(--r);
  padding:.65rem .85rem; transition:transform .15s, box-shadow .15s;
}}
.grp-card:hover {{ transform:translateY(-2px); box-shadow:0 10px 26px rgba(0,0,0,.35); }}
.grp-card .gh {{ font-weight:800; letter-spacing:.05em; margin-bottom:.35rem;
                 display:flex; justify-content:space-between; align-items:baseline; }}
.grp-card .gh small {{ color:var(--mut); font-weight:400; font-size:.7rem; }}
.grow {{ display:grid; grid-template-columns:minmax(0,1fr) 1.5rem 2rem 56px 2.5rem;
         gap:.4rem; align-items:center; font-size:.82rem; padding:.2rem 0;
         border-top:1px solid rgba(36,49,83,.55); }}
.grow.hd {{ border-top:none; color:var(--mut); font-size:.66rem;
            letter-spacing:.04em; }}
.grow .tm {{ white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.grow .num {{ text-align:center; font-variant-numeric:tabular-nums; }}
.grow .pc {{ text-align:right; color:var(--mut); font-size:.74rem; }}
.qdot {{ display:inline-block; width:7px; height:7px; border-radius:50%;
         margin-right:.3rem; vertical-align:1px; }}
.adv {{ height:6px; background:var(--line); border-radius:3px; overflow:hidden; }}
.adv div {{ height:100%; background:linear-gradient(90deg, var(--green), #34d399); }}
.grow.now {{ background:rgba(225,29,72,.13); border-radius:8px;
             box-shadow:inset 2px 0 0 var(--red); }}
.grp-card.now {{ border-color:var(--red); }}
.glive {{ margin-top:.45rem; font-size:.76rem; font-weight:600; color:#fda4af;
          display:flex; gap:.4rem; align-items:center; }}
.glive .dot {{ width:8px; height:8px; border-radius:50%; background:var(--red);
               animation:lvpulse 1.6s infinite; flex:none; }}
.glive .min {{ color:var(--mut); font-weight:400; }}
.lvstar {{ color:var(--red); margin-left:.25rem; animation:lvpulse 1.6s infinite; }}

/* ---- bracket wallchart ---- */
.bk-wrap {{ overflow-x:auto; padding-bottom:.5rem; }}
.bk {{ display:flex; gap:.45rem; min-width:1240px; align-items:stretch; }}
.bk-col {{ display:flex; flex-direction:column; justify-content:space-around;
           flex:1; min-width:124px; }}
.bk-rnd {{ text-align:center; color:var(--mut); font-size:.7rem; font-weight:700;
           letter-spacing:.08em; margin-bottom:.2rem; }}
.bk-tie {{ background:var(--card); border:1px solid var(--line); border-radius:9px;
           padding:.25rem .45rem; margin:.16rem 0; font-size:.75rem; }}
.bk-tie .tm {{ display:flex; justify-content:space-between; gap:.3rem;
               white-space:nowrap; }}
.bk-tie .tm.w {{ font-weight:700; }}
.bk-tie .tm.l {{ color:#64748b; }}
.bk-tie .pp {{ color:var(--mut); font-weight:400; }}
.bk-id {{ color:#5b6a8c; font-size:.6rem; }}
.bk-champ {{ text-align:center; background:linear-gradient(135deg,#1d2a4d,var(--card));
             border:1px solid var(--green); border-radius:10px;
             padding:.5rem .4rem; margin-bottom:.5rem; }}
.bk-champ .c {{ font-size:1rem; font-weight:800; }}
.bk-champ .s {{ color:var(--mut); font-size:.7rem; }}
.bk-tie.live {{ border-color:var(--red);
               box-shadow:0 0 0 1px rgba(225,29,72,.35); }}
.bk-tie.decided {{ border-color:var(--green); }}
.bk-live-badge {{ color:#fff; background:var(--red); border-radius:4px;
                  padding:.02rem .32rem; font-size:.55rem; font-weight:700;
                  animation:lvpulse 1.6s infinite; margin-left:.28rem; }}
.bk-score {{ text-align:center; font-weight:800; font-size:.8rem; margin:.1rem 0;
             font-variant-numeric:tabular-nums; }}
.bk-detail {{ color:var(--mut); font-size:.6rem; font-weight:400; margin-left:.25rem; }}

/* ---- dream-team pitch ---- */
.pitch {{
  position:relative; border-radius:18px; padding:1.2rem .5rem;
  background:
    repeating-linear-gradient(0deg, #0f5132 0 46px, #0c4329 46px 92px);
  border:2px solid rgba(255,255,255,.22);
  box-shadow:inset 0 0 0 7px rgba(255,255,255,.05);
  display:flex; flex-direction:column; gap:.5rem; overflow:hidden;
}}
.pitch::before {{ content:''; position:absolute; left:50%; top:50%;
  width:128px; height:128px; transform:translate(-50%,-50%);
  border:2px solid rgba(255,255,255,.22); border-radius:50%; }}
.pitch::after {{ content:''; position:absolute; left:8px; right:8px; top:50%;
  height:2px; background:rgba(255,255,255,.22); }}
.pbox {{ position:absolute; left:50%; transform:translateX(-50%); width:48%;
  height:62px; border:2px solid rgba(255,255,255,.2); }}
.pbox.top {{ top:-2px; border-top:none; }}
.pbox.bot {{ bottom:-2px; border-bottom:none; }}
.prow {{ position:relative; z-index:2; display:flex; justify-content:space-around;
  gap:.3rem; }}
.ptok {{ display:flex; flex-direction:column; align-items:center; width:88px;
  text-align:center; }}
.ptok img {{ width:52px; height:52px; object-fit:contain;
  filter:drop-shadow(0 3px 5px rgba(0,0,0,.55)); }}
.ptok .mono {{ width:48px; height:48px; border-radius:50%;
  background:linear-gradient(135deg,#1f2d52,#0d1426);
  border:2px solid rgba(255,255,255,.3); display:flex; align-items:center;
  justify-content:center; font-weight:800; font-size:1rem; }}
.ptok .nm {{ font-weight:700; font-size:.74rem; margin-top:.15rem; color:#fff;
  white-space:nowrap; max-width:90px; overflow:hidden; text-overflow:ellipsis;
  text-shadow:0 1px 3px rgba(0,0,0,.7); }}
.ptok .sb {{ font-size:.64rem; color:#dbe4f5; text-shadow:0 1px 2px rgba(0,0,0,.7); }}
.ptok .rt {{ background:var(--green); color:#04231a; font-weight:800;
  font-size:.64rem; border-radius:6px; padding:.02rem .34rem; margin-top:.12rem; }}
@media (max-width:640px) {{
  .ptok {{ width:70px; }}
  .ptok img {{ width:42px; height:42px; }}
  .ptok .mono {{ width:40px; height:40px; font-size:.85rem; }}
  .ptok .nm {{ font-size:.66rem; max-width:72px; }}
  .pitch::before {{ width:92px; height:92px; }}
}}

/* ---- two-team match lineup pitch ---- */
.pitch.match {{ gap:.25rem; padding:.7rem .35rem; }}
.pitch.match .ptok {{ width:64px; }}
.pitch.match .ptok img {{ width:36px; height:36px; }}
.pitch.match .ptok .mono {{ width:32px; height:32px; font-size:.78rem; }}
.pitch.match .ptok .nm {{ font-size:.6rem; max-width:66px; margin-top:.05rem; }}
.pitch.match .ptok .sb {{ font-size:.66rem; min-height:.8rem; }}
.pteam {{ position:relative; z-index:2; display:flex; justify-content:space-between;
  align-items:center; color:#fff; font-weight:700; font-size:.78rem;
  padding:.1rem .4rem; text-shadow:0 1px 3px rgba(0,0,0,.7); }}
.pteam small {{ color:#dbe4f5; font-weight:400; }}
@media (max-width:640px) {{
  .pitch.match .ptok {{ width:46px; }}
  .pitch.match .ptok img {{ width:29px; height:29px; }}
  .pitch.match .ptok .nm {{ font-size:.53rem; max-width:48px; }}
  .pitch.match .ptok .mono {{ width:27px; height:27px; font-size:.66rem; }}
}}

/* ---- misc ---- */
.stButton > button {{ border-radius:12px; font-weight:700; }}
div[data-testid="stMetricValue"] {{ font-size:1.5rem; }}
</style>
""", unsafe_allow_html=True)


def _plot(fig: go.Figure, h: int = 420) -> go.Figure:
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#e8ecf6", family="Outfit, sans-serif"),
                      height=h, margin=dict(l=10, r=10, t=36, b=10))
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


def run_sync(n_sims: int = 20000) -> None:
    """Pull results to-date; refit + re-simulate only if something new arrived."""
    from src import update as upd
    with st.spinner("Syncing results to-date (ESPN / martj42 / football-data.org)…"):
        info, table_new = upd.sync_and_recompute(n_sims=n_sims, verbose=False)
    if table_new is not None:
        st.toast(f"✅ {info['n_changed']} new result(s) imported — odds refreshed.")
        load_predictor.clear()
        st.rerun()
    else:
        srcs = ", ".join(info["sources"]) or "no source reachable"
        st.toast(f"Already up to date ({srcs}) — {info['total']} results in store.")


def group_standings(store, live_matches: list[dict] | None = None
                    ) -> dict[str, pd.DataFrame]:
    """Group tables from recorded results, with any in-play group match folded in
    provisionally at its current score (official primary tiebreakers).

    `live_matches`: in-play board entries (home/away/home_score/away_score). Teams
    in a live match get a `live` flag so the standings can show them as provisional.
    """
    rows = {t: {"team": t, "P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0,
                "Pts": 0, "live": False} for t in CONFIG.teams}

    def apply(home: str, away: str, hs: int, as_: int, live: bool = False) -> None:
        if home not in rows or away not in rows:
            return
        h, a = rows[home], rows[away]
        h["P"] += 1; a["P"] += 1
        h["GF"] += hs; h["GA"] += as_
        a["GF"] += as_; a["GA"] += hs
        if hs > as_:
            h["W"] += 1; a["L"] += 1; h["Pts"] += 3
        elif hs < as_:
            a["W"] += 1; h["L"] += 1; a["Pts"] += 3
        else:
            h["D"] += 1; a["D"] += 1; h["Pts"] += 1; a["Pts"] += 1
        if live:
            h["live"] = a["live"] = True

    for r in store.results:
        if r.stage == "group":
            apply(r.home, r.away, r.home_score, r.away_score)
    for m in live_matches or []:
        if (m["home"] in CONFIG.teams and m["away"] in CONFIG.teams
                and CONFIG.group_of(m["home"]) == CONFIG.group_of(m["away"])):
            apply(m["home"], m["away"], m["home_score"], m["away_score"], live=True)

    out = {}
    for g, teams in CONFIG.groups.items():
        df = pd.DataFrame([rows[t] for t in teams])
        df["GD"] = df["GF"] - df["GA"]
        out[g] = (df.sort_values(["Pts", "GD", "GF"], ascending=False)
                    .reset_index(drop=True))
    return out


def most_likely_bracket(table: pd.DataFrame, standings: dict | None = None
                        ) -> list[tuple[int, str, str]]:
    """Illustrative R32 line-up. Rank each group by *actual* standings so far
    (points, then goal difference, then goals for) with the simulation's
    win_group probability as the final tiebreaker / fallback for matches that
    haven't been played yet, then allocate the 8 best thirds with the real FIFA
    slot table.

    `standings`: optional per-group sorted DataFrames from `group_standings`
    (live + recorded results folded in). When omitted the bracket is built from
    the simulation alone (pre-tournament behaviour)."""
    from src.simulation import tournament_2026 as T
    wg = table.set_index("team")["win_group"].to_dict()
    r32 = table.set_index("team")["round32"].to_dict()
    slot, thirds, third_key = {}, {}, {}
    for g, df in table.groupby("group"):
        if standings is not None and g in standings:
            d = standings[g].copy()
            d["wg"] = d["team"].map(wg).fillna(0.0)
            d = d.sort_values(["Pts", "GD", "GF", "wg"], ascending=False
                              ).reset_index(drop=True)
            slot[f"1{g}"], slot[f"2{g}"] = d.loc[0, "team"], d.loc[1, "team"]
            thirds[g] = d.loc[2, "team"]
            t3 = d.loc[2]
            third_key[g] = (int(t3["Pts"]), int(t3["GD"]), int(t3["GF"]),
                            float(r32.get(t3["team"], 0.0)))
        else:
            d = df.sort_values("win_group", ascending=False).reset_index(drop=True)
            slot[f"1{g}"], slot[f"2{g}"] = d.loc[0, "team"], d.loc[1, "team"]
            thirds[g] = d.loc[2, "team"]
            third_key[g] = (0, 0, 0, float(d.loc[2, "round32"]))
    qualifying = sorted(sorted(thirds, key=third_key.get, reverse=True)[:8])
    alloc = T.allocate_thirds(qualifying)
    ties = []
    for tie_id, hs, as_ in T.BRACKET_R32:
        away = thirds[alloc[tie_id]] if as_ == "3" else slot[as_]
        ties.append((tie_id, slot[hs], away))
    return ties


def predicted_bracket(table: pd.DataFrame, standings: dict | None = None
                      ) -> dict[int, dict]:
    """Play the modal bracket out with the ensemble: at every node the favorite
    advances. p = P(side wins the tie | the matchup happens), draws resolved by
    the same conditional used for ET/penalties in the simulator.

    `standings`: optional live group tables so the R32 line-up reflects results
    to date (see `most_likely_bracket`)."""
    from src.simulation import tournament_2026 as T
    mp = load_predictor(False)
    games: dict[int, dict] = {}
    winners: dict[int, str] = {}

    def play(tid: int, home: str, away: str) -> None:
        out = mp.predict(home, away, True)  # knockouts at neutral venues
        p = out["p_home"] / (out["p_home"] + out["p_away"])
        winners[tid] = home if p >= 0.5 else away
        games[tid] = {"home": home, "away": away, "p": p, "winner": winners[tid]}

    for tid, home, away in most_likely_bracket(table, standings):
        play(tid, home, away)
    for bracket in (T.BRACKET_R16, T.BRACKET_QF, T.BRACKET_SF):
        for tid, s1, s2 in bracket:
            play(tid, winners[s1], winners[s2])
    f_id, s1, s2 = T.FINAL
    play(f_id, winners[s1], winners[s2])
    return games


@st.cache_data(ttl=600, show_spinner=False)
def predicted_bracket_cached(_sig: str) -> dict[int, dict]:
    """Cached predicted bracket (busts when the odds table / results change)."""
    t = load_table()
    return predicted_bracket(t) if t is not None else {}


def _bracket_maps():
    """tie_id -> round name, and feeder_tie -> (next_tie, sibling_feeder)."""
    from src.simulation import tournament_2026 as T
    rnd = {}
    for t in range(73, 89):
        rnd[t] = "Round of 32"
    for t in range(89, 97):
        rnd[t] = "Round of 16"
    for t in (97, 98, 99, 100):
        rnd[t] = "Quarter-final"
    for t in (101, 102):
        rnd[t] = "Semi-final"
    rnd[104] = "Final"
    feeds = {}
    for nxt, a, b in (*T.BRACKET_R16, *T.BRACKET_QF, *T.BRACKET_SF, T.FINAL):
        feeds[a] = (nxt, b)
        feeds[b] = (nxt, a)
    return rnd, feeds


def knockout_context(home: str, away: str, sig: str) -> dict | None:
    """For a knockout match, the projected next-round opponents of the winner.

    Identifies the tie in the predicted bracket by its two teams (best-effort —
    the projection may differ from reality), then reads the sibling feeder tie.
    """
    games = predicted_bracket_cached(sig)
    if not games:
        return None
    rnd, feeds = _bracket_maps()
    pair = {home, away}
    for tid, g in games.items():
        if {g["home"], g["away"]} == pair:
            nxt = feeds.get(tid)
            if not nxt:
                return {"final": True}
            next_tie, sib = nxt
            sg = games.get(sib, {})
            return {"round": rnd[next_tie],
                    "rivalA": sg.get("home"), "rivalB": sg.get("away"),
                    "fav": sg.get("winner")}
    return None


# column layout of the official bracket: left half feeds SF 101, right half SF 102
_BK_COLS: list[tuple[str, list[int]]] = [
    ("R32", [74, 77, 73, 75, 83, 84, 81, 82]),
    ("R16", [89, 90, 93, 94]),
    ("QF", [97, 98]),
    ("SF", [101]),
    ("FINAL", [104]),
    ("SF", [102]),
    ("QF", [99, 100]),
    ("R16", [91, 92, 95, 96]),
    ("R32", [76, 78, 79, 80, 86, 88, 85, 87]),
]


def bracket_html(games: dict[int, dict], champ_prob: float,
                 live_tids: set[int] | None = None,
                 decided_tids: set[int] | None = None) -> str:
    live_tids = live_tids or set()
    decided_tids = decided_tids or set()

    def tie(tid: int) -> str:
        g = games[tid]
        is_live = tid in live_tids
        is_decided = tid in decided_tids
        score = g.get("score")

        rows = ""
        for i, (team, p) in enumerate(((g["home"], g["p"]), (g["away"], 1 - g["p"]))):
            cls = "w" if team == g["winner"] else "l"
            if score is not None:
                right = f'<span class="pp">{score[i]}</span>'
            else:
                right = f'<span class="pp">{p*100:.0f}%</span>'
            rows += (f'<div class="tm {cls}"><span>{flag(team)}</span>'
                     f'{right}</div>')

        live_badge = '<span class="bk-live-badge">LIVE</span>' if is_live else ""
        detail_span = (f'<span class="bk-detail">{g["detail"]}</span>'
                       if is_live and g.get("detail") else "")
        score_line = ""
        if is_live and score is not None:
            score_line = (f'<div class="bk-score">{score[0]}–{score[1]}'
                          f'{detail_span}</div>')

        extra_cls = " live" if is_live else (" decided" if is_decided else "")
        return (f'<div class="bk-tie{extra_cls}">'
                f'<div class="bk-id">M{tid}{live_badge}</div>'
                f'{score_line}{rows}</div>')

    cols = ""
    for label, tids in _BK_COLS:
        body = "".join(tie(t) for t in tids)
        if label == "FINAL":
            champ = games[104]["winner"]
            body = (f'<div class="bk-champ"><div class="s">PREDICTED CHAMPION</div>'
                    f'<div class="c">🏆 {flag(champ)}</div>'
                    f'<div class="s">{champ_prob*100:.1f}% of all simulations</div>'
                    f'</div>') + body
        cols += f'<div class="bk-col"><div class="bk-rnd">{label}</div>{body}</div>'
    return f'<div class="bk-wrap"><div class="bk">{cols}</div></div>'


# ---------------------------------------------------------------- live board
@st.cache_data(ttl=25, show_spinner=False)
def fetch_board():
    from src.data import espn_live
    return espn_live.fetch_scoreboard()


@st.cache_data(ttl=300, show_spinner="Loading player stats…")
def load_players(_board_sig: str):
    """Per-player tournament frame. Keyed by a signature of finished match ids so a
    new final result busts the cache; finished matches are disk-cached underneath."""
    from src.data import espn_players
    return espn_players.build_player_frame(fetch_board())


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
        cls, when = "lv-card", f"{cdmx(m['kickoff']):%b %d}"
    else:
        badge = "<span>Upcoming</span>"
        cls, when = "lv-card", f"{cdmx(m['kickoff']):%b %d · %H:%M} CDMX"

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


def _score_heat(matrix: np.ndarray, home: str, away: str, *, h0: int = 0, a0: int = 0,
                n: int = 6, height: int = 320, key: str = "") -> None:
    sub = matrix[:n, :n]
    fig = px.imshow((sub * 100).round(1), text_auto=".1f",
                    color_continuous_scale=["#0b1220", GREEN],
                    labels=dict(x=f"{away} goals", y=f"{home} goals", color="P %"),
                    x=[str(a0 + j) for j in range(n)],
                    y=[str(h0 + i) for i in range(n)])
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(_plot(fig, height), width='stretch', key=key)


@st.cache_data(ttl=25, show_spinner=False)
def load_lineups(event_id: str):
    from src.data import espn_players
    return espn_players.fetch_match_lineups(event_id)


def _order_lr(players: list[dict]) -> list[dict]:
    """Order a position row left→right using the position-abbreviation suffix."""
    def lr(p: str) -> int:
        p = (p or "").upper()
        if p.endswith("-L") or p in ("LB", "LM", "LW", "LWB"):
            return -1
        if p.endswith("-R") or p in ("RB", "RM", "RW", "RWB"):
            return 1
        return 0
    return sorted(players, key=lambda x: lr(x["pos"]))


def _squad_pitch(lu: dict) -> str:
    """Both starting XIs on one vertical field: away attacks down, home attacks up."""
    def tok(p: dict) -> str:
        if p["jersey"]:
            img = f'<img src="{p["jersey"]}" alt="">'
        else:
            img = f'<div class="mono">{p["number"]}</div>'
        marks = "⚽" * int(p["goals"])
        if p["red"]:
            marks += "🟥"
        elif p["yellow"]:
            marks += "🟨"
        if p["subbed_out"]:
            marks += "↩"
        return (f'<div class="ptok">{img}'
                f'<div class="nm">{p["number"]} {p["surname"]}</div>'
                f'<div class="sb">{marks or "&nbsp;"}</div></div>')

    def row(side: dict, role: str) -> str:
        ps = _order_lr([p for p in side["starters"] if p["role"] == role])
        return f'<div class="prow">{"".join(tok(p) for p in ps)}</div>' if ps else ""

    home, away = lu.get("home", {}), lu.get("away", {})
    if not home or not away:
        return ""
    away_h = (f'<div class="pteam"><span>{flag(away["team"])}</span>'
              f'<small>{away["formation"]} · attacking ↓</small></div>')
    home_h = (f'<div class="pteam"><span>{flag(home["team"])}</span>'
              f'<small>{home["formation"]} · attacking ↑</small></div>')
    body = (away_h
            + "".join(row(away, r) for r in ["GK", "DEF", "MID", "FWD"])
            + "".join(row(home, r) for r in ["FWD", "MID", "DEF", "GK"])
            + home_h)
    return f'<div class="pitch match"><div class="pbox top"></div>' \
           f'<div class="pbox bot"></div>{body}</div>'


def _squad_stats(side: dict) -> pd.DataFrame:
    """Live per-player stat table for one team (starters + players who came on)."""
    rows = []
    for p in side["starters"] + [s for s in side["subs"] if s["subbed_in"]]:
        rows.append({
            "#": p["number"], "player": f'{p["surname"]}',
            "pos": p["pos"], "G": int(p["goals"]), "A": int(p["assists"]),
            "Sh": int(p["shots"]), "SOT": int(p["sot"]),
            "Sv": int(p["saves"]), "F": int(p["fouls"]),
            "C": "🟥" if p["red"] else "🟨" if p["yellow"] else "",
        })
    return pd.DataFrame(rows)


def _mini_group_html(g: str, store, live_list: list[dict], focus: set[str]) -> str:
    """One group's standings (live folded in), the two focus teams emphasised."""
    df = group_standings(store, live_matches=live_list)[g]
    rows = ('<div class="grow hd"><span>GROUP ' + g + '</span>'
            '<span class="num">P</span><span class="num">PTS</span>'
            '<span class="num">GD</span></div>')
    for pos, (_, r) in enumerate(df.iterrows()):
        dot = "#10b981" if pos < 2 else "#f59e0b" if pos == 2 else "transparent"
        hi = "background:rgba(59,130,246,.14);border-radius:8px;" \
            if r["team"] in focus else ""
        star = "<span class='lvstar'>•</span>" if r.get("live") else ""
        rows += (f'<div class="grow" style="{hi}grid-template-columns:'
                 f'minmax(0,1fr) 1.6rem 2rem 2.4rem">'
                 f'<span class="tm"><span class="qdot" style="background:{dot}">'
                 f'</span>{flag(r["team"])}{star}</span>'
                 f'<span class="num">{int(r["P"])}</span>'
                 f'<span class="num">{int(r["Pts"])}</span>'
                 f'<span class="num">{int(r["GD"]):+d}</span></div>')
    return f'<div class="grp-card">{rows}</div>'


def match_context(m: dict) -> None:
    """Group standings (group stage) or projected next-round rivals (knockout)."""
    from src.data.auto_results import infer_stage
    home, away = m["home"], m["away"]
    stage = infer_stage(m["kickoff"], home, away)
    live_list = [m] if m["state"] == "in" else []

    if stage == "group" or (stage is None
                            and CONFIG.group_of(home) == CONFIG.group_of(away)):
        g = CONFIG.group_of(home)
        st.markdown(f"**Group {g} — standings**"
                    + (" *(live, provisional)*" if live_list else ""))
        st.markdown(_mini_group_html(g, _store(), live_list, {home, away}),
                    unsafe_allow_html=True)
    else:
        sig = f"{_ODDS_PATH.stat().st_mtime if _ODDS_PATH.exists() else 0}"
        ctx = knockout_context(home, away, sig)
        if not ctx:
            st.caption("Bracket position for this tie isn't resolved yet.")
        elif ctx.get("final"):
            st.markdown("**🏆 This is the Final — the winner lifts the trophy.**")
        elif ctx.get("rivalA"):
            st.markdown(
                f"**Next up:** the winner advances to the **{ctx['round']}** to face "
                f"the winner of **{flag(ctx['rivalA'])}** vs **{flag(ctx['rivalB'])}** "
                f"— model favours {flag(ctx['fav'])}.")
        else:
            st.caption(f"Winner reaches the **{ctx['round']}**; "
                       "the opponent isn't set yet.")


def match_detail(m: dict, mp) -> None:
    """Every visualization the feed supports for one match: win-probability
    timeline (reconstructed from the goal events), projected/expected scoreline
    heatmap, total-goals distribution, and the event feed."""
    from src.models.inplay import (conditional_outcome, total_goals_dist,
                                   win_prob_timeline)
    home, away = m["home"], m["away"]
    if home not in CONFIG.teams or away not in CONFIG.teams:
        st.info("Teams not in the 2026 field.")
        return
    pre = mp.predict(home, away, not CONFIG.is_host(home))
    events = m.get("events", [])
    kid = f"{home}-{away}"

    match_context(m)

    if events:
        icons = {"goal": "⚽", "yellow": "🟨", "red": "🟥"}
        bits = []
        for e in events:
            who = e["player"] or (home if e["side"] == "home" else away)
            suffix = " (OG)" if e["own_goal"] else " (pen)" if e["penalty"] else ""
            bits.append(f"{e['minute']}' {icons[e['type']]} {who}{suffix}")
        st.markdown("**Events:** " + " · ".join(bits))

    upto = m["minute"] if m["state"] == "in" else 90
    c1, c2 = st.columns([3, 2])

    with c1:
        if m["state"] != "pre":
            st.markdown("**Win probability through the match**")
            tl = win_prob_timeline(pre["lambda_home"], pre["lambda_away"], events,
                                   upto=upto, final_home=m["home_score"],
                                   final_away=m["away_score"])
            if tl is None:
                st.caption("Goal feed doesn't reconcile with the score — "
                           "timeline unavailable for this match.")
            else:
                fig = go.Figure()
                for col, name, color in (("home", flag(home), "rgba(16,185,129,.75)"),
                                         ("draw", "Draw", "rgba(100,116,139,.75)"),
                                         ("away", flag(away), "rgba(59,130,246,.75)")):
                    fig.add_trace(go.Scatter(
                        x=tl["minute"], y=tl[col] * 100, name=name, mode="none",
                        stackgroup="one", fillcolor=color))
                for e in events:
                    if e["type"] == "goal":
                        fig.add_vline(x=e["minute"], line_color="#fff",
                                      line_dash="dot", opacity=.55)
                        fig.add_annotation(x=e["minute"], y=103, text="⚽",
                                           showarrow=False)
                fig.update_layout(yaxis=dict(title="P(final result) %",
                                             range=[0, 106]),
                                  xaxis=dict(title="minute", range=[0, max(upto, 1)]),
                                  legend=dict(orientation="h", y=-0.25))
                st.plotly_chart(_plot(fig, 340), width='stretch', key=f"tl-{kid}")
        else:
            st.markdown("**Expected scoreline** (pre-match)")
            _score_heat(pre["scoreline"], home, away, height=340, key=f"hm-{kid}")

    with c2:
        if m["state"] == "in":
            cond = conditional_outcome(pre["lambda_home"], pre["lambda_away"],
                                       home_score=m["home_score"],
                                       away_score=m["away_score"],
                                       minute=m["minute"])
            st.markdown(f"**Projected final score** (from {m['home_score']}–"
                        f"{m['away_score']}, {m['minute']}')")
            _score_heat(cond["matrix"], home, away, h0=m["home_score"],
                        a0=m["away_score"], n=5, height=240, key=f"cm-{kid}")
            dist = total_goals_dist(cond, m["home_score"], m["away_score"])
        elif m["state"] == "pre":
            cond = conditional_outcome(pre["lambda_home"], pre["lambda_away"],
                                       home_score=0, away_score=0, minute=0)
            dist = total_goals_dist(cond, 0, 0)
        else:
            st.markdown("**What the model expected pre-match**")
            _score_heat(pre["scoreline"], home, away, n=5, height=240,
                        key=f"pm-{kid}")
            dist = None
        if dist is not None:
            st.markdown("**Total goals (final)**")
            fig = go.Figure(go.Bar(
                x=[str(k) if k < 8 else "8+" for k in dist.index],
                y=dist.values * 100, marker_color=BLUE,
                text=[f"{v*100:.0f}%" for v in dist.values],
                textposition="outside"))
            fig.update_layout(yaxis_title="P %", xaxis_title="goals")
            st.plotly_chart(_plot(fig, 240), width='stretch', key=f"tg-{kid}")

    # ---- squads on the pitch + live player stats ----
    if m["state"] != "pre" or m.get("id"):
        try:
            lu = load_lineups(m["id"]) if m.get("id") else {}
        except Exception:
            lu = {}
        pitch = _squad_pitch(lu) if lu else ""
        if pitch:
            st.markdown("**Line-ups**" + (" — live stats" if m["state"] == "in"
                                          else ""))
            st.markdown(pitch, unsafe_allow_html=True)
            st.caption("⚽ = goals · 🟨/🟥 = card · ↩ = substituted off. "
                       "Rows are positional (left→right); jerseys are ESPN kits.")
            sc1, sc2 = st.columns(2)
            for col, ha in ((sc1, "away"), (sc2, "home")):
                with col:
                    st.markdown(f"**{flag(lu[ha]['team'])}**")
                    st.dataframe(_squad_stats(lu[ha]), hide_index=True,
                                 width='stretch',
                                 column_config={"player": st.column_config.TextColumn(
                                     width="medium")})
        elif m["state"] == "pre":
            st.caption("Line-ups appear here once they're announced "
                       "(usually about an hour before kick-off).")


def _countdown(m: dict) -> None:
    """Ticking JS countdown to the next kickoff (CDMX time shown)."""
    import streamlit.components.v1 as components
    ms = int(m["kickoff"].timestamp() * 1000)
    when = f"{cdmx(m['kickoff']):%a %b %d · %H:%M} CDMX"
    components.html(f"""
    <style>@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;700;800&display=swap');</style>
    <div style="display:flex;gap:1.1rem;align-items:center;flex-wrap:wrap;
                font-family:'Outfit',-apple-system,sans-serif;color:#e8ecf6;
                background:#141d33;border:1px solid #243153;border-radius:16px;
                padding:.7rem 1.1rem;">
      <span style="font-size:.72rem;color:#93a4c8;letter-spacing:.1em;">NEXT MATCH</span>
      <span style="font-weight:700;font-size:1rem;">
        {FLAGS.get(m['home'], '')} {m['home']} vs {m['away']} {FLAGS.get(m['away'], '')}
      </span>
      <span id="cd" style="font-size:1.45rem;font-weight:800;color:{GREEN};
                           font-variant-numeric:tabular-nums;">…</span>
      <span style="font-size:.76rem;color:#93a4c8;">{when}</span>
    </div>
    <script>
      const t = {ms}, el = document.getElementById("cd");
      function tick() {{
        let d = t - Date.now();
        if (d <= 0) {{ el.textContent = "KICK-OFF!"; return; }}
        const h = Math.floor(d / 3.6e6), m_ = Math.floor(d % 3.6e6 / 6e4),
              s = Math.floor(d % 6e4 / 1e3);
        el.textContent = (h > 0 ? h + "h " : "") +
          String(m_).padStart(2, "0") + "m " + String(s).padStart(2, "0") + "s";
      }}
      tick(); setInterval(tick, 1000);
    </script>""", height=72)


@st.fragment(run_every=30)
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

    if pre:
        _countdown(pre[0])

    if live:
        st.markdown(f"#### 🔴 In play now ({len(live)})")
        st.markdown('<div class="grid g-live">'
                    + "".join(_match_card(m, mp) for m in live)
                    + "</div>", unsafe_allow_html=True)
        st.caption("Win-probability strip is the model conditional on the current "
                   "score and minute (green = home, grey = draw, blue = away). "
                   "Auto-refreshes every minute.")
    else:
        st.info("No match in play right now — auto-refreshing every minute. ⚽")

    # auto-fold a final result into the odds the moment it shows up finished —
    # recorded straight from the board in hand, so no second (flaky) ESPN fetch
    store_keys = {(frozenset((r.home, r.away))) for r in _store().results}
    missing = [m for m in done
               if frozenset((m["home"], m["away"])) not in store_keys
               and m["home"] in CONFIG.teams and m["away"] in CONFIG.teams]
    if missing and auto:
        handled = st.session_state.setdefault("_autosynced_pairs", set())
        fresh = [m for m in missing
                 if frozenset((m["home"], m["away"])) not in handled]
        if fresh:
            # mark first so a match we can't record (odd stage/name) never loops
            for m in fresh:
                handled.add(frozenset((m["home"], m["away"])))
            from src.data.espn_live import import_board_to_store
            added = import_board_to_store(fresh, _store())
            if added:
                with st.spinner(f"{added} new result(s) in — refreshing the "
                                "simulation…"):
                    from src import update as upd
                    upd.recompute(n_sims=int(sync_sims), regen_vault=False,
                                  verbose=False)
                load_predictor.clear()
                st.toast(f"✅ {added} new result(s) auto-synced — odds refreshed.")
                st.rerun(scope="app")
        # whatever we still couldn't fold in, surface for a manual sync
        leftover = [m for m in missing
                    if frozenset((m["home"], m["away"])) not in
                    {frozenset((r.home, r.away)) for r in _store().results}]
        if leftover:
            st.warning(f"{len(leftover)} finished match(es) couldn't be auto-synced "
                       "— try **Sync results to-date** in the sidebar.")
    elif missing:
        st.warning(f"{len(missing)} finished match(es) not yet in the odds — "
                   "sync from the sidebar to refresh the simulation.")

    # ---- every visualization for one match ----
    choices = live + pre[:4] + list(reversed(done))
    if choices:
        st.markdown("#### 🔬 Match detail")
        state_tag = {"in": "🔴 live", "pre": "upcoming", "post": "FT"}
        ix = st.selectbox(
            "Match", range(len(choices)),
            format_func=lambda i: (f"{choices[i]['home']} "
                                   f"{choices[i]['home_score']}–"
                                   f"{choices[i]['away_score']} "
                                   f"{choices[i]['away']}  ·  "
                                   f"{state_tag[choices[i]['state']]}"),
            key="detail_pick", label_visibility="collapsed")
        match_detail(choices[ix], mp)

    if pre:
        st.markdown("#### ⏭️ Next up")
        st.markdown('<div class="grid g-mini">'
                    + "".join(_match_card(m, mp) for m in pre[:3])
                    + "</div>", unsafe_allow_html=True)
        st.caption("Full schedule with a team filter → **📅 Fixtures** tab.")

    if done:
        st.markdown(f"#### ✅ Played ({len(done)})")
        st.markdown('<div class="grid g-mini">'
                    + "".join(_match_card(m, mp) for m in reversed(done))
                    + "</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------- header
st.markdown("""
<div class="wc-hero">
  <div class="wc-stripe"></div>
  <h1>⚽ WorldCupPred</h1>
  <p>Elo + Dixon-Coles + ML ensemble → thousands of Monte-Carlo futures,
  refreshed with every final whistle.</p>
  <div class="chiprow">
    <span class="chip r">FIFA WORLD CUP 26™</span>
    <span class="chip g">🇺🇸 🇨🇦 🇲🇽 UNITED 26</span>
    <span class="chip b">48 TEAMS · 104 MATCHES</span>
  </div>
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
                     help="Once per session: pull played results from ESPN / martj42 / "
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
    with st.expander("🔔 Phone notifications"):
        topic = CONFIG.env("NTFY_TOPIC")
        st.markdown(
            "Kickoff, goal and full-time alerts on your phone via "
            "[ntfy](https://ntfy.sh) (free):\n\n"
            "1. Install the **ntfy** app (App Store / Play Store).\n"
            "2. **Subscribe to topic** "
            + (f"`{topic}`" if topic else
               "set in the repo's `NTFY_TOPIC` Actions variable") + ".\n"
            "3. Done — a GitHub Action polls the live feed every ~5 min "
            "during matches.\n\n"
            "Configure in repo *Settings → Actions variables*: `NTFY_EVENTS` "
            "(`kickoff,goal,fulltime`) and `NTFY_TEAMS` (e.g. `Mexico,Spain`) "
            "to filter.")
    st.caption("Sources: ESPN scoreboard + martj42/international_results (keyless) · "
               "football-data.org + The Odds API via `.env` keys.")

if auto and not st.session_state.get("_auto_synced"):
    st.session_state["_auto_synced"] = True
    try:
        run_sync(int(sync_sims))
    except Exception as exc:
        st.toast(f"Auto-sync failed: {exc}")

# ---------------------------------------------------------------- tabs
(tab_live, tab_fix, tab_odds, tab_groups, tab_bracket, tab_players, tab_bet,
 tab_history, tab_match, tab_models, tab_kb) = st.tabs(
    ["🔴 Live", "📅 Fixtures", "🏆 Odds", "📊 Groups", "🛣️ Bracket", "⭐ Players",
     "💰 Value", "📈 Trends", "⚔️ Versus", "🧠 Models", "📚 Vault"])

table = load_table()

# ---------------------------------------------------------------- Live
with tab_live:
    live_board()

# ---------------------------------------------------------------- Fixtures
with tab_fix:
    st.subheader("Incoming matches")
    try:
        fx_board = fetch_board()
    except Exception as exc:
        fx_board = []
        st.warning(f"Schedule feed unreachable: {exc}")
    upcoming_all = [m for m in fx_board if m["state"] == "pre"]

    c1, c2 = st.columns([3, 2])
    sel_team = c1.selectbox("Filter by team", ["All teams"] + CONFIG.teams,
                            format_func=lambda t: t if t == "All teams" else flag(t))
    if sel_team != "All teams":
        upcoming_all = [m for m in upcoming_all
                        if sel_team in (m["home"], m["away"])]
        if table is not None and sel_team in set(table["team"]):
            r = table.set_index("team").loc[sel_team]
            c2.metric(f"{flag(sel_team)}",
                      f"{r['champion']*100:.1f}% champion",
                      f"{r['round32']*100:.0f}% reach R32")

    st.caption(f"**{len(upcoming_all)}** match(es) scheduled · times in CDMX · "
               "strips show the model's pre-match W/D/L")
    if upcoming_all:
        mp_fx = load_predictor(False)
        from itertools import groupby
        for day, day_ms in groupby(upcoming_all,
                                   key=lambda m: cdmx(m["kickoff"]).date()):
            day_ms = list(day_ms)
            st.markdown(f"##### {day:%A, %B %d}")
            st.markdown('<div class="grid g-mini">'
                        + "".join(_match_card(x, mp_fx) for x in day_ms)
                        + "</div>", unsafe_allow_html=True)
    elif sel_team != "All teams":
        st.info(f"No more scheduled matches for {sel_team} on the feed — "
                "knockout pairings appear once the groups settle.")

# ---------------------------------------------------------------- Odds
with tab_odds:
    if table is None:
        st.info("No simulation yet. Run `python scripts/run_simulation.py`.")
    else:
        podium = table.head(3)
        medals = ["🥇", "🥈", "🥉"]
        cards = ""
        for i, (_, r) in enumerate(podium.iterrows()):
            cards += (f'<div class="wc-card"><div class="t">{medals[i]} '
                      f'{flag(r["team"])}</div>'
                      f'<div class="v">{r["champion"]*100:.1f}%</div>'
                      f'<div class="s">champion · {r["final"]*100:.0f}% final · '
                      f'group {r["group"]}</div></div>')
        cards += (f'<div class="wc-card"><div class="t">🎲 Tournament state</div>'
                  f'<div class="v">{len(_store())}</div>'
                  f'<div class="s">results recorded · 48 teams · 12 groups</div></div>')
        st.markdown(f'<div class="grid g-pod">{cards}</div>', unsafe_allow_html=True)

        c1, c2 = st.columns([3, 2])
        with c1:
            st.subheader("Champion probability — top 20")
            top = table.head(20).iloc[::-1]
            fig = go.Figure(go.Bar(
                x=top["champion"] * 100,
                y=[flag(t) for t in top["team"]],
                orientation="h",
                marker=dict(color=top["champion"] * 100,
                            colorscale=[[0, "#1d4ed8"], [.5, GREEN], [1, RED]]),
                text=[f"{v*100:.1f}%" for v in top["champion"]],
                textposition="outside",
            ))
            fig.update_layout(xaxis_title="P(champion) %", showlegend=False)
            st.plotly_chart(_plot(fig, 540), width='stretch')
        with c2:
            st.subheader("By confederation")
            conf = table.assign(conf=table["team"].map(CONFIG.confederation_of))
            agg = conf.groupby("conf", as_index=False)["champion"].sum()
            fig = px.pie(agg, names="conf", values="champion", hole=.45,
                         color_discrete_sequence=[BLUE, GREEN, RED, "#f59e0b",
                                                  "#8b5cf6", "#14b8a6"])
            fig.update_traces(textinfo="label+percent")
            st.plotly_chart(_plot(fig, 300), width='stretch')

            st.subheader("Stage odds — compare")
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
@st.fragment(run_every=30)
def groups_board():
    st.caption("Points & goals update live as matches play (provisional) · bar = "
               "P(reach Round of 32) from the latest simulation · "
               "<span class='qdot' style='background:#10b981'></span> direct "
               "<span class='qdot' style='background:#f59e0b'></span> best-third race "
               "· in-play teams glow red (refreshes every 30s)",
               unsafe_allow_html=True)
    store = _store()
    t = load_table()
    odds_ix = t.set_index("team") if t is not None else None

    try:
        board = fetch_board()
    except Exception:
        board = []
    live_teams: dict[str, dict] = {}
    live_by_group: dict[str, list[dict]] = {}
    live_list: list[dict] = []
    for m in board:
        if m["state"] != "in":
            continue
        if m["home"] in CONFIG.teams and m["away"] in CONFIG.teams:
            live_teams[m["home"]] = m
            live_teams[m["away"]] = m
            live_by_group.setdefault(CONFIG.group_of(m["home"]), []).append(m)
            live_list.append(m)

    tables = group_standings(store, live_matches=live_list)
    played_by_group = {g: 0 for g in CONFIG.groups}
    for r in store.results:
        if r.stage == "group" and r.home in CONFIG.teams:
            played_by_group[CONFIG.group_of(r.home)] += 1

    cards = ""
    for g in CONFIG.groups:
        df = tables[g]
        played = played_by_group[g]
        n_live = len(live_by_group.get(g, []))
        rows = ('<div class="grow hd"><span>TEAM</span><span class="num">PTS</span>'
                '<span class="num">GD</span><span>ADVANCE</span>'
                '<span class="pc">%</span></div>')
        for pos, (_, r) in enumerate(df.iterrows()):
            adv = float(odds_ix.loc[r["team"], "round32"]) * 100 if odds_ix is not None else 0
            dot = ("#10b981" if pos < 2 else "#f59e0b" if pos == 2 else "transparent")
            now = " now" if r["team"] in live_teams else ""
            star = "<span class='lvstar'>•</span>" if r.get("live") else ""
            rows += (f'<div class="grow{now}">'
                     f'<span class="tm"><span class="qdot" style="background:{dot}">'
                     f'</span>{flag(r["team"])}{star}</span>'
                     f'<span class="num">{r["Pts"]}</span>'
                     f'<span class="num">{r["GD"]:+d}</span>'
                     f'<span class="adv"><div style="width:{adv:.0f}%"></div></span>'
                     f'<span class="pc">{adv:.0f}</span></div>')
        live_line = ""
        for m in live_by_group.get(g, []):
            live_line += (f'<div class="glive"><span class="dot"></span>'
                          f'<span>{flag(m["home"])} {m["home_score"]} – '
                          f'{m["away_score"]} {flag(m["away"])}</span>'
                          f'<span class="min">{m["detail"]}</span></div>')
        card_cls = " now" if g in live_by_group else ""
        sub = f'{played}/6 played' + (f' · {n_live} live' if n_live else '')
        cards += (f'<div class="grp-card{card_cls}"><div class="gh">GROUP {g}'
                  f'<small>{sub}</small></div>{rows}{live_line}</div>')
    st.markdown(f'<div class="grid g-grp">{cards}</div>', unsafe_allow_html=True)

    played_res = [r for r in store.results if r.stage == "group"]
    if played_res:
        st.markdown("##### Played group matches")
        st.markdown("  \n".join(
            f"`{r.date}` **{flag(r.home)} {r.home_score} – {r.away_score} {flag(r.away)}**"
            for r in played_res))


with tab_groups:
    st.subheader("Group stage — standings & advance odds")
    groups_board()

# ---------------------------------------------------------------- Bracket
@st.fragment(run_every=30)
def bracket_board(table: pd.DataFrame) -> None:
    """Bracket wallchart: auto-refreshes every 30s to fold in live/finished matches."""
    from src.models.inplay import conditional_outcome

    if table is None:
        st.info("No simulation yet.")
        return

    store = _store()

    # Live in-play group matches → folded into the standings so the R32 line-up
    # reflects results to date (mirrors the Groups tab).
    try:
        board = fetch_board()
    except Exception:
        board = []
    live_group: list[dict] = [
        m for m in board
        if m["state"] == "in"
        and m["home"] in CONFIG.teams and m["away"] in CONFIG.teams
        and CONFIG.group_of(m["home"]) == CONFIG.group_of(m["away"])
    ]
    standings = group_standings(store, live_matches=live_group)

    # Rebuild the bracket from the live standings (cheap: 31 ensemble look-ups).
    games = predicted_bracket(table, standings)
    if not games:
        st.info("No simulation yet.")
        return

    # Build pair -> tie_id lookup for fast matching against ESPN data
    pair_to_tid: dict[frozenset, int] = {
        frozenset({g["home"], g["away"]}): tid for tid, g in games.items()
    }

    live_tids: set[int] = set()
    decided_tids: set[int] = set()

    # 1. Apply stored knockout results (authoritative, from the results_store)
    knockout_stages = {"round32", "round16", "quarterfinal", "semifinal", "final"}
    for r in store.results:
        if r.stage not in knockout_stages:
            continue
        tid = pair_to_tid.get(frozenset({r.home, r.away}))
        if tid is None:
            continue
        if r.home_score > r.away_score:
            winner, p = r.home, 1.0
        elif r.home_score < r.away_score:
            winner, p = r.away, 0.0
        else:
            # ET/penalties: keep predicted winner if score is level
            winner = games[tid]["winner"]
            p = 1.0 if winner == games[tid]["home"] else 0.0
        games[tid].update({"winner": winner, "p": p,
                           "score": (r.home_score, r.away_score), "decided": True})
        decided_tids.add(tid)

    # 2. Overlay live ESPN data for knockout ties (in-play conditional
    #    probability or finished score). Group matches already shaped the
    #    line-up above; here we only touch ties that match a live/finished game.
    mp = load_predictor(False)
    for m in board:
        if m["home"] not in CONFIG.teams or m["away"] not in CONFIG.teams:
            continue
        tid = pair_to_tid.get(frozenset({m["home"], m["away"]}))
        if tid is None:
            continue
        g = games[tid]
        if m["state"] == "in":
            pre = mp.predict(m["home"], m["away"], True)
            cond = conditional_outcome(
                pre["lambda_home"], pre["lambda_away"],
                home_score=m["home_score"], away_score=m["away_score"],
                minute=m["minute"],
            )
            # Knockout: no draw possible in the end, normalise home vs away only
            ph, pa = cond["p_home"], cond["p_away"]
            p = ph / (ph + pa) if (ph + pa) > 0 else 0.5
            games[tid].update({
                "p": p,
                "winner": g["home"] if p >= 0.5 else g["away"],
                "score": (m["home_score"], m["away_score"]),
                "detail": m["detail"],
                "live": True,
            })
            live_tids.add(tid)
        elif m["state"] == "post" and tid not in decided_tids:
            if m["home_score"] > m["away_score"]:
                winner, p = m["home"], 1.0
            elif m["home_score"] < m["away_score"]:
                winner, p = m["away"], 0.0
            else:
                winner = g["winner"]
                p = 1.0 if winner == g["home"] else 0.0
            games[tid].update({"winner": winner, "p": p,
                               "score": (m["home_score"], m["away_score"]), "decided": True})
            decided_tids.add(tid)

    champ = games[104]["winner"]
    champ_prob = float(table.set_index("team").loc[champ, "champion"])

    if live_tids:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.4rem;">'
            '<span class="lv-badge">LIVE</span>'
            '<span style="color:var(--mut);font-size:.8rem;">'
            'Bracket is updating with live match probabilities</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown(bracket_html(games, champ_prob,
                             live_tids=live_tids, decided_tids=decided_tids),
                unsafe_allow_html=True)


with tab_bracket:
    if table is None:
        st.info("No simulation yet.")
    else:
        st.subheader("The bracket, as the model predicts it")
        st.caption("Modal wallchart: groups resolved by the live standings (points, "
                   "then the simulation as tiebreaker) with FIFA's official third-place "
                   "slot allocation, then the ensemble's favorite advances at every node "
                   "(percentages = win the tie if it happens). The R32 line-up re-shapes "
                   "as group results come in; live knockout ties show the in-play "
                   "probability and a green border marks a confirmed result. "
                   "Refreshes every 30 s — swipe sideways on a phone.")
        bracket_board(table)

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

# ---------------------------------------------------------------- Players
def _leader_chart(df: pd.DataFrame, value_col: str, title: str, color: str,
                  fmt: str = "{:.0f}", n: int = 12, key: str = "") -> None:
    sub = df[df[value_col] > 0].sort_values(value_col, ascending=False).head(n)
    if sub.empty:
        st.caption(f"No {title.lower()} recorded yet.")
        return
    sub = sub.iloc[::-1]
    labels = [f"{flag(t)} · {p.split()[-1] if len(p.split())>1 else p}"
              for p, t in zip(sub["player"], sub["team"])]
    fig = go.Figure(go.Bar(
        x=sub[value_col], y=[f"{p}" for p in sub["player"]], orientation="h",
        marker_color=color, customdata=sub["team"],
        text=[fmt.format(v) for v in sub[value_col]], textposition="outside",
        hovertemplate="%{y} (%{customdata})<br>" + title + ": %{x}<extra></extra>"))
    fig.update_layout(title=title, yaxis=dict(tickfont=dict(size=11)))
    st.plotly_chart(_plot(fig, 30 + 32 * len(sub)), width='stretch', key=key)


def _pitch(xi: dict) -> str:
    """Render the 4-3-3 Best XI on a CSS soccer field (jersey image per player)."""
    def token(r: pd.Series, gk: bool) -> str:
        if r["jersey"]:
            img = f'<img src="{r["jersey"]}" alt="">'
        else:
            initials = "".join(w[0] for w in str(r["player"]).split()[:2]).upper()
            img = f'<div class="mono">{initials}</div>'
        stat = (f'{int(r["saves"])} saves · {int(r["clean_sheet"])} CS' if gk
                else f'{int(r["goals"])}G · {int(r["assists"])}A')
        name = str(r["player"]).split()[-1]
        return (f'<div class="ptok">{img}'
                f'<div class="nm">{FLAGS.get(r["team"], "")} {name}</div>'
                f'<div class="sb">{stat}</div>'
                f'<div class="rt">★ {r["rating"]:.1f}</div></div>')

    def row(role: str, gk: bool = False) -> str:
        toks = "".join(token(r, gk) for _, r in xi[role].iterrows())
        return f'<div class="prow">{toks}</div>'

    return (f'<div class="pitch"><div class="pbox top"></div>'
            f'<div class="pbox bot"></div>'
            f'{row("FWD")}{row("MID")}{row("DEF")}{row("GK", gk=True)}</div>')


with tab_players:
    st.subheader("Top players & dream team")
    try:
        pboard = fetch_board()
        sig = ",".join(sorted(m["id"] for m in pboard if m["state"] == "post"))
        pdf = load_players(sig)
    except Exception as exc:
        pdf = pd.DataFrame()
        st.warning(f"Player feed unreachable: {exc}")

    if pdf.empty:
        st.info("No player stats yet — they appear here once matches are played. "
                "(Source: ESPN match rosters.)")
    else:
        from src.data import espn_players as ep
        stages_present = [s for s in ["group", "R32", "R16", "QF", "SF", "final"]
                          if s in set(pdf["stage"])]
        nice = {"group": "Group stage", "R32": "Round of 32", "R16": "Round of 16",
                "QF": "Quarter-finals", "SF": "Semi-finals", "final": "Final"}
        c1, c2 = st.columns([2, 3])
        scope = c1.selectbox("Round", ["Whole tournament"]
                             + [nice[s] for s in stages_present])
        if scope != "Whole tournament":
            code = {v: k for k, v in nice.items()}[scope]
            view = pdf[pdf["stage"] == code]
        else:
            view = pdf
        agg = ep.aggregate(view)
        c2.caption(f"{int(view['event_id'].nunique())} match(es) · "
                   f"{len(agg)} players · stats from ESPN rosters")

        sub = st.radio("View", ["⭐ Dream team", "⚽ Scorers", "🅰️ Assists",
                                "🥅 Goalkeepers", "🎯 Shooting", "🏅 Top rated",
                                "🟨 Discipline"], horizontal=True,
                       label_visibility="collapsed")

        if sub == "⭐ Dream team":
            st.caption(f"Best XI (4-3-3) by performance rating — {scope.lower()}. "
                       "Jerseys are ESPN's rendered kits (no headshots in the free "
                       "feed). Rating is our own transparent score (goals ×4, "
                       "assists ×3, on-target ×0.5, saves ×0.5, clean sheet ×2, "
                       "minus cards & own goals); players placed by starting position.")
            xi = ep.best_xi(agg)
            filled = sum(len(xi[r]) for r in xi)
            if filled < 11:
                st.info(f"Only {filled}/11 slots filled so far — the pitch completes "
                        "as more matches (and positions) are played.")
            st.markdown(_pitch(xi), unsafe_allow_html=True)
        elif sub == "⚽ Scorers":
            _leader_chart(agg, "goals", "Goals", GREEN, key="lc-g")
        elif sub == "🅰️ Assists":
            _leader_chart(agg, "assists", "Assists", BLUE, key="lc-a")
        elif sub == "🥅 Goalkeepers":
            gk = agg[agg["role"] == "GK"]
            c1, c2 = st.columns(2)
            with c1:
                _leader_chart(gk, "saves", "Saves", "#f59e0b", key="lc-sv")
            with c2:
                _leader_chart(gk, "clean_sheet", "Clean sheets", GREEN, key="lc-cs")
            show = (gk[gk["apps"] > 0].sort_values("saves", ascending=False)
                    [["player", "team", "apps", "saves", "conceded",
                      "clean_sheet", "save_pct"]].head(15).copy())
            show["team"] = show["team"].map(flag)
            show["save_pct"] *= 100
            st.dataframe(show, hide_index=True, width='stretch', column_config={
                "apps": "apps", "clean_sheet": "clean sheets", "conceded": "GA",
                "save_pct": st.column_config.NumberColumn("save %", format="%.0f%%")})
        elif sub == "🎯 Shooting":
            c1, c2 = st.columns(2)
            with c1:
                _leader_chart(agg, "shots", "Shots", BLUE, key="lc-sh")
            with c2:
                _leader_chart(agg, "sot", "Shots on target", GREEN, key="lc-sot")
        elif sub == "🏅 Top rated":
            _leader_chart(agg, "rating", "Performance rating", RED,
                          fmt="{:.1f}", key="lc-rt")
            st.caption("Composite rating (ours): goals ×4, assists ×3, on-target "
                       "×0.5, saves ×0.5, clean sheet ×2, minus cards & own goals.")
        elif sub == "🟨 Discipline":
            disc = agg.assign(cards=agg["yellow"] + agg["red"] * 2)
            c1, c2 = st.columns(2)
            with c1:
                _leader_chart(disc, "cards", "Card points (Y=1, R=2)", "#facc15",
                              key="lc-cd")
            with c2:
                _leader_chart(agg, "fouls", "Fouls committed", "#f87171",
                              key="lc-fl")

        st.caption("⚠️ Passing accuracy and tackles/interceptions aren't in the free "
                   "feed for every player (only per-match top performers), so they're "
                   "omitted to avoid a misleading leaderboard.")

# ---------------------------------------------------------------- Value
with tab_bet:
    st.subheader("Value vs the market")
    st.caption("Model probabilities against the bookmaker 3-way prices that ride along "
               "in the ESPN feed (de-vigged). **Edge** = model − implied; **EV** = "
               "expected profit per unit. ⚠️ Big edges on longshots usually mean the "
               "model is less sure of the favorite than the market — that's "
               "disagreement, not free money. If you bet: small stakes, fractional "
               "Kelly, never the full fraction. 18+, gamble responsibly.")
    from src import betting as bet

    try:
        board = fetch_board()
    except Exception as exc:
        board = []
        st.warning(f"Live feed unreachable: {exc}")

    mp = load_predictor(False)
    vb = bet.value_board(mp, board)
    if vb.empty:
        st.info("No upcoming matches with bookmaker odds on the feed right now.")
    else:
        c1, c2 = st.columns([2, 1])
        min_edge = c1.slider("Minimum edge (percentage points)", 0.0, 15.0, 3.0, 0.5)
        max_odds = c2.select_slider("Max odds", options=[3, 5, 8, 15, 50], value=8,
                                    help="Filter out extreme longshots where the "
                                         "model is least reliable.")
        picks = vb[(vb["edge"] >= min_edge / 100) & (vb["ev"] > 0)
                   & (vb["odds"] <= max_odds)].copy()
        st.markdown(f"**{len(picks)} value pick(s)** "
                    f"(of {len(vb)} priced outcomes, {vb['match'].nunique()} matches)")
        if len(picks):
            show = picks.copy()
            show["when"] = show["kickoff"].map(lambda t: f"{cdmx(t):%b %d %H:%M}")
            show["match"] = show.apply(
                lambda r: f"{FLAGS.get(r['match'].split(' vs ')[0], '')} "
                          f"{r['match']}", axis=1)
            for c in ["implied", "model", "edge", "ev"]:
                show[c] = show[c] * 100
            show["stake"] = show["kelly"] / 4 * 100  # quarter Kelly, % of bankroll
            st.dataframe(
                show[["when", "match", "pick", "odds", "implied", "model",
                      "edge", "ev", "stake"]],
                width='stretch', hide_index=True,
                column_config={
                    "when": "kickoff (CDMX)", "odds": st.column_config.NumberColumn(
                        "odds", format="%.2f"),
                    "implied": st.column_config.NumberColumn("implied %", format="%.1f%%"),
                    "model": st.column_config.NumberColumn("model %", format="%.1f%%"),
                    "edge": st.column_config.ProgressColumn(
                        "edge", format="%.1f pp", min_value=0,
                        max_value=float(show["edge"].max())),
                    "ev": st.column_config.NumberColumn("EV", format="%.1f%%"),
                    "stake": st.column_config.NumberColumn(
                        "¼-Kelly stake", format="%.1f%% bank"),
                })
        with st.expander("Full market comparison (all priced outcomes)"):
            full = vb.copy()
            full["when"] = full["kickoff"].map(lambda t: f"{cdmx(t):%b %d %H:%M}")
            for c in ["implied", "model", "edge", "ev"]:
                full[c] = (full[c] * 100).round(1)
            st.dataframe(full[["when", "match", "pick", "odds", "implied",
                               "model", "edge", "ev", "book"]],
                         width='stretch', hide_index=True, height=420)

    st.divider()
    st.subheader("🧮 Bet builder — price anything")
    st.caption("Pick a bet, the model tells you its probability and the fair odds; "
               "enter the bookmaker's price to see your edge.")

    kind = st.radio("Bet type", ["Match market", "Tournament outright"],
                    horizontal=True, label_visibility="collapsed")

    p_model, bet_label = None, ""
    if kind == "Match market":
        upcoming = [m for m in board if m["state"] == "pre"
                    and m["home"] in CONFIG.teams and m["away"] in CONFIG.teams]
        c1, c2 = st.columns([3, 2])
        if upcoming:
            opts = [f"{m['home']} vs {m['away']}  ·  {cdmx(m['kickoff']):%b %d %H:%M}"
                    for m in upcoming]
            pick_ix = c1.selectbox("Fixture", range(len(opts)),
                                   format_func=lambda i: opts[i])
            mh, ma = upcoming[pick_ix]["home"], upcoming[pick_ix]["away"]
        else:
            mh = c1.selectbox("Team A", CONFIG.teams, index=0, format_func=flag)
            ma = c1.selectbox("Team B", CONFIG.teams, index=1, format_func=flag)
        market_ix = c2.selectbox("Market", range(len(bet.FIXTURE_MARKETS)),
                                 format_func=lambda i: bet.FIXTURE_MARKETS[i][1])
        code, mlabel = bet.FIXTURE_MARKETS[market_ix]

        line, score = 2.5, None
        if code in ("O", "U"):
            line = st.select_slider("Goal line", options=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                                    value=2.5)
        if code == "CS":
            cc1, cc2 = st.columns(2)
            score = (cc1.number_input(f"{mh} goals", 0, 10, 1),
                     cc2.number_input(f"{ma} goals", 0, 10, 0))
        if mh != ma:
            out = mp.predict(mh, ma, not CONFIG.is_host(mh))
            p_model = bet.market_prob(out, code, line=line, score=score)
            extra = (f" {line}" if code in ("O", "U")
                     else f" {score[0]}-{score[1]}" if code == "CS" else "")
            who = {"1": mh, "2": ma, "X": "Draw", "1X": f"{mh}/draw",
                   "X2": f"draw/{ma}", "12": f"{mh}/{ma}"}.get(code, mlabel)
            bet_label = f"{mh} vs {ma}: {who if code in ('1','2','X','1X','X2','12') else mlabel}{extra}"
    else:
        c1, c2 = st.columns(2)
        team = c1.selectbox("Team", table["team"].tolist() if table is not None
                            else CONFIG.teams, format_func=flag)
        omkt = c2.selectbox("Market", ["Champion", "Reach final", "Reach semi-final",
                                       "Reach quarter-final", "Win group"])
        if table is not None:
            r = table.set_index("team").loc[team]
            p_model = float({"Champion": r["champion"], "Reach final": r["final"],
                             "Reach semi-final": r["semifinal"],
                             "Reach quarter-final": r["quarterfinal"],
                             "Win group": r["win_group"]}[omkt])
            bet_label = f"{team}: {omkt}"

    if p_model is not None and p_model > 0:
        fair = 1 / p_model
        c1, c2, c3 = st.columns(3)
        c1.metric("Model probability", f"{p_model*100:.1f}%")
        c2.metric("Fair odds", f"{fair:.2f}")
        offered = c3.number_input("Bookmaker decimal odds", min_value=1.01,
                                  value=float(round(fair, 2)), step=0.05)
        edge = p_model - 1 / offered
        ev = p_model * offered - 1
        kf = bet.kelly_fraction(p_model, offered)
        c1, c2, c3 = st.columns(3)
        c1.metric("Edge vs implied", f"{edge*100:+.1f} pp")
        c2.metric("EV per $100", f"${ev*100:+.1f}")
        c3.metric("¼-Kelly stake", f"{kf/4*100:.1f}% of bankroll")

        if st.button("➕ Add to parlay"):
            st.session_state.setdefault("parlay", [])
            st.session_state["parlay"].append({"label": bet_label, "p": p_model})
            st.rerun()
    elif p_model is not None:
        st.info("The model gives this essentially zero probability.")

    legs = st.session_state.get("parlay", [])
    if legs:
        st.divider()
        st.subheader("🎫 Parlay slip")
        p_combo = float(np.prod([leg["p"] for leg in legs]))
        for i, leg in enumerate(legs, 1):
            st.markdown(f"{i}. **{leg['label']}** — {leg['p']*100:.1f}%")
        c1, c2, c3 = st.columns(3)
        c1.metric("Combined probability", f"{p_combo*100:.2f}%")
        c2.metric("Fair parlay odds", f"{1/p_combo:.2f}" if p_combo > 0 else "∞")
        offered_p = c3.number_input("Offered parlay odds", min_value=1.01,
                                    value=float(round(1 / max(p_combo, 1e-6), 2)),
                                    step=0.1, key="parlay_odds")
        ev_p = p_combo * offered_p - 1
        st.metric("Parlay EV per $100", f"${ev_p*100:+.1f}")
        st.caption("Legs assumed independent — true across different matches; do not "
                   "parlay two bets on the same game.")
        if st.button("Clear slip"):
            st.session_state["parlay"] = []
            st.rerun()

# ---------------------------------------------------------------- Trends
with tab_history:
    st.subheader("Champion odds over time")
    st.caption("One snapshot per saved simulation (sync, manual entry, or CLI run).")
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

# ---------------------------------------------------------------- Versus
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

# ---------------------------------------------------------------- Vault
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
