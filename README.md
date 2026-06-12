# WorldCupPred — FIFA World Cup 2026 Winner Prediction

Estimate every nation's probability of **winning the 2026 FIFA World Cup** (and of
reaching each stage), refresh those odds as real results come in, and document the whole
thing in an auto-generated Obsidian knowledge base.

The engine is a **statistical core feeding a Monte Carlo tournament simulation**:

- **International Elo** + **Dixon-Coles bivariate Poisson** produce a calibrated match model
  `predict_match(home, away, neutral) -> P(win/draw/loss) + scoreline distribution`.
- A **HistGradientBoosting** classifier and a **keras** neural net (with learned per-nation
  embeddings) form an **ensemble** layer on top.
- A **Monte Carlo** simulator plays the real 48-team 2026 format 50,000 times to get title
  and stage probabilities — and can **simulate from now**, fixing already-played results.

Data is **free-first**: the historical backbone (martj42 international results) and
eloratings.net need no keys; betting odds and live scores plug in via free-tier API keys
when available (`.env`), with scraping fallbacks.

---

## Quickstart

```bash
pip install -r requirements.txt

python scripts/refresh_data.py        # download + build the match-level dataset
python scripts/run_simulation.py      # fit models, simulate, write the title-odds table
streamlit run app.py                  # interactive console: enter scores, retrain, view odds
```

Optional: copy `.env.example` to `.env` and add free API keys to unlock live odds/scores.

## The live loop (during the tournament)

As matches are played, open the Streamlit app's **Update scores** tab (or run
`python -m src.update --match "Spain 3-1 Cape Verde"`):

1. the result is appended to the results store,
2. Elo updates incrementally and the base model optionally retrains,
3. the Monte Carlo **re-simulates the remaining fixtures**,
4. title odds refresh and the Obsidian vault notes regenerate.

## Project layout

```
config/      settings.yaml, groups_2026.yaml (the verified final draw), confederations.yaml
data/        raw -> interim -> processed (.parquet)
src/
  data/        scrapers + API clients + dataset builder
  features/    match-level feature engineering
  models/      elo, dixon_coles, ml_outcome, dl_outcome, ensemble, registry
  simulation/  tournament_2026 (format/bracket), monte_carlo, scenarios
  evaluation/  metrics (RPS/Brier), backtest (WC2018/2022, Euro2020/2024)
  vault/       Obsidian note generator
  update.py    live-update CLI    predict.py  title-odds table    config.py  single source of truth
scripts/     refresh_data.py, run_simulation.py
vault/       Obsidian knowledge base (auto-generated)
app.py       Streamlit operations console
tests/       tournament / monte carlo / elo unit tests
```

## Data sources

| Source | Key | Use |
|---|---|---|
| [martj42/international_results](https://github.com/martj42/international_results) | none | training backbone (all internationals 1872→) |
| [eloratings.net](https://www.eloratings.net) | none | international Elo cross-check |
| FIFA World Ranking (Wikipedia) | none | rank-diff feature |
| [The Odds API](https://the-odds-api.com) | free | betting odds (calibration + benchmark) |
| [football-data.org](https://www.football-data.org) | free | 2026 fixtures & live scores |
| Transfermarkt | none (scrape) | squad market value (talent proxy) |

## Methodology & status

See the Obsidian vault (`vault/`) for methodology notes, data-source cards, and model cards,
and `C:\Users\alexi\.claude\plans\make-a-full-plan-snappy-turtle.md` for the full build plan.

Build phases: **P0 scaffold ✓** · P1 data · P2 match engine · P3 simulation · P4 ML/DL
ensemble · P5 live loop · P6 vault · P7 app · P8 optional APIs + tests.

## Disclaimer

For research and entertainment. Predictions are probabilistic estimates, not betting advice.
