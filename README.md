# ⚽ Football Match-Outcome Predictor (Elo + form ML)

A focused **machine-learning** project: predict the result of an international football
match (**Home win / Draw / Away win**) from two complementary signals: an **Elo rating**
(opponent-adjusted team strength) and each team's **recent goal form** (last 3 matches),
plus head-to-head, confederation and venue.

Why match-level and not "who wins the World Cup"? There have only been **22 World Cups -> 22
champions**, far too few to learn from. Match prediction has many more labelled examples, and
during a tournament you simply apply it match by match.

---

## Project layout

```
world-cup/
├── README.md
├── requirements.txt
├── .env.example           <- copy to .env, add your API-Football key (optional)
│
├── data/                  <- raw CSVs (+ live_results.csv fetched from the API)
│   ├── kaggle_2026/       <- World Cup finals + FIFA rankings + 2026 schedule
│   ├── maven_analytics/   <- international match history (1872-2022) + 2022 squads
│   └── live_results.csv   <- recent results pulled from API-Football (2022-2026)
│
├── src/                   <- the package (all logic)
│   ├── io.py              <- safe CSV reading
│   ├── teams.py           <- canonical team-name map
│   ├── load.py            <- clean, name-normalized loaders
│   ├── elodata.py         <- eloratings.net Elo (pre-match, leak-free) fetch + cache
│   ├── features.py        <- Elo + goal form + h2h + confederation (no leakage)
│   ├── model.py           <- 5 tuned models + draw-aware prediction
│   ├── evaluate.py        <- accuracy / log-loss / RPS, model comparison, WC backtest
│   └── livedata.py        <- fetch recent results from API-Football
│
├── notebooks/
│   ├── explore_data.ipynb <- data exploration (EDA)
│   └── match_model.ipynb  <- train, compare models, measure accuracy
│
├── scripts/
│   ├── daily_update.py       <- daily: API results + eloratings refresh + regen charts
│   ├── setup_schedule.ps1    <- register/remove the daily Windows Scheduled Task
│   ├── update_and_predict.py <- fetch latest results + predict upcoming fixtures
│   ├── tune_models.py        <- grid-search hyperparameter tuning (all 5 models)
│   ├── register_model.py     <- train + log + register a model version in MLflow
│   ├── feature_analysis.py   <- analysis charts (one image per chart) -> reports/
│   ├── viz.py                <- polished presentation visuals -> reports/
│   ├── viz_groups.py         <- 2026 group-stage prediction charts -> reports/
│   ├── viz_results.py        <- prediction tracker: initial picks vs actual results
│   └── viz_r32.py            <- Round-of-32 knockout predictions -> reports/round_of_32/
│
└── docs/
    ├── PIPELINE.md        <- full pipeline & methodology (purpose, features, validation)
    └── DATA_GUIDE.md      <- plain-English guide to every file & football term
```

📖 **For the full methodology** (purpose, data, feature engineering, models, validation),
see **[docs/PIPELINE.md](docs/PIPELINE.md)**.

**Design rule:** logic lives in `src/`; notebooks/scripts import it. Only `src/load.py` reads `data/`.

---

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The live-data fetch is optional. To use it, copy `.env.example` to `.env` and add a free
[API-Football](https://www.api-football.com/) key. Everything else runs from the bundled CSVs.

## Run

Open `notebooks/match_model.ipynb` in VS Code, pick the `.venv` kernel, **Run All**. Or from Python:

```python
from src import features, evaluate
table, _ = features.build_training_table()
print(evaluate.compare_models(table, "2021-01-01"))   # accuracy / log-loss / RPS per model
```

Generate the charts:

```powershell
.venv\Scripts\python.exe scripts/feature_analysis.py   # analysis charts -> reports/
.venv\Scripts\python.exe scripts/viz_groups.py         # 2026 group-stage predictions -> reports/
```

---

## The model

**Features** (`src/features.py`): all known before kick-off, no leakage:

| Feature | Meaning |
|---|---|
| `elo_diff`, `home_elo`, `away_elo` | **eloratings.net Elo** (professional, importance-weighted strength), the dominant feature |
| `home/away_goals_for_avg` | avg goals **scored**, last 3 matches (recent attacking form) |
| `home/away_goals_against_avg` | avg goals **conceded**, last 3 matches (recent defensive form) |
| `h2h_home_wins / draws / away_wins` | share of past meetings between the two teams |
| `home/away_confederation` | region code (UEFA, CONMEBOL, ...) |
| `is_neutral` | neutral venue (no home advantage)? used to give the 2026 hosts a home boost |

**Models compared** (`src/model.py`): all 5 hyperparameter-tuned (`scripts/tune_models.py`):
`LogisticRegression`, `XGBoost`, `GradientBoosting`, `RandomForest`, `ExtraTrees`. They cluster
~60-61%; **Logistic** is the model the prediction scripts use - best on accuracy, log-loss *and*
RPS, simplest, and best-calibrated probabilities (which matter most for the % bars).
Single-label output is **draw-aware** (a plain argmax almost never predicts a draw).

**Validation** (`src/evaluate.py`): always a **time split** (train past, test future), averaged
walk-forward. Plus a `world_cup_backtest()` sanity check on the 2022 finals. Metrics:
**accuracy**, **log-loss**, **RPS**, calibration, permutation importance.

---

## Data sources

- **[Kaggle / FIFA World Cup](https://www.kaggle.com/)**: World Cup finals (1930-2022), FIFA
  rankings, and the official 2026 group-stage schedule.
- **[Maven Analytics / International Football Results](https://www.mavenanalytics.io/data-playground)**:
 ~17,000 friendlies, qualifiers and continental matches (1872-2022), plus 2022 squads.
- **[API-Football](https://www.api-football.com/)** (api-sports.io): recent / ongoing results
  to keep the model current (optional; requires a free key).
- **[eloratings.net](https://eloratings.net/)**: professional, importance-weighted national-team
  Elo ratings (served as raw TSV). Fetched + cached to `data/eloratings_history.csv` by `src/elodata.py`.

Data is included for convenience and remains the property of its original providers; please
credit them if you reuse it.

---

## Daily automation

`scripts/daily_update.py` refreshes everything in one run: pulls finished results from
API-Football, refreshes the eloratings Elo history + recent results, rebuilds features,
regenerates the group-stage prediction charts, and updates the **prediction tracker**
(`reports/viz_prediction_tracker.png`) — the model's pre-tournament picks graded against
the actual results as they come in — plus a **per-day chart** for each match-day
(`reports/daily/YYYY-MM-DD.png`): that day's fixtures as Win/Draw/Lose probability bars,
with the actual result outlined once played. It uses only ~3 of the 100 daily API requests.

The "initial" predictions are frozen once into `data/initial_predictions.csv` (model
trained only on pre-kickoff data, so no leakage) and never overwritten, so the tracker is
an honest record of how the original forecasts held up.

Schedule it to run once a day on Windows (no admin needed):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_schedule.ps1            # daily at 09:00
powershell -ExecutionPolicy Bypass -File scripts\setup_schedule.ps1 -At 21:30  # custom time
powershell -ExecutionPolicy Bypass -File scripts\setup_schedule.ps1 -Remove    # uninstall
```

Output is appended to `logs/daily_update.log`. Run it by hand anytime with
`.venv\Scripts\python.exe scripts\daily_update.py`.

## Round-of-32 predictions

The knockout bracket isn't in the static schedule (it depends on final group standings),
and neither football API carries it on a free plan — API-Football is empty for the 2026
World Cup, and football-data.org gates the World Cup behind a token/tier. So the R32
matchups come from **fixturedownload.com**, a free, no-auth feed that fills in the real
ties as the bracket is decided (`src/knockout.py`).

`scripts/viz_r32.py` pulls those 16 ties, predicts each with the same logistic +
draw-aware model, and writes to its **own folder** (the daily charts are untouched):

```powershell
.venv\Scripts\python.exe scripts\viz_r32.py
```

- `reports/round_of_32/round_of_32.png` — all 16 ties at a glance
- `reports/round_of_32/YYYY-MM-DD.png` — one PNG per match-day (same look as `daily/`)

Once a tie is played, its actual result is outlined and graded, exactly like the daily
tracker. To switch the source to the **football-data.org** API instead, add a free
`FOOTBALL_DATA_TOKEN=...` to `.env` (it's used automatically when present, with
fixturedownload as the fallback).

## Model versioning (MLflow)

Every daily retrain is logged as a new **MLflow** model version, so you keep a full,
comparable history instead of silently overwriting the model. Each run records the params
(algo, feature list, form/train windows, draw-boost, a data fingerprint, the git commit),
the walk-forward + 2022-World-Cup-backtest metrics, and the fitted model artifact. The
freshly trained model is registered under `worldcup-outcome-model` and tagged `@champion`.

```powershell
.venv\Scripts\python.exe scripts\register_model.py                # logistic -> new version + @champion
.venv\Scripts\python.exe scripts\register_model.py --algo xgb     # log another algo to compare
.venv\Scripts\python.exe scripts\register_model.py --no-champion  # register but don't promote
```

Load "the current production model" anywhere by alias — no refitting:

```python
from src import registry
clf = registry.load_champion()           # -> a ready OutcomeClassifier
```

Browse runs, metrics-over-time and the registry in the web UI:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db    # then open http://127.0.0.1:5000
```

The tracking store (`mlflow.db`) and artifacts (`mlartifacts/`) are local and gitignored.