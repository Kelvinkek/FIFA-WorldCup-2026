# ⚽ Football Match-Outcome Predictor (Elo + form ML)

A focused **machine-learning** project: predict the result of an international football
match — **Home win / Draw / Away win** — from two complementary signals: an **Elo rating**
(opponent-adjusted team strength) and each team's **recent goal form** (last 10 matches),
plus head-to-head, confederation and venue.

Why match-level and not "who wins the World Cup"? There have only been **22 World Cups → 22
champions**, far too few to learn from. Match prediction has many more labelled examples — and
during a tournament you simply apply it match by match.

---

## Project layout

```
world-cup/
├── README.md
├── requirements.txt
├── .env.example           ← copy to .env, add your API-Football key (optional)
│
├── data/                  ← raw CSVs (+ live_results.csv fetched from the API)
│   ├── kaggle_2026/       ← World Cup finals + FIFA rankings + 2026 schedule
│   ├── maven_analytics/   ← international match history (1872–2022) + 2022 squads
│   └── live_results.csv   ← recent results pulled from API-Football (2022–2026)
│
├── src/                   ← the package (all logic)
│   ├── io.py              ← safe CSV reading
│   ├── teams.py           ← canonical team-name map
│   ├── load.py            ← clean, name-normalized loaders
│   ├── features.py        ← Elo ratings + goal form + h2h + confederation (no leakage)
│   ├── model.py           ← 5 tuned models + draw-aware prediction
│   ├── evaluate.py        ← accuracy / log-loss / RPS, model comparison, WC backtest
│   └── livedata.py        ← fetch recent results from API-Football
│
├── notebooks/
│   ├── explore_data.ipynb ← data exploration (EDA)
│   └── match_model.ipynb  ← train, compare models, measure accuracy
│
├── scripts/
│   ├── update_and_predict.py ← fetch latest results + predict upcoming fixtures
│   ├── tune_models.py     ← grid-search hyperparameter tuning (all 5 models)
│   ├── feature_analysis.py ← analysis charts (one image per chart) → reports/
│   └── viz.py             ← polished presentation visuals → reports/
│
└── docs/
    └── DATA_GUIDE.md      ← plain-English guide to every file & football term
```

**Design rule:** logic lives in `src/`; notebooks/scripts import it. Only `src/load.py` reads `data/`.

---

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

Open `notebooks/match_model.ipynb` in VS Code, pick the `.venv` kernel, **Run All**. Or:

```python
from src import features, evaluate
table, _ = features.build_training_table()
print(evaluate.compare_models(table, "2021-01-01"))   # accuracy / log-loss / RPS per model
```

Analysis charts: `python scripts/feature_analysis.py` → `reports/*.png`.

---

## The model

**Features** (`src/features.py`) — all known before kick-off, no leakage:

| Feature | Meaning |
|---|---|
| `elo_diff`, `home_elo`, `away_elo` | **Elo strength** (opponent-adjusted) — the dominant feature |
| `home/away_goals_for_avg` | avg goals **scored**, last 10 matches (recent attacking form) |
| `home/away_goals_against_avg` | avg goals **conceded**, last 10 matches (recent defensive form) |
| `h2h_home_wins / draws / away_wins` | share of past meetings between the two teams |
| `home/away_confederation` | region code (UEFA, CONMEBOL, …) |
| `is_neutral` | neutral venue? |

**Models compared** (`src/model.py`) — all 5 hyperparameter-tuned (`scripts/tune_models.py`):
`RandomForest`, `GradientBoosting`, `XGBoost`, `ExtraTrees`, `LogisticRegression`.
Single-label output is **draw-aware** (a plain argmax almost never predicts a draw).

**Validation** (`src/evaluate.py`): always a **time split** (train past, test future), averaged
walk-forward. Plus a `world_cup_backtest()` sanity check on the 2022 finals. Metrics:
**accuracy**, **log-loss**, **RPS**, calibration, permutation importance.

---

## Headline result

Walk-forward test (2022/2023/2024 splits; ~4,400 matches, all teams):

| Model | Accuracy | Log-loss |
|---|---|---|
| Extra Trees | ~59.4% | 0.900 |
| **Random Forest** | **~59.3%** | 0.889 |
| XGBoost | ~59.1% | 0.889 |
| Gradient Boosting | ~58.9% | **0.888** |
| Logistic | ~58.5% | 0.894 |
| *always predict home* | ~44% | — |

All five cluster at **~59%** — the result reflects the *data*, not one lucky model. On the **2022
World Cup backtest** it drops to **~52%** (a real tournament is harder: elite-vs-elite, neutral).

**`elo_diff` is the dominant feature** (~20× any other); goal form adds a complementary lift.
Elo + form together beat either alone, and adding Elo also improves draw prediction for
evenly-matched sides.

## Notes

- Elo and goal form are *results-derived* (built from past matches) but **leak-free** — they only
  use games played before the one predicted.
- Next levers: fix draw prediction (class weighting), opponent-weighted form, xG features (paid feed).
