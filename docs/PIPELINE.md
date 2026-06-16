# Pipeline & Methodology

A complete walkthrough of **what this project does, why, and how** - from raw CSV to a
match prediction. Pair this with [DATA_GUIDE.md](DATA_GUIDE.md) (plain-English data dictionary)
and the top-level [README](../README.md) (quick start).

---

## 1. Purpose - what problem are we solving?

**Goal:** predict the outcome of an international football match - **Home win / Draw / Away win** -
and apply it to the **2026 FIFA World Cup**.

**Why match-level, not "who wins the World Cup"?** There have only been **22 World Cups → 22
champions** - far too few examples to train a model to pick a tournament winner directly. A *match*
is the right unit: there are **thousands** of them, so it's a proper supervised-learning problem.
A tournament prediction is then just *many* match predictions stitched together (e.g. summing
expected points to project group tables).

**Why it's worth doing:**
- It's an honest, end-to-end **data-science exercise**: messy multi-source data → cleaning →
  feature engineering → model selection → leakage-free validation → a usable forecast.
- The result is **interpretable and defensible** - every prediction traces back to two intuitive
  signals (team strength + recent form), not a black box.
- It demonstrates the *discipline* that matters in real ML: no data leakage, honest baselines,
  walk-forward validation, and resisting the urge to over-engineer.

---

## 2. The data

Three sources are merged into **one chronological match table**:

| Source | Contents | Coverage |
|---|---|---|
| `data/maven_analytics/international_matches.csv` | all internationals (friendlies, qualifiers, continental cups) | 1872–2022 |
| `data/kaggle_2026/matches_1930_2022.csv` | World Cup finals | 1930–2022 |
| `data/live_results.csv` | recent results fetched from **API-Football** | 2022–2026 |
| `data/eloratings_history.csv` | **eloratings.net** pre-match Elo (fetched per team) | 1872–today |

Plus supporting files: the **2026 schedule** (`kaggle_2026/schedule_2026.csv`, the real
post-draw fixtures) and **FIFA rankings** (used only to map teams → confederation).

**Why these sources?** Maven gives 150 years of depth; Kaggle adds the World Cup finals; the API
fills the recent 2022–2026 gap; and **eloratings.net** supplies professional, importance-weighted
team strength (see `src/elodata.py`). For *training* we keep matches from **2015 onward** - recent
enough to be relevant, deep enough to learn from. After joining eloratings' pre-match Elo, **~3,900
matches** carry the full feature set (a handful of obscure/regional teams without an eloratings page
drop out).

---

## 3. The pipeline at a glance

```
 io.py  →  teams.py  →  load.py  →  features.py  →  model.py  →  evaluate.py
(read)    (fix names)   (clean)     (engineer)      (predict)    (validate)
```

Each stage is one module in `src/`. The golden rule: **logic lives in `src/`; notebooks and
scripts only import it; only `src/load.py` touches `data/`.**

### Stage by stage

1. **Read** (`io.py`) - load CSVs safely (auto-detect encoding; the 2022 squads file is cp1252).
2. **Normalize names** (`teams.py`) - the sources spell teams differently (`Iran` vs `IR Iran`).
   A canonical map fixes this so joins don't silently drop rows.
3. **Clean & load** (`load.py`) - one tidy loader per file, returning name-normalized DataFrames.
4. **Engineer features** (`features.py`) - turn matches into model inputs, *without leakage*.
5. **Train & predict** (`model.py`) - 5 tuned classifiers behind one interface.
6. **Validate** (`evaluate.py`) - time-split accuracy / log-loss / RPS, plus a World Cup backtest.

---

## 4. Feature engineering - the heart of it

Every feature uses **only information available before kick-off** (no leakage). The 13 features:

| Feature group | Columns | What it captures |
|---|---|---|
| **Elo strength** ⭐ | `elo_diff`, `home_elo`, `away_elo` | *opponent-adjusted* team strength from **eloratings.net** (professional, importance-weighted) - it knows you beat Brazil, not San Marino |
| **Recent goal form** ⭐ | `home/away_goals_for_avg`, `home/away_goals_against_avg` | avg goals scored / conceded over the last **3** matches |
| **Head-to-head** | `h2h_home_wins`, `h2h_draws`, `h2h_away_wins` | share of past meetings between the two teams |
| **Confederation** | `home/away_confederation` | region code (UEFA, CONMEBOL, …) |
| **Venue** | `is_neutral` | neutral venue flag |

**Two signals do the work, and they're complementary:**
- **Elo** (from **eloratings.net**, joined pre-match via `src/elodata.py`) - a professional rating
  where, after each match, the winner takes points from the loser (more for a bigger win, a stronger
  opponent, and a more important match). It captures *long-term, opponent-adjusted* strength.
  **`elo_diff` is by far the most important feature** (~20× any other).
- **Goal form** (`add_form`) - a team's recent scoring/conceding trend. It captures *short-term*
  momentum that Elo is slow to reflect. Window tuned to **3 matches** (best log-loss; 3–10 are
  near-equivalent, 15+ goes stale).

A homemade Elo (`compute_elo`) is kept in `features.py` for reference, but the eloratings ratings
replaced it - switching lifted walk-forward accuracy from **~59.3% to ~60.5%** (log-loss 0.889 → 0.873).
(A `match_importance` feature was tested and dropped: it's constant for a World-Cup-only predictor,
and the eloratings Elo already bakes match importance into the rating.)

**No-leakage discipline:** form uses `.shift()` so a match never sees its own result; Elo is the
rating *before* the match; h2h counts only *prior* meetings. This is the single most important
correctness property - without it the model would look brilliant in testing and fail in reality.

**What we deliberately left out** (tested, didn't help): squad/player features, possession/shots/xG
(no free source at scale for internationals), rest days, Elo momentum, clean-sheet rate - all
either redundant with Elo+form or unavailable. *More features ≠ better.*

---

## 5. The models

Five classifiers, all **hyperparameter-tuned** by walk-forward grid search (`scripts/tune_models.py`),
exposed through one class `OutcomeClassifier(algo=...)`:

| Model | Type | Notes |
|---|---|---|
| **Logistic Regression** | linear | **the default, used for predictions - best accuracy/log-loss/RPS + best-calibrated** |
| XGBoost | boosted trees | competitive |
| Gradient Boosting | boosted trees | competitive |
| Random Forest | bagged trees | competitive (alias `rf` / `random_forest`) |
| Extra Trees | randomised trees | good accuracy but worst calibration (over-confident probs) |

**Why these (and not deep learning)?** This is **small, tabular data** (~3,900 rows, 13 features) -
the regime where tree ensembles and linear models dominate. A neural net was tested and came
**last** (it needs far more data). All five tuned models cluster at **~60–61%**, which is reassuring:
the result reflects the *data*, not one lucky algorithm.

**Draw-aware prediction.** A plain "pick the most likely outcome" almost never predicts a draw
(a draw is rarely the single top probability), even though ~23% of matches are drawn. So
`predict_label()` gently boosts the draw probability (`DRAW_BOOST`) before choosing - recovering
~6× more drawn games at no accuracy cost. The raw probabilities (`predict_match()`) are left
untouched, so log-loss scoring stays clean.

---

## 6. Validation - how we know it works

**Always a time split, never random.** Random splits leak the future into the past. We train on
matches *before* a cut-off date and test on matches *after* it.

- **Walk-forward** (`evaluate.walk_forward`) - repeat the split at **2022, 2023, 2024** and
  average. This is the **primary** metric (large, diverse test sets).
- **World Cup backtest** (`evaluate.world_cup_backtest`) - train on everything before the 2022 WC,
  predict its 64 matches. A **supplementary** real-world sanity check (small, noisy - read it as a
  reality check, not the headline).

**Metrics:** accuracy, **log-loss** (rewards calibrated probabilities), **RPS** (football-standard),
plus **calibration** curves and **permutation feature importance**.

### Results

| Test | Accuracy | Log-loss |
|---|---|---|
| **Walk-forward** (general matches) | **~60.5%** | ~0.85 |
| World Cup backtest (2022 finals) | ~50% | ~1.04 |
| *always predict home* | ~52% | - |
| *random guess* | ~33% | 1.10 |

The gap (61% vs 50%) is honest and instructive: a top-heavy tournament like the World Cup is
**harder** than general fixtures (elite-vs-elite, neutral venues, no easy lopsided games).

---

## 7. How to run it

```powershell
# setup
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```python
# compare the 5 models
from src import features, evaluate
table, _ = features.build_training_table()
print(evaluate.compare_models(table, "2023-01-01"))
```

```powershell
# predict upcoming 2026 fixtures (add --fetch to pull the latest results first)
.venv\Scripts\python.exe scripts\update_and_predict.py --fetch

# analysis charts / presentation visuals
.venv\Scripts\python.exe scripts\feature_analysis.py
.venv\Scripts\python.exe scripts\viz.py
.venv\Scripts\python.exe scripts\viz_groups.py     # 2026 group-stage prediction charts
.venv\Scripts\python.exe scripts\viz_results.py    # prediction tracker + per-day charts
```

Or open `notebooks/match_model.ipynb` and **Run All** for the narrated end-to-end story.

---

## 8. Honest limitations

- **~61% is near the ceiling for free data.** Football is genuinely high-variance; even a perfect
  model can't predict draws and upsets reliably. Confirmed repeatedly - more features and more
  tuning barely move it (the eloratings switch added ~1.4 points, the most of anything tried).
- **No xG / injuries / lineups** - these would help but need a paid data feed (or are infeasible to
  backfill on the free API's 100 requests/day).
- **Draws are inherently hard** - the model predicts them at a realistic rate now, but you can't
  raise accuracy by predicting *more* of them (it's a true trade-off).
- **The WC backtest is small (64 matches)** - treat the ~52% as indicative, not precise.

---

## 9. Maintaining it for future competitions

The model is only as fresh as its data. One script does the whole daily refresh:

```powershell
.venv\Scripts\python.exe scripts\daily_update.py
```

It (1) pulls finished matches from API-Football, (2) refreshes **eloratings.net** Elo + recent
results, (3) rebuilds features, (4) regenerates the group-stage charts, and (5) updates the
**prediction tracker** (`reports/viz_prediction_tracker.png`) plus a **per-day chart** for each
match-day (`reports/daily/YYYY-MM-DD.png`).

To run it unattended once a day, register the Windows Scheduled Task:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_schedule.ps1     # daily at 09:00
```

During a tournament each result updates Elo/form, which sharpens the next prediction; between
tournaments the eloratings data simply doesn't change until teams play again. Retraining is cheap
(seconds), so the daily cadence is effectively free.

---

### TL;DR

> Merge 150 years of international results → engineer **eloratings.net Elo + recent goal form +
> recent goal form** (no leakage) → train **5 tuned models** (Logistic the default) → validate
> by **time-split walk-forward** → predict any match's **Home/Draw/Away** odds, draw-aware. Result:
> **~60.5%** general accuracy, **~50%** on a real World Cup - honest, interpretable, and reproducible.
