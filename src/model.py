"""Match-prediction model — trained without leaking the future.

`OutcomeClassifier` predicts P(home win), P(draw), P(away win) for a match.
It exposes `.fit(table, cutoff)` (train only on matches strictly before `cutoff`,
so a backtest never sees the matches it predicts) and `.predict_match(...)`.

The feature vector for a match is built by `match_features()` so training and
prediction use an identical layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

# Feature columns — Elo (opponent-adjusted strength) + recent goal form + h2h +
# confederation + neutral. Elo and goal form are the two dominant, complementary signals.
FEATURES = [
    "elo_diff", "home_elo", "away_elo",
    "home_goals_for_avg", "home_goals_against_avg",
    "away_goals_for_avg", "away_goals_against_avg",
    "h2h_home_wins", "h2h_draws", "h2h_away_wins",
    "home_confederation", "away_confederation", "is_neutral",
]

# Draws are rarely the single most-likely outcome, so a plain argmax almost never
# predicts one. Nudging the draw probability up by this factor before picking a single
# label recovers ~6x more drawn games at no accuracy cost. Used by predict_label().
DRAW_BOOST = 1.4


def match_features(home_form, away_form, neutral: bool, h2h=(1/3, 1/3, 1/3)) -> dict:
    """Assemble one match's features for a brand-new fixture.

    `home_form`/`away_form` are rows from `features.team_form()` (elo, goals_for_avg,
    goals_against_avg, confederation); `h2h` is (home_share, draw_share, away_share)
    from `features.h2h_table()`.
    """
    he, ae = float(home_form["elo"]), float(away_form["elo"])
    return {
        "elo_diff": he - ae, "home_elo": he, "away_elo": ae,
        "home_goals_for_avg": float(home_form["goals_for_avg"]),
        "home_goals_against_avg": float(home_form["goals_against_avg"]),
        "away_goals_for_avg": float(away_form["goals_for_avg"]),
        "away_goals_against_avg": float(away_form["goals_against_avg"]),
        "h2h_home_wins": float(h2h[0]), "h2h_draws": float(h2h[1]), "h2h_away_wins": float(h2h[2]),
        "home_confederation": int(home_form["confederation"]),
        "away_confederation": int(away_form["confederation"]),
        "is_neutral": float(bool(neutral)),
    }


def _xy(table: pd.DataFrame, cutoff) -> pd.DataFrame:
    """Rows strictly before `cutoff` with all features present."""
    return table[table["date"] < pd.Timestamp(cutoff)].dropna(subset=FEATURES)


@dataclass
class OutcomeClassifier:
    """Classifier over {H, D, A}. `algo` selects the ML model.

    Hyperparameters tuned by walk-forward grid search (`scripts/tune_models.py`) on
    the form-feature dataset (~4,400 matches). Random Forest is the default/best.

    Five models for comparison:
    algo='random_forest' -> RandomForestClassifier (best)
    algo='logistic'      -> LogisticRegression (linear)
    algo='gbm'           -> HistGradientBoostingClassifier (boosting)
    algo='extra_trees'   -> ExtraTreesClassifier (extra-randomised trees)
    algo='xgb'           -> XGBClassifier (gradient boosting, XGBoost)
    """
    algo: str = "random_forest"

    def _make(self):
        if self.algo == "logistic":
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=3000, C=0.01),
            )
        if self.algo == "rf":
            return RandomForestClassifier(
                n_estimators=300, max_depth=None, min_samples_leaf=15,
                n_jobs=-1, random_state=0,
            )
        if self.algo == "extra_trees":
            return ExtraTreesClassifier(
                n_estimators=300, max_depth=12, min_samples_leaf=5,
                n_jobs=-1, random_state=0,
            )
        if self.algo == "xgb":
            return XGBClassifier(
                n_estimators=400, max_depth=2, learning_rate=0.02,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
                eval_metric="mlogloss", random_state=0, n_jobs=-1,
            )
        return HistGradientBoostingClassifier(
            learning_rate=0.02, max_depth=2, max_iter=150,
            l2_regularization=1.0, min_samples_leaf=20,
        )

    def fit(self, table: pd.DataFrame, cutoff) -> "OutcomeClassifier":
        train = _xy(table, cutoff)
        X, y = train[FEATURES].to_numpy(), train["outcome"]
        self.clf_ = self._make()
        if self.algo == "xgb":
            # XGBoost needs integer class labels for multiclass.
            self.le_ = LabelEncoder().fit(y)
            self.clf_.fit(X, self.le_.transform(y))
            self.classes_ = list(self.le_.classes_)
        else:
            self.clf_.fit(X, y)
            self.classes_ = list(self.clf_.classes_)
        return self

    def predict_proba_df(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Class probabilities for many matches at once (rows carry FEATURES)."""
        probs = self.clf_.predict_proba(rows[FEATURES].to_numpy())
        return pd.DataFrame(probs, columns=self.classes_, index=rows.index)

    def predict_match(self, feats: dict) -> dict:
        x = np.array([[feats[f] for f in FEATURES]])
        probs = self.clf_.predict_proba(x)[0]
        return {c: float(p) for c, p in zip(self.classes_, probs)}

    def predict_label(self, feats: dict, draw_boost: float = DRAW_BOOST) -> str:
        """Single most-likely outcome, draw-aware.

        Plain argmax almost never picks a draw (a draw is rarely the single most
        likely outcome). Nudging the draw probability up by `draw_boost` before
        choosing recovers ~6x more drawn games at no accuracy cost — see
        scripts/tune_models.py / the draw-boost sweep. Probabilities from
        `predict_match` are left raw (they're already well-calibrated for log-loss).
        """
        p = dict(self.predict_match(feats))
        p["D"] = p.get("D", 0.0) * draw_boost
        return max(p, key=p.get)

    def feature_importance(self, rows: pd.DataFrame, n_repeats: int = 5) -> pd.Series:
        """Permutation importance on the given rows (model-agnostic)."""
        from sklearn.inspection import permutation_importance

        r = permutation_importance(
            self.clf_, rows[FEATURES].to_numpy(), rows["outcome"].to_numpy(),
            n_repeats=n_repeats, random_state=0, scoring="neg_log_loss",
        )
        return pd.Series(r.importances_mean, index=FEATURES).sort_values(ascending=False)
