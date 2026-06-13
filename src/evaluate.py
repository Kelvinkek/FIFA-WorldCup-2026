"""Validation & accuracy scoring for the match-outcome classifier.

Key idea (no leakage): always train on matches *before* a cut-off date and test on
matches at/after it - never a random split.

Metrics (lower log-loss / rps is better):
  accuracy  - fraction of matches whose most-likely outcome was correct
  log_loss  - rewards calibrated probabilities
  rps       - Ranked Probability Score (the football-standard metric)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOMES = ["H", "D", "A"]


def metrics_from_probs(probs: pd.DataFrame, actual: pd.Series) -> dict:
    """Accuracy / log-loss / RPS from a probability DataFrame (cols H, D, A)."""
    P = probs[OUTCOMES].to_numpy()
    pred = probs[OUTCOMES].idxmax(axis=1).to_numpy()
    acc = float((pred == actual.to_numpy()).mean())

    eps = 1e-15
    col = {o: i for i, o in enumerate(OUTCOMES)}
    idx = actual.map(col).to_numpy()
    p_true = np.clip(P[np.arange(len(P)), idx], eps, 1.0)
    ll = float(-np.log(p_true).mean())

    onehot = np.eye(len(OUTCOMES))[idx]
    rps = float((((np.cumsum(P, 1) - np.cumsum(onehot, 1)) ** 2).sum(1) / (len(OUTCOMES) - 1)).mean())
    return {"n": int(len(actual)), "accuracy": acc, "log_loss": ll, "rps": rps}


def compare_models(table: pd.DataFrame, train_before: str) -> pd.DataFrame:
    """Train each model on matches before `train_before`, test on those at/after it.

    Returns one row per model (logistic / random_forest / gbm) with the metrics.
    """
    from .model import FEATURES, OutcomeClassifier

    test = table[table["date"] >= pd.Timestamp(train_before)].dropna(subset=FEATURES)
    actual = test["outcome"]

    rows = []
    for name, m in _candidates().items():
        m.fit(table, train_before)
        rows.append({"model": name, **metrics_from_probs(m.predict_proba_df(test)[OUTCOMES], actual)})
    return pd.DataFrame(rows).set_index("model")


def _candidates() -> dict:
    from .model import OutcomeClassifier
    return {
        "logistic": OutcomeClassifier("logistic"),
        "random_forest": OutcomeClassifier("rf"),
        "gbm": OutcomeClassifier("gbm"),
        "extra_trees": OutcomeClassifier("extra_trees"),
        "xgboost": OutcomeClassifier("xgb"),
    }


def world_cup_backtest(table: pd.DataFrame, year: int = 2022,
                       start_date: str = "2022-11-20") -> pd.DataFrame:
    """Supplementary real-world check: train on everything BEFORE the tournament,
    then predict that World Cup's finals matches. Small sample (~64), so read it as a
    sanity check, not the primary metric (use walk_forward for that)."""
    from .model import FEATURES

    test = table[(table["competition"] == "FIFA World Cup")
                 & (table["date"].dt.year == year)].dropna(subset=FEATURES)
    rows = []
    for name, m in _candidates().items():
        m.fit(table, start_date)
        rows.append({"model": name, **metrics_from_probs(m.predict_proba_df(test)[OUTCOMES], test["outcome"])})
    return pd.DataFrame(rows).set_index("model")


def walk_forward(table: pd.DataFrame, cutoffs: list[str]) -> pd.DataFrame:
    """Average each model's scores across several train/test split dates."""
    frames = [compare_models(table, c).assign(cutoff=c) for c in cutoffs]
    allr = pd.concat(frames)
    return allr.groupby(level=0)[["accuracy", "log_loss", "rps"]].mean().round(4)


def calibration_table(probs: pd.DataFrame, actual: pd.Series,
                      outcome: str = "H", bins: int = 10) -> pd.DataFrame:
    """Reliability of predicted probabilities: predicted vs observed frequency."""
    p = probs[outcome].to_numpy()
    y = (actual.to_numpy() == outcome).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        mask = which == b
        if mask.sum() == 0:
            continue
        out.append({"pred_mean": float(p[mask].mean()),
                    "obs_freq": float(y[mask].mean()), "n": int(mask.sum())})
    return pd.DataFrame(out)
