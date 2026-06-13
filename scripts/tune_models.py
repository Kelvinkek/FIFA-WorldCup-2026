"""Fine-tune the 5 models by walk-forward grid search.

    python scripts/tune_models.py

For each model (logistic, random forest, gradient boosting, extra trees, XGBoost) it
tries a grid of hyperparameters, scores every config by the mean log-loss across
several train/test split dates (no leakage), and prints the best. Those winners are
baked into `src/model.py` (OutcomeClassifier._make).
"""

import sys
import itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import features, model, evaluate

table, _ = features.build_training_table()
F = model.FEATURES
CUTS = ["2022-01-01", "2023-01-01", "2024-01-01"]


def wf(make):
    lls, accs = [], []
    for cut in CUTS:
        tr = table[table["date"] < pd.Timestamp(cut)].dropna(subset=F)
        te = table[table["date"] >= pd.Timestamp(cut)].dropna(subset=F)
        m = make().fit(tr[F], tr["outcome"])
        P = pd.DataFrame(m.predict_proba(te[F]), columns=list(m.classes_), index=te.index)
        r = evaluate.metrics_from_probs(P[["H", "D", "A"]], te["outcome"])
        lls.append(r["log_loss"]); accs.append(r["accuracy"])
    return float(np.mean(lls)), float(np.mean(accs))


def search(name, grid, make):
    best = None
    for params in grid:
        ll, acc = wf(lambda p=params: make(p))
        if best is None or ll < best[0]:
            best = (ll, acc, params)
    print(f"{name}: log-loss {best[0]:.4f}  acc {best[1]*100:.1f}%  params={best[2]}")
    return best


if __name__ == "__main__":
    search("LOGISTIC", [{"C": c} for c in [0.01, 0.03, 0.05, 0.1, 0.3, 1, 3]],
           lambda p: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, **p)))

    rf_grid = [{"n_estimators": n, "max_depth": d, "min_samples_leaf": l}
               for n, d, l in itertools.product([300, 600], [3, 5, 8, None], [5, 15, 30])]
    search("RANDOM FOREST", rf_grid,
           lambda p: RandomForestClassifier(n_jobs=-1, random_state=0, **p))

    gb_grid = [{"learning_rate": lr, "max_depth": d, "max_iter": i,
                "l2_regularization": l2, "min_samples_leaf": leaf}
               for lr, d, i, l2, leaf in itertools.product(
                   [0.02, 0.05], [2, 3], [150, 300], [1.0, 10.0], [20, 40])]
    search("GRADIENT BOOSTING", gb_grid,
           lambda p: HistGradientBoostingClassifier(**p))

    from sklearn.ensemble import ExtraTreesClassifier
    et_grid = [{"n_estimators": n, "max_depth": d, "min_samples_leaf": l}
               for n, d, l in itertools.product([300, 600], [None, 12, 20], [5, 15, 30])]
    search("EXTRA TREES", et_grid,
           lambda p: ExtraTreesClassifier(n_jobs=-1, random_state=0, **p))

    # XGBoost needs integer labels — wrap so the search helper can score it.
    from xgboost import XGBClassifier
    from sklearn.preprocessing import LabelEncoder

    class _XGBWrap:
        def __init__(self, **p): self.p = p
        def fit(self, X, y):
            self.le = LabelEncoder().fit(y); self.m = XGBClassifier(
                eval_metric="mlogloss", random_state=0, n_jobs=-1, colsample_bytree=0.8, **self.p)
            self.m.fit(X, self.le.transform(y)); self.classes_ = self.le.classes_; return self
        def predict_proba(self, X): return self.m.predict_proba(X)

    xgb_grid = [{"n_estimators": n, "max_depth": d, "learning_rate": lr,
                 "subsample": s, "reg_lambda": rl}
                for n, d, lr, s, rl in itertools.product(
                    [200, 400], [2, 3, 4], [0.02, 0.05], [0.8, 1.0], [1.0, 5.0])]
    search("XGBOOST", xgb_grid, lambda p: _XGBWrap(**p))
