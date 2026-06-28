"""MLflow model versioning for the match-outcome classifier.

Every training run is logged to a **local** MLflow store (./mlruns) with:
  - params : algo, feature list, form/train windows, draw-boost, training-row count,
             data fingerprint (hash of the eloratings inputs), git commit
  - metrics: walk-forward accuracy / log-loss / RPS + the 2022 World-Cup backtest
  - artifact: the fitted model (standard sklearn flavor) + meta.json

The freshly trained model is then registered in the MLflow **Model Registry** under
`REGISTERED_NAME` and given the `@champion` alias, so prediction code can pull "the
current production model" by alias (`load_champion()`) instead of refitting, and you
get a full, comparable history of every daily retrain.

    from src import registry
    info = registry.log_training_run(algo="logistic")   # train + log + register
    clf  = registry.load_champion()                      # -> OutcomeClassifier

Browse the history with:   mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import timedelta

import pandas as pd

from .io import ROOT
from . import features
from .model import FEATURES, DRAW_BOOST, OutcomeClassifier
from .features import FORM_WINDOW, MIN_YEAR
from .evaluate import OUTCOMES, metrics_from_probs

# MLflow 3.x retired the bare-filesystem store, so we use a local SQLite tracking DB
# (mlflow.db) with artifacts written alongside it (mlartifacts/). Both are gitignored.
MLFLOW_DB = ROOT / "mlflow.db"
ARTIFACT_DIR = ROOT / "mlartifacts"
TRACKING_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"
EXPERIMENT = "worldcup-outcome"
REGISTERED_NAME = "worldcup-outcome-model"
CHAMPION_ALIAS = "champion"
DEFAULT_CUTOFFS = ["2022-01-01", "2023-01-01", "2024-01-01"]
WC_BACKTEST_START = "2022-11-20"


def _client_and_mlflow():
    """Configure MLflow to use the local file store and return (mlflow, MlflowClient)."""
    import mlflow
    from mlflow.tracking import MlflowClient

    ARTIFACT_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_DIR.as_uri())
    mlflow.set_experiment(EXPERIMENT)
    return mlflow, MlflowClient()


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _data_fingerprint(table: pd.DataFrame) -> dict:
    """A reproducibility stamp: shape/date-range of the training table plus an md5 of
    the eloratings inputs, so two runs on identical data share a fingerprint."""
    h = hashlib.md5()
    for name in ("eloratings_history.csv", "eloratings_results.csv", "live_results.csv"):
        p = ROOT / "data" / name
        if p.exists():
            h.update(p.read_bytes())
    trainable = table.dropna(subset=FEATURES)
    return {
        "rows_total": int(len(table)),
        "rows_trainable": int(len(trainable)),
        "date_min": str(table["date"].min().date()),
        "date_max": str(table["date"].max().date()),
        "elo_md5": h.hexdigest()[:12],
    }


def _walk_forward_one(table: pd.DataFrame, algo: str, cutoffs: list[str]) -> dict:
    """Walk-forward metrics for a SINGLE algo (averaged over cutoffs) - cheaper than
    evaluate.walk_forward, which retrains all five candidates at every split."""
    accs, lls, rpss, ns = [], [], [], []
    for c in cutoffs:
        test = table[table["date"] >= pd.Timestamp(c)].dropna(subset=FEATURES)
        clf = OutcomeClassifier(algo).fit(table, c)
        mt = metrics_from_probs(clf.predict_proba_df(test)[OUTCOMES], test["outcome"])
        accs.append(mt["accuracy"]); lls.append(mt["log_loss"])
        rpss.append(mt["rps"]); ns.append(mt["n"])
    n = len(cutoffs)
    return {
        "wf_accuracy": sum(accs) / n, "wf_log_loss": sum(lls) / n,
        "wf_rps": sum(rpss) / n, "wf_test_rows_avg": sum(ns) / n,
    }


def _wc_backtest_one(table: pd.DataFrame, algo: str) -> dict:
    """2022 World-Cup finals backtest for a single algo (sanity check, small sample)."""
    test = table[(table["competition"] == "FIFA World Cup")
                 & (table["date"].dt.year == 2022)].dropna(subset=FEATURES)
    if test.empty:
        return {}
    clf = OutcomeClassifier(algo).fit(table, WC_BACKTEST_START)
    mt = metrics_from_probs(clf.predict_proba_df(test)[OUTCOMES], test["outcome"])
    return {f"wc2022_{k}": v for k, v in mt.items()}


def log_training_run(algo: str = "logistic", cutoffs: list[str] | None = None,
                     register: bool = True, set_champion: bool = True) -> dict:
    """Train `algo`, evaluate it, log everything to MLflow, and register the deployable
    model (fit on ALL data) as a new version + @champion alias.

    Returns a dict with the run_id, registered version and the headline metrics.
    """
    mlflow, client = _client_and_mlflow()
    cutoffs = cutoffs or DEFAULT_CUTOFFS

    table, _ = features.build_training_table()
    wf = _walk_forward_one(table, algo, cutoffs)
    wc = _wc_backtest_one(table, algo)
    fp = _data_fingerprint(table)

    # The deployable model trains on EVERYTHING (cutoff just past the last match).
    deploy_cutoff = (table["date"].max() + timedelta(days=1)).strftime("%Y-%m-%d")
    deploy = OutcomeClassifier(algo).fit(table, deploy_cutoff)

    info: dict = {"algo": algo, **wf, **wc}
    with mlflow.start_run(run_name=f"{algo}-{fp['date_max']}") as run:
        mlflow.set_tags({
            "algo": algo, "git_commit": _git_commit(),
            "classes": json.dumps(deploy.classes_),
            "features": json.dumps(FEATURES),
            "deploy_cutoff": deploy_cutoff,
        })
        mlflow.log_params({
            "algo": algo, "n_features": len(FEATURES),
            "form_window": FORM_WINDOW, "min_year": MIN_YEAR,
            "draw_boost": DRAW_BOOST, "cutoffs": ",".join(cutoffs),
            **{f"data_{k}": v for k, v in fp.items()},
        })
        mlflow.log_metrics({k: float(v) for k, v in {**wf, **wc}.items()})

        model_info = mlflow.sklearn.log_model(
            deploy.clf_, name="model",
            registered_model_name=REGISTERED_NAME if register else None,
        )
        mlflow.log_dict({
            "algo": algo, "classes": deploy.classes_, "features": FEATURES,
            "draw_boost": DRAW_BOOST, "deploy_cutoff": deploy_cutoff,
            "metrics": {**wf, **wc}, "data": fp,
        }, "meta.json")
        info["run_id"] = run.info.run_id

    if register:
        # Find the version just created (highest under this name) and tag it.
        versions = client.search_model_versions(f"name='{REGISTERED_NAME}'")
        latest = max(versions, key=lambda v: int(v.version))
        info["version"] = latest.version
        client.set_model_version_tag(REGISTERED_NAME, latest.version, "algo", algo)
        client.set_model_version_tag(REGISTERED_NAME, latest.version, "classes",
                                     json.dumps(deploy.classes_))
        client.set_model_version_tag(REGISTERED_NAME, latest.version, "wf_accuracy",
                                     f"{wf['wf_accuracy']:.4f}")
        if set_champion:
            client.set_registered_model_alias(REGISTERED_NAME, CHAMPION_ALIAS, latest.version)
            info["champion"] = True
    return info


def load_champion() -> OutcomeClassifier:
    """Load the @champion model version and rebuild an OutcomeClassifier around it.

    Predictions use this instead of refitting - so every prediction is traceable to a
    specific, registered model version. Inference needs only (algo, fitted estimator,
    class order); the label-encoder is fit-only, so it isn't required here.
    """
    mlflow, client = _client_and_mlflow()
    mv = client.get_model_version_by_alias(REGISTERED_NAME, CHAMPION_ALIAS)
    estimator = mlflow.sklearn.load_model(f"models:/{REGISTERED_NAME}@{CHAMPION_ALIAS}")
    clf = OutcomeClassifier(mv.tags.get("algo", "logistic"))
    clf.clf_ = estimator
    clf.classes_ = json.loads(mv.tags["classes"])
    return clf


def champion_info() -> dict | None:
    """Metadata for the current champion version (or None if nothing is registered)."""
    _, client = _client_and_mlflow()
    try:
        mv = client.get_model_version_by_alias(REGISTERED_NAME, CHAMPION_ALIAS)
    except Exception:
        return None
    return {"version": mv.version, "run_id": mv.run_id, **dict(mv.tags)}
