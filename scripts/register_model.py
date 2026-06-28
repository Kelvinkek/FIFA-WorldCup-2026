"""Train, evaluate and register a new model version in MLflow.

    python scripts/register_model.py                 # logistic (default), set as champion
    python scripts/register_model.py --algo xgb      # log a different algo for comparison
    python scripts/register_model.py --no-champion   # log+register but don't promote

Every run is recorded in ./mlruns. Browse the full history (metrics over time, the
registry, side-by-side runs) with:

    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src import registry


def main() -> None:
    ap = argparse.ArgumentParser(description="Register a model version in MLflow")
    ap.add_argument("--algo", default="logistic",
                    help="logistic | xgb | gbm | random_forest | extra_trees")
    ap.add_argument("--no-champion", action="store_true",
                    help="register the version but do not promote it to @champion")
    args = ap.parse_args()

    print(f"Training + logging '{args.algo}' to MLflow ({registry.TRACKING_URI}) ...")
    info = registry.log_training_run(args.algo, set_champion=not args.no_champion)

    print("\nLogged run:")
    print(f"  run_id        {info['run_id']}")
    print(f"  version       {info.get('version', '(not registered)')}")
    print(f"  champion      {info.get('champion', False)}")
    print(f"  walk-forward  acc={info['wf_accuracy']:.4f}  "
          f"log_loss={info['wf_log_loss']:.4f}  rps={info['wf_rps']:.4f}")
    if "wc2022_accuracy" in info:
        print(f"  WC-2022 test  acc={info['wc2022_accuracy']:.4f}  "
              f"log_loss={info['wc2022_log_loss']:.4f}")
    print("\nBrowse with:  mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
