"""Daily data refresh + prediction regen for the World Cup model.

Intended to be run once a day by Windows Task Scheduler (see scripts/setup_schedule.ps1).
Steps:
  1. Pull finished results from API-Football (current-season World Cup, friendlies,
     Nations League) -> data/live_results.csv
  2. Refresh eloratings.net Elo history + recent results   -> data/eloratings_*.csv
  3. Rebuild features + regenerate group-stage charts      -> reports/
  4. Regenerate the prediction tracker (initial vs actual) -> reports/

Budget: API-Football free plan = 100 requests/day; this uses ~3. eloratings is free.
Every step is wrapped so one failure (e.g. network) never aborts the rest. Output is
timestamped so the Task Scheduler log is readable.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src import elodata, features, livedata, model

# Current-season competitions worth checking daily (keeps the request budget tiny).
DAILY_TARGETS = [(1, 2026), (10, 2026), (5, 2025)]  # World Cup, Friendlies, Nations League


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def main() -> None:
    log("=== daily update started ===")

    # 1. latest results from API-Football
    try:
        st = livedata.account_status()
        used, cap = st["requests"]["current"], st["requests"]["limit_day"]
        log(f"API-Football: {used}/{cap} requests used today")
        livedata.update_live_csv(DAILY_TARGETS)
    except Exception as e:
        log(f"! API-Football step skipped: {e!r}")

    # 2. refresh eloratings pre-match Elo (only the teams in our data)
    try:
        m = features.unified_matches()
        recent = m[m["date"].dt.year >= 2014]
        teams_needed = sorted(set(recent["home"]) | set(recent["away"]))
        hist = elodata.refresh_elo_history(teams_needed, verbose=False)
        log(f"eloratings refreshed: {len(hist):,} team-matches, "
            f"latest {hist['date'].max().date()}")
    except Exception as e:
        log(f"! eloratings step skipped: {e!r}")

    # 3. rebuild features + regenerate predictions
    try:
        t, _ = features.build_training_table()
        trainable = len(t.dropna(subset=model.FEATURES))
        log(f"training table rebuilt: {trainable:,} trainable rows "
            f"(through {t['date'].max().date()})")
        subprocess.run([sys.executable, str(Path(__file__).with_name("viz_groups.py"))], check=False)
        log("group-stage charts regenerated")
    except Exception as e:
        log(f"! rebuild step skipped: {e!r}")

    # 4. prediction tracker: frozen initial predictions vs actual results
    try:
        subprocess.run([sys.executable, str(Path(__file__).with_name("viz_results.py"))], check=False)
        log("prediction tracker regenerated")
    except Exception as e:
        log(f"! tracker step skipped: {e!r}")

    log("=== daily update finished ===")


if __name__ == "__main__":
    main()
