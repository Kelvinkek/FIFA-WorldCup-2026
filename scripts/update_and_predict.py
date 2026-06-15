"""Predict upcoming 2026 fixtures with the form-based model.

    python scripts/update_and_predict.py            # predict (no API call)
    python scripts/update_and_predict.py --fetch    # pull latest results first
    python scripts/update_and_predict.py -n 20

Form features cover any team with match history, so (unlike the old squad model)
this predicts the whole 2026 schedule.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")

import pandas as pd

from src import features, model, load
from src.teams import display

HOSTS = {"United States", "Canada", "Mexico"}  # 2026 co-hosts -> home advantage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="pull latest results from API first")
    ap.add_argument("-n", type=int, default=12, help="how many fixtures to predict")
    args = ap.parse_args()

    if args.fetch:
        from src import livedata
        print("Fetching latest finished matches from API-Football…")
        livedata.update_live_csv()
        print()

    table, _ = features.build_training_table()
    print(f"Trained on {len(table):,} matches (latest {table['date'].max().date()}).")
    clf = model.OutcomeClassifier("random_forest").fit(table, str(table["date"].max() + pd.Timedelta(days=1)))

    form = features.team_form()
    h2h = features.h2h_table()

    def predict(home, away):
        if home not in form.index or away not in form.index:
            return None
        rec = h2h.get(tuple(sorted((home, away))), (1/3, 1/3, 1/3))
        # h2h is stored for the sorted pair; flip if home isn't the first alphabetically
        if tuple(sorted((home, away)))[0] != home:
            rec = (rec[2], rec[1], rec[0])
        neutral = home not in HOSTS
        return clf.predict_match(model.match_features(form.loc[home], form.loc[away], neutral, rec))

    sch = load.load_schedule_2026().sort_values(["Date", "Time"]).reset_index(drop=True)
    print(f"\n{'UPCOMING 2026 FIXTURES - form-model prediction':^62}")
    print("-" * 62)
    shown = 0
    for _, r in sch.iterrows():
        h, a = r["home_team"], r["away_team"]
        p = predict(h, a)
        if p is None:
            continue
        # draw-aware pick (plain argmax almost never calls a draw)
        boosted = dict(p); boosted["D"] *= model.DRAW_BOOST
        pick = max(boosted, key=boosted.get)
        ph, pa = display(h), display(a)
        label = {"H": ph, "D": "Draw", "A": pa}[pick]
        host = " (host)" if h in HOSTS else ""
        print(f"{r['Date'].date()}  {ph:>16}{host} vs {pa:<16}  "
              f"{ph} {p['H']*100:3.0f}% / D {p['D']*100:3.0f}% / {pa} {p['A']*100:3.0f}%  -> {label}")
        shown += 1
        if shown >= args.n:
            break


if __name__ == "__main__":
    main()
