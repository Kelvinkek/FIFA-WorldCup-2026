"""Round-of-32 prediction charts -> reports/round_of_32/  (separate from the daily PNGs).

    python scripts/viz_r32.py

Fixtures come from src/knockout.py (fixturedownload.com open feed by default; football-
data.org if a token is set). Each tie is predicted with the same logistic + draw-aware
model used everywhere else. Output (its own folder, so the daily PNGs are untouched):

  reports/round_of_32/round_of_32.png   - all 16 ties at a glance
  reports/round_of_32/YYYY-MM-DD.png    - one PNG per match-day (same look as daily/)

Once a tie is played (results arrive via eloratings), its actual result is outlined and
graded - identical styling to the daily tracker, just a different round and folder.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # import sibling viz_results
import warnings; warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd

from src import features, model, knockout
from src.io import ROOT
import viz_results as vr          # reuse grade() + the rendering helpers for identical styling

OUT_DIR = ROOT / "reports" / "round_of_32"
HOSTS = {"United States", "Canada", "Mexico"}   # host nations keep home advantage


def predict_fixtures(fx: pd.DataFrame) -> pd.DataFrame:
    """Predict H/D/A for each R32 tie with the current logistic model + draw-aware pick.

    Uses the latest team form/Elo (these are upcoming fixtures), the same as the group
    charts. Output columns match viz_results' snapshot so we can reuse grade()."""
    table, _ = features.build_training_table()
    cutoff = (table["date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # train on all data
    clf = model.OutcomeClassifier("logistic").fit(table, cutoff)
    form = features.team_form()
    h2h = features.h2h_table()
    rows = []
    for _, r in fx.iterrows():
        h, a = r.home, r.away
        if h not in form.index or a not in form.index:
            print(f"  skip (no model features): {h} vs {a}")
            continue
        rec = h2h.get(tuple(sorted((h, a))), (1/3, 1/3, 1/3))
        if tuple(sorted((h, a)))[0] != h:
            rec = (rec[2], rec[1], rec[0])
        p = clf.predict_match(model.match_features(form.loc[h], form.loc[a], h not in HOSTS, rec))
        boosted = dict(p); boosted["D"] *= model.DRAW_BOOST
        rows.append({"date": r.date, "home": h, "away": a,
                     "p_H": p["H"], "p_D": p["D"], "p_A": p["A"],
                     "pick": max(boosted, key=boosted.get)})
    return pd.DataFrame(rows)


def render_overview(graded: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "round_of_32.png"
    nplayed = int(graded["played"].sum())
    sub = (f"all {len(graded)} ties  ·  {nplayed} played, {len(graded) - nplayed} to come"
           "  ·  logistic model, draw-aware pick")
    vr._render(graded, "FIFA World Cup 2026 — Round of 32", sub, out)
    print(f"Saved round_of_32/{out.name}: {len(graded)} ties ({nplayed} played)")


def render_day(date: pd.Timestamp, day: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{date:%Y-%m-%d}.png"
    nplayed = int(day["played"].sum())
    sub = f"Round of 32  ·  {len(day)} matches  ·  {nplayed} played, {len(day) - nplayed} to come"
    if nplayed:
        sub += f"  ·  {int(day['correct'].sum())}/{nplayed} correct"
    vr._render(day, "FIFA World Cup 2026", sub, out, day_label=date.strftime("%A %d %B %Y"))
    print(f"Saved round_of_32/{out.name}: {len(day)} matches ({nplayed} played)")


if __name__ == "__main__":
    fx = knockout.refresh_round_of_32()
    if fx.empty:
        print("No Round-of-32 fixtures decided yet - nothing to predict.")
        raise SystemExit
    graded = vr.grade(predict_fixtures(fx))     # grades any tie already played; rest = "to play"
    render_overview(graded)
    for d, day in graded.groupby("date"):
        render_day(d, day.reset_index(drop=True))
