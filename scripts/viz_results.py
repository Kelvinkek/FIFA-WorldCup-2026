"""Prediction tracker — the model's INITIAL predictions vs the ACTUAL results.

    python scripts/viz_results.py  ->  reports/viz_prediction_tracker.png

On first run it freezes the model's pre-tournament prediction for every 2026 group
fixture into data/initial_predictions.csv (trained only on data before kickoff, so no
leakage). That snapshot is never overwritten — it's the "initial" call we grade against.

Each run it pulls the actual results (from eloratings.net, via data/eloratings_results.csv,
refreshed daily by daily_update.py), matches them to the frozen predictions, and draws a
scorecard: who we got right/wrong, running accuracy, and Brier / log-loss.
"""

import sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src import features, model, load, elodata
from src.teams import display
from src.io import ROOT

SNAPSHOT = ROOT / "data" / "initial_predictions.csv"
OUT = ROOT / "reports" / "viz_prediction_tracker.png"
DAILY_DIR = ROOT / "reports" / "daily"
OUT.parent.mkdir(exist_ok=True)

HOSTS = {"United States", "Canada", "Mexico"}
WIN, DRAW, LOSE, MUTED, INK = "#1b9e5a", "#e8a33d", "#3b6fd4", "#9aa0a6", "#1f2a37"
GOOD, BAD = "#1b9e5a", "#d6453d"
CUTOFF = "2026-06-11"   # tournament kickoff — train strictly before this


def build_snapshot() -> pd.DataFrame:
    """Freeze the model's pre-tournament prediction for every group fixture."""
    table, _ = features.build_training_table()
    clf = model.OutcomeClassifier("logistic").fit(table, CUTOFF)
    form = features.team_form(); h2h = features.h2h_table()
    sch = load.load_schedule_2026()
    rows = []
    for _, r in sch.iterrows():
        h, a = r.home_team, r.away_team
        if h not in form.index or a not in form.index:
            continue
        rec = h2h.get(tuple(sorted((h, a))), (1/3, 1/3, 1/3))
        if tuple(sorted((h, a)))[0] != h:
            rec = (rec[2], rec[1], rec[0])
        p = clf.predict_match(model.match_features(form.loc[h], form.loc[a], h not in HOSTS, rec))
        boosted = dict(p); boosted["D"] *= model.DRAW_BOOST
        rows.append({"date": r.Date, "home": h, "away": a,
                     "p_H": p["H"], "p_D": p["D"], "p_A": p["A"],
                     "pick": max(boosted, key=boosted.get)})
    df = pd.DataFrame(rows)
    df.to_csv(SNAPSHOT, index=False)
    print(f"Froze initial predictions -> {SNAPSHOT.name} ({len(df)} fixtures)")
    return df


def load_snapshot() -> pd.DataFrame:
    if SNAPSHOT.exists():
        return pd.read_csv(SNAPSHOT, parse_dates=["date"])
    return build_snapshot()


def grade(pred: pd.DataFrame) -> pd.DataFrame:
    """Match every frozen prediction to its real result (by team pair); grade if played."""
    res = elodata.load_results()
    res = res[res["date"] >= pd.Timestamp("2026-06-01")]          # tournament window
    by_pair = {frozenset((r.home, r.away)): (r.home, int(r.hg), int(r.ag))
               for _, r in res.iterrows()}
    out = []
    for _, p in pred.iterrows():
        row = {"date": p.date, "home": p.home, "away": p.away,
               "p_H": p.p_H, "p_D": p.p_D, "p_A": p.p_A, "pick": p.pick,
               "played": False, "hg": None, "ag": None, "actual": None,
               "correct": False, "prob_actual": np.nan, "logloss": np.nan, "brier": np.nan}
        m = by_pair.get(frozenset((p.home, p.away)))
        if m:
            ah, hg, ag = m
            if ah != p.home:                                       # orient to prediction's home/away
                hg, ag = ag, hg
            actual = "H" if hg > ag else "A" if hg < ag else "D"
            pa = {"H": p.p_H, "D": p.p_D, "A": p.p_A}[actual]
            row.update(played=True, hg=hg, ag=ag, actual=actual, correct=(p.pick == actual),
                       prob_actual=pa, logloss=-np.log(max(pa, 1e-9)),
                       brier=float(sum((np.array([p.p_H, p.p_D, p.p_A]) -
                                        np.array([actual == o for o in "HDA"], float)) ** 2)))
        out.append(row)
    return pd.DataFrame(out).sort_values(["date", "home"]).reset_index(drop=True)


def _draw_match_row(ax, yi: float, r) -> None:
    """One horizontal Win/Draw/Lose probability bar; outline the actual result if played.
    The matchup is labelled centred just below its bar."""
    H, D, A = r.p_H * 100, r.p_D * 100, r.p_A * 100
    ax.barh(yi, H, color=WIN, edgecolor="white", height=0.46)
    ax.barh(yi, D, left=H, color=DRAW, edgecolor="white", height=0.46)
    ax.barh(yi, A, left=H + D, color=LOSE, edgecolor="white", height=0.46)
    for val, centre in [(H, H / 2), (D, H + D / 2), (A, H + D + A / 2)]:
        if val >= 7:
            ax.text(centre, yi, f"{val:.0f}%", ha="center", va="center",
                    color="white", fontsize=8.5, fontweight="bold")
    if r.played:
        left, width = {"H": (0, H), "D": (H, D), "A": (H + D, A)}[r.actual]
        ax.barh(yi, width, left=left, height=0.46, fill=False,
                edgecolor="#111111", linewidth=2.4, zorder=6)
        mark, mcol = ("✓", GOOD) if r.correct else ("✗", BAD)
        ax.text(102, yi, f"{int(r.hg)}–{int(r.ag)}", va="center", ha="left", fontsize=11,
                fontweight="bold", color=INK, clip_on=False)
        ax.text(113, yi, mark, va="center", ha="center", fontsize=14,
                fontweight="bold", color=mcol, clip_on=False)
    else:
        ax.text(102, yi, "to play", va="center", ha="left", fontsize=10,
                style="italic", color=MUTED, clip_on=False)
    # matchup label centred just below the bar
    ax.text(50, yi - 0.38, f"{display(r.home)}  vs  {display(r.away)}", ha="center", va="center",
            fontsize=10, fontweight="bold", color=INK)


def _legend(fig, fig_h: float) -> None:
    import matplotlib.patches as mpatches
    fig.legend(handles=[mpatches.Patch(color=WIN, label="Predicted home win"),
                        mpatches.Patch(color=DRAW, label="Predicted draw"),
                        mpatches.Patch(color=LOSE, label="Predicted away win"),
                        mpatches.Patch(facecolor="white", edgecolor="#111111", lw=2,
                                       label="Actual result (outlined)")],
               loc="lower center", ncol=4, fontsize=11, frameon=False,
               bbox_to_anchor=(0.5, 0.18 / fig_h))


def _render(df: pd.DataFrame, title: str, subtitle: str, outpath: Path,
            stats: list | None = None, day_label: str | None = None) -> None:
    """Shared horizontal-bar figure for a set of matches (tracker or single day)."""
    n = len(df)
    header_in = 1.95 if stats else (1.75 if day_label else 1.2)
    row_in, foot_in = 0.9, 0.7
    fig_h = header_in + max(n, 1) * row_in + foot_in
    fig, ax = plt.subplots(figsize=(13.5, fig_h)); fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.04, right=0.87, top=1 - header_in / fig_h, bottom=foot_in / fig_h)

    def yf(inch):
        return 1 - inch / fig_h

    fig.text(0.5, yf(0.4), title, ha="center", fontsize=21, fontweight="bold", color=INK)
    sub_y = 0.72
    if day_label:                                    # bold full weekday/date line
        fig.text(0.5, yf(0.76), day_label, ha="center", fontsize=15, fontweight="bold", color=INK)
        sub_y = 1.08
    fig.text(0.5, yf(sub_y), subtitle, ha="center", fontsize=12, color=MUTED)
    if stats:
        for i, (lab, val, col) in enumerate(stats):
            x = 0.1 + i * 0.198
            fig.text(x, yf(1.3), val, ha="center", fontsize=18, fontweight="bold", color=col)
            fig.text(x, yf(1.58), lab, ha="center", fontsize=10, color=MUTED)

    if n == 0:
        ax.axis("off")
        plt.savefig(outpath, dpi=120, bbox_inches="tight"); plt.close(fig)
        return

    y = np.arange(n)[::-1]
    for yi, (_, r) in zip(y, df.iterrows()):
        _draw_match_row(ax, yi, r)
    for yi in y[:-1]:                                # divider between matches
        ax.axhline(yi - 0.5, color="#e3e7ec", lw=1.0, zorder=0)
    ax.set_xlim(0, 100); ax.set_ylim(-0.72, n - 0.45)
    ax.set_yticks([]); ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    _legend(fig, fig_h)
    plt.savefig(outpath, dpi=120, bbox_inches="tight"); plt.close(fig)


def render_tracker(graded: pd.DataFrame) -> None:
    """Cumulative tracker over every match played so far."""
    played = graded[graded["played"]]
    asof = datetime.now().strftime("%a %d %b %Y")
    if played.empty:
        _render(played, "FIFA World Cup 2026 — Prediction Tracker",
                f"no completed matches yet  ·  as of {asof}", OUT)
        print(f"Saved {OUT.name} (0 results)"); return
    acc = played["correct"].mean()
    stats = [("Matches played", f"{len(played)}", INK),
             ("Correct picks", f"{played['correct'].sum()} / {len(played)}", GOOD if acc >= .5 else INK),
             ("Accuracy", f"{acc*100:.0f}%", GOOD if acc >= .5 else BAD),
             ("Avg log-loss", f"{played['logloss'].mean():.3f}", INK),
             ("Avg Brier", f"{played['brier'].mean():.3f}", INK)]
    _render(played, "FIFA World Cup 2026 — Prediction Tracker",
            f"initial predictions frozen at kickoff  ·  as of {asof}", OUT, stats)
    print(f"Saved {OUT.name}: {len(played)} graded, accuracy {acc*100:.1f}%")


def render_day(date: pd.Timestamp, day: pd.DataFrame) -> None:
    """One PNG for a single match-day: that day's fixtures, prediction vs result."""
    DAILY_DIR.mkdir(exist_ok=True)
    out = DAILY_DIR / f"{date:%Y-%m-%d}.png"
    nplayed = int(day["played"].sum())
    sub = f"{len(day)} matches  ·  {nplayed} played, {len(day) - nplayed} to come"
    if nplayed:
        sub += f"  ·  {int(day['correct'].sum())}/{nplayed} correct"
    _render(day, "FIFA World Cup 2026", sub, out,
            day_label=date.strftime("%A %d %B %Y"))   # e.g. "Sunday 14 June 2026"
    print(f"Saved daily/{out.name}: {len(day)} matches ({nplayed} played)")


if __name__ == "__main__":
    graded = grade(load_snapshot())
    render_tracker(graded)
    for d, day in graded.groupby("date"):                 # every match-day, incl. upcoming
        render_day(d, day.reset_index(drop=True))          # future days show all "to play"
