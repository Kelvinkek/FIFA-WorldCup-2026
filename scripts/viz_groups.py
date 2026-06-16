"""Group-stage prediction visual - every match as a Win/Draw/Lose bar.

    python scripts/viz_groups.py  ->  reports/viz_group_predictions.png

Each match is a 100%-stacked bar: green = home win, amber = draw, blue = away win.
The model's pick (draw-aware) is printed at the right - DRAW picks are amber so the
likely-drawn games stand out.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")

import re
from datetime import timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import features, model, load
from src.teams import display
from src.io import ROOT

OUTDIR = ROOT / "reports"
OUTDIR.mkdir(exist_ok=True)

WIN, DRAW, LOSE = "#1b9e5a", "#e8a33d", "#3b6fd4"
MUTED = "#6b7280"
HOSTS = {"United States", "Canada", "Mexico"}

# The schedule's Time is "venue-local (UTC+3 reference)", e.g. "13:00 (22:00)".
# We rebuild the true UTC instant from the two clocks, then convert to Brisbane
# (AEST, UTC+10, no DST). The two clocks pin down the venue's UTC offset exactly.
def to_brisbane(date, time):
    m = re.match(r"\s*(\d{1,2}):(\d{2})\s*\((\d{1,2}):(\d{2})\)", str(time))
    if not m:
        return None
    lh, lm, ph, pm = map(int, m.groups())
    venue = date.normalize() + timedelta(hours=lh, minutes=lm)
    diff = ((ph * 60 + pm) - (lh * 60 + lm)) % 1440      # how far the +3 clock leads venue
    venue_off = 3 * 60 - diff                            # venue offset from UTC (minutes)
    utc = venue - timedelta(minutes=venue_off)
    return utc + timedelta(hours=10)                     # -> Brisbane

table, _ = features.build_training_table()
clf = model.OutcomeClassifier("logistic").fit(table, "2026-06-11")
form = features.team_form(); h2h = features.h2h_table()

def predict(h, a):
    rec = h2h.get(tuple(sorted((h, a))), (1/3, 1/3, 1/3))
    if tuple(sorted((h, a)))[0] != h:
        rec = (rec[2], rec[1], rec[0])
    p = clf.predict_match(model.match_features(form.loc[h], form.loc[a], h not in HOSTS, rec))
    boosted = dict(p); boosted["D"] *= model.DRAW_BOOST
    return p, max(boosted, key=boosted.get)

sch = load.load_schedule_2026().sort_values(["Date", "Time"])
adj = {}
for _, r in sch.iterrows():
    adj.setdefault(r.home_team, set()).add(r.away_team); adj.setdefault(r.away_team, set()).add(r.home_team)
seen, groups = set(), []
for t in adj:
    if t in seen: continue
    stk = [t]; comp = []
    while stk:
        x = stk.pop()
        if x in seen: continue
        seen.add(x); comp.append(x); stk += [y for y in adj[x] if y not in seen]
    groups.append(sorted(comp))
groups.sort(key=lambda g: 0 if "Mexico" in g else 1)

import matplotlib.patches as mpatches


def draw_batch(batch, first_letter, fname, subtitle):
    """One image of up to 6 groups (3 rows x 2 cols), no in-bar percentages."""
    fig, axes = plt.subplots(3, 2, figsize=(22, 24))
    fig.suptitle(f"FIFA World Cup 2026 - Group-Stage Predictions  ({subtitle})",
                 fontsize=22, fontweight="bold", y=1.0)
    for gi, g in enumerate(batch):
        ax = axes[gi // 2][gi % 2]
        gs = set(g)
        matches = [(r.home_team, r.away_team, r.Date, r.Time) for _, r in sch.iterrows()
                   if r.home_team in gs and r.away_team in gs]
        # within a matchday the bars are PITCH apart; between matchdays the gap is doubled
        # (2*PITCH) with a divider centred in it; the date sticks above its matchday's bars.
        items = sorted(((h, a, to_brisbane(date, time)) for (h, a, date, time) in matches),
                       key=lambda r: r[2])
        PITCH = 2.9
        ys, headers, dividers = [], [], []
        prev_day = None
        for (h, a, bne) in items:
            day = bne.date()
            if prev_day is None:
                y = 0.0
                headers.append((y + 1.25, bne))
            elif day != prev_day:
                y = ys[-1] - 1.5 * PITCH                                  # 1.5x gap between matchdays
                headers.append((y + 1.25, bne))                          # date stuck above the bars
                dividers.append(((ys[-1] - 1.35) + (y + 1.25)) / 2)      # midway: last bar <-> next date
            else:
                y = ys[-1] - PITCH
            ys.append(y)
            prev_day = day
        ys = np.array(ys)
        for yi, (h, a, bne) in zip(ys, items):
            p, pick = predict(h, a)
            H, D, A = p["H"] * 100, p["D"] * 100, p["A"] * 100
            ax.barh(yi, H, color=WIN, edgecolor="white", height=1.5)
            ax.barh(yi, D, left=H, color=DRAW, edgecolor="white", height=1.5)
            ax.barh(yi, A, left=H + D, color=LOSE, edgecolor="white", height=1.5)
            # percentage inside each segment (skip very thin ones to avoid clutter)
            for val, centre in [(H, H / 2), (D, H + D / 2), (A, H + D + A / 2)]:
                if val >= 8:
                    ax.text(centre, yi, f"{val:.0f}%", ha="center", va="center",
                            color="white", fontsize=9, fontweight="bold")
            col = {"H": WIN, "D": DRAW, "A": LOSE}[pick]
            txt = "DRAW" if pick == "D" else display(h if pick == "H" else a)
            ax.text(102, yi, txt, ha="left", va="center", fontsize=10, fontweight="bold",
                    color=col, clip_on=False)
            host = "*" if h in HOSTS else ""
            ax.text(50, yi - 1.35, f"{display(h)}{host}  vs  {display(a)}   ·   {bne.strftime('%H:%M')}",
                    ha="center", va="center", fontsize=10, fontweight="bold", color="#1f2a37")
        for hy, bne in headers:                  # bold full-text matchday header, above its bars
            ax.text(0, hy, bne.strftime("%A %d %B"), ha="left", va="center",
                    fontsize=11, fontweight="bold", color="#16335c")
        for dv in dividers:                      # plain divider in the middle of the matchday gap
            ax.axhline(dv, color="#cfd6de", lw=1.1, zorder=0)
        ax.set_xlim(0, 100); ax.set_ylim(ys.min() - 1.9, ys[0] + 2.1)
        ax.set_yticks([]); ax.set_xticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(f"Group {chr(first_letter + gi)}:  {', '.join(display(t) for t in g)}",
                     fontsize=13, fontweight="bold", loc="center", color="#16335c", pad=10)

    fig.legend(handles=[mpatches.Patch(color=WIN, label="Home win"),
                        mpatches.Patch(color=DRAW, label="Draw"),
                        mpatches.Patch(color=LOSE, label="Away win")],
               loc="lower center", ncol=3, fontsize=13, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.text(0.5, -0.028, "Kickoff times in Brisbane time (AEST, UTC+10)   ·   * = host nation",
             ha="center", fontsize=11, color=MUTED)
    plt.tight_layout(rect=[0.0, 0.03, 1, 0.97])
    plt.subplots_adjust(wspace=0.42, hspace=0.18)
    plt.savefig(OUTDIR / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")


draw_batch(groups[:6], 65, "viz_group_predictions_A-F.png", "Groups A–F")
draw_batch(groups[6:], 71, "viz_group_predictions_G-L.png", "Groups G–L")
