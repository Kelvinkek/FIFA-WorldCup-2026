"""Group-stage prediction visual — every match as a Win/Draw/Lose bar.

    python scripts/viz_groups.py  ->  reports/viz_group_predictions.png

Each match is a 100%-stacked bar: green = home win, amber = draw, blue = away win.
The model's pick (draw-aware) is printed at the right — DRAW picks are amber so the
likely-drawn games stand out.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import features, model, load
from src.io import ROOT

OUTDIR = ROOT / "reports"
OUTDIR.mkdir(exist_ok=True)

WIN, DRAW, LOSE = "#1b9e5a", "#e8a33d", "#3b6fd4"
HOSTS = {"United States", "Canada", "Mexico"}
SHORT = {"Korea Republic": "Korea Rep", "Bosnia-Herzegovina": "Bosnia",
         "Côte d'Ivoire": "Ivory Coast", "United States": "USA",
         "South Africa": "S. Africa", "Saudi Arabia": "S. Arabia",
         "New Zealand": "N. Zealand", "Cape Verde": "Cape Verde", "Switzerland": "Switzerland"}
def s(t): return SHORT.get(t, t)

table, _ = features.build_training_table()
clf = model.OutcomeClassifier("random_forest").fit(table, "2026-06-11")
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
    fig, axes = plt.subplots(3, 2, figsize=(21, 14))
    fig.suptitle(f"FIFA World Cup 2026 — Group-Stage Predictions  ({subtitle})",
                 fontsize=22, fontweight="bold", y=1.0)
    for gi, g in enumerate(batch):
        ax = axes[gi // 2][gi % 2]
        gs = set(g)
        matches = [(r.home_team, r.away_team) for _, r in sch.iterrows()
                   if r.home_team in gs and r.away_team in gs]
        y = np.arange(len(matches))[::-1]
        labels = []
        for yi, (h, a) in zip(y, matches):
            p, pick = predict(h, a)
            H, D, A = p["H"] * 100, p["D"] * 100, p["A"] * 100
            ax.barh(yi, H, color=WIN, edgecolor="white", height=0.72)
            ax.barh(yi, D, left=H, color=DRAW, edgecolor="white", height=0.72)
            ax.barh(yi, A, left=H + D, color=LOSE, edgecolor="white", height=0.72)
            # percentage inside each segment (skip very thin ones to avoid clutter)
            for val, centre in [(H, H / 2), (D, H + D / 2), (A, H + D + A / 2)]:
                if val >= 8:
                    ax.text(centre, yi, f"{val:.0f}%", ha="center", va="center",
                            color="white", fontsize=9, fontweight="bold")
            col = {"H": WIN, "D": DRAW, "A": LOSE}[pick]
            txt = "DRAW" if pick == "D" else (s(h) if pick == "H" else s(a))
            ax.text(103, yi, txt, ha="left", va="center", fontsize=11, fontweight="bold", color=col)
            host = "*" if h in HOSTS else ""
            labels.append(f"{s(h)}{host}  v  {s(a)}")
        ax.set_xlim(0, 100); ax.set_ylim(-0.6, len(matches) - 0.4)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=11)
        ax.set_xticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(f"Group {chr(first_letter + gi)}:  {', '.join(s(t) for t in g)}",
                     fontsize=13, fontweight="bold", loc="left", color="#16335c", pad=8)

    fig.legend(handles=[mpatches.Patch(color=WIN, label="Home win"),
                        mpatches.Patch(color=DRAW, label="Draw"),
                        mpatches.Patch(color=LOSE, label="Away win")],
               loc="lower center", ncol=3, fontsize=13, frameon=False, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0.0, 0.03, 1, 0.97])
    plt.subplots_adjust(wspace=0.72, hspace=0.4)
    plt.savefig(OUTDIR / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")


draw_batch(groups[:6], 65, "viz_group_predictions_A-F.png", "Groups A–F")
draw_batch(groups[6:], 71, "viz_group_predictions_G-L.png", "Groups G–L")
