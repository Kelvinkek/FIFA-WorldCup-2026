"""Polished, presentation-quality visuals for the Elo + form model.

    python scripts/viz.py

Writes high-impact charts to reports/:
  viz_form_ranking.png     – 2026 teams by current net goal form (the model's input)
  viz_model_scorecard.png  – model accuracy vs honest baselines
  viz_feature_impact.png   – what the model learned (signed importance)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401

from src import features, model, evaluate, load
from src.teams import display
from src.io import ROOT

OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)

# ---- shared house style ----
INK = "#1f2a37"; MUTED = "#6b7280"; GRID = "#e5e7eb"
ACCENT = "#2563eb"; GOOD = "#059669"; WARN = "#d97706"; BAD = "#dc2626"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.linewidth": 1.0,
    "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
})


def _clean(ax):
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def titled(fig, title, subtitle):
    """Bold title + muted subtitle above the plot, no overlap."""
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.suptitle(title, fontsize=15, fontweight="bold", color=INK, x=0.04, ha="left", y=0.985)
    fig.text(0.04, 0.915, subtitle, fontsize=10.5, color=MUTED, ha="left")


# ============ 1. Current goal-form ranking (the model's core input) ============
sched_teams = set(load.load_schedule_2026()["home_team"]) | set(load.load_schedule_2026()["away_team"])
form = features.team_form()
form = form[form.index.isin(sched_teams)].copy()
form["net_form"] = form["goals_for_avg"] - form["goals_against_avg"]
top = form.sort_values("net_form", ascending=False).head(20)
norm = (top["net_form"] - top["net_form"].min()) / (top["net_form"].max() - top["net_form"].min())
colors = plt.cm.Greens(0.35 + 0.6 * norm.values)

fig, ax = plt.subplots(figsize=(9, 9))
y = np.arange(len(top))[::-1]
ax.barh(y, top["net_form"], color=colors, edgecolor="white", height=0.78)
for yi, (team, val) in zip(y, top["net_form"].items()):
    ax.text(val + 0.02 if val >= 0 else val - 0.02, yi, display(team),
            va="center", ha="left" if val >= 0 else "right",
            fontsize=10.5, color=INK, fontweight="bold")
ax.set_yticks([]); _clean(ax)
ax.axvline(0, color=MUTED, lw=1)
ax.set_xlabel("net recent form  (avg goals scored − conceded, last 3 matches)")
titled(fig, "Current Form - 2026 World Cup teams",
       "The model's core input: who's been scoring more than they concede lately")
fig.savefig(OUT / "viz_form_ranking.png", dpi=140); plt.close(fig)
print("saved viz_form_ranking.png")


# ============ 2. Model scorecard vs honest baselines ============
table, _ = features.build_training_table()
wf = evaluate.walk_forward(table, ["2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"])
# baselines
test_all = table[table["date"] >= pd.Timestamp("2021-01-01")]
home_rate = (test_all["outcome"] == "H").mean()
rows = [("Always predict\nhome", home_rate, WARN),
        ("Random\nguess", 1 / 3, BAD)]
# all 5 tuned models, logistic first (it's the default used for predictions)
for m in ["logistic", "xgboost", "gbm", "random_forest", "extra_trees"]:
    rows.append((m.replace("_", "\n"), wf.loc[m, "accuracy"], GOOD if wf.loc[m, "accuracy"] >= home_rate else ACCENT))

labels = [r[0] for r in rows]; vals = [r[1] for r in rows]; cols = [r[2] for r in rows]
fig, ax = plt.subplots(figsize=(10, 5.5))
bars = ax.bar(labels, [v * 100 for v in vals], color=cols, edgecolor="white", width=0.66)
ax.axhline(home_rate * 100, ls="--", color=WARN, lw=1.2)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v * 100 + 0.8, f"{v*100:.1f}%",
            ha="center", fontweight="bold", color=INK)
ax.set_ylim(0, 72); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.tick_params(length=0)
ax.set_ylabel("test accuracy (%)")
titled(fig, "Model scorecard - does it beat trivial guessing?",
       "The tuned models beat 'always predict home' by several points - real signal, but football has a hard ceiling")
fig.savefig(OUT / "viz_model_scorecard.png", dpi=140); plt.close(fig)
print("saved viz_model_scorecard.png")


# ============ 3. Signed feature impact ============
gbm = model.OutcomeClassifier("gbm").fit(table, "2023-01-01")
test = table[table["date"] >= pd.Timestamp("2023-01-01")].dropna(subset=model.FEATURES)
imp = gbm.feature_importance(test, n_repeats=10).sort_values()
cols = [GOOD if v > 0 else BAD for v in imp.values]
fig, ax = plt.subplots(figsize=(9, 8))
ax.barh(range(len(imp)), imp.values, color=cols, edgecolor="white", height=0.72)
ax.set_yticks(range(len(imp))); ax.set_yticklabels([c.replace("diff_", "") for c in imp.index], fontsize=10)
ax.axvline(0, color=MUTED, lw=1); _clean(ax)
ax.set_xlabel("permutation importance  (positive = helps, negative = noise)")
titled(fig, "What the model learned (Elo + form features)",
       "Green features add signal; red ones are noise the model would be better without")
fig.savefig(OUT / "viz_feature_impact.png", dpi=140); plt.close(fig)
print("saved viz_feature_impact.png")
