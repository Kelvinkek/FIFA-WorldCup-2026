"""Feature-analysis charts for the form-based model — saved as INDIVIDUAL images.

    python scripts/feature_analysis.py

Writes one PNG per chart into reports/:
  01_feature_importance.png   04_home_advantage.png
  02_form_vs_winrate.png      05_outcome_balance.png
  03_feature_correlation.png  06_model_comparison.png
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

from src import features, model, evaluate
from src.io import ROOT

OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)

table, _ = features.build_training_table()
CUT = "2023-01-01"
test = table[table["date"] >= pd.Timestamp(CUT)].dropna(subset=model.FEATURES)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=130)
    plt.close(fig)
    print(f"  saved {name}")


# 01 — permutation feature importance
gbm = model.OutcomeClassifier("gbm").fit(table, CUT)
imp = gbm.feature_importance(test, n_repeats=8)
fig, ax = plt.subplots(figsize=(8, 5))
cols = ["#059669" if v > 0 else "#dc2626" for v in imp.sort_values().values]
imp.sort_values().plot.barh(ax=ax, color=cols)
ax.set_title("Form-feature importance (permutation)")
ax.set_xlabel("drop in log-loss when shuffled")
save(fig, "01_feature_importance.png")

# 02 — home goal-form vs home win rate
t = table.dropna(subset=["home_goals_for_avg"]).copy()
t["bin"] = pd.cut(t["home_goals_for_avg"], bins=np.linspace(0, 4, 9))
rate = t.groupby("bin", observed=True).apply(lambda d: (d["outcome"] == "H").mean())
mids = [iv.mid for iv in rate.index]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(mids, rate.values, marker="o", color="#cc4444")
ax.axhline(0.5, ls="--", c="gray", lw=1)
ax.set_title("Home win rate vs home recent scoring form")
ax.set_xlabel("home_goals_for_avg (avg goals scored, last 5)"); ax.set_ylabel("P(home win)")
save(fig, "02_form_vs_winrate.png")

# 03 — feature correlation
cdf = table[model.FEATURES].copy()
cdf["home_win"] = (table["outcome"] == "H").astype(int)
C = cdf.corr()
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(C, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(C))); ax.set_xticklabels(C.columns, rotation=90, fontsize=8)
ax.set_yticks(range(len(C))); ax.set_yticklabels(C.columns, fontsize=8)
ax.set_title("Feature correlation")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
save(fig, "03_feature_correlation.png")

# 04 — home advantage (home vs neutral)
home = table[~table["neutral"].astype(bool)]["outcome"].value_counts(normalize=True).reindex(["H", "D", "A"])
neut = table[table["neutral"].astype(bool)]["outcome"].value_counts(normalize=True).reindex(["H", "D", "A"])
x = np.arange(3); w = 0.38
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(x - w / 2, home.values * 100, w, label="home game", color="#669933")
ax.bar(x + w / 2, neut.values * 100, w, label="neutral", color="#999999")
ax.set_xticks(x); ax.set_xticklabels(["Home win", "Draw", "Away win"])
ax.set_ylabel("%"); ax.set_title("Home advantage (home vs neutral venue)"); ax.legend()
save(fig, "04_home_advantage.png")

# 05 — outcome balance
fig, ax = plt.subplots(figsize=(6, 5))
(table["outcome"].value_counts(normalize=True).reindex(["H", "D", "A"]) * 100).plot.bar(
    ax=ax, color=["#cc4444", "#999999", "#3366aa"])
ax.set_xticklabels(["Home win", "Draw", "Away win"], rotation=0)
ax.set_ylabel("%"); ax.set_title("Outcome balance (the classes to predict)")
save(fig, "05_outcome_balance.png")

# 06 — model comparison
cmp = evaluate.compare_models(table, CUT)
fig, ax = plt.subplots(figsize=(7, 5))
cmp["log_loss"].sort_values(ascending=False).plot.barh(ax=ax, color="#693")
ax.set_xlabel("log-loss (lower = better)")
ax.set_title("Model comparison (form features)")
save(fig, "06_model_comparison.png")

print("\nFeature importance:")
print(imp.round(4).to_string())
