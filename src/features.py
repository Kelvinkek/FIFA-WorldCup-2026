"""Feature engineering for the match-outcome model.

Team strength comes from **eloratings.net Elo** (professional, importance-weighted,
joined pre-match via src/elodata.py), combined with **recent goal form** (goals scored /
conceded over the last N matches), head-to-head history, confederation, a neutral-venue
flag, and match importance. All features use only information available *before* the
match being predicted (Elo is pre-match; form is shifted), so there is no leakage.

`compute_elo()` (a homemade Elo) is kept as a reference/fallback but is no longer in the
pipeline - attach_eloratings() supplies the Elo features instead.

Pipeline:
    unified_matches()      -> one chronological table of every match we have
    attach_eloratings()    -> join eloratings pre-match Elo (home_elo/away_elo/elo_diff)
    build_training_table() -> match rows + Elo/form/h2h/confederation/importance + outcome
    team_form()            -> each team's latest form + current Elo (for new fixtures)
    h2h_table()            -> head-to-head record for any pair of teams
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import load, elodata
from .io import DATA_DIR

MIN_YEAR = 2015          # oldest matches kept for TRAINING (form uses older history too)
FORM_WINDOW = 5          # recent matches that define "form" (tuned: 5 best; >15 goes stale)
CONFEDS = ["OFC", "AFC", "CAF", "CONCACAF", "UEFA", "CONMEBOL"]
ELO_BASE, ELO_K, ELO_HOME_ADV = 1500.0, 40.0, 65.0


def compute_elo(m, k=ELO_K, home_adv=ELO_HOME_ADV, base=ELO_BASE):
    """Time-aware **opponent-adjusted** strength. Returns (matches + home_elo/away_elo/
    elo_diff [pre-match, no leakage], final_ratings_dict). Complements goal form, which
    is not opponent-adjusted."""
    ratings: dict[str, float] = {}
    he = np.empty(len(m)); ae = np.empty(len(m))
    for i, (h, a, hg, ag, neu) in enumerate(zip(m["home"], m["away"], m["hg"], m["ag"], m["neutral"])):
        rh = ratings.get(h, base); ra = ratings.get(a, base)
        he[i] = rh; ae[i] = ra
        adv = 0.0 if neu else home_adv
        exp = 1.0 / (1.0 + 10 ** ((ra - (rh + adv)) / 400.0))
        sc = 1.0 if hg > ag else 0.0 if hg < ag else 0.5
        change = k * (np.log1p(abs(hg - ag)) if hg != ag else 1.0) * (sc - exp)
        ratings[h] = rh + change; ratings[a] = ra - change
    out = m.copy()
    out["home_elo"] = he; out["away_elo"] = ae; out["elo_diff"] = he - ae
    return out, ratings


def attach_eloratings(m: pd.DataFrame) -> pd.DataFrame:
    """Attach eloratings.net **pre-match** Elo (home_elo/away_elo/elo_diff).

    The professional, importance-weighted ratings from eloratings.net, joined on
    (date, team). Rows eloratings doesn't cover (mostly obscure/regional sides) get
    NaN and fall out of training via the FEATURES dropna. Replaces compute_elo()."""
    h = elodata.load_elo_history()
    he = h.rename(columns={"team": "home", "elo_before": "home_elo"})[["date", "home", "home_elo"]]
    ae = h.rename(columns={"team": "away", "elo_before": "away_elo"})[["date", "away", "away_elo"]]
    out = m.merge(he, on=["date", "home"], how="left").merge(ae, on=["date", "away"], how="left")
    out["elo_diff"] = out["home_elo"] - out["away_elo"]
    return out


def unified_matches(include_wc_finals: bool = True) -> pd.DataFrame:
    """Every match we have, one tidy chronological table.

    Columns: date, home, away, hg, ag, neutral, competition.
    Maven international matches + Kaggle World Cup finals + live API results.
    """
    intl = load.load_international()
    parts = [pd.DataFrame({
        "date": intl["Date"], "home": intl["Home Team"], "away": intl["Away Team"],
        "hg": intl["Home Goals"], "ag": intl["Away Goals"],
        "neutral": ~intl["Home Stadium"].astype(bool), "competition": intl["Tournament"],
    })]
    if include_wc_finals:
        wc = load.load_wc_matches()
        host = wc["Host"].astype(str).str.strip()
        parts.append(pd.DataFrame({
            "date": wc["Date"], "home": wc["home_team"], "away": wc["away_team"],
            "hg": wc["home_score"], "ag": wc["away_score"],
            "neutral": wc["home_team"].str.strip() != host, "competition": "FIFA World Cup",
        }))
    live = DATA_DIR / "live_results.csv"
    if live.exists():
        lv = pd.read_csv(live, parse_dates=["date"])
        parts.append(lv[["date", "home", "away", "hg", "ag", "neutral", "competition"]])

    m = pd.concat(parts, ignore_index=True)
    m = m.dropna(subset=["date", "home", "away", "hg", "ag"])
    m = m.drop_duplicates(subset=["date", "home", "away"], keep="last")
    m = m.sort_values("date").reset_index(drop=True)
    m["hg"] = m["hg"].astype(int)
    m["ag"] = m["ag"].astype(int)
    return m


def _confederation_codes() -> dict[str, int]:
    rk = load.load_rankings("2026-06-08")
    conf = dict(zip(rk["team"], rk["association"]))
    code = {c: i for i, c in enumerate(CONFEDS)}
    return {t: code.get(c, -1) for t, c in conf.items()}


def add_form(m: pd.DataFrame, window: int = FORM_WINDOW) -> pd.DataFrame:
    """Add recent goals scored/conceded for each side (mean over prior `window`).

    Computed over the FULL history then read onto each match, shifted so the current
    match is excluded (no leakage). Adds home/away _goals_for_avg / _goals_against_avg.
    """
    long = pd.concat([
        pd.DataFrame({"i": m.index, "date": m["date"], "team": m["home"], "gf": m["hg"], "ga": m["ag"], "side": "home"}),
        pd.DataFrame({"i": m.index, "date": m["date"], "team": m["away"], "gf": m["ag"], "ga": m["hg"], "side": "away"}),
    ], ignore_index=True).sort_values("date")
    grp = long.groupby("team")
    long["gf_avg"] = grp["gf"].transform(lambda s: s.shift().rolling(window, min_periods=3).mean())
    long["ga_avg"] = grp["ga"].transform(lambda s: s.shift().rolling(window, min_periods=3).mean())

    out = m.copy()
    h = long[long["side"] == "home"].set_index("i")
    a = long[long["side"] == "away"].set_index("i")
    out["home_goals_for_avg"] = h["gf_avg"].reindex(out.index)
    out["home_goals_against_avg"] = h["ga_avg"].reindex(out.index)
    out["away_goals_for_avg"] = a["gf_avg"].reindex(out.index)
    out["away_goals_against_avg"] = a["ga_avg"].reindex(out.index)
    return out


def add_h2h(m: pd.DataFrame) -> pd.DataFrame:
    """Head-to-head: share of prior meetings won by home / drawn / won by away."""
    recs: dict[tuple, dict] = {}
    hw, dd, aw = [], [], []
    for h, a, hg, ag in zip(m["home"], m["away"], m["hg"], m["ag"]):
        key = tuple(sorted((h, a)))
        r = recs.get(key, {key[0]: 0, key[1]: 0, "D": 0})
        tot = r[h] + r[a] + r["D"]
        if tot:
            hw.append(r[h] / tot); dd.append(r["D"] / tot); aw.append(r[a] / tot)
        else:
            hw.append(1 / 3); dd.append(1 / 3); aw.append(1 / 3)
        r["D" if hg == ag else (h if hg > ag else a)] += 1
        recs[key] = r
    out = m.copy()
    out["h2h_home_wins"], out["h2h_draws"], out["h2h_away_wins"] = hw, dd, aw
    return out


def add_confederation(m: pd.DataFrame) -> pd.DataFrame:
    codes = _confederation_codes()
    out = m.copy()
    out["home_confederation"] = out["home"].map(codes).fillna(-1).astype(int)
    out["away_confederation"] = out["away"].map(codes).fillna(-1).astype(int)
    return out


def build_training_table() -> tuple[pd.DataFrame, None]:
    """Labelled match table with form/h2h/confederation features.

    Form is computed over the full history, then we keep MIN_YEAR+ rows for training.
    Target: outcome in {H, D, A}.
    """
    m = unified_matches()
    m = attach_eloratings(m)
    m = add_form(m)
    m = add_h2h(m)
    m = add_confederation(m)
    m["is_neutral"] = m["neutral"].astype(float)
    m["match_importance"] = m["competition"].map(elodata.importance_of)
    m = m[m["date"].dt.year >= MIN_YEAR].reset_index(drop=True)
    m["outcome"] = np.where(m["hg"] > m["ag"], "H", np.where(m["hg"] < m["ag"], "A", "D"))
    return m, None


# ---- helpers for predicting brand-new fixtures ----

def team_form(window: int = FORM_WINDOW) -> pd.DataFrame:
    """Each team's LATEST state: recent form + confederation + current Elo.

    Elo is the live eloratings.net rating (same source as training, so scales match)."""
    m = unified_matches()
    ratings = elodata.current_elo()
    long = pd.concat([
        pd.DataFrame({"date": m["date"], "team": m["home"], "gf": m["hg"], "ga": m["ag"]}),
        pd.DataFrame({"date": m["date"], "team": m["away"], "gf": m["ag"], "ga": m["hg"]}),
    ]).sort_values("date")
    g = long.groupby("team").agg(
        goals_for_avg=("gf", lambda s: s.tail(window).mean()),
        goals_against_avg=("ga", lambda s: s.tail(window).mean()),
    )
    g["confederation"] = pd.Series(_confederation_codes()).reindex(g.index).fillna(-1).astype(int)
    g["elo"] = pd.Series(ratings).reindex(g.index).fillna(ELO_BASE)
    return g


def h2h_table() -> dict[tuple, tuple]:
    """Final head-to-head shares for each unordered team pair."""
    m = unified_matches()
    recs: dict[tuple, dict] = {}
    for h, a, hg, ag in zip(m["home"], m["away"], m["hg"], m["ag"]):
        key = tuple(sorted((h, a)))
        r = recs.setdefault(key, {key[0]: 0, key[1]: 0, "D": 0})
        r["D" if hg == ag else (h if hg > ag else a)] += 1
    out = {}
    for key, r in recs.items():
        tot = r[key[0]] + r[key[1]] + r["D"]
        out[key] = (r[key[0]] / tot, r["D"] / tot, r[key[1]] / tot) if tot else (1/3, 1/3, 1/3)
    return out
