"""Clean, merge-ready loaders.

Each function reads a raw CSV via `io.read_csv_safe`, applies the canonical
team-name map from `teams.py`, and returns a tidy DataFrame. Notebooks and the
modelling code should import from here rather than touching `data/` directly.
"""

from __future__ import annotations

import pandas as pd

from . import io
from .teams import normalize_teams

K = io.DATA_DIR / "kaggle_2026"
M = io.DATA_DIR / "maven_analytics"


def load_rankings(snapshot: str = "2026-06-08") -> pd.DataFrame:
    """FIFA ranking snapshot (Kaggle). `snapshot` in {'2026-06-08','2022-10-06'}."""
    df, _ = io.read_csv_safe(K / f"fifa_ranking_{snapshot}.csv")
    df = normalize_teams(df, ["team"])
    return df


def load_wc_matches() -> pd.DataFrame:
    """World Cup matches 1930-2022 (Kaggle, rich). Parsed Date, names normalised."""
    df, _ = io.read_csv_safe(K / "matches_1930_2022.csv")
    df = normalize_teams(df, ["home_team", "away_team"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def load_international() -> pd.DataFrame:
    """All international matches 1872-2022 (Maven). Team names canonicalised.

    Note: this file holds qualifiers, friendlies and continental cups - NOT the
    World Cup finals (those live in `load_wc_matches`). So `is_wc_qualifier`
    flags qualification matches, the closest WC-related rows present here.
    Adds a numeric `year` for convenience.
    """
    df, _ = io.read_csv_safe(M / "international_matches.csv")
    df = normalize_teams(df, ["Home Team", "Away Team"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["year"] = df["Date"].dt.year
    df["is_wc_qualifier"] = df["Tournament"].eq("FIFA World Cup qualification")
    return df


def load_squads() -> pd.DataFrame:
    """2022 World Cup squads (Maven, cp1252). Adds derived non-WC career stats."""
    df, _ = io.read_csv_safe(M / "2022_world_cup_squads.csv")
    df = normalize_teams(df, ["Team"])
    # Goals scored for the national team outside the World Cup.
    df["non_wc_goals"] = (df["Goals"] - df["WC Goals"]).clip(lower=0)
    return df


def load_schedule_2026() -> pd.DataFrame:
    """2026 World Cup schedule (Kaggle). Team names canonicalised."""
    df, _ = io.read_csv_safe(K / "schedule_2026.csv")
    df = normalize_teams(df, ["home_team", "away_team"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def load_tournaments() -> pd.DataFrame:
    """Tournament-level summary (Kaggle world_cup.csv - has a complete 2022 row)."""
    df, _ = io.read_csv_safe(K / "world_cup.csv")
    return df
