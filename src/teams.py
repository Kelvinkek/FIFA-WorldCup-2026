"""Canonical team-name mapping — the cross-source join key.

The two data sources spell many teams differently (e.g. Maven "Iran" vs Kaggle
"IR Iran"). We standardise on the **Kaggle** spelling because the Kaggle files
carry the FIFA ranking and 2026 schedule we ultimately join everything onto.

`TEAM_NAME_MAP` maps every known *alternate* spelling -> canonical Kaggle name.
Names not in the map are returned unchanged (e.g. Maven-only regional/historical
sides like "Catalonia" or "Saarland" that have no Kaggle counterpart).

The mappings below were derived by diffing every team string across all 12 CSVs.
"""

from __future__ import annotations

import pandas as pd

# alternate spelling (mostly Maven) -> canonical (Kaggle) spelling
TEAM_NAME_MAP: dict[str, str] = {
    # confirmed Maven -> Kaggle counterparts
    "China": "China PR",
    "Iran": "IR Iran",
    "Ivory Coast": "Côte d'Ivoire",
    "North Korea": "Korea DPR",
    "South Korea": "Korea Republic",
    "Turkey": "Türkiye",
    "East Germany": "Germany DR",
    "German DR": "Germany DR",
    "DR Congo": "Congo DR",
    "Taiwan": "Chinese Taipei",
    "Vietnam Republic": "Vietnam",
    "Yemen DPR": "Yemen",
    "United States Virgin Islands": "US Virgin Islands",
    "Hong Kong": "Hong Kong, China",
    "Gambia": "The Gambia",
    "Brunei": "Brunei Darussalam",
    "Kyrgyzstan": "Kyrgyz Republic",
    # renamed nations — historical match data uses the old name, 2026 uses the new
    "Czech Republic": "Czechia",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    # other common variants worth normalising defensively
    "USA": "United States",
    "Korea": "Korea Republic",
    "Republic of Ireland": "Ireland",
}


def canonical(name: str) -> str:
    """Return the canonical spelling for a single team name."""
    if not isinstance(name, str):
        return name
    return TEAM_NAME_MAP.get(name.strip(), name.strip())


def normalize_teams(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Return a copy of df with the given team-name columns canonicalised."""
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].map(canonical)
    return out
