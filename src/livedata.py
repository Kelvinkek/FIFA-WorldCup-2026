"""Fetch finished international matches from API-Football and save them as CSV.

Keeps the model current: pull completed results, write them to data/live_results.csv
in the same shape `features.unified_matches()` reads, so Elo and form fold the new
games in automatically.

Auth: API-Football (api-sports.io). Key + host are read from a gitignored .env:
    API_FOOTBALL_KEY=...
    API_FOOTBALL_HOST=v3.football.api-sports.io

Free plan = 100 requests/day, so each call counts - one (league, season) = one request.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests

from .io import ROOT, DATA_DIR
from .teams import canonical

LIVE_CSV = DATA_DIR / "live_results.csv"
FINISHED = {"FT", "AET", "PEN"}  # full-time, after extra-time, penalties

# International competitions that inform national-team strength (league ids from
# API-Football). Club leagues are ignored. (league_id, name)
INTL_LEAGUES = {
    1: "World Cup", 4: "Euro Championship", 5: "UEFA Nations League",
    6: "Africa Cup of Nations", 7: "Asian Cup", 9: "Copa America",
    10: "Friendlies", 22: "CONCACAF Gold Cup", 960: "Euro Qualification",
    29: "WC Qualification Africa", 30: "WC Qualification Asia",
    31: "WC Qualification CONCACAF", 32: "WC Qualification Europe",
    33: "WC Qualification Oceania", 34: "WC Qualification South America",
}

WC_QUAL = [29, 30, 31, 32, 33, 34]

# Rich pull covering 2015+ (the 2018 WC qualifying campaign onward): competitive
# qualifiers + continental cups + Nations League + friendlies. ~35 requests.
DEFAULT_TARGETS = (
    [(10, s) for s in range(2017, 2027)]                 # Friendlies 2017-2026
    + [(5, s) for s in (2018, 2020, 2022, 2024, 2025)]   # Nations League
    + [(q, s) for q in WC_QUAL for s in (2021, 2024)]    # WC qualifiers (2022 & 2026 cycles)
    + [(960, 2023)]                                      # Euro 2024 qualifiers
    + [(4, s) for s in (2016, 2020, 2024)]               # Euros
    + [(9, s) for s in (2016, 2019, 2021, 2024)]         # Copa America
    + [(6, s) for s in (2019, 2021, 2023)]               # Africa Cup of Nations
    + [(7, s) for s in (2019, 2023)]                     # Asian Cup
    + [(22, s) for s in (2017, 2019, 2021, 2023, 2025)]  # CONCACAF Gold Cup
)


def _config() -> tuple[str, str]:
    env = ROOT / ".env"
    cfg = {}
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    key = cfg.get("API_FOOTBALL_KEY") or os.environ.get("API_FOOTBALL_KEY", "")
    host = cfg.get("API_FOOTBALL_HOST") or "v3.football.api-sports.io"
    if not key:
        raise RuntimeError("No API key. Put API_FOOTBALL_KEY=... in a .env file.")
    return key, host


def _get(path: str, **params) -> dict:
    key, host = _config()
    r = requests.get(f"https://{host}{path}", headers={"x-apisports-key": key},
                     params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def account_status() -> dict:
    """Account plan + how many of today's requests are used."""
    return _get("/status").get("response", {})


def fetch_fixtures(league: int, season: int) -> pd.DataFrame:
    """Finished fixtures for one (league, season), in the unified layout."""
    data = _get("/fixtures", league=league, season=season)
    rows = []
    for m in data.get("response", []):
        if m["fixture"]["status"]["short"] not in FINISHED:
            continue
        g = m["goals"]
        if g["home"] is None or g["away"] is None:
            continue
        rows.append({
            "date": m["fixture"]["date"][:10],
            "home": canonical(m["teams"]["home"]["name"]),
            "away": canonical(m["teams"]["away"]["name"]),
            "hg": int(g["home"]),
            "ag": int(g["away"]),
            "neutral": True,  # international tournaments/friendlies: treat as neutral
            "competition": m["league"]["name"],
        })
    return pd.DataFrame(rows)


def fetch_finished(targets: list[tuple[int, int]] | None = None) -> pd.DataFrame:
    """Fetch finished matches across several (league, season) targets."""
    targets = targets or DEFAULT_TARGETS
    frames = []
    for league, season in targets:
        try:
            df = fetch_fixtures(league, season)
        except requests.HTTPError as e:
            print(f"  {INTL_LEAGUES.get(league, league)} {season}: HTTP {e.response.status_code}")
            continue
        print(f"  {INTL_LEAGUES.get(league, league)} {season}: {len(df)} finished matches")
        frames.append(df)
        time.sleep(7)  # free plan ~10 req/min - stay under the rate limit
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        out["date"] = pd.to_datetime(out["date"])
        out = out.sort_values("date").reset_index(drop=True)
    return out


def fetch_injuries(league: int, season: int) -> pd.DataFrame:
    """Current injuries for a competition: one row per injured player.

    Columns: team, player, reason, date. Use league=1 (World Cup), 5 (Nations
    League), etc. One request. (The genuinely *new* signal Elo can't see.)
    """
    data = _get("/injuries", league=league, season=season)
    rows = []
    for it in data.get("response", []):
        rows.append({
            "team": canonical(it["team"]["name"]),
            "player": it["player"]["name"],
            "reason": it["player"].get("reason"),
            "date": it["fixture"]["date"][:10] if it.get("fixture") else None,
        })
    return pd.DataFrame(rows)


def update_live_csv(targets: list[tuple[int, int]] | None = None) -> Path:
    """Fetch finished matches and write/merge them into data/live_results.csv."""
    new = fetch_finished(targets)
    if new.empty:
        print("No finished matches returned.")
        return LIVE_CSV

    if LIVE_CSV.exists():
        old = pd.read_csv(LIVE_CSV, parse_dates=["date"])
        new = pd.concat([old, new], ignore_index=True)
    new = (new.drop_duplicates(subset=["date", "home", "away"], keep="last")
              .sort_values("date").reset_index(drop=True))
    new.to_csv(LIVE_CSV, index=False)
    print(f"Saved {len(new)} matches -> {LIVE_CSV}  "
          f"({new['date'].min().date()} to {new['date'].max().date()})")
    return LIVE_CSV
