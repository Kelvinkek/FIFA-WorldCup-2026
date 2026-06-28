"""Knockout-stage fixtures (Round of 32) from an open source.

The official knockout bracket is NOT in the static schedule - it depends on the final
group standings - and the two football APIs don't help here: API-Football's free plan
returns nothing for the 2026 World Cup, and football-data.org gates the World Cup
behind a token/tier. So the default source is **fixturedownload.com**, a free, no-auth
JSON feed that publishes the full 2026 bracket and fills in the real matchups as each
round is decided.

Two providers, same output (a tidy fixtures DataFrame cached to data/round_of_32.csv):

  source="fixturedownload"  -> open JSON feed, no auth                 (default, tested)
  source="football-data"    -> football-data.org API, needs a free
                               FOOTBALL_DATA_TOKEN in .env             (drop-in if their
                                                                        free tier covers WC)

`source="auto"` uses football-data.org when a token is present, else fixturedownload,
falling back to fixturedownload on any error - so the pipeline never breaks.
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timedelta, timezone

import pandas as pd

from .io import DATA_DIR, ROOT
from .teams import canonical

R32_CACHE = DATA_DIR / "round_of_32.csv"
BRISBANE = timezone(timedelta(hours=10))   # AEST - the project's display timezone

# --- fixturedownload.com (open, no auth) ---
FD_FEED = "https://fixturedownload.com/feed/json/fifa-world-cup-2026"
FD_ROUND_OF_32 = 4   # RoundNumber: 1-3 = group match-days, 4 = Round of 32

# --- football-data.org (needs a free token) ---
FOOTBALL_DATA_URL = "https://api.football-data.org/v4/competitions/WC/matches"
FOOTBALL_DATA_STAGE = "LAST_32"


def _read_env(key: str) -> str:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and line.split("=", 1)[0].strip() == key:
                return line.split("=", 1)[1].strip()
    return os.environ.get(key, "")


def _to_rows(records: list[dict]) -> pd.DataFrame:
    """Normalise provider records -> the cached fixture layout (Brisbane-localised)."""
    rows = []
    for rec in records:
        utc = rec["utc"]
        bne = utc.astimezone(BRISBANE)
        rows.append({
            "match_no": rec.get("match_no"),
            "date_utc": utc.strftime("%Y-%m-%d %H:%M"),
            "date": bne.strftime("%Y-%m-%d"),          # Brisbane match-day (project standard)
            "time": bne.strftime("%H:%M"),             # Brisbane kickoff
            "home": canonical(rec["home"]),
            "away": canonical(rec["away"]),
            "location": rec.get("location"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date_utc").reset_index(drop=True)
    return df


def _from_fixturedownload() -> pd.DataFrame:
    req = urllib.request.Request(FD_FEED, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        feed = json.loads(r.read().decode("utf-8"))
    records = []
    for m in feed:
        if str(m.get("RoundNumber")) != str(FD_ROUND_OF_32):
            continue
        h, a = m.get("HomeTeam"), m.get("AwayTeam")
        if not h or not a:                              # slot not decided yet
            continue
        records.append({
            "match_no": m.get("MatchNumber"),
            "utc": datetime.strptime(m["DateUtc"], "%Y-%m-%d %H:%M:%SZ").replace(tzinfo=timezone.utc),
            "home": h, "away": a, "location": m.get("Location"),
        })
    return _to_rows(records)


def _from_football_data() -> pd.DataFrame:
    """football-data.org provider. Requires FOOTBALL_DATA_TOKEN in .env and a tier that
    includes the World Cup; raises otherwise (the caller can fall back)."""
    token = _read_env("FOOTBALL_DATA_TOKEN")
    if not token:
        raise RuntimeError("No FOOTBALL_DATA_TOKEN in .env")
    req = urllib.request.Request(
        f"{FOOTBALL_DATA_URL}?stage={FOOTBALL_DATA_STAGE}",
        headers={"X-Auth-Token": token, "User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
    records = []
    for m in data.get("matches", []):
        h = (m.get("homeTeam") or {}).get("name")
        a = (m.get("awayTeam") or {}).get("name")
        if not h or not a:                              # slot not decided yet
            continue
        records.append({
            "match_no": m.get("id"),
            "utc": datetime.strptime(m["utcDate"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc),
            "home": h, "away": a, "location": (m.get("venue") or None),
        })
    return _to_rows(records)


def refresh_round_of_32(source: str = "auto", verbose: bool = True) -> pd.DataFrame:
    """Fetch the Round-of-32 fixtures and cache them to data/round_of_32.csv."""
    use_api = source == "football-data" or (source == "auto" and _read_env("FOOTBALL_DATA_TOKEN"))
    df = pd.DataFrame()
    if use_api:
        try:
            df = _from_football_data()
            if verbose:
                print(f"Round of 32 via football-data.org: {len(df)} fixtures")
        except Exception as e:
            if source == "football-data":
                raise
            if verbose:
                print(f"  football-data.org unavailable ({e!r}); using fixturedownload")
    if df.empty:
        df = _from_fixturedownload()
        if verbose:
            print(f"Round of 32 via fixturedownload.com: {len(df)} fixtures")

    df.to_csv(R32_CACHE, index=False)
    if verbose and not df.empty:
        print(f"  cached -> {R32_CACHE.name}  ({df['date'].min().date()} to {df['date'].max().date()})")
    elif verbose:
        print("  no decided Round-of-32 fixtures yet")
    return df


def load_round_of_32() -> pd.DataFrame:
    """Cached R32 fixtures (refreshing from the web if the cache is missing)."""
    if R32_CACHE.exists():
        return pd.read_csv(R32_CACHE, parse_dates=["date"])
    return refresh_round_of_32(verbose=False)
