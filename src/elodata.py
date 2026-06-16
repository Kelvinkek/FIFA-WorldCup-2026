"""eloratings.net data — professional, importance-weighted national-team Elo.

eloratings.net serves raw TSV files (no scraping/JS needed):
  - en.teams.tsv        : 2-letter code -> team name
  - <Team_Name>.tsv     : one team's FULL match history, with the team's Elo *after*
                          each match (spaces in the name become underscores)

For leak-free features we need each team's Elo **before** a match. Inside a team's own
file the rows are chronological, so the pre-match Elo of a row = the post-match Elo of
the previous row (a simple shift). We harvest that into a (date, team) -> elo_before
table and join it onto our match list.

Public API:
  refresh_elo_history(teams=None)  -> fetch + rebuild data/eloratings_history.csv
  load_elo_history()               -> cached DataFrame [date, team, elo_before]
  elo_before_map()                 -> {(Timestamp, canonical_team): elo_before}
  current_elo()                    -> {canonical_team: latest Elo}  (from World.tsv)
"""

from __future__ import annotations

import gzip
import urllib.parse
import urllib.request

import pandas as pd

from . import teams
from .io import DATA_DIR

BASE = "https://www.eloratings.net"
CACHE = DATA_DIR / "eloratings_history.csv"
RESULTS_CACHE = DATA_DIR / "eloratings_results.csv"
RESULTS_SINCE = "2025-01-01"   # only cache recent results (for the prediction tracker)
_HDR = {"User-Agent": "Mozilla/5.0"}


# ----------------------------- low-level fetch -----------------------------

def _get(path: str) -> str:
    """Fetch a TSV file from eloratings.net (handles gzip + the U+2212 minus)."""
    url = f"{BASE}/{path}"
    raw = urllib.request.urlopen(urllib.request.Request(url, headers=_HDR), timeout=30).read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", "replace").replace("−", "-")  # normalise minus sign


def _slug(name: str) -> str:
    return urllib.parse.quote(name.replace(" ", "_"))


# ----------------------------- reference tables -----------------------------

def code_to_name() -> dict[str, str]:
    """eloratings 2-letter code -> team name (skips `_loc` venue pseudo-codes)."""
    out = {}
    for line in _get("en.teams.tsv").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and not parts[0].endswith("_loc"):
            out[parts[0]] = parts[1]
    return out


# ----------------------------- history build -----------------------------

def _parse_team_file(text: str, code2name: dict[str, str], subject: str) -> pd.DataFrame:
    """Parse one <Team>.tsv into rows: date, team, elo_before/elo_after for `subject`.

    Each row is an actual fixture: f[3]=home code, f[4]=away code, f[10]=home Elo
    (post-match), f[11]=away Elo (post-match). We pick whichever side is the subject
    team, so an away game records the subject's Elo (f[11]), not the opponent's.
    """
    recs = []
    for line in text.splitlines():
        f = line.split("\t")
        if len(f) < 12:
            continue
        try:
            date = pd.Timestamp(int(f[0]), int(f[1]), int(f[2]))
            home = teams.canonical(code2name.get(f[3], f[3]))
            away = teams.canonical(code2name.get(f[4], f[4]))
            if home == subject:
                elo_after = int(f[10])
            elif away == subject:
                elo_after = int(f[11])
            else:
                continue
        except (ValueError, KeyError):
            continue
        recs.append((date, subject, elo_after))
    df = pd.DataFrame(recs, columns=["date", "team", "elo_after"]).sort_values("date")
    # pre-match Elo = previous match's post-match Elo (within this team's own history)
    df["elo_before"] = df["elo_after"].shift()
    return df


def _parse_results(text: str, code2name: dict[str, str]) -> pd.DataFrame:
    """Parse one <Team>.tsv into match rows: date, home, away, hg, ag, comp_code."""
    recs = []
    for line in text.splitlines():
        f = line.split("\t")
        if len(f) < 8:
            continue
        try:
            date = pd.Timestamp(int(f[0]), int(f[1]), int(f[2]))
            home = teams.canonical(code2name.get(f[3], f[3]))
            away = teams.canonical(code2name.get(f[4], f[4]))
            hg, ag = int(f[5]), int(f[6])
        except (ValueError, KeyError):
            continue
        recs.append((date, home, away, hg, ag, f[7]))
    return pd.DataFrame(recs, columns=["date", "home", "away", "hg", "ag", "comp_code"])


def refresh_elo_history(teams_wanted: list[str] | None = None, verbose: bool = True) -> pd.DataFrame:
    """Fetch per-team histories and (re)build the cached tables.

    Writes two CSVs from the same downloads:
      - data/eloratings_history.csv : (date, team, elo_before) for joining onto matches
      - data/eloratings_results.csv : recent match results with scores (for the tracker)

    `teams_wanted` is a list of *canonical* names; if None, every eloratings national
    team is fetched. Returns the Elo-history DataFrame.
    """
    code2name = code_to_name()
    # canonical name -> eloratings display name (for building the file slug)
    canon_to_elo = {teams.canonical(n): n for n in code2name.values()}
    wanted = teams_wanted or list(canon_to_elo)

    frames, results, missing = [], [], []
    for i, canon in enumerate(sorted(set(wanted))):
        elo_name = canon_to_elo.get(canon, canon)
        try:
            txt = _get(f"{_slug(elo_name)}.tsv")
        except Exception:
            missing.append(canon)
            continue
        df = _parse_team_file(txt, code2name, subject=canon)
        frames.append(df[["date", "team", "elo_before", "elo_after"]])
        rdf = _parse_results(txt, code2name)
        results.append(rdf[rdf["date"] >= pd.Timestamp(RESULTS_SINCE)])
        if verbose and (i + 1) % 25 == 0:
            print(f"  ...fetched {i + 1}/{len(set(wanted))} teams")

    hist = pd.concat(frames, ignore_index=True).dropna(subset=["elo_before"])
    hist = hist.drop_duplicates(subset=["date", "team"], keep="last").sort_values("date")
    CACHE.parent.mkdir(exist_ok=True)
    hist.to_csv(CACHE, index=False)

    res = (pd.concat(results, ignore_index=True)
             .drop_duplicates(subset=["date", "home", "away"]).sort_values("date"))
    res.to_csv(RESULTS_CACHE, index=False)

    if verbose:
        print(f"Saved {CACHE.name}: {len(hist):,} team-matches, "
              f"{hist['team'].nunique()} teams; {len(missing)} not found")
        print(f"Saved {RESULTS_CACHE.name}: {len(res):,} recent results "
              f"(since {RESULTS_SINCE})")
    return hist


def load_results() -> pd.DataFrame:
    if not RESULTS_CACHE.exists():
        raise FileNotFoundError(f"{RESULTS_CACHE} missing — run elodata.refresh_elo_history() first")
    return pd.read_csv(RESULTS_CACHE, parse_dates=["date"])


def load_elo_history() -> pd.DataFrame:
    if not CACHE.exists():
        raise FileNotFoundError(f"{CACHE} missing — run elodata.refresh_elo_history() first")
    return pd.read_csv(CACHE, parse_dates=["date"])


def elo_before_map() -> dict[tuple, float]:
    """{(date, canonical_team): elo_before} for joining onto a match list."""
    h = load_elo_history()
    return {(d, t): e for d, t, e in zip(h["date"], h["team"], h["elo_before"])}


def current_elo() -> dict[str, float]:
    """Latest Elo for every team, from World.tsv (code col 2, Elo col 3)."""
    code2name = code_to_name()
    out = {}
    for line in _get("World.tsv").splitlines():
        f = line.split("\t")
        if len(f) >= 4 and f[2] in code2name:
            try:
                out[teams.canonical(code2name[f[2]])] = float(f[3])
            except ValueError:
                continue
    return out


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    refresh_elo_history()
