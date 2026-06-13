"""Low-level CSV reading helpers - the raw I/O layer.

No cleaning or business logic here: just locate files and read them with the
right encoding. Cleaning lives in `load.py`, name mapping in `teams.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Repo root is the parent of this src/ folder.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# Files known to need a non-UTF-8 decoder. Everything else defaults to utf-8
# and falls back to cp1252 / latin-1 if that fails.
ENCODING_OVERRIDES = {
    "2022_world_cup_squads.csv": "cp1252",
}


def configure_console() -> None:
    """Force UTF-8 stdout so Windows cp1252 consoles don't crash on names."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def find_csvs(root: Path = DATA_DIR) -> list[Path]:
    """All CSVs under root, sorted."""
    return sorted(root.rglob("*.csv"))


def read_csv_safe(path: Path) -> tuple[pd.DataFrame, str]:
    """Read a CSV, picking a working encoding. Returns (df, encoding_used)."""
    preferred = ENCODING_OVERRIDES.get(Path(path).name)
    candidates = [preferred] if preferred else []
    candidates += ["utf-8", "cp1252", "latin-1"]
    last_err: Exception | None = None
    for enc in candidates:
        if enc is None:
            continue
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            return df, enc
        except (UnicodeDecodeError, ValueError) as err:
            last_err = err
            continue
    raise RuntimeError(f"Could not read {Path(path).name}: {last_err}")


def human_size(num_bytes: int) -> str:
    """Format a byte count as a human-readable string."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
