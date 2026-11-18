"""Load the business dataset with correct dtypes and survival columns."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "data" / "raw" / "businesses_synthetic.csv"
DATE_COLS = ["issue_date", "cancel_date", "last_renewal_date"]


def load_businesses(path: Path | str = DEFAULT_PATH) -> pd.DataFrame:
    """Read the CSV, parse dates and ensure `duration_months` (>0) and `event` (0/1) exist.

    Raw DED data may lack the survival columns, so they are derived from the
    dates when missing: event = license cancelled; duration = months from issue
    to cancellation, or to the latest date in the data for active (censored) licenses.
    """
    df = pd.read_csv(path, parse_dates=DATE_COLS)
    if "event" not in df or "duration_months" not in df:
        cutoff = df[["issue_date", "cancel_date", "last_renewal_date"]].max().max()
        end = df["cancel_date"].fillna(cutoff)
        df["event"] = df["cancel_date"].notna().astype(int)
        df["duration_months"] = ((end - df["issue_date"]).dt.days / 30.44).round().astype(int)
    # lifelines needs strictly positive durations
    df["duration_months"] = df["duration_months"].clip(lower=0.5)
    df["event"] = df["event"].astype(int)
    df["issue_year"] = df["issue_date"].dt.year
    return df
