"""OHLCV frame helpers.

Every loader and feed returns a DataFrame indexed by the candle's UTC open
time with lowercase ``open, high, low, close, volume`` columns, oldest first.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV)
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance returns (field, ticker) columns for downloads.
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={c: str(c).lower().replace(" ", "_") for c in df.columns})
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"price data is missing columns {missing}")
    out = df[OHLCV].astype(float).dropna(subset=["close"])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def load_csv(path: str | Path) -> pd.DataFrame:
    """Read candles exported from any vendor or broker: a date/datetime/
    timestamp column (UTC unless it carries an offset) plus OHLCV columns."""
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if c.lower() in ("date", "datetime", "timestamp", "time")), None)
    if date_col is None:
        raise ValueError(f"{path}: needs a date/datetime/timestamp column")
    df[date_col] = pd.to_datetime(df[date_col], utc=True)
    return normalize_ohlcv(df.set_index(date_col))
