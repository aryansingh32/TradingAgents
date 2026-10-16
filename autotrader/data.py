"""Price history loading.

Every loader returns a DataFrame indexed by timestamp with lowercase
``open, high, low, close, volume`` columns, oldest first.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
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


def load_yfinance(symbol: str, interval: str = "1d", lookback_days: int = 1500) -> pd.DataFrame:
    import yfinance as yf

    start = datetime.now(UTC) - timedelta(days=lookback_days)
    raw = yf.download(
        symbol, start=start.strftime("%Y-%m-%d"), interval=interval,
        progress=False, auto_adjust=True,
    )
    return normalize_ohlcv(raw)


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if c.lower() in ("date", "datetime", "timestamp", "time")), None)
    if date_col is None:
        raise ValueError(f"{path}: needs a date/datetime/timestamp column")
    df[date_col] = pd.to_datetime(df[date_col])
    return normalize_ohlcv(df.set_index(date_col))


def load_history(symbol: str, source: str = "yfinance", interval: str = "1d",
                 lookback_days: int = 1500, data_dir: str | Path = "autotrader_data") -> pd.DataFrame:
    """Load history from ``source``: ``yfinance`` or ``csv``.

    The csv source reads ``<data_dir>/<symbol>.csv`` (``/`` in the symbol
    becomes ``_``), which lets the bot run on data exported from any vendor
    or broker, and on machines that cannot reach Yahoo.
    """
    if source == "yfinance":
        return load_yfinance(symbol, interval, lookback_days)
    if source == "csv":
        return load_csv(Path(data_dir) / f"{symbol.replace('/', '_')}.csv")
    raise ValueError(f"unknown data source {source!r}")
