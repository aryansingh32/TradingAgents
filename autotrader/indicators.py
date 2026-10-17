"""Technical indicators on OHLCV frames (pure pandas, no look-ahead)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    # No losses in the window means maximum strength.
    return out.where(loss != 0, 100.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def bollinger(close: pd.Series, period: int = 20, width: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return pd.DataFrame({"mid": mid, "upper": mid + width * std, "lower": mid - width * std})


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Wilder's ADX with +DI / -DI."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - df["close"].shift()).abs(),
         (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    alpha = 1 / period
    atr_w = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr_w
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr_w
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return pd.DataFrame({"adx": dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean(),
                         "plus_di": plus_di, "minus_di": minus_di})


def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.Series:
    """Supertrend direction: +1 while price holds above the lower band, -1
    after it closes below it (and vice versa)."""
    a = atr(df, period).to_numpy()
    hl2 = ((df["high"] + df["low"]) / 2).to_numpy()
    close = df["close"].to_numpy()
    upper, lower = hl2 + mult * a, hl2 - mult * a
    direction = np.full(len(df), np.nan)
    fu, fl, d = np.nan, np.nan, 1.0
    for i in range(len(df)):
        if np.isnan(a[i]):
            continue
        if np.isnan(fu):
            fu, fl = upper[i], lower[i]
        else:
            fu = upper[i] if upper[i] < fu or close[i - 1] > fu else fu
            fl = lower[i] if lower[i] > fl or close[i - 1] < fl else fl
        if close[i] > fu:
            d = 1.0
        elif close[i] < fl:
            d = -1.0
        direction[i] = d
    return pd.Series(direction, index=df.index)


def obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["close"].diff()).fillna(0)
    return (sign * df["volume"]).cumsum()
