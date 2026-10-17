"""More signal families used by popular bots and discretionary traders.

Each ``*_scores`` function returns one score per bar in [-1, 1], or NaN where
the family has nothing to say (that bar then simply doesn't count). The same
functions serve the live bot (last bar) and the backtest (every bar), and use
only data up to each bar.

* patterns: candlestick reversal patterns at the end of a move (engulfing,
  hammer / shooting star, morning / evening star), fading over three bars;
* volume: price vs rolling VWAP, on-balance-volume trend, and volume spikes
  in the direction of the bar;
* trend_strength: Supertrend direction scaled by ADX (strong trends count
  fully, weak trends barely);
* mean_reversion: the RSI(2) / Bollinger dip-buy inside a long-term uptrend
  (Connors-style). It is a *setup*: when it fires strongly in a confirmed
  uptrend the bot may buy even if momentum readings are still negative,
  because buying weakness is the point of the setup.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from autotrader import indicators as ind
from autotrader.signals import Opinion


def _decay(raw: pd.Series, weights=(1.0, 0.6, 0.3)) -> pd.Series:
    """Carry a sparse event signal for a few bars, fading; NaN when none."""
    out = pd.Series(np.nan, index=raw.index)
    for lag, w in reversed(list(enumerate(weights))):
        shifted = raw.shift(lag) * w
        out = shifted.where(shifted.notna() & (shifted != 0), out)
    return out


def pattern_scores(df: pd.DataFrame) -> pd.Series:
    o, h, lo, c = (df[k] for k in ("open", "high", "low", "close"))
    body = (c - o).abs()
    rng = (h - lo).replace(0, np.nan)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - lo
    fell = c.shift(1) < c.shift(4)  # the move before the pattern
    rose = c.shift(1) > c.shift(4)
    green, red = c > o, c < o
    po, pc = o.shift(1), c.shift(1)

    bull_engulf = green & (pc < po) & (c >= po) & (o <= pc) & fell
    bear_engulf = red & (pc > po) & (c <= po) & (o >= pc) & rose
    hammer = (lower >= 2 * body) & (upper <= 0.3 * body + 0.1 * rng) & fell
    star = (upper >= 2 * body) & (lower <= 0.3 * body + 0.1 * rng) & rose
    small_mid = (c.shift(1) - o.shift(1)).abs() < 0.3 * (h.shift(1) - lo.shift(1))
    morning = (c.shift(2) < o.shift(2)) & small_mid & green & (c > (o.shift(2) + c.shift(2)) / 2) & fell.shift(1, fill_value=False)
    evening = (c.shift(2) > o.shift(2)) & small_mid & red & (c < (o.shift(2) + c.shift(2)) / 2) & rose.shift(1, fill_value=False)

    raw = pd.Series(0.0, index=df.index)
    raw = raw.mask(hammer, 0.7).mask(star, -0.7)
    raw = raw.mask(bull_engulf | morning, 1.0).mask(bear_engulf | evening, -1.0)
    return _decay(raw)


def volume_scores(df: pd.DataFrame, period: int = 20) -> pd.Series:
    vol = df["volume"]
    if not (vol > 0).any():
        return pd.Series(np.nan, index=df.index)
    close = df["close"]
    vwap = (close * vol).rolling(period).sum() / vol.rolling(period).sum().replace(0, np.nan)
    o = ind.obv(df)
    obv_trend = np.tanh((o - ind.ema(o, period)) / (o.rolling(period).std().replace(0, np.nan)))
    rel = vol / vol.rolling(period).mean().replace(0, np.nan)
    bar_dir = np.sign(close - df["open"])
    spike = (bar_dir * ((rel - 1.5).clip(lower=0, upper=1.5) / 1.5)).where(rel > 1.5, 0.0)
    score = 0.4 * np.sign(close - vwap) + 0.3 * obv_trend.fillna(0) + 0.3 * spike
    return score.where(vwap.notna())


def trend_strength_scores(df: pd.DataFrame) -> pd.Series:
    st = ind.supertrend(df)
    strength = (ind.adx(df)["adx"] / 40).clip(upper=1.0)
    return (st * strength).where(st.notna() & strength.notna())


def mean_reversion_scores(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    uptrend = close > ind.sma(close, 200)
    rsi2 = ind.rsi(close, 2)
    bb = ind.bollinger(close)
    pct_b = (close - bb["lower"]) / (bb["upper"] - bb["lower"]).replace(0, np.nan)
    score = pd.Series(np.nan, index=df.index)
    score = score.mask(uptrend & (rsi2 < 10), 0.7)
    score = score.mask(uptrend & ((rsi2 < 5) | ((rsi2 < 10) & (pct_b < 0))), 1.0)
    return score


EXTRA = {
    "patterns": pattern_scores,
    "volume": volume_scores,
    "trend_strength": trend_strength_scores,
    "mean_reversion": mean_reversion_scores,
}

_REASON = {
    "patterns": lambda v: ("bullish reversal candle (engulfing / hammer / morning star)" if v > 0
                           else "bearish reversal candle (engulfing / shooting star / evening star)"),
    "volume": lambda v: ("buying volume: above VWAP, OBV rising" if v > 0 else "selling volume: below VWAP, OBV falling"),
    "trend_strength": lambda v: (f"Supertrend {'up' if v > 0 else 'down'}, ADX trend strength {abs(v):.0%}"),
    "mean_reversion": lambda v: "RSI(2) dip inside a 200-bar uptrend (mean-reversion setup)",
}


def extra_opinions(df: pd.DataFrame, weights: dict[str, float]) -> list[Opinion]:
    out = []
    for name, fn in EXTRA.items():
        if weights.get(name, 0) <= 0 or len(df) < 210:
            continue
        v = fn(df).iloc[-1]
        if pd.notna(v):
            out.append(Opinion(name, float(v), [_REASON[name](v)]))
    return out
