"""The bot's two price-history opinions: a technical read and an analog forecast.

Each returns an ``Opinion`` with a score in [-1, 1] (+1 strongly bullish,
-1 strongly bearish) and the reasons behind it, computed only from bars up to
and including the last row of the frame it is given, so the same functions
are safe to call bar by bar in a backtest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from autotrader import indicators as ind

MIN_BARS_TECHNICAL = 60
MIN_BARS_FORECAST = 250


@dataclass
class Opinion:
    name: str
    score: float
    reasons: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        self.score = float(np.clip(self.score, -1.0, 1.0))


def technical_opinion(df: pd.DataFrame) -> Opinion | None:
    """Trend, momentum, mean-reversion and breakout, averaged."""
    if len(df) < MIN_BARS_TECHNICAL:
        return None
    close = df["close"]
    price = float(close.iloc[-1])
    long_span = 200 if len(df) >= 220 else 100
    ema_fast = ind.ema(close, 50).iloc[-1]
    ema_slow = ind.ema(close, long_span).iloc[-1]
    atr_now = ind.atr(df).iloc[-1]
    hist = ind.macd(close)["hist"].iloc[-1]
    rsi_now = ind.rsi(close).iloc[-1]
    prior_high = df["high"].iloc[-21:-1].max()
    prior_low = df["low"].iloc[-21:-1].min()

    parts: list[float] = []
    reasons: list[str] = []

    if not np.isnan(ema_slow):
        trend = (0.5 if price > ema_slow else -0.5) + (0.5 if ema_fast > ema_slow else -0.5)
        parts.append(trend)
        reasons.append(
            f"price {'above' if price > ema_slow else 'below'} EMA{long_span}, "
            f"EMA50 {'above' if ema_fast > ema_slow else 'below'} EMA{long_span}"
        )
    if not (np.isnan(hist) or np.isnan(atr_now) or atr_now == 0):
        momentum = float(np.tanh(3 * hist / atr_now))
        parts.append(momentum)
        reasons.append(f"MACD histogram {'rising' if hist > 0 else 'falling'} ({momentum:+.2f})")
    if not np.isnan(rsi_now):
        if rsi_now > 75:
            parts.append(-0.5)
            reasons.append(f"RSI {rsi_now:.0f} overbought")
        elif rsi_now < 25:
            parts.append(0.5)
            reasons.append(f"RSI {rsi_now:.0f} oversold")
        else:
            parts.append((rsi_now - 50) / 50)
            reasons.append(f"RSI {rsi_now:.0f}")
    if price > prior_high:
        parts.append(1.0)
        reasons.append("20-bar breakout to the upside")
    elif price < prior_low:
        parts.append(-1.0)
        reasons.append("20-bar breakdown")

    if not parts:
        return None
    return Opinion(
        "technical", float(np.mean(parts)), reasons,
        {"rsi": float(rsi_now), "atr": float(atr_now), "ema50": float(ema_fast), "ema_long": float(ema_slow)},
    )


def analog_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    logret = np.log(close).diff()
    atr = ind.atr(df)
    feats = pd.DataFrame(
        {
            "ret5": np.log(close / close.shift(5)),
            "ret20": np.log(close / close.shift(20)),
            "ret60": np.log(close / close.shift(60)),
            "vol20": logret.rolling(20).std(),
            "dist_ema50": (close - ind.ema(close, 50)) / atr,
            "rsi": ind.rsi(close) / 100,
            "vol_ratio": logret.rolling(20).std() / logret.rolling(120).std(),
        },
        index=df.index,
    )
    return feats.replace([np.inf, -np.inf], np.nan)


def forecast_opinion(df: pd.DataFrame, horizon: int = 10, neighbors: int = 50,
                     features: pd.DataFrame | None = None) -> Opinion | None:
    """History-analog forecast: "the last N times the market looked like this,
    what happened over the next ``horizon`` bars?"

    Each bar is described by its recent returns, volatility, distance from
    trend and RSI. The ``neighbors`` past bars most similar to today (whose
    outcome is already known) vote with their forward returns. The score
    blends the share of analogs that went up with their median move scaled
    by volatility.

    ``features`` may be precomputed with ``analog_features`` on a longer frame
    (all features are causal); only its rows up to ``df``'s last bar are used.
    """
    if len(df) < MIN_BARS_FORECAST:
        return None
    feats = analog_features(df) if features is None else features.iloc[: len(df)]
    close = df["close"]
    current = feats.iloc[-1]
    if current.isna().any():
        return None
    # Only bars whose forward return is fully in the past are candidates.
    n_known = len(df) - horizon
    fwd = np.log(close.shift(-horizon).iloc[:n_known] / close.iloc[:n_known])
    known = feats.iloc[:n_known].assign(fwd=fwd).dropna()
    if len(known) < max(neighbors, 100):
        return None
    hist_feats = known.drop(columns="fwd")
    mean, std = hist_feats.mean(), hist_feats.std().replace(0, 1)
    z_hist = (hist_feats - mean) / std
    z_now = (current - mean) / std
    dist = np.sqrt(((z_hist - z_now) ** 2).sum(axis=1))
    k = min(neighbors, len(dist))
    nearest = known.loc[dist.nsmallest(k).index, "fwd"]

    prob_up = float((nearest > 0).mean())
    median = float(nearest.median())
    p10, p90 = float(nearest.quantile(0.1)), float(nearest.quantile(0.9))
    scale = float(current["vol20"]) * np.sqrt(horizon) or 1e-9
    score = 0.5 * (2 * prob_up - 1) + 0.5 * float(np.tanh(median / scale))
    pct = lambda x: f"{(np.exp(x) - 1) * 100:+.1f}%"  # noqa: E731
    return Opinion(
        "forecast", score,
        [
            f"{k} closest historical analogs: {prob_up:.0%} rose over the next {horizon} bars",
            f"median move {pct(median)}, 10th-90th percentile {pct(p10)} to {pct(p90)}",
        ],
        {"prob_up": prob_up, "median_return": float(np.exp(median) - 1),
         "p10_return": float(np.exp(p10) - 1), "p90_return": float(np.exp(p90) - 1),
         "horizon": horizon, "analogs": k},
    )
