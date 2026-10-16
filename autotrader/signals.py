"""The bot's price-history opinions: a technical read, an analog forecast, and
the higher-timeframe trend filter.

Every opinion has a score in [-1, 1] (+1 strongly bullish, -1 strongly
bearish). Each one is computed for a whole history at once (``*_scores``) for
fast backtests, and for the last bar only (``*_opinion``) in live trading;
both paths run the same code, so a backtest sees exactly what the live bot
would have seen at each closed candle (tested).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

from autotrader import indicators as ind

MIN_BARS_TECHNICAL = 220
MIN_BARS_FORECAST = 300


@dataclass
class Opinion:
    name: str
    score: float
    reasons: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        self.score = float(np.clip(self.score, -1.0, 1.0))


# -- technical ------------------------------------------------------------------

def technical_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Per-bar technical score: the mean of trend, momentum, RSI and (when
    present) a 20-bar breakout component."""
    close = df["close"]
    ema50, ema200 = ind.ema(close, 50), ind.ema(close, 200)
    atr = ind.atr(df)
    hist = ind.macd(close)["hist"]
    rsi = ind.rsi(close)
    prior_high = df["high"].shift(1).rolling(20).max()
    prior_low = df["low"].shift(1).rolling(20).min()

    trend = np.where(close > ema200, 0.5, -0.5) + np.where(ema50 > ema200, 0.5, -0.5)
    trend = pd.Series(trend, index=df.index).where(ema200.notna())
    momentum = np.tanh(3 * hist / atr.replace(0, np.nan))
    rsi_part = ((rsi - 50) / 50).mask(rsi > 75, -0.5).mask(rsi < 25, 0.5)
    breakout = pd.Series(np.nan, index=df.index).mask(close > prior_high, 1.0).mask(close < prior_low, -1.0)

    parts = pd.DataFrame({"trend": trend, "momentum": momentum, "rsi": rsi_part, "breakout": breakout})
    score = parts.mean(axis=1, skipna=True)
    score[parts[["trend", "momentum", "rsi"]].isna().any(axis=1)] = np.nan
    score.iloc[: MIN_BARS_TECHNICAL - 1] = np.nan
    return pd.DataFrame({"score": score, "trend": trend, "momentum": momentum, "rsi": rsi,
                         "breakout": breakout, "atr": atr, "ema50": ema50, "ema200": ema200})


def technical_opinion(df: pd.DataFrame) -> Opinion | None:
    if len(df) < MIN_BARS_TECHNICAL:
        return None
    row = technical_scores(df).iloc[-1]
    if np.isnan(row["score"]):
        return None
    price = df["close"].iloc[-1]
    reasons = [
        f"price {'above' if price > row['ema200'] else 'below'} EMA200, "
        f"EMA50 {'above' if row['ema50'] > row['ema200'] else 'below'} EMA200",
        f"MACD histogram {'rising' if row['momentum'] > 0 else 'falling'} ({row['momentum']:+.2f})",
        f"RSI {row['rsi']:.0f}" + (" overbought" if row["rsi"] > 75 else " oversold" if row["rsi"] < 25 else ""),
    ]
    if row["breakout"] == 1:
        reasons.append("20-bar breakout to the upside")
    elif row["breakout"] == -1:
        reasons.append("20-bar breakdown")
    return Opinion("technical", row["score"], reasons,
                   {k: float(row[k]) for k in ("rsi", "atr", "ema50", "ema200")})


# -- analog forecast --------------------------------------------------------------

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


def _analog_at(F: np.ndarray, fwd: np.ndarray, ok: np.ndarray, t: int, horizon: int,
               neighbors: int, window: int) -> tuple[float, float, float, float, float, int] | None:
    """Analog vote for bar ``t`` using only bars whose forward return was
    known by ``t``'s close. Returns (score, prob_up, median, p10, p90, k)."""
    current = F[t]
    if np.isnan(current).any():
        return None
    hi = t - horizon + 1  # candidates i satisfy i + horizon <= t
    lo = max(0, hi - window)
    if hi - lo < max(neighbors, 100):
        return None
    cand = np.arange(lo, hi)[ok[lo:hi]]
    if len(cand) < max(neighbors, 100):
        return None
    X = F[cand]
    mean, std = X.mean(axis=0), X.std(axis=0, ddof=1)
    std[std == 0] = 1.0
    dist = (((X - mean) / std - (current - mean) / std) ** 2).sum(axis=1)
    k = min(neighbors, len(cand))
    nearest = fwd[cand[np.argpartition(dist, k - 1)[:k]]]
    prob_up = float((nearest > 0).mean())
    median = float(np.median(nearest))
    scale = float(current[3]) * np.sqrt(horizon) or 1e-9  # vol20
    score = 0.5 * (2 * prob_up - 1) + 0.5 * float(np.tanh(median / scale))
    return score, prob_up, median, float(np.quantile(nearest, 0.1)), float(np.quantile(nearest, 0.9)), k


def _analog_arrays(df: pd.DataFrame, horizon: int, features: pd.DataFrame | None):
    feats = analog_features(df) if features is None else features.iloc[: len(df)]
    F = feats.to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    fwd = np.full(len(df), np.nan)
    if len(df) > horizon:
        fwd[:-horizon] = np.log(close[horizon:] / close[:-horizon])
    ok = ~np.isnan(F).any(axis=1) & ~np.isnan(fwd)
    return F, fwd, ok


def forecast_scores(df: pd.DataFrame, horizon: int = 10, neighbors: int = 50, window: int = 8000,
                    start: int = MIN_BARS_FORECAST) -> pd.Series:
    """Analog forecast score for every bar from ``start`` on (NaN before)."""
    F, fwd, ok = _analog_arrays(df, horizon, None)
    out = np.full(len(df), np.nan)
    for t in range(max(start, MIN_BARS_FORECAST - 1), len(df)):
        res = _analog_at(F, fwd, ok, t, horizon, neighbors, window)
        if res:
            out[t] = res[0]
    return pd.Series(out, index=df.index, name="forecast")


def forecast_opinion(df: pd.DataFrame, horizon: int = 10, neighbors: int = 50,
                     features: pd.DataFrame | None = None, window: int = 8000) -> Opinion | None:
    """History-analog forecast: "the last N times the market looked like this,
    what happened over the next ``horizon`` bars?"

    Each bar is described by its recent returns, volatility, distance from
    trend and RSI. The ``neighbors`` past bars most similar to the last one
    (whose outcome is already known) vote with their forward returns. The
    score blends the share that went up with their median move scaled by
    volatility. It is a probability read of comparable history, not a
    prediction: when analogs disagree the score sits near zero.
    """
    if len(df) < MIN_BARS_FORECAST:
        return None
    F, fwd, ok = _analog_arrays(df, horizon, features)
    res = _analog_at(F, fwd, ok, len(df) - 1, horizon, neighbors, window)
    if res is None:
        return None
    score, prob_up, median, p10, p90, k = res
    pct = lambda x: f"{(np.exp(x) - 1) * 100:+.1f}%"  # noqa: E731
    return Opinion(
        "forecast", score,
        [f"{k} closest historical analogs: {prob_up:.0%} rose over the next {horizon} bars",
         f"median move {pct(median)}, 10th-90th percentile {pct(p10)} to {pct(p90)}"],
        {"prob_up": prob_up, "median_return": float(np.exp(median) - 1),
         "p10_return": float(np.exp(p10) - 1), "p90_return": float(np.exp(p90) - 1),
         "horizon": horizon, "analogs": k},
    )


# -- higher-timeframe trend filter ----------------------------------------------

def trend_up_series(htf: pd.DataFrame, ema_span: int = 200) -> pd.Series:
    """True where the higher-timeframe close is above its EMA and the EMA is
    rising over the last 5 bars. Uses EMA100 when history is too short for
    the requested span."""
    span = ema_span if len(htf) >= ema_span + 20 else min(100, ema_span)
    ema = ind.ema(htf["close"], span)
    up = (htf["close"] > ema) & (ema > ema.shift(5))
    return up.where(ema.shift(5).notna())


def align_to(htf_values: pd.Series, htf_step: timedelta, ltf_index: pd.Index, ltf_step: timedelta) -> pd.Series:
    """For each lower-timeframe bar, the latest higher-timeframe value whose
    bar had closed by the lower bar's close (no look-ahead)."""
    left = pd.DataFrame({"t": ltf_index + ltf_step}).reset_index(drop=True)
    right = pd.DataFrame({"t": htf_values.index + htf_step, "v": htf_values.to_numpy()})
    left["t"] = left["t"].astype("datetime64[ns, UTC]")
    right["t"] = right["t"].astype("datetime64[ns, UTC]")
    merged = pd.merge_asof(left, right.sort_values("t"), on="t", direction="backward")
    return pd.Series(merged["v"].to_numpy(), index=ltf_index)


def trend_opinion(htf: pd.DataFrame, ema_span: int = 200) -> tuple[bool | None, str]:
    if len(htf) < 120:
        return None, "not enough higher-timeframe history for the trend filter"
    up = trend_up_series(htf, ema_span).iloc[-1]
    if pd.isna(up):
        return None, "trend filter undefined"
    return bool(up), ("higher timeframe in an uptrend" if up else
                      "higher timeframe not in an uptrend: no new longs")


def realized_vol(df: pd.DataFrame, bars_per_year: float, lookback: int = 60) -> float:
    r = np.log(df["close"]).diff().tail(lookback)
    return float(r.std() * np.sqrt(bars_per_year)) if len(r) > 10 else float("nan")


def bars_per_year(interval: str, market: str) -> float:
    from autotrader.feeds import interval_delta

    step = interval_delta(interval)
    if market == "crypto":
        return timedelta(days=365) / step
    if step >= timedelta(days=1):
        return 252.0
    session = timedelta(hours=6, minutes=30) if market == "us" else timedelta(hours=6, minutes=15)
    return 252.0 * max(1.0, session / step)
