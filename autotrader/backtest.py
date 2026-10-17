"""Fast walk-forward backtest of the bot's price-history rules.

``prepare`` computes every opinion for every bar once (the expensive part);
``simulate`` then replays the trading rules over those arrays in
milliseconds, so the optimizer can try hundreds of parameter sets.

Fidelity rules:

* signals use closed candles only, with the same code as the live bot;
* an entry decided at a bar's close fills at the next bar's open;
* stops and take profits fill when a later bar's range touches them (a gap
  through the level fills at the open; the stop first if a bar touches both);
* costs per market (fees, taxes, spread) are charged on both sides;
* the higher-timeframe trend filter only sees higher-timeframe bars that had
  closed by the decision time;
* the protections (cooldown, stop-loss guard, drawdown and losing-streak
  guards) and the daily loss limit run exactly as live.

The LLM debate and the crypto context are not replayed (no reliable history
for them), so a backtest measures the technical + forecast + trend core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

from autotrader.config import RiskConfig, SessionConfig, SignalConfig, TradeManagementConfig
from autotrader.ensemble import weights_of
from autotrader.feeds import interval_delta
from autotrader.protections import Protections
from autotrader.risk import RiskManager, entry_window_open, round_qty, vol_multiplier
from autotrader.signals import (
    MIN_BARS_TECHNICAL,
    align_to,
    bars_per_year,
    forecast_scores,
    technical_scores,
    trend_up_series,
)
from autotrader.signals_extra import EXTRA


@dataclass
class Prepared:
    df: pd.DataFrame
    interval: str
    market: str
    scores: dict[str, np.ndarray]  # opinion name -> per-bar score (nan = no opinion)
    trend_up: np.ndarray  # 1.0 up, 0.0 not up, nan unknown
    atr: np.ndarray
    vol: np.ndarray

    @property
    def technical(self) -> np.ndarray:
        return self.scores["technical"]

    @property
    def forecast(self) -> np.ndarray:
        return self.scores["forecast"]


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def resample_ohlcv(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    rule = {"4h": "4h", "1d": "1D"}[interval]
    out = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return out.dropna(subset=["close"])


def prepare(df: pd.DataFrame, sig: SignalConfig, interval: str = "1d", market: str = "crypto",
            htf: pd.DataFrame | None = None) -> Prepared:
    """Compute all per-bar opinions once. ``htf`` is the higher-timeframe
    history for the trend filter; when omitted it is resampled from ``df``."""
    tech = technical_scores(df)
    fc = (forecast_scores(df, sig.forecast_horizon, sig.forecast_neighbors, sig.forecast_window)
          if sig.forecast_weight > 0 else pd.Series(np.nan, index=df.index))
    scores = {"technical": tech["score"].to_numpy(), "forecast": fc.to_numpy()}
    for name, fn in EXTRA.items():
        scores[name] = fn(df).to_numpy(dtype=float)
    step = interval_delta(interval)
    htf_step = interval_delta(sig.trend_interval)
    if htf is None:
        htf = df if htf_step == step else resample_ohlcv(df, sig.trend_interval)
    trend = align_to(trend_up_series(htf, sig.trend_ema).astype(float), htf_step, df.index, step)
    bpy = bars_per_year(interval, market)
    vol = np.log(df["close"]).diff().rolling(60).std() * np.sqrt(bpy)
    return Prepared(df, interval, market, scores, trend.to_numpy(dtype=float),
                    tech["atr"].to_numpy(), vol.to_numpy())


def combined_score(p: Prepared, sig: SignalConfig) -> np.ndarray:
    """Weighted mean of the opinions present at each bar (as ``combine``)."""
    weights = weights_of(sig)
    n = len(p.df)
    num, den = np.zeros(n), np.zeros(n)
    for name, arr in p.scores.items():
        w = weights.get(name, 0.0)
        if w <= 0:
            continue
        ok = ~np.isnan(arr)
        num += np.where(ok, w * np.nan_to_num(arr), 0.0)
        den += np.where(ok, w, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def simulate(p: Prepared, risk_cfg: RiskConfig, sig: SignalConfig, cost: float = 0.001,
             starting_cash: float = 100000.0, start: int | None = None, end: int | None = None,
             session: SessionConfig | None = None,
             trade_cfg: TradeManagementConfig | None = None) -> BacktestResult:
    df = p.df
    trade_cfg = trade_cfg or TradeManagementConfig()
    session = session or SessionConfig()
    rm = RiskManager(risk_cfg)
    guards = Protections(risk_cfg)
    step = interval_delta(p.interval)
    intraday_stock = p.market != "crypto" and step < timedelta(days=1)
    idx = df.index
    opens, highs, lows, closes = (df[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close"))
    first = max(start if start is not None else 0, MIN_BARS_TECHNICAL)
    last = len(df) if end is None else min(end, len(df))

    score = combined_score(p, sig)
    trend_block = (p.trend_up == 0.0) if sig.trend_filter else np.zeros(len(df), bool)
    want_buy = (score >= sig.buy_threshold) & ~trend_block
    if sig.mean_reversion_weight > 0 and "mean_reversion" in p.scores:
        mr = np.nan_to_num(p.scores["mean_reversion"], nan=-1.0)
        want_buy |= (mr >= sig.mean_reversion_trigger) & (p.trend_up == 1.0)
    want_buy = want_buy.tolist()
    want_sell = (score <= sig.sell_threshold).tolist()
    opens, highs, lows, closes_l = opens.tolist(), highs.tolist(), lows.tolist(), closes.tolist()

    cash, qty, prot, entry_i, entry_cost = starting_cash, 0.0, None, -1, 0.0
    qty0, realized = 0.0, 0.0  # initial size, proceeds of partial exits
    pending_buy = pending_sell = False
    trades: list[dict] = []
    curve = np.empty(max(last - first, 0))
    day, day_start = None, starting_cash

    def close_trade(price: float, i: int, reason: str):
        nonlocal cash, qty, prot
        last_leg = qty * price * (1 - cost)
        cash += last_leg
        proceeds = realized + last_leg
        pnl = proceeds - entry_cost
        r_mult = pnl / (qty0 * prot.risk_per_unit)
        trades.append({"entry_time": idx[entry_i], "exit_time": idx[i], "entry": prot.entry, "exit": price,
                       "qty": qty0, "pnl": pnl, "return": proceeds / entry_cost - 1, "r": r_mult,
                       "reason": reason, "bars": i - entry_i, "partial": prot.partial_done})
        guards.record_exit("s", idx[i].to_pydatetime(), pnl, reason)
        qty, prot = 0.0, None

    days = (idx.as_unit("s").asi8 // 86_400).tolist()  # UTC day number per bar
    for i in range(first, last):
        d = days[i]
        if d != day:
            day, day_start = d, cash + qty * opens[i]

        if pending_sell and qty > 0:
            close_trade(opens[i], i, "signal_sell")
        if pending_buy and qty == 0:
            fill_time = idx[i].to_pydatetime()
            equity = cash
            ok, _ = rm.can_open(0, day_start, equity)
            if ok:
                ok, _ = guards.allows_entry("s", fill_time, equity, step)
            if ok and intraday_stock:
                ok = entry_window_open(p.market, fill_time, session)
            if ok:
                mult = guards.risk_multiplier(equity) * vol_multiplier(risk_cfg, p.vol[i - 1])
                plan = rm.plan_entry(opens[i], p.atr[i - 1], equity, cash / (1 + cost), p.market, mult)
                if plan:
                    qty, prot, entry_i = plan.qty, plan.protection, i
                    qty0, realized = qty, 0.0
                    entry_cost = qty * opens[i] * (1 + cost)
                    cash -= entry_cost
        pending_buy = pending_sell = False

        if qty > 0:
            hit = rm.exit_reason(prot, lows[i], highs[i])
            if hit:
                # A gap through the level fills at the open, not the level.
                fill = max(opens[i], hit[1]) if hit[0] == "take_profit" else min(opens[i], hit[1])
                close_trade(fill, i, hit[0])
            else:
                level = rm.partial_level(prot, trade_cfg.partial_tp_r)
                if level is not None and highs[i] >= level:
                    part = round_qty(qty * trade_cfg.partial_tp_fraction, p.market)
                    if 0 < part < qty:
                        fill = max(opens[i], level)
                        realized += part * fill * (1 - cost)
                        cash += part * fill * (1 - cost)
                        qty -= part
                    prot.partial_done = True
                    if trade_cfg.breakeven_after_partial:
                        prot.stop = max(prot.stop, prot.entry * (1 + 2 * cost))
                rm.update_trailing(prot, highs[i])

        if qty > 0 and want_sell[i]:
            pending_sell = True
        elif qty == 0 and want_buy[i]:
            pending_buy = True
        equity_now = cash + qty * closes_l[i]
        guards.record_equity(equity_now)
        curve[i - first] = equity_now

    if qty > 0:
        close_trade(closes_l[last - 1], last - 1, "end_of_test")
        curve[-1] = cash

    equity = pd.Series(curve, index=idx[first:last], name="equity")
    return BacktestResult(equity, trades,
                          stats(equity, trades, closes[first:last], p.interval, p.market, starting_cash))


def stats(equity: pd.Series, trades: list[dict], closes: np.ndarray, interval: str, market: str,
          start_cash: float) -> dict:
    if equity.empty:
        return {}
    bpy = bars_per_year(interval, market)
    rets = equity.pct_change().dropna()
    years = max(len(equity) / bpy, 1e-9)
    total = float(equity.iloc[-1] / start_cash - 1)
    rs = np.array([t["r"] for t in trades]) if trades else np.array([])
    wins, losses = rs[rs > 0], rs[rs <= 0]
    streak = longest = 0
    for r in rs:
        streak = streak + 1 if r <= 0 else 0
        longest = max(longest, streak)
    gross_win = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = -sum(t["pnl"] for t in trades if t["pnl"] <= 0)
    expectancy = float(rs.mean()) if len(rs) else 0.0
    sd = float(rs.std(ddof=1)) if len(rs) > 1 else 0.0
    return {
        "total_return": total,
        "cagr": float((1 + total) ** (1 / years) - 1) if total > -1 else -1.0,
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "sharpe": float(rets.mean() / rets.std() * np.sqrt(bpy)) if rets.std() > 0 else 0.0,
        "trades": len(trades),
        "win_rate": float(len(wins) / len(rs)) if len(rs) else 0.0,
        "avg_win_r": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_r": float(losses.mean()) if len(losses) else 0.0,
        "expectancy_r": expectancy,
        # Van Tharp's System Quality Number: mean R / stdev R * sqrt(trades).
        "sqn": expectancy / sd * float(np.sqrt(len(rs))) if sd > 0 else 0.0,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win else 0.0),
        "max_losing_streak": int(longest),
        "exposure": float(sum(t["bars"] for t in trades) / max(len(equity), 1)),
        "buy_and_hold_return": float(closes[-1] / closes[0] - 1) if len(closes) else 0.0,
    }


def run_backtest(df: pd.DataFrame, risk_cfg: RiskConfig | None = None, sig_cfg: SignalConfig | None = None,
                 market: str = "crypto", interval: str = "1d", cost: float = 0.001,
                 starting_cash: float = 100000.0, htf: pd.DataFrame | None = None,
                 session: SessionConfig | None = None,
                 trade_cfg: TradeManagementConfig | None = None) -> BacktestResult:
    sig_cfg = sig_cfg or SignalConfig()
    p = prepare(df, sig_cfg, interval, market, htf)
    return simulate(p, risk_cfg or RiskConfig(), sig_cfg, cost, starting_cash, session=session,
                    trade_cfg=trade_cfg)
