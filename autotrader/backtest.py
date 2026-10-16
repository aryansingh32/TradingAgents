"""Walk-forward backtest of the price-history opinions with the bot's risk rules.

At each bar the opinions see only bars up to its close; an entry fills at the
next bar's open, and stops / take profits fill when a later bar's range
touches them (the stop first when a bar touches both). The LLM debate is not
replayed (it would cost one multi-minute LLM run per bar), so a backtest
measures the technical + forecast part of the bot; use TradingAgents' own
``tradingagents.backtest`` to evaluate the debate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from autotrader import indicators as ind
from autotrader.config import RiskConfig, SignalConfig
from autotrader.ensemble import BUY, SELL, combine
from autotrader.risk import RiskManager
from autotrader.signals import (
    MIN_BARS_FORECAST,
    analog_features,
    forecast_opinion,
    technical_opinion,
)


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def _periods_per_year(index: pd.Index, market: str) -> float:
    if len(index) < 2:
        return 252.0
    step = pd.Series(index).diff().median()
    if step >= pd.Timedelta(days=1):
        return 365.0 if market == "crypto" else 252.0
    return pd.Timedelta(days=365 if market == "crypto" else 252) / step


def run_backtest(df: pd.DataFrame, risk_cfg: RiskConfig | None = None, sig_cfg: SignalConfig | None = None,
                 market: str = "crypto", starting_cash: float = 100000.0, fee_rate: float = 0.001,
                 start_bar: int | None = None) -> BacktestResult:
    risk_cfg = risk_cfg or RiskConfig()
    sig_cfg = sig_cfg or SignalConfig()
    rm = RiskManager(risk_cfg)
    feats = analog_features(df) if sig_cfg.forecast_weight > 0 else None
    atr = ind.atr(df)
    opens, highs, lows, closes = (df[c].to_numpy() for c in ("open", "high", "low", "close"))

    cash, qty, prot, entry_bar = starting_cash, 0.0, None, None
    pending_buy = pending_sell = False
    trades: list[dict] = []
    curve: list[float] = []
    first = start_bar if start_bar is not None else MIN_BARS_FORECAST
    day_start_equity, current_day = starting_cash, None

    def sell(price: float, i: int, reason: str):
        nonlocal cash, qty, prot
        proceeds = qty * price * (1 - fee_rate)
        cost = qty * prot.entry * (1 + fee_rate)
        cash += proceeds
        trades.append({"entry_time": df.index[entry_bar], "exit_time": df.index[i], "entry": prot.entry,
                       "exit": price, "qty": qty, "pnl": proceeds - cost,
                       "return": proceeds / cost - 1, "reason": reason})
        qty, prot = 0.0, None

    for i in range(first, len(df)):
        day = df.index[i].date() if hasattr(df.index[i], "date") else i
        if day != current_day:
            current_day, day_start_equity = day, cash + qty * opens[i]

        # Orders decided at the previous close fill at this bar's open.
        if pending_sell and qty > 0:
            sell(opens[i], i, "signal_sell")
        if pending_buy and qty == 0:
            equity = cash
            ok, _ = rm.can_open(0, day_start_equity, equity)
            plan = rm.plan_entry(opens[i], float(atr.iloc[i - 1]), equity, cash, market) if ok else None
            if plan:
                qty, prot, entry_bar = plan.qty, plan.protection, i
                cash -= qty * opens[i] * (1 + fee_rate)
        pending_buy = pending_sell = False

        if qty > 0:
            hit = rm.exit_reason(prot, lows[i], highs[i])
            if hit:
                # A gap through the level fills at the open, not the level.
                level = hit[1]
                fill = min(opens[i], level) if hit[0] == "stop_loss" else max(opens[i], level)
                sell(fill, i, hit[0])
            else:
                rm.update_trailing(prot, highs[i])

        window = df.iloc[: i + 1]
        opinions = [
            technical_opinion(window) if sig_cfg.technical_weight > 0 else None,
            forecast_opinion(window, sig_cfg.forecast_horizon, sig_cfg.forecast_neighbors, feats)
            if feats is not None else None,
        ]
        decision = combine(opinions, sig_cfg)
        if qty > 0 and decision.action == SELL:
            pending_sell = True
        elif qty == 0 and decision.action == BUY:
            pending_buy = True
        curve.append(cash + qty * closes[i])

    if qty > 0:
        sell(closes[-1], len(df) - 1, "end_of_test")
        curve[-1] = cash

    equity = pd.Series(curve, index=df.index[first:], name="equity")
    return BacktestResult(equity, trades, _stats(equity, trades, closes[first:], market, starting_cash))


def _stats(equity: pd.Series, trades: list[dict], closes: np.ndarray, market: str, start: float) -> dict:
    if equity.empty:
        return {}
    rets = equity.pct_change().dropna()
    ppy = _periods_per_year(equity.index, market)
    years = max(len(equity) / ppy, 1e-9)
    total = float(equity.iloc[-1] / start - 1)
    drawdown = (equity / equity.cummax() - 1).min()
    wins = [t for t in trades if t["pnl"] > 0]
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = -sum(t["pnl"] for t in trades if t["pnl"] <= 0)
    return {
        "total_return": total,
        "cagr": (1 + total) ** (1 / years) - 1 if total > -1 else -1.0,
        "max_drawdown": float(drawdown),
        "sharpe": float(rets.mean() / rets.std() * np.sqrt(ppy)) if rets.std() > 0 else 0.0,
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win else 0.0,
        "exposure": float((equity.diff().fillna(0) != 0).mean()),
        "buy_and_hold_return": float(closes[-1] / closes[0] - 1),
    }
