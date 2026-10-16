"""Market monitor: one live table across crypto, US and Indian symbols with
the history (returns over several horizons, drawdown from the high,
volatility), the higher-timeframe trend, the analog forecast, positioning
context for crypto, and the bot's current read."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from autotrader.config import BotConfig, SymbolConfig
from autotrader.ensemble import combine
from autotrader.feeds import interval_delta
from autotrader.signals import (
    bars_per_year,
    forecast_opinion,
    realized_vol,
    technical_opinion,
    trend_opinion,
)


def _ret(close: pd.Series, delta: timedelta) -> float | None:
    past = close[close.index <= close.index[-1] - delta]
    return float(close.iloc[-1] / past.iloc[-1] - 1) if len(past) else None


def snapshot(sym: SymbolConfig, df: pd.DataFrame, htf: pd.DataFrame | None, cfg: BotConfig,
             price: float | None = None, context=None) -> dict:
    close = df["close"]
    sc = cfg.signals
    tech = technical_opinion(df)
    fc = forecast_opinion(df, sc.forecast_horizon, sc.forecast_neighbors, window=sc.forecast_window)
    ctx = None
    if context is not None and sym.market == "crypto":
        try:
            ctx = context.opinion(sym.price_symbol, _ret(close, timedelta(hours=23)))
        except Exception:
            ctx = None
    trend_up, note = trend_opinion(htf, sc.trend_ema) if htf is not None else (None, "")
    decision = combine([tech, fc, ctx], sc, trend_up, note)
    step = interval_delta(cfg.interval)
    return {
        "symbol": sym.symbol,
        "price": price if price is not None else float(close.iloc[-1]),
        "last_candle": (df.index[-1] + step).isoformat(),
        "1d": _ret(close, timedelta(days=1)), "1w": _ret(close, timedelta(days=7)),
        "1m": _ret(close, timedelta(days=30)), "1y": _ret(close, timedelta(days=365)),
        "from_high": float(close.iloc[-1] / close.max() - 1),
        "vol_annual": realized_vol(df, bars_per_year(cfg.interval, sym.market)),
        "trend_up": trend_up,
        "technical": tech.score if tech else None,
        "forecast_prob_up": fc.details["prob_up"] if fc else None,
        "forecast_median": fc.details["median_return"] if fc else None,
        "context": ctx.score if ctx else None,
        "score": decision.score,
        "action": decision.action,
    }


def render(rows: list[dict], cfg: BotConfig) -> None:
    from rich.console import Console
    from rich.table import Table

    def pct(v):
        if v is None:
            return "-"
        color = "green" if v > 0 else "red" if v < 0 else "white"
        return f"[{color}]{v:+.1%}[/{color}]"

    h = cfg.signals.forecast_horizon
    table = Table(title=f"AutoTrader market monitor ({cfg.interval} candles)", header_style="bold")
    for col in ("Symbol", "Price", "1D", "1W", "1M", "1Y", "From high", "Vol/yr", "Trend",
                f"P(up {h})", "Median", "Context", "Score", "Action"):
        table.add_column(col, justify="left" if col == "Symbol" else "right", no_wrap=True)
    for r in rows:
        if "error" in r:
            table.add_row(r["symbol"], *["-"] * 12, f"[red]{r['error'][:40]}[/red]")
            continue
        action_color = {"BUY": "green", "SELL": "red"}.get(r["action"], "yellow")
        trend = {True: "[green]up[/green]", False: "[red]down[/red]"}.get(r["trend_up"], "-")
        table.add_row(
            r["symbol"], f"{r['price']:,.2f}", pct(r["1d"]), pct(r["1w"]), pct(r["1m"]), pct(r["1y"]),
            pct(r["from_high"]), f"{r['vol_annual']:.0%}" if r["vol_annual"] == r["vol_annual"] else "-",
            trend, f"{r['forecast_prob_up']:.0%}" if r["forecast_prob_up"] is not None else "-",
            pct(r["forecast_median"]), f"{r['context']:+.2f}" if r["context"] is not None else "-",
            f"{r['score']:+.2f}", f"[{action_color}]{r['action']}[/{action_color}]",
        )
    Console(width=max(160, Console().width)).print(table)
