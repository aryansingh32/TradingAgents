"""Market monitor: one table across crypto, US and Indian symbols with the
history (returns over several horizons, drawdown from the high, volatility,
trend regime) and the bot's current read and analog forecast."""

from __future__ import annotations

import numpy as np
import pandas as pd

from autotrader.config import BotConfig
from autotrader.ensemble import combine
from autotrader.signals import forecast_opinion, technical_opinion


def _ret(close: pd.Series, bars: int) -> float | None:
    return float(close.iloc[-1] / close.iloc[-bars - 1] - 1) if len(close) > bars else None


def snapshot(symbol: str, df: pd.DataFrame, cfg: BotConfig) -> dict:
    close = df["close"]
    ppy = 365 if cfg.symbols and any(s.market == "crypto" and s.symbol == symbol for s in cfg.symbols) else 252
    tech = technical_opinion(df)
    fc = forecast_opinion(df, cfg.signals.forecast_horizon, cfg.signals.forecast_neighbors)
    decision = combine([tech, fc], cfg.signals)
    vol = float(np.log(close).diff().tail(60).std() * np.sqrt(ppy)) if len(close) > 60 else None
    return {
        "symbol": symbol,
        "price": float(close.iloc[-1]),
        "1d": _ret(close, 1), "1w": _ret(close, 5), "1m": _ret(close, 21), "1y": _ret(close, 252),
        "from_high": float(close.iloc[-1] / close.max() - 1),
        "vol_annual": vol,
        "technical": tech.score if tech else None,
        "forecast_prob_up": fc.details["prob_up"] if fc else None,
        "forecast_median": fc.details["median_return"] if fc else None,
        "score": decision.score,
        "action": decision.action,
    }


def render(rows: list[dict], horizon: int) -> None:
    from rich.console import Console
    from rich.table import Table

    def pct(v):
        if v is None:
            return "-"
        color = "green" if v > 0 else "red" if v < 0 else "white"
        return f"[{color}]{v:+.1%}[/{color}]"

    table = Table(title="AutoTrader market monitor", header_style="bold")
    for col in ("Symbol", "Price", "1D", "1W", "1M", "1Y", "From high", "Vol (yr)",
                f"P(up {horizon} bars)", "Median fcst", "Score", "Action"):
        table.add_column(col, justify="right" if col != "Symbol" else "left")
    for r in rows:
        if "error" in r:
            table.add_row(r["symbol"], *["-"] * 10, f"[red]{r['error'][:30]}[/red]")
            continue
        action_color = {"BUY": "green", "SELL": "red"}.get(r["action"], "yellow")
        table.add_row(
            r["symbol"], f"{r['price']:,.2f}", pct(r["1d"]), pct(r["1w"]), pct(r["1m"]), pct(r["1y"]),
            pct(r["from_high"]), f"{r['vol_annual']:.0%}" if r["vol_annual"] else "-",
            f"{r['forecast_prob_up']:.0%}" if r["forecast_prob_up"] is not None else "-",
            pct(r["forecast_median"]), f"{r['score']:+.2f}", f"[{action_color}]{r['action']}[/{action_color}]",
        )
    Console().print(table)
