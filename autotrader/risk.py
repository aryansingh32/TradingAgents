"""Position sizing, protective stops and the daily loss limit."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

from autotrader.config import RiskConfig

# Regular sessions. Exchange holidays are not modelled: on a holiday the
# broker rejects or queues the order and the bot logs the error.
_SESSIONS = {
    "us": (ZoneInfo("America/New_York"), time(9, 30), time(16, 0)),
    "india": (ZoneInfo("Asia/Kolkata"), time(9, 15), time(15, 30)),
}


def market_open(market: str, now: datetime) -> bool:
    if market == "crypto":
        return True
    tz, start, end = _SESSIONS[market]
    local = now.astimezone(tz)
    return local.weekday() < 5 and start <= local.time() < end


@dataclass
class Protection:
    """The protective levels the bot keeps for an open long position."""

    entry: float
    stop: float
    take: float
    atr: float
    highest: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TradePlan:
    qty: float
    protection: Protection


def round_qty(qty: float, market: str) -> float:
    # Stocks trade in whole shares; crypto to 6 decimals.
    if market == "crypto":
        return math.floor(qty * 1e6) / 1e6
    return float(math.floor(qty))


class RiskManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg

    def levels(self, price: float, atr: float) -> Protection:
        stop_dist = self.cfg.atr_stop_mult * atr
        return Protection(entry=price, stop=price - stop_dist,
                          take=price + self.cfg.reward_risk * stop_dist, atr=atr, highest=price)

    def plan_entry(self, price: float, atr: float, equity: float, cash: float,
                   market: str) -> TradePlan | None:
        """Size a long entry so that hitting the stop loses ``risk_per_trade``
        of equity, capped by ``max_position_pct`` and available cash."""
        if price <= 0 or atr <= 0 or equity <= 0:
            return None
        stop_dist = self.cfg.atr_stop_mult * atr
        if stop_dist >= price:
            return None
        qty = equity * self.cfg.risk_per_trade / stop_dist
        qty = min(qty, equity * self.cfg.max_position_pct / price, max(cash, 0) * 0.99 / price)
        qty = round_qty(qty, market)
        if qty <= 0:
            return None
        return TradePlan(qty, self.levels(price, atr))

    def can_open(self, open_positions: int, day_start_equity: float, equity: float) -> tuple[bool, str]:
        if open_positions >= self.cfg.max_open_positions:
            return False, f"max open positions ({self.cfg.max_open_positions}) reached"
        if day_start_equity > 0 and equity <= day_start_equity * (1 - self.cfg.max_daily_loss_pct):
            return False, (f"daily loss limit hit ({(equity / day_start_equity - 1):.2%} today); "
                           "no new entries until tomorrow")
        return True, ""

    def update_trailing(self, prot: Protection, high: float) -> Protection:
        """Raise the stop as price makes new highs; never lower it."""
        prot.highest = max(prot.highest, high)
        if self.cfg.trail_after_atr > 0 and prot.highest - prot.entry >= self.cfg.trail_after_atr * prot.atr:
            prot.stop = max(prot.stop, prot.highest - self.cfg.atr_stop_mult * prot.atr)
        return prot

    @staticmethod
    def exit_reason(prot: Protection, low: float, high: float) -> tuple[str, float] | None:
        """Which protective level the bar touched, and the exit price. When a
        bar spans both, assume the stop filled first (the conservative read)."""
        if low <= prot.stop:
            return "stop_loss", prot.stop
        if high >= prot.take:
            return "take_profit", prot.take
        return None
