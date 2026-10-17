"""Smart DCA: the most-used retail bot, done with discipline.

Buys a fixed quote amount on a schedule and never sells. The amount grows
when the asset trades below its 200-day EMA (``dip_multiplier``) and when the
crypto Fear & Greed index shows extreme fear (``fear_multiplier``), and
shrinks in extreme greed (``greed_multiplier``), capped at ``max_multiplier``.
Buying more when cheap lowers the average cost versus plain DCA; it does not
avoid drawdowns, and it is meant for long-term holdings only.

Unlike martingale "safety orders", the amount never depends on the position
being underwater, so a falling market cannot make it double down without
limit.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime, timedelta

from autotrader import indicators as ind
from autotrader.config import AccumulateConfig, SymbolConfig
from autotrader.risk import round_qty

log = logging.getLogger(__name__)


def multiplier(acc: AccumulateConfig, price: float, ema200: float | None, fear_greed: int | None) -> tuple[float, list[str]]:
    mult, why = 1.0, []
    if ema200 and price < ema200:
        mult *= acc.dip_multiplier
        why.append(f"price {price / ema200 - 1:+.0%} vs 200-day EMA")
    if fear_greed is not None:
        if fear_greed <= 20:
            mult *= acc.fear_multiplier
            why.append(f"extreme fear ({fear_greed})")
        elif fear_greed >= 80:
            mult *= acc.greed_multiplier
            why.append(f"extreme greed ({fear_greed})")
    return min(mult, acc.max_multiplier), why


class Accumulator:
    def __init__(self, plans: list[AccumulateConfig], brokers: dict, md, state: dict, journal, fear_greed=None):
        self.plans, self.brokers, self.md, self.journal = plans, brokers, md, journal
        self.state = state.setdefault("dca", {})
        self.fear_greed = fear_greed  # callable -> int | None

    def step(self, now: datetime | None = None) -> None:
        now = now or datetime.now(UTC)
        for acc in self.plans:
            last = self.state.get(acc.symbol)
            if last and now - datetime.fromisoformat(last) < timedelta(hours=acc.every_hours):
                continue
            try:
                self._buy(acc, now)
            except Exception as exc:
                log.exception("accumulate %s failed", acc.symbol)
                self.journal("error", symbol=acc.symbol, error=f"accumulate: {exc!r}")

    def _buy(self, acc: AccumulateConfig, now: datetime) -> None:
        sym = SymbolConfig(symbol=acc.symbol, market=acc.market, broker=acc.broker, data_symbol=acc.data_symbol)
        daily = self.md.candles(acc.price_symbol, acc.market, "1d", lookback=timedelta(days=400))
        try:
            price = self.md.quote(acc.price_symbol, acc.market).price
        except Exception:
            price = float(daily["close"].iloc[-1])
        ema = ind.ema(daily["close"], 200).iloc[-1] if len(daily) >= 200 else None
        fg = None
        if acc.market == "crypto" and self.fear_greed is not None:
            try:
                fg = self.fear_greed()
            except Exception:
                fg = None
        mult, why = multiplier(acc, price, None if ema is None or math.isnan(ema) else float(ema), fg)
        qty = round_qty(acc.amount * mult / price, acc.market)
        broker = self.brokers[acc.broker]
        if qty <= 0:
            self.journal("accumulate_skipped", symbol=acc.symbol, reason="amount below one unit")
        else:
            broker.mark(sym, price)
            result = broker.buy(sym, qty, price, None)
            self.journal("accumulate", symbol=acc.symbol, qty=qty, price=price, amount=round(qty * price, 2),
                         multiplier=round(mult, 2), why=why, result=result)
        self.state[acc.symbol] = now.isoformat()
