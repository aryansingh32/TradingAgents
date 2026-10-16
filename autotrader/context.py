"""Crypto positioning and sentiment context.

Price alone does not show how crowded a trade is. Three public gauges do:

* perpetual futures funding rate (Binance USDT-M): very high funding means
  leveraged longs are paying heavily to stay in, which often precedes
  long liquidations; negative funding means shorts are crowded;
* open interest change against price change: price up on fast-rising open
  interest is a leverage-driven move; falling price with falling open
  interest is deleveraging;
* the Crypto Fear & Greed Index (alternative.me).

These are used as a contrarian tilt at extremes and are neutral otherwise.
Research treats them as context for leverage and sentiment, not as standalone
predictors, so their default weight is small.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from autotrader.feeds import BinanceFeed, FeedError, HttpFeed
from autotrader.signals import Opinion

log = logging.getLogger(__name__)


class CryptoContext(HttpFeed):
    name = "crypto-context"
    ttl_seconds = 900

    def __init__(self, futures_url: str = "https://fapi.binance.com",
                 fng_url: str = "https://api.alternative.me/fng/", session=None):
        super().__init__(session)
        self.futures_url, self.fng_url = futures_url.rstrip("/"), fng_url
        self._cache: dict[str, tuple[float, Opinion | None]] = {}

    def _funding(self, pair: str) -> tuple[float, float]:
        """(latest funding rate, its percentile against the last ~30 days)."""
        hist = self._get(f"{self.futures_url}/fapi/v1/fundingRate", params={"symbol": pair, "limit": 90})
        rates = np.array([float(h["fundingRate"]) for h in hist])
        now = float(self._get(f"{self.futures_url}/fapi/v1/premiumIndex",
                              params={"symbol": pair})["lastFundingRate"])
        return now, float((rates < now).mean()) if len(rates) else 0.5

    def _oi_change(self, pair: str) -> float:
        rows = self._get(f"{self.futures_url}/futures/data/openInterestHist",
                         params={"symbol": pair, "period": "1h", "limit": 25})
        if len(rows) < 2:
            raise FeedError("open interest history too short")
        first, last = float(rows[0]["sumOpenInterest"]), float(rows[-1]["sumOpenInterest"])
        return last / first - 1

    def _fear_greed(self) -> int:
        return int(self._get(self.fng_url, params={"limit": 1})["data"][0]["value"])

    def opinion(self, symbol: str, price_change_24h: float | None) -> Opinion | None:
        hit = self._cache.get(symbol)
        if hit and time.monotonic() - hit[0] < self.ttl_seconds:
            return hit[1]
        op = self._compute(symbol, price_change_24h)
        self._cache[symbol] = (time.monotonic(), op)
        return op

    def _compute(self, symbol: str, price_change_24h: float | None) -> Opinion | None:
        pair = BinanceFeed.pair(symbol)
        parts, reasons, details = [], [], {}
        try:
            rate, pct = self._funding(pair)
            details.update(funding_rate=rate, funding_percentile=pct)
            apr = rate * 3 * 365
            if rate > 0.0003 and pct >= 0.9:
                parts.append(-0.6)
                reasons.append(f"funding {rate:.4%} per 8h ({apr:.0%} APR), top decile: crowded longs")
            elif rate < 0:
                parts.append(0.3)
                reasons.append(f"negative funding {rate:.4%}: shorts are paying, squeeze risk")
            else:
                parts.append(0.0)
                reasons.append(f"funding {rate:.4%} per 8h: neutral")
        except Exception as exc:
            log.info("funding unavailable for %s: %s", pair, exc)
        try:
            oi = self._oi_change(pair)
            details["oi_change_24h"] = oi
            if price_change_24h is not None:
                if price_change_24h > 0.02 and oi > 0.05:
                    parts.append(-0.3)
                    reasons.append(f"price +{price_change_24h:.1%} on open interest +{oi:.1%}: leverage-driven")
                elif price_change_24h < -0.02 and oi < -0.05:
                    parts.append(0.3)
                    reasons.append(f"price {price_change_24h:.1%} with open interest {oi:.1%}: deleveraging flush")
                else:
                    parts.append(0.0)
        except Exception as exc:
            log.info("open interest unavailable for %s: %s", pair, exc)
        try:
            fng = self._fear_greed()
            details["fear_greed"] = fng
            if fng <= 20:
                parts.append(0.5)
                reasons.append(f"Fear & Greed {fng}: extreme fear")
            elif fng >= 80:
                parts.append(-0.5)
                reasons.append(f"Fear & Greed {fng}: extreme greed")
            else:
                parts.append(0.0)
                reasons.append(f"Fear & Greed {fng}")
        except Exception as exc:
            log.info("fear & greed unavailable: %s", exc)
        if not parts:
            return None
        return Opinion("context", float(np.mean(parts)), reasons, details)
