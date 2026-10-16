"""Indian brokers (Zerodha, Angel One, Upstox, Dhan, Fyers, ...) through a
self-hosted OpenAlgo server (https://github.com/marketcalls/openalgo).

OpenAlgo gives one REST API in front of 30+ Indian brokers, so the bot does
not need a separate adapter per broker. Run OpenAlgo, log in to your broker
there (SEBI rules require a daily OAuth/2FA login and a static IP registered
with the broker for API orders), and put its API key in ``OPENALGO_API_KEY``.

Stops are enforced by the bot on every loop (``native_stops = False``), so
keep ``loop_seconds`` short during market hours.
"""

from __future__ import annotations

import os

from autotrader.brokers.base import BrokerError, HttpBroker
from autotrader.config import SymbolConfig
from autotrader.risk import Protection


class OpenAlgoBroker(HttpBroker):
    def __init__(self, name: str, host: str = "http://127.0.0.1:5000", key_env: str = "OPENALGO_API_KEY",
                 strategy: str = "AutoTrader", session=None):
        super().__init__(name, host, session)
        self.api_key = os.environ.get(key_env)
        if not self.api_key:
            raise BrokerError(f"openalgo: set {key_env}")
        self.strategy = strategy

    def _post(self, path: str, payload: dict) -> dict:
        data = self._request("POST", f"/api/v1/{path}",
                             json={"apikey": self.api_key, "strategy": self.strategy, **payload})
        if data.get("status") not in (None, "success"):
            raise BrokerError(f"openalgo {path}: {data}")
        return data

    @staticmethod
    def _where(symbol: SymbolConfig) -> dict:
        # CNC = delivery (hold overnight); MIS = intraday, squared off by the broker.
        return {"symbol": symbol.symbol, "exchange": symbol.exchange or "NSE",
                "product": symbol.product or "CNC"}

    def _funds(self) -> dict:
        return self._post("funds", {}).get("data", {})

    def cash(self) -> float:
        return float(self._funds().get("availablecash", 0))

    def equity(self) -> float:
        # Cash plus the market value of delivery holdings.
        holdings = self._post("holdings", {}).get("data", {}).get("statistics", {})
        return self.cash() + float(holdings.get("totalholdingvalue", 0) or 0)

    def position_qty(self, symbol: SymbolConfig) -> float:
        data = self._post("openposition", self._where(symbol))
        return float(data.get("quantity", 0) or 0)

    def buy(self, symbol: SymbolConfig, qty: float, price: float, protection: Protection) -> dict:
        return self._post("placeorder", {**self._where(symbol), "action": "BUY",
                                         "pricetype": "MARKET", "quantity": str(int(qty))})

    def close(self, symbol: SymbolConfig, price: float) -> dict:
        qty = self.position_qty(symbol)
        if qty <= 0:
            return {"filled_qty": 0}
        return self._post("placeorder", {**self._where(symbol), "action": "SELL",
                                         "pricetype": "MARKET", "quantity": str(int(qty))})
