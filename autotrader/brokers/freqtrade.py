"""Crypto exchanges through a running freqtrade bot's REST API.

freqtrade connects to Binance, Kraken, Bybit, OKX, Kucoin, Hyperliquid and
more, and keeps its own stop loss (``stoploss``, ``trailing_stop``,
optionally ``stoploss_on_exchange``), so positions stay protected while this
bot is offline. Enable ``api_server`` and ``force_entry_enable`` in the
freqtrade config; credentials come from ``FREQTRADE_USERNAME`` and
``FREQTRADE_PASSWORD``. Symbols use freqtrade pairs, e.g. ``BTC/USDT``.
"""

from __future__ import annotations

import os

from autotrader.brokers.base import BrokerError, HttpBroker
from autotrader.config import SymbolConfig
from autotrader.risk import Protection


class FreqtradeBroker(HttpBroker):
    native_stops = True

    def __init__(self, name: str, url: str = "http://127.0.0.1:8080", user_env: str = "FREQTRADE_USERNAME",
                 password_env: str = "FREQTRADE_PASSWORD", session=None):
        super().__init__(name, url, session)
        user, password = os.environ.get(user_env), os.environ.get(password_env)
        if not user or not password:
            raise BrokerError(f"freqtrade: set {user_env} and {password_env}")
        self.session.auth = (user, password)

    def _open_trades(self, pair: str) -> list[dict]:
        return [t for t in self._request("GET", "/api/v1/status") if t.get("pair") == pair]

    def equity(self) -> float:
        return float(self._request("GET", "/api/v1/balance")["total"])

    def cash(self) -> float:
        bal = self._request("GET", "/api/v1/balance")
        stake = bal.get("stake")
        free = next((c.get("free", 0) for c in bal.get("currencies", []) if c.get("currency") == stake), None)
        return float(free if free is not None else bal["total"])

    def position_qty(self, symbol: SymbolConfig) -> float:
        return sum(float(t.get("amount", 0)) for t in self._open_trades(symbol.symbol))

    def buy(self, symbol: SymbolConfig, qty: float, price: float, protection: Protection | None) -> dict:
        return self._request("POST", "/api/v1/forceenter", json={
            "pair": symbol.symbol, "side": "long", "stakeamount": round(qty * price, 2),
            "entry_tag": "autotrader",
        })

    def close(self, symbol: SymbolConfig, price: float) -> dict:
        results = [self._request("POST", "/api/v1/forceexit", json={"tradeid": str(t["trade_id"])})
                   for t in self._open_trades(symbol.symbol)]
        return {"exits": results}

    def reduce(self, symbol: SymbolConfig, qty: float, price: float, new_stop: float) -> dict:
        trades = self._open_trades(symbol.symbol)
        if not trades:
            raise BrokerError(f"freqtrade: no open trade for {symbol.symbol}")
        return self._request("POST", "/api/v1/forceexit", json={
            "tradeid": str(trades[0]["trade_id"]), "amount": qty})
