"""Alpaca (US stocks and ETFs) via its REST API.

Entries are bracket orders, so the stop loss and take profit live at Alpaca
and fire even when the bot is offline. Keys come from ``APCA_API_KEY_ID`` and
``APCA_API_SECRET_KEY`` (or the env var names given in the config). Paper
endpoint by default.
"""

from __future__ import annotations

import os

from autotrader.brokers.base import BrokerError, HttpBroker
from autotrader.config import SymbolConfig
from autotrader.risk import Protection

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"


class AlpacaBroker(HttpBroker):
    native_stops = True

    def __init__(self, name: str, live: bool = False, key_env: str = "APCA_API_KEY_ID",
                 secret_env: str = "APCA_API_SECRET_KEY", base_url: str | None = None, session=None):
        super().__init__(name, base_url or (LIVE_URL if live else PAPER_URL), session)
        key, secret = os.environ.get(key_env), os.environ.get(secret_env)
        if not key or not secret:
            raise BrokerError(f"alpaca: set {key_env} and {secret_env}")
        self.session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})

    def _account(self) -> dict:
        return self._request("GET", "/v2/account")

    def equity(self) -> float:
        return float(self._account()["equity"])

    def cash(self) -> float:
        return float(self._account()["buying_power"])

    def position_qty(self, symbol: SymbolConfig) -> float:
        try:
            return float(self._request("GET", f"/v2/positions/{symbol.symbol}")["qty"])
        except BrokerError as exc:
            if "-> 404" in str(exc):
                return 0.0
            raise

    def buy(self, symbol: SymbolConfig, qty: float, price: float, protection: Protection) -> dict:
        order = {
            "symbol": symbol.symbol, "qty": str(int(qty)), "side": "buy", "type": "market",
            # GTC so the protective legs survive overnight.
            "time_in_force": "gtc", "order_class": "bracket",
            "take_profit": {"limit_price": f"{protection.take:.2f}"},
            "stop_loss": {"stop_price": f"{protection.stop:.2f}"},
        }
        return self._request("POST", "/v2/orders", json=order)

    def close(self, symbol: SymbolConfig, price: float) -> dict:
        # Cancel the open bracket legs first; they hold the shares.
        for order in self._request("GET", "/v2/orders", params={"status": "open", "symbols": symbol.symbol}):
            self._request("DELETE", f"/v2/orders/{order['id']}")
        return self._request("DELETE", f"/v2/positions/{symbol.symbol}")
