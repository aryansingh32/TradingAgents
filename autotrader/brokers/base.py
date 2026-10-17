"""Broker interface the engine trades through."""

from __future__ import annotations

from abc import ABC, abstractmethod

import requests

from autotrader.config import SymbolConfig
from autotrader.risk import Protection


class BrokerError(RuntimeError):
    pass


class Broker(ABC):
    #: True when the broker itself holds the stop loss / take profit orders,
    #: so they fire even while the bot is offline. The bot also enforces its
    #: own (trailing) levels on every loop either way.
    native_stops = False

    def __init__(self, name: str):
        self.name = name

    def mark(self, symbol: SymbolConfig, price: float) -> None:  # noqa: B027
        """Record the latest price (used by brokers that value positions locally)."""

    @abstractmethod
    def equity(self) -> float: ...

    @abstractmethod
    def cash(self) -> float: ...

    @abstractmethod
    def position_qty(self, symbol: SymbolConfig) -> float: ...

    @abstractmethod
    def buy(self, symbol: SymbolConfig, qty: float, price: float, protection: Protection | None) -> dict: ...

    @abstractmethod
    def close(self, symbol: SymbolConfig, price: float) -> dict: ...

    def reduce(self, symbol: SymbolConfig, qty: float, price: float, new_stop: float) -> dict:
        """Sell part of a position (scale out), keeping ``new_stop`` on the rest."""
        raise NotImplementedError(f"{self.name}: partial exits are not supported")


class HttpBroker(Broker):
    timeout = 20

    def __init__(self, name: str, base_url: str, session: requests.Session | None = None):
        super().__init__(name)
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()

    def _request(self, method: str, path: str, **kwargs):
        resp = self.session.request(method, f"{self.base_url}{path}", timeout=self.timeout, **kwargs)
        if resp.status_code >= 400:
            raise BrokerError(f"{self.name} {method} {path} -> {resp.status_code}: {resp.text[:300]}")
        return resp.json() if resp.content else {}
