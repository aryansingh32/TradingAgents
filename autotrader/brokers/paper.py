"""Simulated account: fills at the current price with fees and slippage."""

from __future__ import annotations

import json
from pathlib import Path

from autotrader.brokers.base import Broker, BrokerError
from autotrader.config import SymbolConfig
from autotrader.risk import Protection


class PaperBroker(Broker):
    def __init__(self, name: str, state_file: Path, starting_cash: float = 100000.0,
                 fee_rate: float = 0.001, slippage: float = 0.0005):
        super().__init__(name)
        self.state_file = Path(state_file)
        self.fee_rate = fee_rate
        self.slippage = slippage
        try:
            self.state = json.loads(self.state_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self.state = {"cash": float(starting_cash), "positions": {}, "prices": {}}

    def _save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def mark(self, symbol: SymbolConfig, price: float) -> None:
        self.state["prices"][symbol.symbol] = price
        self._save()

    def cash(self) -> float:
        return self.state["cash"]

    def equity(self) -> float:
        value = sum(p["qty"] * self.state["prices"].get(s, p["avg_price"])
                    for s, p in self.state["positions"].items())
        return self.state["cash"] + value

    def position_qty(self, symbol: SymbolConfig) -> float:
        return self.state["positions"].get(symbol.symbol, {}).get("qty", 0.0)

    def buy(self, symbol: SymbolConfig, qty: float, price: float, protection: Protection) -> dict:
        fill = price * (1 + self.slippage)
        cost = qty * fill * (1 + self.fee_rate)
        if cost > self.state["cash"]:
            raise BrokerError(f"paper: not enough cash for {qty} {symbol.symbol} ({cost:.2f})")
        self.state["cash"] -= cost
        pos = self.state["positions"].setdefault(symbol.symbol, {"qty": 0.0, "avg_price": 0.0})
        pos["avg_price"] = (pos["avg_price"] * pos["qty"] + fill * qty) / (pos["qty"] + qty)
        pos["qty"] += qty
        self.state["prices"][symbol.symbol] = price
        self._save()
        return {"filled_qty": qty, "fill_price": fill}

    def close(self, symbol: SymbolConfig, price: float) -> dict:
        pos = self.state["positions"].pop(symbol.symbol, None)
        if not pos:
            return {"filled_qty": 0.0}
        fill = price * (1 - self.slippage)
        proceeds = pos["qty"] * fill * (1 - self.fee_rate)
        self.state["cash"] += proceeds
        self.state["prices"][symbol.symbol] = price
        self._save()
        return {"filled_qty": pos["qty"], "fill_price": fill,
                "pnl": proceeds - pos["qty"] * pos["avg_price"]}
