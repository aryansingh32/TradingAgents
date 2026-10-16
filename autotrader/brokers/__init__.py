"""Broker adapters. ``build_broker`` turns a config section into an adapter."""

from __future__ import annotations

from pathlib import Path

from autotrader.brokers.base import Broker, BrokerError

BROKER_TYPES = ("paper", "alpaca", "openalgo", "freqtrade")


def build_broker(name: str, section: dict, data_dir: Path, live: bool) -> Broker:
    section = dict(section)
    kind = section.pop("type", name)
    if kind == "paper":
        from autotrader.brokers.paper import PaperBroker

        return PaperBroker(name, data_dir / f"paper_{name}.json", **section)
    if not live:
        # In paper mode every real broker is pointed at its sandbox, or refused
        # when it has none, so a paper config can never place a real order.
        if kind == "alpaca":
            from autotrader.brokers.alpaca import AlpacaBroker

            section.pop("live", None)
            return AlpacaBroker(name, live=False, **section)
        raise BrokerError(
            f"broker {name!r} ({kind}) has no paper endpoint; in paper mode use "
            "type 'paper' for it (or freqtrade's own dry_run / OpenAlgo's analyzer mode "
            "with mode 'live')."
        )
    if kind == "alpaca":
        from autotrader.brokers.alpaca import AlpacaBroker

        return AlpacaBroker(name, live=section.pop("live", True), **section)
    if kind == "openalgo":
        from autotrader.brokers.openalgo import OpenAlgoBroker

        return OpenAlgoBroker(name, **section)
    if kind == "freqtrade":
        from autotrader.brokers.freqtrade import FreqtradeBroker

        return FreqtradeBroker(name, **section)
    raise BrokerError(f"unknown broker type {kind!r}; expected one of {BROKER_TYPES}")


__all__ = ["Broker", "BrokerError", "build_broker", "BROKER_TYPES"]
