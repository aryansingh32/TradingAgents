"""The TradingAgents multi-agent debate as one of the bot's opinions.

A debate run reads news, social sentiment, fundamentals and technicals, has
bull and bear researchers argue, and ends with a portfolio manager's rating.
It takes minutes and costs LLM credits, so each symbol's rating is cached and
refreshed at most every ``refresh_hours``.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from autotrader.signals import Opinion

log = logging.getLogger(__name__)

RATING_SCORES = {"Buy": 1.0, "Overweight": 0.5, "Hold": 0.0, "Underweight": -0.5, "Sell": -1.0}


def ta_ticker(symbol: str, market: str) -> str:
    """Ticker form TradingAgents expects: ``BTC-USD`` for crypto, Yahoo
    suffixes (``RELIANCE.NS``) for Indian stocks."""
    if market == "crypto":
        base = symbol.replace("-", "/").split("/")[0].upper()
        return f"{base}-USD"
    if market == "india" and "." not in symbol:
        return f"{symbol}.NS"
    return symbol


class LLMAdvisor:
    def __init__(self, cache_dir: Path, refresh_hours: float = 24.0,
                 analysts: list[str] | None = None, graph_factory=None):
        self.cache_file = Path(cache_dir) / "llm_ratings.json"
        self.refresh = timedelta(hours=refresh_hours)
        self.analysts = analysts or ["market", "social", "news", "fundamentals"]
        self._graph_factory = graph_factory
        self._graph = None

    def _load(self) -> dict:
        try:
            return json.loads(self.cache_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save(self, cache: dict) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    def _get_graph(self):
        if self._graph is None:
            if self._graph_factory is not None:
                self._graph = self._graph_factory()
            else:
                from tradingagents.default_config import DEFAULT_CONFIG
                from tradingagents.graph.trading_graph import TradingAgentsGraph

                self._graph = TradingAgentsGraph(
                    selected_analysts=tuple(self.analysts), config=DEFAULT_CONFIG.copy()
                )
        return self._graph

    def opinion(self, symbol: str, market: str, now: datetime | None = None) -> Opinion | None:
        now = now or datetime.now(UTC)
        ticker = ta_ticker(symbol, market)
        cache = self._load()
        entry = cache.get(ticker)
        if entry and now - datetime.fromisoformat(entry["at"]) < self.refresh:
            rating = entry["rating"]
        else:
            try:
                _, rating = self._get_graph().propagate(
                    ticker, now.strftime("%Y-%m-%d"),
                    asset_type="crypto" if market == "crypto" else "stock",
                )
            except Exception:  # an LLM outage must not stop the bot
                log.exception("TradingAgents run failed for %s", ticker)
                if not entry:
                    return None
                rating = entry["rating"]
            else:
                cache[ticker] = {"rating": rating, "at": now.isoformat()}
                self._save(cache)
        if rating not in RATING_SCORES:  # "REVIEW": no parseable decision
            return None
        return Opinion("llm", RATING_SCORES[rating], [f"TradingAgents debate rating: {rating}"],
                       {"rating": rating})
