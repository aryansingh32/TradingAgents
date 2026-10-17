"""Round-the-clock news: a per-symbol news opinion and an event-risk guard,
both fed by World Monitor (see ``tradingagents.dataflows.vendors.worldmonitor``).

News opinion: stories that mention the symbol in the last day, each scored
by a finance word list (beats/upgrade/approval vs miss/downgrade/probe),
weighted by World Monitor's importance score, corroboration and recency, plus
any market-implication card's direction for the ticker. It is a light tilt;
the full reading of the news is the TradingAgents debate's job.

Event-risk guard, the rules desks follow around scheduled and unscheduled
risk; it blocks new entries (exits always run) when:

* a critical, corroborated breaking story or a high-severity market-stress /
  VIX-spike signal appeared in the last few hours (shock guard);
* a high-impact macro release (Fed, CPI, payrolls, RBI...) for the market's
  country falls on today's date (macro guard; crypto follows US releases);
* the stock reports earnings within ``earnings_days`` (earnings guard);
* today is a configured blackout date.
"""

from __future__ import annotations

import logging
import math
import re
import time
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from autotrader.config import NewsConfig, SymbolConfig
from autotrader.signals import Opinion
from tradingagents.dataflows.vendors.worldmonitor import (
    WorldMonitorClient,
    WorldMonitorUnavailable,
    mentions,
    search_terms,
)

log = logging.getLogger(__name__)

POSITIVE = {
    "beat", "beats", "surge", "surges", "soar", "soars", "rally", "rallies", "record", "upgrade",
    "upgraded", "outperform", "approval", "approved", "approves", "inflow", "inflows", "growth",
    "profit", "profits", "raises", "raised", "bullish", "buyback", "partnership", "wins", "jump",
    "jumps", "gain", "gains", "strong", "expands", "adoption", "accumulate", "etf approval",
}
NEGATIVE = {
    "miss", "misses", "plunge", "plunges", "crash", "crashes", "slump", "downgrade", "downgraded",
    "underperform", "probe", "investigation", "lawsuit", "sued", "fraud", "hack", "hacked", "exploit",
    "outflow", "outflows", "loss", "losses", "cuts", "cut", "bearish", "ban", "bans", "default",
    "bankruptcy", "recall", "fine", "fined", "sanction", "sanctions", "layoffs", "warning", "warns",
    "delist", "selloff", "sell-off", "liquidation", "liquidations", "weak", "drops", "falls",
}
_WORD = re.compile(r"[a-z][a-z\-']+")

MARKET_COUNTRIES = {"us": {"US", "USA"}, "crypto": {"US", "USA"}, "india": {"IN", "IND", "INDIA"}}
_TZ = {"us": ZoneInfo("America/New_York"), "india": ZoneInfo("Asia/Kolkata"), "crypto": UTC}
_STRESS = {"CROSS_SOURCE_SIGNAL_TYPE_MARKET_STRESS", "CROSS_SOURCE_SIGNAL_TYPE_VIX_SPIKE",
           "CROSS_SOURCE_SIGNAL_TYPE_COMPOSITE_ESCALATION"}
_HIGH_SEVERITY = {"CROSS_SOURCE_SIGNAL_SEVERITY_HIGH", "CROSS_SOURCE_SIGNAL_SEVERITY_CRITICAL"}


def headline_polarity(text: str) -> float:
    words = _WORD.findall(text.lower())
    pos = sum(w in POSITIVE for w in words)
    neg = sum(w in NEGATIVE for w in words)
    return 0.0 if pos == neg else (pos - neg) / (pos + neg)


class NewsMonitor:
    def __init__(self, cfg: NewsConfig, client: WorldMonitorClient | None = None, clock=None):
        self.cfg = cfg
        self.client = client or WorldMonitorClient(base_url=cfg.base_url)
        self.clock = clock or (lambda: datetime.now(UTC))
        self._at = -math.inf
        self.items: list[dict] = []
        self.signals: list[dict] = []
        self.cards: list[dict] = []
        self.events: list[dict] = []
        self.earnings: list[dict] = []

    @property
    def available(self) -> bool:
        return self.client.configured

    def refresh(self, force: bool = False) -> None:
        if not force and time.monotonic() - self._at < self.cfg.refresh_seconds:
            return
        self._at = time.monotonic()
        now = self.clock()
        reads = {
            "items": lambda: self.client.news(self.cfg.variant),
            "signals": self.client.cross_source_signals,
            "cards": self.client.market_implications,
            "events": lambda: self.client.economic_calendar(now - timedelta(days=1), now + timedelta(days=2)),
            "earnings": lambda: self.client.earnings_calendar(now, now + timedelta(days=self.cfg.earnings_days + 3)),
        }
        for attr, read in reads.items():
            try:
                setattr(self, attr, read())
            except WorldMonitorUnavailable as exc:
                log.info("world monitor %s unavailable: %s", attr, exc)

    # -- opinion ---------------------------------------------------------
    def opinion(self, sym: SymbolConfig) -> Opinion | None:
        self.refresh()
        now_ms = self.clock().timestamp() * 1000
        horizon_ms = self.cfg.lookback_hours * 3_600_000
        terms = search_terms(sym.price_symbol) | search_terms(sym.symbol)
        total, weight, reasons = 0.0, 0.0, []
        for item in self.items:
            age = now_ms - int(item.get("publishedAt") or 0)
            if age > horizon_ms or not mentions(item, terms):
                continue
            pol = headline_polarity(f"{item.get('title', '')} {item.get('snippet', '')}")
            level = (item.get("threat") or {}).get("level", "")
            if level in ("THREAT_LEVEL_HIGH", "THREAT_LEVEL_CRITICAL") and pol >= 0:
                pol -= 0.5  # a severe event about the asset is rarely good for it
            w = ((item.get("importanceScore") or 30) / 100) * (1 + math.log1p(item.get("corroborationCount") or 0)) \
                * math.exp(-age / horizon_ms)
            total += w * pol
            weight += w
            if pol and len(reasons) < 3:
                reasons.append(f"{'+' if pol > 0 else '-'} {item.get('title', '')[:90]} ({item.get('source', '')})")
        for card in self.cards:
            if str(card.get("ticker", "")).upper() in {t.upper() for t in terms}:
                d = str(card.get("direction", "")).lower()
                pol = 1.0 if d in ("bullish", "up", "long", "positive") else -1.0 if d in ("bearish", "down", "short", "negative") else 0.0
                if pol:
                    total += pol
                    weight += 1.0
                    reasons.append(f"market implication: {card.get('direction')} ({card.get('title', '')[:70]})")
        if weight == 0:
            return None
        score = math.tanh(2 * total / max(weight, 1.0))
        return Opinion("news", score, reasons or ["neutral coverage"], {"stories_weight": round(weight, 2)})

    # -- guard ------------------------------------------------------------
    def entry_block(self, sym: SymbolConfig, now: datetime | None = None) -> str | None:
        self.refresh()
        now = now or self.clock()
        c = self.cfg
        local_day = now.astimezone(_TZ[sym.market]).date().isoformat()
        if local_day in c.blackout_dates:
            return f"blackout date {local_day}"
        if c.shock_guard:
            since = (now - timedelta(hours=c.shock_hours)).timestamp() * 1000
            for item in self.items:
                if int(item.get("publishedAt") or 0) < since:
                    continue
                if (item.get("threat") or {}).get("level") == "THREAT_LEVEL_CRITICAL" \
                        and (item.get("corroborationCount") or 0) >= c.shock_min_sources:
                    return f"breaking critical story: {item.get('title', '')[:90]}"
            for s in self.signals:
                if s.get("type") in _STRESS and s.get("severity") in _HIGH_SEVERITY \
                        and int(s.get("detectedAt") or 0) >= since:
                    return f"market stress signal: {s.get('summary', '')[:90]}"
        if c.macro_guard:
            countries = MARKET_COUNTRIES[sym.market]
            for e in self.events:
                if str(e.get("impact", "")).lower() == "high" and str(e.get("country", "")).upper() in countries \
                        and str(e.get("date", ""))[:10] == local_day:
                    return f"high-impact release today: {e.get('event')} ({e.get('country')})"
        if c.earnings_guard and sym.market != "crypto":
            terms = {t.upper() for t in search_terms(sym.price_symbol) | search_terms(sym.symbol)}
            limit = (now + timedelta(days=c.earnings_days)).date().isoformat()
            for e in self.earnings:
                if str(e.get("symbol", "")).upper() in terms and local_day <= str(e.get("date", ""))[:10] <= limit:
                    return f"earnings on {e.get('date')} {e.get('hour', '')}".strip()
        return None
