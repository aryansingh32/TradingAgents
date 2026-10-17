"""World Monitor vendor: real-time global intelligence for the news analyst.

World Monitor (https://github.com/koala73/worldmonitor, AGPL-3.0) fuses
hundreds of live feeds (news wires, conflict and unrest trackers, markets,
shipping, flights, outages, disasters) into scored, deduplicated signals. This
vendor reads five of its REST services:

* ``news/v1/list-feed-digest``: the live headline digest, each story with an
  importance score, corroboration count, threat level, story phase
  (breaking / developing / sustained / fading) and the tickers it mentions;
* ``intelligence/v1/list-cross-source-signals``: cross-domain escalation
  signals (market stress, VIX spikes, commodity shocks, shipping disruption,
  sanctions, composite escalation);
* ``intelligence/v1/list-market-implications``: analyst cards linking events
  to tickers with a direction (Pro plan; skipped when not entitled);
* ``economic/v1/get-economic-calendar`` and ``market/v1/list-earnings-calendar``:
  scheduled high-impact releases and earnings.

Server-to-server calls need an API key in ``WORLDMONITOR_API_KEY`` (sent as
``X-WorldMonitor-Key``). A self-hosted instance works too: point
``WORLDMONITOR_BASE_URL`` at it. Without a key, or on any network error, the
tool returns a short "unavailable" note instead of failing the analysis.

World Monitor serves the live state of the world only, so a historical
analysis date withholds it, as the Polymarket vendor does.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta

import requests

from tradingagents.dataflows.date_window import is_historical

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.worldmonitor.app"
REQUEST_TIMEOUT = 20
DEFAULT_LIMIT = 12

_THREAT_RANK = {"THREAT_LEVEL_CRITICAL": 4, "THREAT_LEVEL_HIGH": 3, "THREAT_LEVEL_MEDIUM": 2,
                "THREAT_LEVEL_LOW": 1}
_SEVERITY_RANK = {"CROSS_SOURCE_SIGNAL_SEVERITY_CRITICAL": 4, "CROSS_SOURCE_SIGNAL_SEVERITY_HIGH": 3,
                  "CROSS_SOURCE_SIGNAL_SEVERITY_MEDIUM": 2, "CROSS_SOURCE_SIGNAL_SEVERITY_LOW": 1}


class WorldMonitorUnavailable(RuntimeError):
    pass


def _short(enum_value: str | None, prefix: str) -> str:
    return (enum_value or "").removeprefix(prefix).lower() or "-"


class WorldMonitorClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 session: requests.Session | None = None, timeout: float = REQUEST_TIMEOUT):
        self.api_key = api_key if api_key is not None else os.environ.get("WORLDMONITOR_API_KEY", "")
        self.base_url = (base_url or os.environ.get("WORLDMONITOR_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        # A self-hosted instance may run without user keys.
        return bool(self.api_key) or self.base_url != DEFAULT_BASE_URL

    def get(self, path: str, params: dict | None = None) -> dict:
        if not self.configured:
            raise WorldMonitorUnavailable("WORLDMONITOR_API_KEY is not set")
        headers = {"X-WorldMonitor-Key": self.api_key} if self.api_key else {}
        try:
            resp = self.session.request("GET", f"{self.base_url}{path}", params=params or {},
                                        headers=headers, timeout=self.timeout)
        except requests.RequestException as exc:
            raise WorldMonitorUnavailable(f"network error: {exc}") from exc
        if resp.status_code in (401, 402, 403):
            raise WorldMonitorUnavailable(f"{path} not available on this key ({resp.status_code})")
        if resp.status_code >= 400:
            raise WorldMonitorUnavailable(f"{path} -> HTTP {resp.status_code}")
        try:
            return resp.json()
        except ValueError as exc:
            raise WorldMonitorUnavailable(f"{path}: invalid JSON") from exc

    # -- typed reads ------------------------------------------------------
    def news(self, variant: str = "finance", lang: str = "en") -> list[dict]:
        """Digest items across all categories, newest first, each tagged with
        its category."""
        data = self.get("/api/news/v1/list-feed-digest", {"variant": variant, "lang": lang})
        items = []
        for category, bucket in (data.get("categories") or {}).items():
            for item in (bucket or {}).get("items") or []:
                items.append({**item, "category": category})
        items.sort(key=lambda i: int(i.get("publishedAt") or 0), reverse=True)
        return items

    def cross_source_signals(self) -> list[dict]:
        return self.get("/api/intelligence/v1/list-cross-source-signals").get("signals") or []

    def market_implications(self) -> list[dict]:
        return self.get("/api/intelligence/v1/list-market-implications").get("cards") or []

    def economic_calendar(self, start: datetime, end: datetime) -> list[dict]:
        data = self.get("/api/economic/v1/get-economic-calendar",
                        {"fromDate": start.strftime("%Y-%m-%d"), "toDate": end.strftime("%Y-%m-%d")})
        return [] if data.get("unavailable") else data.get("events") or []

    def earnings_calendar(self, start: datetime, end: datetime) -> list[dict]:
        data = self.get("/api/market/v1/list-earnings-calendar",
                        {"fromDate": start.strftime("%Y-%m-%d"), "toDate": end.strftime("%Y-%m-%d")})
        return [] if data.get("unavailable") else data.get("earnings") or []


def mentions(item: dict, terms: set[str]) -> bool:
    """Whether a digest item is about one of ``terms`` (tickers or names)."""
    if not terms:
        return False
    tickers = {str(t).upper() for t in item.get("tickers") or []}
    if tickers & {t.upper() for t in terms}:
        return True
    text = f" {item.get('title', '')} {item.get('snippet', '')} ".lower()
    return any(f" {t.lower()} " in text or f" {t.lower()}'" in text or f"({t.lower()})" in text
               for t in terms if len(t) >= 3)


def search_terms(focus: str | None) -> set[str]:
    """Ticker and name variants for a focus such as ``NVDA``, ``BTC-USD`` or
    ``RELIANCE.NS``."""
    if not focus:
        return set()
    base = focus.upper().split(".")[0].replace("/", "-").split("-")[0]
    terms = {focus.upper(), base}
    names = {"BTC": {"bitcoin"}, "ETH": {"ethereum", "ether"}, "SOL": {"solana"}, "XRP": {"ripple"}}
    terms |= names.get(base, set())
    return terms


def _fmt_time(ms) -> str:
    try:
        return datetime.fromtimestamp(int(ms) / 1000, UTC).strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError, OSError):
        return "-"


def get_world_monitor(focus: str | None = None, limit: int | None = None,
                      as_of_date: str | None = None, client: WorldMonitorClient | None = None) -> str:
    """Markdown brief of the live world state relevant to trading.

    Args:
        focus: Optional ticker or asset (e.g. "NVDA", "BTC-USD") to surface
            stories and implications that mention it first.
        limit: Max headlines (default 12).
        as_of_date: The analysis date; a past date withholds the live data.
    """
    if is_historical(as_of_date):
        return (f"World Monitor data is withheld for {as_of_date}: it serves only the live "
                f"state of the world, which would put post-decision information into a "
                f"{as_of_date} analysis.")
    client = client or WorldMonitorClient()
    limit = limit or DEFAULT_LIMIT
    now = datetime.now(UTC)
    terms = search_terms(focus)
    sections = ["## World Monitor: live global intelligence\n"
                "Scored, deduplicated stories from hundreds of feeds. Importance 0-100 blends "
                "severity, source tier and corroboration; threat level is World Monitor's "
                "classification of the event, not a price direction.\n"]
    any_ok = False

    try:
        items = client.news("finance")
        any_ok = True
        day_ago = (now - timedelta(hours=36)).timestamp() * 1000
        recent = [i for i in items if int(i.get("publishedAt") or 0) >= day_ago] or items
        focused = [i for i in recent if mentions(i, terms)]
        rest = sorted((i for i in recent if i not in focused),
                      key=lambda i: (_THREAT_RANK.get((i.get("threat") or {}).get("level"), 0),
                                     i.get("importanceScore") or 0), reverse=True)
        chosen = (focused + rest)[:limit]
        if chosen:
            lines = ["| Time | Story | Source | Importance | Threat | Phase | Sources | Tickers |",
                     "|---|---|---|---|---|---|---|---|"]
            for i in chosen:
                threat = _short((i.get("threat") or {}).get("level"), "THREAT_LEVEL_")
                phase = _short((i.get("storyMeta") or {}).get("phase"), "STORY_PHASE_")
                title = str(i.get("title", "")).replace("|", "/")
                tick = ", ".join(i.get("tickers") or []) or "-"
                mark = "**" if i in focused else ""
                lines.append(f"| {_fmt_time(i.get('publishedAt'))} | {mark}{title}{mark} | {i.get('source', '-')} | "
                             f"{i.get('importanceScore', '-')} | {threat} | {phase} | "
                             f"{i.get('corroborationCount', '-')} | {tick} |")
            head = f"### Headlines ({len(focused)} mention {focus})" if focus else "### Headlines"
            sections.append(head + "\n" + "\n".join(lines))
    except WorldMonitorUnavailable as exc:
        sections.append(f"### Headlines\nUnavailable: {exc}")

    try:
        signals = sorted(client.cross_source_signals(),
                         key=lambda s: s.get("severityScore") or 0, reverse=True)[:6]
        any_ok = True
        if signals:
            sections.append("### Cross-source escalation signals\n" + "\n".join(
                f"- [{_short(s.get('severity'), 'CROSS_SOURCE_SIGNAL_SEVERITY_')}] "
                f"{_short(s.get('type'), 'CROSS_SOURCE_SIGNAL_TYPE_')} in {s.get('theater', '-')}: "
                f"{s.get('summary', '')}" for s in signals))
    except WorldMonitorUnavailable as exc:
        logger.info("world monitor signals unavailable: %s", exc)

    try:
        cards = client.market_implications()
        any_ok = True
        if terms:
            cards = sorted(cards, key=lambda c: str(c.get("ticker", "")).upper() not in terms)
        if cards:
            sections.append("### Market implications\n" + "\n".join(
                f"- {c.get('ticker', '-')} {c.get('direction', '')} ({c.get('confidence', '-')} confidence, "
                f"{c.get('timeframe', '-')}): {c.get('title', '')}. Driver: {c.get('driver', '-')}. "
                f"Risk: {c.get('riskCaveat', '-')}" for c in cards[:6]))
    except WorldMonitorUnavailable as exc:
        logger.info("world monitor implications unavailable: %s", exc)

    try:
        events = [e for e in client.economic_calendar(now, now + timedelta(days=7))
                  if str(e.get("impact", "")).lower() == "high"]
        any_ok = True
        if events:
            sections.append("### High-impact economic events (next 7 days)\n" + "\n".join(
                f"- {e.get('date')} {e.get('country')}: {e.get('event')} (estimate {e.get('estimate') or '-'}, "
                f"previous {e.get('previous') or '-'})" for e in events[:10]))
    except WorldMonitorUnavailable as exc:
        logger.info("world monitor calendar unavailable: %s", exc)

    if terms:
        try:
            earnings = [e for e in client.earnings_calendar(now, now + timedelta(days=14))
                        if str(e.get("symbol", "")).upper() in terms]
            any_ok = True
            if earnings:
                sections.append("### Upcoming earnings\n" + "\n".join(
                    f"- {e.get('date')} {e.get('hour', '')}: {e.get('company')} ({e.get('symbol')}), "
                    f"EPS estimate {e.get('epsEstimate', '-')}" for e in earnings))
        except WorldMonitorUnavailable as exc:
            logger.info("world monitor earnings unavailable: %s", exc)

    if not any_ok:
        return ("World Monitor is unavailable (set WORLDMONITOR_API_KEY, or WORLDMONITOR_BASE_URL "
                "for a self-hosted instance). Proceed without its global-intelligence signal.")
    return "\n\n".join(sections)
