"""World Monitor vendor: parsing of its documented response shapes, focus
ranking, graceful degradation, and the historical-date guard."""

import json
from datetime import UTC, datetime

import pytest

from tradingagents.agents import tools
from tradingagents.dataflows.date_window import get_current_date
from tradingagents.dataflows.vendors import worldmonitor as wm

pytestmark = pytest.mark.unit

NOW_MS = int(datetime.now(UTC).timestamp() * 1000)


class Resp:
    def __init__(self, data, status=200):
        self.status_code, self._data = status, data

    def json(self):
        return self._data


class Session:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def request(self, method, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params, headers))
        for suffix, data in self.routes.items():
            if url.endswith(suffix):
                return data if isinstance(data, Resp) else Resp(data)
        return Resp({}, 404)


DIGEST = {"categories": {
    "markets": {"items": [
        {"title": "Fed signals pause", "source": "Reuters", "publishedAt": NOW_MS - 3_600_000,
         "importanceScore": 70, "corroborationCount": 5, "tickers": [],
         "threat": {"level": "THREAT_LEVEL_MEDIUM"}, "storyMeta": {"phase": "STORY_PHASE_DEVELOPING"}},
        {"title": "Nvidia beats on data-center demand", "source": "Bloomberg", "publishedAt": NOW_MS - 600_000,
         "importanceScore": 60, "corroborationCount": 3, "tickers": ["NVDA"],
         "threat": {"level": "THREAT_LEVEL_LOW"}, "storyMeta": {"phase": "STORY_PHASE_BREAKING"}},
    ]},
    "crypto": {"items": [
        {"title": "Bitcoin ETF outflows deepen", "source": "CoinDesk", "publishedAt": NOW_MS - 1_000_000,
         "importanceScore": 55, "tickers": [], "threat": {"level": "THREAT_LEVEL_HIGH"}}]},
}}
ROUTES = {
    "/api/news/v1/list-feed-digest": DIGEST,
    "/api/intelligence/v1/list-cross-source-signals": {"signals": [
        {"type": "CROSS_SOURCE_SIGNAL_TYPE_VIX_SPIKE", "severity": "CROSS_SOURCE_SIGNAL_SEVERITY_HIGH",
         "severityScore": 80, "theater": "Global Markets", "summary": "VIX up 30%"}]},
    "/api/intelligence/v1/list-market-implications": Resp({}, 403),
    "/api/economic/v1/get-economic-calendar": {"events": [
        {"date": "2026-10-20", "country": "US", "event": "CPI", "impact": "high"},
        {"date": "2026-10-21", "country": "US", "event": "Housing starts", "impact": "low"}]},
    "/api/market/v1/list-earnings-calendar": {"earnings": [
        {"date": "2026-10-22", "hour": "amc", "company": "NVIDIA", "symbol": "NVDA", "epsEstimate": 1.1}]},
}


def client(routes=ROUTES, key="wm_test"):
    return wm.WorldMonitorClient(api_key=key, session=Session(routes))


def test_brief_puts_focus_first_and_includes_every_section():
    c = client()
    out = wm.get_world_monitor("NVDA", as_of_date=get_current_date(), client=c)
    table = out.split("### Headlines")[1]
    assert table.index("Nvidia beats") < table.index("Fed signals pause")
    assert "**Nvidia beats on data-center demand**" in out and "breaking" in out
    assert "vix_spike in Global Markets" in out
    assert "CPI" in out and "Housing starts" not in out  # high impact only
    assert "NVIDIA (NVDA)" in out
    assert "Market implications" not in out  # 403 on this key: skipped, not fatal
    assert c.session.calls[0][2] == {"X-WorldMonitor-Key": "wm_test"}
    assert c.session.calls[0][1] == {"variant": "finance", "lang": "en"}


def test_bitcoin_focus_matches_by_name():
    out = wm.get_world_monitor("BTC-USD", as_of_date=get_current_date(), client=client())
    assert "(1 mention BTC-USD)" in out and "**Bitcoin ETF outflows deepen**" in out


def test_degrades_without_key_or_network():
    no_key = wm.WorldMonitorClient(api_key="", base_url=wm.DEFAULT_BASE_URL, session=Session(ROUTES))
    assert "unavailable" in wm.get_world_monitor("NVDA", as_of_date=get_current_date(), client=no_key)
    assert no_key.session.calls == []
    self_hosted = wm.WorldMonitorClient(api_key="", base_url="http://localhost:3000", session=Session(ROUTES))
    assert "Headlines" in wm.get_world_monitor(None, as_of_date=get_current_date(), client=self_hosted)


def test_withholds_live_data_from_a_historical_run():
    c = client()
    out = wm.get_world_monitor("NVDA", as_of_date="2025-06-01", client=c)
    assert "withheld" in out and c.session.calls == []


def test_mentions_and_terms():
    assert {"BTC", "bitcoin"} <= wm.search_terms("BTC/USDT")
    assert wm.search_terms("RELIANCE.NS") == {"RELIANCE.NS", "RELIANCE"}
    assert wm.mentions({"title": "x", "tickers": ["nvda"]}, {"NVDA"})
    assert not wm.mentions({"title": "Bitcoiners rejoice"}, {"bitcoin"})
    assert wm.mentions({"title": "Bitcoin rallies"}, {"bitcoin"})


def test_tool_is_registered_and_date_injected():
    props = tools.get_world_monitor.tool_call_schema.model_json_schema()["properties"]
    assert "trade_date" not in props and set(props) == {"focus", "limit"}
    json.dumps(props)  # schema is serialisable for every provider
