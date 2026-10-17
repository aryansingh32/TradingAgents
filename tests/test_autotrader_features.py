"""Tests for AutoTrader's news guard, extra signals, partial exits, smart DCA,
universe scanner, notifications and remote commands."""

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from autotrader.accumulate import Accumulator, multiplier
from autotrader.backtest import run_backtest
from autotrader.config import (
    AccumulateConfig,
    NewsConfig,
    NotifyConfig,
    RiskConfig,
    SignalConfig,
    SymbolConfig,
    TradeManagementConfig,
    UniverseConfig,
    config_from_dict,
)
from autotrader.engine import Engine
from autotrader.ensemble import BUY, HOLD, combine
from autotrader.news import NewsMonitor, headline_polarity
from autotrader.notify import Notifier
from autotrader.scanner import crypto_top_volume
from autotrader.signals import Opinion
from autotrader.signals_extra import EXTRA, mean_reversion_scores, pattern_scores
from tests.test_autotrader import FakeMarket, journal, make_df, trending_df

pytestmark = pytest.mark.unit

NOW = datetime(2026, 10, 16, 15, 0, tzinfo=UTC)  # Friday, 11:00 New York
NOW_MS = int(NOW.timestamp() * 1000)


class FakeWM:
    """World Monitor client stand-in returning documented response shapes."""

    configured = True

    def __init__(self, items=(), signals=(), cards=(), events=(), earnings=()):
        self._items, self._signals, self._cards = list(items), list(signals), list(cards)
        self._events, self._earnings = list(events), list(earnings)

    def news(self, variant="finance"):
        return self._items

    def cross_source_signals(self):
        return self._signals

    def market_implications(self):
        return self._cards

    def economic_calendar(self, start, end):
        return self._events

    def earnings_calendar(self, start, end):
        return self._earnings


def story(title, minutes_ago=30, tickers=(), threat="THREAT_LEVEL_LOW", sources=1, importance=60):
    return {"title": title, "source": "Wire", "publishedAt": NOW_MS - minutes_ago * 60_000,
            "tickers": list(tickers), "importanceScore": importance, "corroborationCount": sources,
            "threat": {"level": threat}}


def monitor(client, **cfg):
    return NewsMonitor(NewsConfig(**cfg), client=client, clock=lambda: NOW)


BTC = SymbolConfig("BTC/USDT", market="crypto")
NVDA = SymbolConfig("NVDA", market="us")
RELIANCE = SymbolConfig("RELIANCE.NS", market="india")


# -- news -----------------------------------------------------------------------

def test_headline_polarity():
    assert headline_polarity("Nvidia beats estimates, raises guidance") > 0
    assert headline_polarity("Exchange hacked, withdrawals halted amid probe") < 0
    assert headline_polarity("Markets open") == 0


def test_news_opinion_reads_stories_about_the_symbol():
    wm = FakeWM(items=[story("Bitcoin ETF inflows surge to record", sources=4),
                       story("Oil falls as supply grows"),
                       story("Old bitcoin crash story", minutes_ago=60 * 30)])
    op = monitor(wm).opinion(BTC)
    assert op.score > 0.3 and "inflows" in op.reasons[0]
    assert monitor(FakeWM(items=[story("Oil falls")])).opinion(BTC) is None
    cards = [{"ticker": "NVDA", "direction": "bearish", "title": "export curbs"}]
    assert monitor(FakeWM(cards=cards)).opinion(NVDA).score < 0


def test_news_guard_blocks_entries_around_event_risk():
    shock = FakeWM(items=[story("Major exchange collapses", threat="THREAT_LEVEL_CRITICAL", sources=5)])
    assert "breaking critical" in monitor(shock).entry_block(BTC, NOW)
    lone = FakeWM(items=[story("Unverified rumour", threat="THREAT_LEVEL_CRITICAL", sources=1)])
    assert monitor(lone).entry_block(BTC, NOW) is None
    stress = FakeWM(signals=[{"type": "CROSS_SOURCE_SIGNAL_TYPE_VIX_SPIKE",
                              "severity": "CROSS_SOURCE_SIGNAL_SEVERITY_HIGH",
                              "detectedAt": NOW_MS - 600_000, "summary": "VIX +35%"}])
    assert "market stress" in monitor(stress).entry_block(NVDA, NOW)

    cpi = FakeWM(events=[{"date": "2026-10-16", "country": "US", "event": "CPI", "impact": "high"}])
    assert "CPI" in monitor(cpi).entry_block(NVDA, NOW)
    assert "CPI" in monitor(cpi).entry_block(BTC, NOW)  # crypto follows US macro
    assert monitor(cpi).entry_block(RELIANCE, NOW) is None
    assert monitor(cpi, macro_guard=False).entry_block(NVDA, NOW) is None

    earn = FakeWM(earnings=[{"symbol": "NVDA", "date": "2026-10-17", "hour": "amc"}])
    assert "earnings" in monitor(earn).entry_block(NVDA, NOW)
    assert monitor(earn, earnings_days=0).entry_block(NVDA, NOW) is None
    assert "blackout" in monitor(FakeWM(), blackout_dates=["2026-10-16"]).entry_block(NVDA, NOW)


# -- extra signals ------------------------------------------------------------------

@pytest.mark.parametrize("name", list(EXTRA))
def test_extra_signals_use_only_past_bars(name):
    df = make_df(n=500, seed=7)
    df["volume"] = np.random.default_rng(1).uniform(500, 1500, len(df))
    full = EXTRA[name](df)
    for t in (300, 420, 499):
        part = EXTRA[name](df.iloc[: t + 1]).iloc[-1]
        assert (np.isnan(part) and np.isnan(full.iloc[t])) or part == pytest.approx(full.iloc[t])


def test_bullish_engulfing_after_a_decline():
    o = [10, 9.5, 9.0, 8.5, 8.0]
    c = [9.6, 9.1, 8.6, 8.1, 8.9]
    df = pd.DataFrame({"open": o, "close": c, "volume": 1.0},
                      index=pd.date_range("2026-01-01", periods=5, freq="D", tz="UTC"))
    df["high"] = df[["open", "close"]].max(axis=1) + 0.05
    df["low"] = df[["open", "close"]].min(axis=1) - 0.05
    assert pattern_scores(df).iloc[-1] == 1.0


def test_mean_reversion_fires_on_a_dip_inside_an_uptrend():
    up = trending_df(n=300, up=True, freq="D")
    dip = up.copy()
    for k, f in enumerate([0.97, 0.95, 0.93]):
        dip.iloc[-3 + k, :4] = up.iloc[-4]["close"] * f
    assert mean_reversion_scores(dip).iloc[-1] >= 0.7
    assert np.isnan(mean_reversion_scores(trending_df(n=300, up=False, freq="D")).iloc[-1])


def test_mean_reversion_setup_needs_a_confirmed_uptrend():
    cfg = SignalConfig()
    ops = [Opinion("technical", -0.4), Opinion("mean_reversion", 1.0)]
    assert combine(ops, cfg, trend_up=True).action == BUY
    assert combine(ops, cfg, trend_up=None).action == HOLD
    assert combine(ops, cfg, trend_up=False).action == HOLD
    assert combine([*ops, Opinion("llm", -1.0)], cfg, trend_up=True).action != BUY


# -- partial exits -------------------------------------------------------------------

def test_backtest_partial_take_profit_moves_stop_to_breakeven():
    df = make_df(n=700, drift=0.003, vol=0.015, seed=11)
    sig = SignalConfig(forecast_weight=0.0, trend_interval="1d")
    with_partial = run_backtest(df, RiskConfig(reward_risk=6), sig, cost=0.0,
                                trade_cfg=TradeManagementConfig(partial_tp_r=1.0))
    assert any(t["partial"] for t in with_partial.trades)
    # Once scaled out at 1R with the stop at breakeven, a trade cannot end as a loss.
    for t in with_partial.trades:
        if t["partial"]:
            assert t["r"] >= 0.5 - 1e-9


def engine(tmp_path, market, **extra):
    raw = {"data_dir": str(tmp_path), "interval": "1h",
           "symbols": [{"symbol": "BTC/USDT", "market": "crypto"}],
           "signals": {"technical_weight": 1.0, "forecast_weight": 0.0, "llm_weight": 0.0,
                       "context_weight": 0.0, "news_weight": 0.0, "patterns_weight": 0.0,
                       "volume_weight": 0.0, "trend_strength_weight": 0.0, "mean_reversion_weight": 0.0,
                       "trend_interval": "1h"},
           "risk": {"max_spread_pct": 0.01},
           "news": {"enabled": False},
           "brokers": {"paper": {"starting_cash": 10000, "fee_rate": 0, "slippage": 0}}}
    raw.update(extra)
    return Engine(config_from_dict(raw), market_data=market, context=False)


def test_engine_scales_out_at_1r(tmp_path):
    market = FakeMarket(trending_df(up=True))
    eng = engine(tmp_path, market)
    sym, broker = eng.cfg.symbols[0], eng.brokers["paper"]
    eng.step()
    qty = broker.position_qty(sym)
    prot = eng.state["protections"]["BTC/USDT"]
    market.price = prot["entry"] + 1.05 * (prot["entry"] - prot["initial_stop"])
    eng.risk_tick()
    assert broker.position_qty(sym) == pytest.approx(qty / 2, abs=1e-5)
    prot = eng.state["protections"]["BTC/USDT"]
    assert prot["partial_done"] and prot["stop"] >= prot["entry"]
    assert any(e["event"] == "partial_exit" for e in journal(tmp_path))


# -- commands, news block, heartbeat --------------------------------------------------

def test_pause_panic_and_heartbeat(tmp_path):
    market = FakeMarket(trending_df(up=True))
    eng = engine(tmp_path, market)
    sym, broker = eng.cfg.symbols[0], eng.brokers["paper"]
    assert "paused" in eng.handle_command("pause")
    eng.step()
    assert broker.position_qty(sym) == 0
    assert "paused" in journal(tmp_path)[-1]["reason"]
    eng.handle_command("resume")
    eng.state["last_bar"] = {}  # force a fresh decision
    eng.decisions = {}
    eng.step()
    assert broker.position_qty(sym) > 0
    assert (tmp_path / "heartbeat").exists()
    reply = eng.handle_command("panic")
    assert "BTC/USDT" in reply and broker.position_qty(sym) == 0 and eng.state["paused"]
    assert "equity" in eng.handle_command("status")


def test_news_guard_blocks_engine_entry(tmp_path):
    class Shock:
        def opinion(self, sym):
            return None

        def entry_block(self, sym, now):
            return "breaking critical story: exchange halts withdrawals"

    market = FakeMarket(trending_df(up=True))
    cfg = engine(tmp_path, market).cfg
    eng = Engine(cfg, market_data=market, context=False, news=Shock())
    eng.step()
    assert eng.brokers["paper"].position_qty(cfg.symbols[0]) == 0
    assert journal(tmp_path)[-1]["event"] == "entry_blocked_news"


# -- smart DCA ------------------------------------------------------------------------

def test_dca_multiplier():
    acc = AccumulateConfig("BTC/USDT", dip_multiplier=2, fear_multiplier=1.5, greed_multiplier=0.5,
                           max_multiplier=2.5)
    assert multiplier(acc, 100, 90, 50)[0] == 1.0
    assert multiplier(acc, 80, 90, 50)[0] == 2.0
    assert multiplier(acc, 80, 90, 10)[0] == 2.5  # capped
    assert multiplier(acc, 100, 90, 90)[0] == 0.5


def test_accumulator_buys_on_schedule(tmp_path):
    from autotrader.brokers.paper import PaperBroker

    broker = PaperBroker("dca", tmp_path / "dca.json", starting_cash=10000, fee_rate=0, slippage=0)
    daily = trending_df(n=300, up=False, freq="D")  # below its 200-day EMA: buys double
    md = FakeMarket(daily)
    events = []
    acc = Accumulator([AccumulateConfig("BTC/USDT", broker="dca", amount=100, every_hours=24)],
                      {"dca": broker}, md, {}, lambda e, **f: events.append((e, f)), fear_greed=lambda: 50)
    acc.step(NOW)
    acc.step(NOW + timedelta(hours=1))
    assert [e for e, _ in events] == ["accumulate"]
    assert events[0][1]["multiplier"] == 2.0
    assert events[0][1]["amount"] == pytest.approx(200, rel=0.01)
    acc.step(NOW + timedelta(hours=25))
    assert len(events) == 2


def test_accumulated_symbol_cannot_also_be_traded_on_the_same_broker():
    with pytest.raises(ValueError, match="cannot also be traded"):
        config_from_dict({"symbols": [{"symbol": "BTC/USDT"}],
                          "accumulate": [{"symbol": "BTC/USDT", "broker": "paper"}]})
    config_from_dict({"symbols": [], "accumulate": [{"symbol": "BTC/USDT"}]})


# -- scanner and notifier ----------------------------------------------------------------

class Resp:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


def test_crypto_universe_by_volume():
    rows = [{"symbol": "BTCUSDT", "quoteVolume": "9e9"}, {"symbol": "USDCUSDT", "quoteVolume": "8e9"},
            {"symbol": "ETHUSDT", "quoteVolume": "5e9"}, {"symbol": "BTCUPUSDT", "quoteVolume": "4e9"},
            {"symbol": "SOLUSDT", "quoteVolume": "1e9"}, {"symbol": "ETHBTC", "quoteVolume": "9e9"},
            {"symbol": "TINYUSDT", "quoteVolume": "1e6"}]
    session = type("S", (), {"get": lambda self, url, timeout=None: Resp(rows)})()
    picks = crypto_top_volume(UniverseConfig(top_n=3, min_quote_volume=5e7), session)
    assert picks == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


def test_telegram_commands_only_from_the_owner(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    updates = {"result": [
        {"update_id": 1, "message": {"chat": {"id": 42}, "text": "/pause"}},
        {"update_id": 2, "message": {"chat": {"id": 99}, "text": "/panic"}},
        {"update_id": 3, "message": {"chat": {"id": 42}, "text": "/status@my_bot"}},
        {"update_id": 4, "message": {"chat": {"id": 42}, "text": "/rm -rf"}},
    ]}
    sent = []

    class S:
        def get(self, url, params=None, timeout=None):
            sent.append(params)
            return Resp(updates)

        def post(self, url, json=None, timeout=None):
            sent.append(json)

    n = Notifier(NotifyConfig(telegram=True), session=S())
    assert n.poll_commands() == ["pause", "status"]
    n.poll_commands()
    assert sent[-1]["offset"] == 5
    n.send("entry", "bought")
    n.send("not_listed", "ignored")
    assert sent[-1] == {"chat_id": "42", "text": "bought"}
    assert json.dumps(sent)


def test_dca_only_config_loads_its_market_feed():
    from autotrader.engine import markets_used

    cfg = config_from_dict({"symbols": [], "accumulate": [{"symbol": "BTC/USDT"}],
                            "universe": {"enabled": True, "market": "us", "watchlist": ["SPY"]}})
    assert markets_used(cfg) == {"crypto", "us"}
