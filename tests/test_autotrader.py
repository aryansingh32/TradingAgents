"""Tests for the autotrader bot: feeds, signals, risk, protections, brokers,
engine, backtest and optimizer.

Network calls are replaced by fake HTTP sessions that return payloads in each
API's documented response format; price paths are generated so each test
controls the market condition it checks.
"""

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from autotrader import indicators as ind
from autotrader.backtest import prepare, run_backtest, simulate
from autotrader.brokers import BrokerError, build_broker
from autotrader.brokers.alpaca import AlpacaBroker
from autotrader.brokers.freqtrade import FreqtradeBroker
from autotrader.brokers.openalgo import OpenAlgoBroker
from autotrader.brokers.paper import PaperBroker
from autotrader.config import (
    LIVE_CONFIRM_ENV,
    LIVE_CONFIRM_VALUE,
    RiskConfig,
    SessionConfig,
    SignalConfig,
    SymbolConfig,
    config_from_dict,
    load_config,
)
from autotrader.context import CryptoContext
from autotrader.engine import Engine
from autotrader.ensemble import BUY, HOLD, SELL, combine
from autotrader.feeds import (
    AlpacaDataFeed,
    BinanceFeed,
    CoinbaseFeed,
    FallbackFeed,
    FeedError,
    KrakenFeed,
    MarketData,
    OpenAlgoFeed,
    Quote,
    StaleData,
    check_fresh,
    closed_only,
    crypto_parts,
)
from autotrader.llm_advisor import LLMAdvisor, ta_ticker
from autotrader.optimize import walk_forward
from autotrader.protections import Protections
from autotrader.risk import (
    Protection,
    RiskManager,
    entry_window_open,
    market_open,
    mis_squareoff_due,
    vol_multiplier,
)
from autotrader.signals import (
    Opinion,
    align_to,
    analog_features,
    forecast_opinion,
    forecast_scores,
    technical_opinion,
    technical_scores,
    trend_up_series,
)

pytestmark = pytest.mark.unit


def make_df(n=600, drift=0.001, vol=0.02, seed=0, start=100.0, freq="D", t0="2022-01-01"):
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(drift + rng.normal(0, vol, n)))
    idx = pd.date_range(t0, periods=n, freq=freq, tz="UTC")
    df = pd.DataFrame({"open": np.r_[close[0], close[:-1]], "close": close}, index=idx)
    df["high"] = df[["open", "close"]].max(axis=1) * 1.005
    df["low"] = df[["open", "close"]].min(axis=1) * 0.995
    df["volume"] = 1000.0
    return df


def trending_df(n=400, up=True, freq="h", end=None):
    close = 100 * np.exp(np.linspace(0, 1.0 if up else -1.0, n)) * (1 + 0.01 * np.sin(np.arange(n)))
    if end is None:
        idx = pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC")
    else:
        idx = pd.date_range(end=end, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open": close, "close": close, "high": close * 1.01, "low": close * 0.99,
                         "volume": 1.0}, index=idx)


# -- fake HTTP ------------------------------------------------------------------

class FakeResponse:
    def __init__(self, data, status=200):
        self._data, self.status_code = data, status
        self.content = b"x"
        self.text = json.dumps(data)

    def json(self):
        return self._data


class FakeSession:
    def __init__(self, routes):
        self.routes, self.calls, self.headers, self.auth = routes, [], {}, None

    def request(self, method, url, timeout=None, **kwargs):
        self.calls.append((method, url, kwargs))
        for (m, suffix), data in self.routes.items():
            if m == method and url.split("?")[0].endswith(suffix):
                data = data(kwargs) if callable(data) else data
                return FakeResponse(*data) if isinstance(data, tuple) else FakeResponse(data)
        return FakeResponse({"status": "success"})


# -- feeds ----------------------------------------------------------------------

def test_crypto_symbol_mapping():
    assert crypto_parts("BTC/USDT") == ("BTC", "USDT")
    assert crypto_parts("BTC-USD") == ("BTC", "USD")
    assert crypto_parts("ETHUSDT") == ("ETH", "USDT")
    assert BinanceFeed.pair("BTC-USD") == "BTCUSDT"
    assert CoinbaseFeed.product("BTC/USDT") == "BTC-USD"
    assert KrakenFeed.pair("BTC/USDT") == "XBTUSD"


def _binance_rows(start_ms, n, step_ms=3_600_000):
    return [[start_ms + i * step_ms, "100", "101", "99", str(100 + i), "5", start_ms + (i + 1) * step_ms - 1,
             "0", 1, "0", "0", "0"] for i in range(n)]


def test_binance_klines_paginate_and_parse():
    def klines(kwargs):
        since = kwargs["params"]["startTime"]
        remaining = (kwargs["params"]["endTime"] - since) // 3_600_000
        return _binance_rows(since, min(1000, remaining))

    s = FakeSession({("GET", "/api/v3/klines"): klines,
                     ("GET", "/api/v3/ticker/bookTicker"): {"bidPrice": "99.9", "askPrice": "100.1"}})
    feed = BinanceFeed(session=s)
    df = feed.candles("BTC/USDT", "1h", datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 20, tzinfo=UTC))
    assert len(df) == 50 * 24 and df.index.is_monotonic_increasing
    assert df.index[0] == pd.Timestamp("2026-01-01", tz="UTC")
    assert len([c for c in s.calls if c[1].endswith("klines")]) == 2
    q = feed.quote("BTC/USDT")
    assert q.price == pytest.approx(100.0) and q.spread_pct == pytest.approx(0.002)


def test_coinbase_and_kraken_parse_their_formats():
    t0 = int(datetime(2026, 1, 1, tzinfo=UTC).timestamp())
    cb = FakeSession({("GET", "/candles"): [[t0 + 3600, 9, 12, 10, 11, 2], [t0, 8, 11, 9, 10, 1]],
                      ("GET", "/ticker"): {"price": "11", "bid": "10.9", "ask": "11.1"}})
    df = CoinbaseFeed(session=cb).candles("BTC-USD", "1h", datetime(2026, 1, 1, tzinfo=UTC),
                                          datetime(2026, 1, 1, 2, tzinfo=UTC))
    assert list(df["open"]) == [9, 10] and list(df["low"]) == [8, 9]
    kr = FakeSession({("GET", "/0/public/OHLC"): {"error": [], "result": {
        "XXBTZUSD": [[t0, "1", "3", "0.5", "2", "1.5", "7", 3]], "last": t0}}})
    df = KrakenFeed(session=kr).candles("BTC/USD", "1h", datetime(2026, 1, 1, tzinfo=UTC))
    assert df.iloc[0].to_dict() == {"open": 1, "high": 3, "low": 0.5, "close": 2, "volume": 7}
    bad = FakeSession({("GET", "/0/public/OHLC"): {"error": ["EQuery:Unknown asset pair"]}})
    with pytest.raises(FeedError):
        KrakenFeed(session=bad).candles("FOO/USD", "1h", datetime(2026, 1, 1, tzinfo=UTC))


def test_alpaca_and_openalgo_data(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    pages = iter([{"bars": [{"t": "2026-10-15T13:30:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10}],
                   "next_page_token": "p2"},
                  {"bars": [{"t": "2026-10-15T14:30:00Z", "o": 1.5, "h": 2, "l": 1, "c": 1.8, "v": 5}],
                   "next_page_token": None}])
    s = FakeSession({("GET", "/v2/stocks/SPY/bars"): lambda kw: next(pages),
                     ("GET", "/trades/latest"): {"trade": {"p": 501.2, "t": "2026-10-15T15:00:00Z"}},
                     ("GET", "/quotes/latest"): {"quote": {"bp": 501.1, "ap": 501.3}}})
    feed = AlpacaDataFeed(session=s)
    df = feed.candles("SPY", "1h", datetime(2026, 10, 15, tzinfo=UTC))
    assert list(df["close"]) == [1.5, 1.8]
    assert s.calls[0][2]["params"]["feed"] == "iex" and s.calls[1][2]["params"]["page_token"] == "p2"
    assert feed.quote("SPY").price == 501.2

    monkeypatch.setenv("OPENALGO_API_KEY", "key")
    oa = FakeSession({("POST", "/api/v1/history"): {"status": "success", "data": [
        {"timestamp": 1760500800, "open": 2500, "high": 2510, "low": 2490, "close": 2505, "volume": 100}]},
        ("POST", "/api/v1/quotes"): {"status": "success", "data": {"ltp": 2506, "bid": 2505.9, "ask": 2506.1}}})
    feed = OpenAlgoFeed(session=oa)
    df = feed.candles("RELIANCE.NS", "1h", datetime(2026, 10, 1, tzinfo=UTC))
    assert df["close"].iloc[0] == 2505
    body = oa.calls[0][2]["json"]
    assert body["symbol"] == "RELIANCE" and body["exchange"] == "NSE" and body["interval"] == "1h"
    assert feed.quote("RELIANCE").price == 2506


def test_closed_candles_and_staleness():
    df = make_df(n=10, freq="h", t0="2026-10-16 00:00")
    now = datetime(2026, 10, 16, 9, 30, tzinfo=UTC)
    closed = closed_only(df, "1h", now)
    assert closed.index[-1] == pd.Timestamp("2026-10-16 08:00", tz="UTC")  # 09:00 bar still forming
    check_fresh(closed, "1h", now, 2, "crypto", "BTC")
    with pytest.raises(StaleData):
        check_fresh(closed, "1h", now + timedelta(hours=4), 2, "crypto", "BTC")
    # Daily stock bars: Friday's candle is still fresh on Monday morning.
    daily = make_df(n=5, t0="2026-10-12")  # Mon..Fri
    check_fresh(daily, "1d", datetime(2026, 10, 19, 15, tzinfo=UTC), 2, "us", "SPY")


def test_fallback_feed_and_incremental_cache():
    class Broken:
        name = "broken"

        def candles(self, *a):
            raise FeedError("down")

        def quote(self, *a):
            raise FeedError("down")

    class Good:
        name = "good"

        def __init__(self):
            self.starts = []
            self.full = make_df(n=300, freq="h", t0="2026-10-04")

        def candles(self, symbol, interval, start, end=None):
            self.starts.append(start)
            return self.full[self.full.index >= pd.Timestamp(start)]

        def quote(self, symbol):
            return Quote(1.0, source="good")

    good = Good()
    feed = FallbackFeed([Broken(), good])
    assert feed.quote("X").source == "good"
    now = good.full.index[-1] + timedelta(hours=1, minutes=5)
    md = MarketData({"crypto": feed}, lookback_days=30, max_age_bars=2, clock=lambda: now)
    first = md.candles("BTC", "crypto", "1h")
    assert len(first) == 300
    md.candles("BTC", "crypto", "1h")
    assert good.starts[-1] == first.index[-1].to_pydatetime()  # second fetch only from the last bar


# -- signals ----------------------------------------------------------------------

def test_indicators_ranges():
    df = make_df()
    r = ind.rsi(df["close"]).dropna()
    assert ((r >= 0) & (r <= 100)).all()
    assert (ind.atr(df).dropna() > 0).all()


def test_technical_reads_trend_and_matches_vectorized():
    up, down = trending_df(up=True), trending_df(up=False)
    assert technical_opinion(up).score > 0.3
    assert technical_opinion(down).score < -0.3
    assert technical_opinion(make_df(n=100)) is None
    df = make_df(n=500)
    series = technical_scores(df)["score"]
    for t in (230, 320, 499):
        assert technical_opinion(df.iloc[: t + 1]).score == pytest.approx(np.clip(series.iloc[t], -1, 1))


def test_forecast_has_no_look_ahead_and_backtest_matches_live():
    df = make_df(n=600)
    series = forecast_scores(df, horizon=10, neighbors=40, start=400)
    for t in (420, 599):
        live = forecast_opinion(df.iloc[: t + 1], horizon=10, neighbors=40)
        assert live.score == pytest.approx(series.iloc[t])
    # Changing the future must not change a past forecast.
    changed = df.copy()
    changed.iloc[500:, :4] *= 3
    assert forecast_scores(changed, 10, 40, start=499).iloc[499] == pytest.approx(series.iloc[499])
    feats = analog_features(df)
    assert forecast_opinion(df.iloc[:500], 10, 40, features=feats).score == pytest.approx(series.iloc[499])


def test_trend_alignment_uses_only_closed_higher_bars():
    hourly = make_df(n=24 * 3, freq="h", t0="2026-10-01")
    daily = pd.Series([1.0, 0.0, 1.0], index=pd.date_range("2026-10-01", periods=3, freq="D", tz="UTC"))
    aligned = align_to(daily, timedelta(days=1), hourly.index, timedelta(hours=1))
    # The 23:00 bar of Oct 1 closes at midnight, when Oct 1's daily bar closes.
    assert np.isnan(aligned.iloc[0]) and aligned.iloc[22] != aligned.iloc[22]
    assert aligned.iloc[23] == 1.0 and aligned.iloc[24 + 22] == 1.0 and aligned.iloc[24 + 23] == 0.0
    assert trend_up_series(trending_df(n=400, freq="D")).iloc[-1]
    assert not trend_up_series(trending_df(n=400, up=False, freq="D")).iloc[-1]


def test_combine_weights_trend_filter_and_llm_veto():
    cfg = SignalConfig(technical_weight=0.5, forecast_weight=0.5, llm_weight=0.0, context_weight=0.0)
    d = combine([Opinion("technical", 0.8), Opinion("forecast", 0.4), Opinion("llm", -1)], cfg)
    assert d.score == pytest.approx(0.6) and d.action == BUY
    assert combine([Opinion("technical", 0.8)], cfg, trend_up=False).action == HOLD
    assert combine([Opinion("technical", -0.8)], cfg, trend_up=False).action == SELL
    cfg = SignalConfig(technical_weight=0.4, forecast_weight=0.3, llm_weight=0.3, buy_threshold=0.2)
    vetoed = combine([Opinion("technical", 1), Opinion("forecast", 1), Opinion("llm", -1)], cfg)
    assert vetoed.score == pytest.approx(0.4) and vetoed.action == HOLD
    assert combine([None, None], cfg).action == HOLD


def test_crypto_context_tilts_only_at_extremes():
    s = FakeSession({
        ("GET", "/fapi/v1/fundingRate"): [{"fundingRate": "0.0001"}] * 80 + [{"fundingRate": "0.0002"}] * 10,
        ("GET", "/fapi/v1/premiumIndex"): {"lastFundingRate": "0.0009"},
        ("GET", "/futures/data/openInterestHist"): [{"sumOpenInterest": "100"}, {"sumOpenInterest": "110"}],
        ("GET", "/fng/"): {"data": [{"value": "85"}]}})
    op = CryptoContext(session=s).opinion("BTC/USDT", 0.05)
    assert op.score < -0.4 and op.details["fear_greed"] == 85
    calm = FakeSession({
        ("GET", "/fapi/v1/fundingRate"): [{"fundingRate": "0.0001"}] * 90,
        ("GET", "/fapi/v1/premiumIndex"): {"lastFundingRate": "0.0001"},
        ("GET", "/futures/data/openInterestHist"): [{"sumOpenInterest": "100"}, {"sumOpenInterest": "101"}],
        ("GET", "/fng/"): {"data": [{"value": "50"}]}})
    assert CryptoContext(session=calm).opinion("BTC/USDT", 0.01).score == 0.0
    down = FakeSession({})
    down.request = lambda *a, **k: (_ for _ in ()).throw(ConnectionError("blocked"))
    ctx = CryptoContext(session=down)
    ctx.retries = 0
    assert ctx.opinion("BTC/USDT", 0.0) is None


# -- risk and protections -----------------------------------------------------------

def test_position_sizing_risks_fixed_fraction():
    rm = RiskManager(RiskConfig(risk_per_trade=0.01, max_position_pct=1.0, atr_stop_mult=2, reward_risk=3))
    plan = rm.plan_entry(price=100, atr=5, equity=10000, cash=10000, market="us")
    assert plan.qty == 10  # 1% of 10k = 100 at risk over a 10 stop
    assert plan.protection.stop == 90 and plan.protection.take == 130
    assert rm.plan_entry(100, 5, 10000, 10000, "us", size_mult=0.5).qty == 5
    capped = RiskManager(RiskConfig(max_position_pct=0.05)).plan_entry(100, 1, 10000, 10000, "us")
    assert capped.qty * 100 <= 500
    assert rm.plan_entry(100, 60, 10000, 10000, "us") is None
    assert vol_multiplier(RiskConfig(vol_target_annual=0.5), 1.0) == 0.5
    assert vol_multiplier(RiskConfig(vol_target_annual=0.5, min_size_mult=0.25), 4.0) == 0.25
    assert vol_multiplier(RiskConfig(), 4.0) == 1.0


def test_trailing_stop_only_rises_and_stop_wins_ties():
    rm = RiskManager(RiskConfig(atr_stop_mult=2, trail_after_atr=1))
    p = Protection(entry=100, stop=90, take=200, atr=5, highest=100)
    rm.update_trailing(p, 104)
    assert p.stop == 90
    rm.update_trailing(p, 120)
    assert p.stop == 110 and p.initial_stop == 90
    rm.update_trailing(p, 115)
    assert p.stop == 110
    assert rm.exit_reason(p, low=105, high=250) == ("trailing_stop", 110)
    assert rm.exit_reason(p, low=111, high=250) == ("take_profit", 200)
    assert rm.exit_reason(p, low=111, high=150) is None


def test_sessions():
    s = SessionConfig()
    assert market_open("crypto", datetime(2026, 10, 17, 3, tzinfo=UTC))
    assert market_open("us", datetime(2026, 10, 16, 15, 0, tzinfo=UTC))
    assert not market_open("us", datetime(2026, 10, 16, 21, 0, tzinfo=UTC))
    assert not entry_window_open("us", datetime(2026, 10, 16, 13, 35, tzinfo=UTC), s)  # 09:35 NY
    assert entry_window_open("us", datetime(2026, 10, 16, 14, 0, tzinfo=UTC), s)
    assert not entry_window_open("us", datetime(2026, 10, 16, 19, 50, tzinfo=UTC), s)  # 15:50 NY
    assert market_open("india", datetime(2026, 10, 16, 5, 0, tzinfo=UTC))
    assert not entry_window_open("india", datetime(2026, 10, 16, 3, 50, tzinfo=UTC), s)  # 09:20 IST
    assert not market_open("india", datetime(2026, 10, 17, 5, 0, tzinfo=UTC))
    assert mis_squareoff_due(datetime(2026, 10, 16, 9, 41, tzinfo=UTC), s)  # 15:11 IST
    assert not mis_squareoff_due(datetime(2026, 10, 16, 9, 30, tzinfo=UTC), s)


def test_protections():
    cfg = RiskConfig(cooldown_bars=3, stoploss_guard_count=2, stoploss_guard_hours=24,
                     stoploss_guard_pause_hours=6, drawdown_reduce_pct=0.1, drawdown_pause_pct=0.2,
                     loss_streak_reduce=2)
    g = Protections(cfg)
    t = datetime(2026, 10, 16, 12, tzinfo=UTC)
    hour = timedelta(hours=1)
    g.record_equity(10000)
    g.record_exit("BTC", t, -50, "stop_loss")
    assert not g.allows_entry("BTC", t + 2 * hour, 10000, hour)[0]  # cooldown
    assert g.allows_entry("BTC", t + 3 * hour, 10000, hour)[0]
    assert g.allows_entry("ETH", t + hour, 10000, hour)[0]
    g.record_exit("ETH", t + hour, -50, "stop_loss")
    ok, why = g.allows_entry("SOL", t + 2 * hour, 10000, hour)
    assert not ok and "stop-loss guard" in why
    assert g.allows_entry("SOL", t + 8 * hour, 10000, hour)[0]
    assert g.risk_multiplier(10000) == 0.5  # two losses in a row
    assert g.risk_multiplier(8900) == 0.25  # plus an 11% drawdown
    assert not g.allows_entry("SOL", t + 8 * hour, 7900, hour)[0]
    g.record_exit("SOL", t + 9 * hour, 100, "take_profit")
    assert g.risk_multiplier(10000) == 1.0


# -- config / safety ----------------------------------------------------------------

def test_live_mode_needs_explicit_confirmation(monkeypatch, tmp_path):
    raw = {"mode": "live", "data_dir": str(tmp_path), "symbols": [{"symbol": "BTC-USD"}]}
    monkeypatch.delenv(LIVE_CONFIRM_ENV, raising=False)
    with pytest.raises(RuntimeError, match="Live mode"):
        Engine(config_from_dict(raw), brokers={}, market_data=object(), context=False)
    monkeypatch.setenv(LIVE_CONFIRM_ENV, LIVE_CONFIRM_VALUE)
    Engine(config_from_dict(raw), brokers={}, market_data=object(), context=False)


def test_config_validation_and_tuned_params(tmp_path):
    with pytest.raises(ValueError, match="market"):
        SymbolConfig("X", market="forex")
    with pytest.raises(ValueError, match="missing"):
        config_from_dict({"symbols": [{"symbol": "X", "broker": "alpaca"}]})
    path = tmp_path / "bot.json"
    path.write_text(json.dumps({"symbols": [{"symbol": "BTC/USDT"}]}))
    path.with_suffix(".tuned.json").write_text(json.dumps({"risk": {"atr_stop_mult": 3.5},
                                                            "signals": {"buy_threshold": 0.45}}))
    cfg = load_config(path)
    assert cfg.risk.atr_stop_mult == 3.5 and cfg.signals.buy_threshold == 0.45
    assert cfg.feeds["crypto"]["type"] == "crypto" and cfg.costs["india"] == 0.0012


def test_paper_mode_never_builds_a_real_money_broker(tmp_path, monkeypatch):
    with pytest.raises(BrokerError, match="no paper endpoint"):
        build_broker("india", {"type": "openalgo"}, tmp_path, live=False)
    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    alpaca = build_broker("us", {"type": "alpaca", "live": True}, tmp_path, live=False)
    assert "paper-api" in alpaca.base_url


# -- brokers ------------------------------------------------------------------------

PROT = Protection(entry=100, stop=90, take=120, atr=5, highest=100)


def test_paper_broker_round_trip(tmp_path):
    b = PaperBroker("paper", tmp_path / "p.json", starting_cash=10000, fee_rate=0, slippage=0)
    sym = SymbolConfig("BTC-USD")
    b.buy(sym, 10, 100, PROT)
    b.mark(sym, 110)
    assert b.equity() == pytest.approx(10100)
    assert b.close(sym, 110)["pnl"] == pytest.approx(100)
    assert PaperBroker("paper", tmp_path / "p.json").cash() == pytest.approx(10100)


def test_alpaca_sends_bracket_order(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    s = FakeSession({("GET", "/v2/positions/SPY"): ({"message": "not found"}, 404),
                     ("GET", "/v2/orders"): [{"id": "o1"}]})
    b = AlpacaBroker("us", session=s)
    spy = SymbolConfig("SPY", market="us")
    assert b.position_qty(spy) == 0
    b.buy(spy, 3, 100, PROT)
    order = s.calls[-1][2]["json"]
    assert order["order_class"] == "bracket" and order["stop_loss"]["stop_price"] == "90.00"
    b.close(spy, 100)
    assert ("DELETE", "https://paper-api.alpaca.markets/v2/orders/o1") in [(m, u) for m, u, _ in s.calls]


def test_openalgo_places_nse_order(monkeypatch):
    monkeypatch.setenv("OPENALGO_API_KEY", "key")
    s = FakeSession({("POST", "/api/v1/openposition"): {"status": "success", "quantity": "7"}})
    b = OpenAlgoBroker("india", session=s)
    sym = SymbolConfig("RELIANCE", market="india", exchange="NSE", product="CNC")
    b.buy(sym, 5, 2500, PROT)
    assert s.calls[-1][2]["json"] == {"apikey": "key", "strategy": "AutoTrader", "symbol": "RELIANCE",
                                      "exchange": "NSE", "product": "CNC", "action": "BUY",
                                      "pricetype": "MARKET", "quantity": "5"}
    b.close(sym, 2500)
    assert s.calls[-1][2]["json"]["action"] == "SELL" and s.calls[-1][2]["json"]["quantity"] == "7"


def test_freqtrade_force_enter_and_exit(monkeypatch):
    monkeypatch.setenv("FREQTRADE_USERNAME", "u")
    monkeypatch.setenv("FREQTRADE_PASSWORD", "p")
    s = FakeSession({("GET", "/api/v1/status"): [{"pair": "BTC/USDT", "trade_id": 4, "amount": 0.1}]})
    b = FreqtradeBroker("crypto", session=s)
    sym = SymbolConfig("BTC/USDT")
    assert b.position_qty(sym) == pytest.approx(0.1)
    b.buy(sym, 0.05, 60000, PROT)
    assert s.calls[-1][2]["json"]["stakeamount"] == 3000.0
    b.close(sym, 60000)
    assert s.calls[-1][2]["json"] == {"tradeid": "4"}


class FakeGraph:
    def __init__(self, rating):
        self.rating, self.calls = rating, []

    def propagate(self, ticker, date, asset_type="stock"):
        self.calls.append((ticker, date, asset_type))
        return {}, self.rating


def test_llm_advisor_caches_and_maps_ratings(tmp_path):
    graph = FakeGraph("Overweight")
    adv = LLMAdvisor(tmp_path, refresh_hours=24, graph_factory=lambda: graph)
    now = datetime(2026, 10, 16, tzinfo=UTC)
    assert adv.opinion("BTC/USDT", "crypto", now).score == 0.5
    adv.opinion("BTC/USDT", "crypto", now + timedelta(hours=1))
    assert graph.calls == [("BTC-USD", "2026-10-16", "crypto")]
    adv.opinion("BTC/USDT", "crypto", now + timedelta(hours=25))
    assert len(graph.calls) == 2
    assert LLMAdvisor(tmp_path, graph_factory=lambda: FakeGraph("REVIEW")).opinion("SPY", "us", now) is None
    assert ta_ticker("RELIANCE", "india") == "RELIANCE.NS"


# -- engine ------------------------------------------------------------------------

class FakeMarket:
    """MarketData stand-in: a candle frame and a live price per symbol."""

    def __init__(self, df, price=None, stale=False):
        self.df, self.price, self.stale = df, price, stale

    def candles(self, symbol, market, interval, lookback=None):
        if self.stale:
            raise StaleData("no fresh candles")
        return self.df

    def quote(self, symbol, market):
        return Quote(self.price if self.price is not None else float(self.df["close"].iloc[-1]), 99.99, 100.01)


def engine_for(tmp_path, market, **signals):
    cfg = config_from_dict({
        "data_dir": str(tmp_path), "interval": "1h",
        "symbols": [{"symbol": "BTC/USDT", "market": "crypto"}],
        "signals": {"technical_weight": 1.0, "forecast_weight": 0.0, "llm_weight": 0.0,
                    "context_weight": 0.0, "trend_interval": "1h", **signals},
        "risk": {"max_spread_pct": 0.01},
        "brokers": {"paper": {"starting_cash": 10000, "fee_rate": 0, "slippage": 0}},
    })
    return Engine(cfg, market_data=market, context=False)


def journal(tmp_path):
    return [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text().splitlines()]


def test_engine_buys_once_per_candle_then_stops_out_on_live_price(tmp_path):
    market = FakeMarket(trending_df(up=True))
    eng = engine_for(tmp_path, market)
    sym, broker = eng.cfg.symbols[0], eng.brokers["paper"]
    eng.step()
    qty = broker.position_qty(sym)
    assert qty > 0
    eng.step()  # same candle: no second entry, no re-decision
    assert broker.position_qty(sym) == qty
    assert sum(1 for e in journal(tmp_path) if e["event"] == "decision") == 1
    stop = eng.state["protections"]["BTC/USDT"]["stop"]
    market.price = stop * 0.99  # live price falls through the stop between candles
    eng.risk_tick()
    assert broker.position_qty(sym) == 0
    exits = [e for e in journal(tmp_path) if e["event"] == "exit"]
    assert exits[0]["reason"] == "stop_loss" and exits[0]["r"] < 0
    assert eng.guards["paper"].state.loss_streak == 1


def test_engine_signal_sell_exits_and_downtrend_does_not_buy(tmp_path):
    market = FakeMarket(trending_df(up=True))
    eng = engine_for(tmp_path, market)
    eng.step()
    eng.state["protections"]["BTC/USDT"]["stop"] = 0.0
    eng.state["protections"]["BTC/USDT"]["initial_stop"] = 0.0
    market.df = trending_df(up=False, end=market.df.index[-1] + timedelta(hours=1))
    eng.step()
    assert eng.brokers["paper"].position_qty(eng.cfg.symbols[0]) == 0
    assert any(e.get("reason") == "signal_sell" for e in journal(tmp_path))


def test_engine_trend_filter_blocks_buys(tmp_path):
    # Hourly chart says buy, but the daily chart is in a downtrend: no new long.
    class TwoTimeframes(FakeMarket):
        def candles(self, symbol, market, interval, lookback=None):
            return trending_df(n=400, up=False, freq="D") if interval == "1d" else self.df

    eng = engine_for(tmp_path, TwoTimeframes(trending_df(up=True)), trend_interval="1d")
    d = eng.step()["BTC/USDT"]
    assert d.score > eng.cfg.signals.buy_threshold
    assert d.action == HOLD and eng.brokers["paper"].position_qty(eng.cfg.symbols[0]) == 0
    assert any("not in an uptrend" in n for n in d.notes)


def test_engine_skips_stale_data_and_closed_markets(tmp_path):
    eng = engine_for(tmp_path, FakeMarket(trending_df(), stale=True))
    assert eng.step() == {}
    assert journal(tmp_path)[-1]["event"] == "stale_data"
    cfg = config_from_dict({"data_dir": str(tmp_path), "symbols": [{"symbol": "SPY", "market": "us"}]})
    eng = Engine(cfg, market_data=FakeMarket(trending_df(), stale=True), context=False)
    assert eng.step(datetime(2026, 10, 17, 15, tzinfo=UTC)) == {}


def test_engine_blocks_wide_spread(tmp_path):
    class WideSpread(FakeMarket):
        def quote(self, symbol, market):
            return Quote(100, 98, 102)

    eng = engine_for(tmp_path, WideSpread(trending_df(up=True)))
    eng.step()
    assert eng.brokers["paper"].position_qty(eng.cfg.symbols[0]) == 0
    assert "spread" in journal(tmp_path)[-1]["reason"]


# -- backtest / optimizer ---------------------------------------------------------------

def test_backtest_stats_costs_and_no_overlap():
    df = make_df(n=800, drift=0.002, seed=3)
    sig = SignalConfig(forecast_neighbors=30, trend_interval="1d")
    free = run_backtest(df, sig_cfg=sig, cost=0.0)
    costly = run_backtest(df, sig_cfg=sig, cost=0.005)
    s = free.stats
    assert {"expectancy_r", "sqn", "max_losing_streak", "avg_win_r"} <= set(s)
    assert s["max_drawdown"] <= 0 and s["trades"] > 0
    assert costly.stats["total_return"] < s["total_return"]
    for a, b in zip(free.trades, free.trades[1:], strict=False):
        assert b["entry_time"] >= a["exit_time"]


def test_backtest_gap_through_stop_fills_at_open():
    n = 300
    close = np.r_[np.linspace(100, 160, n - 1), 80.0]
    idx = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    df = pd.DataFrame({"open": np.r_[close[0], close[:-1]], "close": close, "volume": 1.0}, index=idx)
    df.iloc[-1, df.columns.get_loc("open")] = 80.0  # gap down
    df["high"], df["low"] = df[["open", "close"]].max(axis=1) * 1.001, df[["open", "close"]].min(axis=1) * 0.999
    sig = SignalConfig(forecast_weight=0.0, trend_filter=False, buy_threshold=0.0, trend_interval="1d")
    res = run_backtest(df, RiskConfig(reward_risk=50, trail_after_atr=0), sig, cost=0.0)
    last = res.trades[-1]
    assert last["reason"] == "stop_loss" and last["exit"] == pytest.approx(80.0)


def test_walk_forward_reports_out_of_sample():
    df = make_df(n=900, drift=0.0015, seed=5)
    sig = SignalConfig(forecast_weight=0.0, trend_interval="1d")
    p = prepare(df, sig, "1d", "crypto")
    grid = {"signals.buy_threshold": [0.2, 0.4], "risk.atr_stop_mult": [1.5, 3.0]}
    rep = walk_forward([(p, 0.001)], RiskConfig(), sig, grid, folds=3, min_trades=2)
    assert rep["combos_tested"] == 4 and len(rep["folds"]) == 3
    assert rep["out_of_sample"][0]["trades"] >= 0
    # Out-of-sample trades come only from the test blocks.
    first_test_start = df.index[int(len(df) / 4)]
    res = simulate(p, RiskConfig(), sig, start=int(len(df) / 4))
    assert all(t["entry_time"] >= first_test_start for t in res.trades)
