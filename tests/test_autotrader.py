"""Tests for the autotrader bot: signals, risk, brokers, engine and backtest."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from autotrader import indicators as ind
from autotrader.backtest import run_backtest
from autotrader.brokers import BrokerError, build_broker
from autotrader.brokers.alpaca import AlpacaBroker
from autotrader.brokers.freqtrade import FreqtradeBroker
from autotrader.brokers.openalgo import OpenAlgoBroker
from autotrader.brokers.paper import PaperBroker
from autotrader.config import (
    LIVE_CONFIRM_ENV,
    LIVE_CONFIRM_VALUE,
    RiskConfig,
    SignalConfig,
    SymbolConfig,
    config_from_dict,
)
from autotrader.engine import Engine
from autotrader.ensemble import BUY, HOLD, SELL, combine
from autotrader.llm_advisor import LLMAdvisor, ta_ticker
from autotrader.risk import Protection, RiskManager, market_open
from autotrader.signals import Opinion, analog_features, forecast_opinion, technical_opinion

pytestmark = pytest.mark.unit


def make_df(n=600, drift=0.001, vol=0.02, seed=0, start=100.0):
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(drift + rng.normal(0, vol, n)))
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    df = pd.DataFrame({"open": np.r_[close[0], close[:-1]], "close": close}, index=idx)
    df["high"] = df[["open", "close"]].max(axis=1) * 1.005
    df["low"] = df[["open", "close"]].min(axis=1) * 0.995
    df["volume"] = 1000.0
    return df


def trending_df(n=400, up=True):
    close = 100 * np.exp(np.linspace(0, 1.0 if up else -1.0, n)) * (1 + 0.01 * np.sin(np.arange(n)))
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    df = pd.DataFrame({"open": close, "close": close, "high": close * 1.01, "low": close * 0.99,
                       "volume": 1.0}, index=idx)
    return df


# -- indicators / signals ---------------------------------------------------

def test_indicators_ranges():
    df = make_df()
    r = ind.rsi(df["close"]).dropna()
    assert ((r >= 0) & (r <= 100)).all()
    assert (ind.atr(df).dropna() > 0).all()


def test_technical_reads_trend_direction():
    assert technical_opinion(trending_df(up=True)).score > 0.3
    assert technical_opinion(trending_df(up=False)).score < -0.3
    assert technical_opinion(make_df(n=30)) is None


def test_forecast_has_no_look_ahead():
    """Precomputed features over the full history give the same forecast as
    computing on the truncated frame, so later bars never leak in."""
    df = make_df(n=700)
    feats = analog_features(df)
    cut = df.iloc[:500]
    a = forecast_opinion(cut, horizon=10, neighbors=40)
    b = forecast_opinion(cut, horizon=10, neighbors=40, features=feats)
    assert a is not None
    assert a.score == pytest.approx(b.score)
    assert 0 <= a.details["prob_up"] <= 1
    # Changing the future must not change today's forecast.
    future_changed = df.copy()
    future_changed.iloc[500:, :4] *= 3
    c = forecast_opinion(cut, 10, 40, analog_features(future_changed))
    assert c.score == pytest.approx(a.score)


def test_combine_weights_and_llm_veto():
    cfg = SignalConfig(technical_weight=0.5, forecast_weight=0.5, llm_weight=0.0)
    d = combine([Opinion("technical", 0.8), Opinion("forecast", 0.4), Opinion("llm", -1)], cfg)
    assert d.score == pytest.approx(0.6) and d.action == BUY
    cfg = SignalConfig(technical_weight=0.4, forecast_weight=0.3, llm_weight=0.3, buy_threshold=0.2)
    vetoed = combine([Opinion("technical", 1), Opinion("forecast", 1), Opinion("llm", -1)], cfg)
    assert vetoed.score == pytest.approx(0.4) and vetoed.action == HOLD
    assert combine([None, Opinion("forecast", -0.9)], cfg).action == SELL
    assert combine([None, None], cfg).action == HOLD


# -- risk ---------------------------------------------------------------------

def test_position_sizing_risks_fixed_fraction():
    rm = RiskManager(RiskConfig(risk_per_trade=0.01, max_position_pct=1.0, atr_stop_mult=2, reward_risk=3))
    plan = rm.plan_entry(price=100, atr=5, equity=10000, cash=10000, market="us")
    # stop 10 below; 1% of 10k = 100 at risk -> 10 shares
    assert plan.qty == 10
    assert plan.protection.stop == 90 and plan.protection.take == 130
    capped = RiskManager(RiskConfig(max_position_pct=0.05)).plan_entry(100, 1, 10000, 10000, "us")
    assert capped.qty * 100 <= 500
    assert RiskManager(RiskConfig()).plan_entry(100, 5, 10000, 10000, "crypto").qty == pytest.approx(10.0)
    assert rm.plan_entry(100, 60, 10000, 10000, "us") is None  # stop below zero


def test_trailing_stop_only_rises_and_stop_wins_ties():
    rm = RiskManager(RiskConfig(atr_stop_mult=2, trail_after_atr=1))
    p = Protection(entry=100, stop=90, take=200, atr=5, highest=100)
    rm.update_trailing(p, 104)
    assert p.stop == 90  # not yet 1 ATR in profit
    rm.update_trailing(p, 120)
    assert p.stop == 110
    rm.update_trailing(p, 115)
    assert p.stop == 110
    assert rm.exit_reason(p, low=105, high=250) == ("stop_loss", 110)
    assert rm.exit_reason(p, low=111, high=250) == ("take_profit", 200)
    assert rm.exit_reason(p, low=111, high=150) is None


def test_daily_loss_limit_and_max_positions():
    rm = RiskManager(RiskConfig(max_daily_loss_pct=0.03, max_open_positions=2))
    assert rm.can_open(0, 10000, 9800)[0]
    assert not rm.can_open(0, 10000, 9700)[0]
    assert not rm.can_open(2, 10000, 10000)[0]


def test_market_hours():
    assert market_open("crypto", datetime(2026, 10, 17, 3, tzinfo=UTC))  # Saturday
    assert market_open("us", datetime(2026, 10, 16, 15, 0, tzinfo=UTC))  # 11:00 New York
    assert not market_open("us", datetime(2026, 10, 16, 21, 0, tzinfo=UTC))
    assert market_open("india", datetime(2026, 10, 16, 5, 0, tzinfo=UTC))  # 10:30 IST
    assert not market_open("india", datetime(2026, 10, 17, 5, 0, tzinfo=UTC))  # Saturday


# -- config / safety --------------------------------------------------------

def test_live_mode_needs_explicit_confirmation(monkeypatch, tmp_path):
    raw = {"mode": "live", "data_dir": str(tmp_path), "symbols": [{"symbol": "BTC-USD"}]}
    monkeypatch.delenv(LIVE_CONFIRM_ENV, raising=False)
    with pytest.raises(RuntimeError, match="Live mode"):
        Engine(config_from_dict(raw), brokers={}, data_loader=lambda s: make_df())
    monkeypatch.setenv(LIVE_CONFIRM_ENV, LIVE_CONFIRM_VALUE)
    Engine(config_from_dict(raw), brokers={}, data_loader=lambda s: make_df())


def test_config_validation():
    with pytest.raises(ValueError, match="market"):
        SymbolConfig("X", market="forex")
    with pytest.raises(ValueError, match="missing"):
        config_from_dict({"symbols": [{"symbol": "X", "broker": "alpaca"}]})
    with pytest.raises(ValueError, match="at least one"):
        config_from_dict({"symbols": []})


def test_paper_mode_never_builds_a_real_money_broker(tmp_path, monkeypatch):
    with pytest.raises(BrokerError, match="no paper endpoint"):
        build_broker("india", {"type": "openalgo"}, tmp_path, live=False)
    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    alpaca = build_broker("us", {"type": "alpaca", "live": True}, tmp_path, live=False)
    assert "paper-api" in alpaca.base_url


# -- brokers ------------------------------------------------------------------

class FakeResponse:
    def __init__(self, data, status=200):
        self._data, self.status_code = data, status
        self.content = b"x"
        self.text = str(data)

    def json(self):
        return self._data


class FakeSession:
    def __init__(self, routes):
        self.routes, self.calls, self.headers, self.auth = routes, [], {}, None

    def request(self, method, url, timeout=None, **kwargs):
        self.calls.append((method, url, kwargs))
        for (m, suffix), data in self.routes.items():
            if m == method and url.endswith(suffix):
                return FakeResponse(*data) if isinstance(data, tuple) else FakeResponse(data)
        return FakeResponse({"status": "success"})


PROT = Protection(entry=100, stop=90, take=120, atr=5, highest=100)


def test_paper_broker_round_trip(tmp_path):
    b = PaperBroker("paper", tmp_path / "p.json", starting_cash=10000, fee_rate=0, slippage=0)
    sym = SymbolConfig("BTC-USD")
    b.buy(sym, 10, 100, PROT)
    b.mark(sym, 110)
    assert b.equity() == pytest.approx(10100)
    assert b.close(sym, 110)["pnl"] == pytest.approx(100)
    assert PaperBroker("paper", tmp_path / "p.json").cash() == pytest.approx(10100)  # persisted


def test_alpaca_sends_bracket_order(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    s = FakeSession({("GET", "/v2/positions/SPY"): ({"message": "not found"}, 404),
                     ("GET", "/v2/orders"): [{"id": "o1"}]})
    b = AlpacaBroker("us", session=s)
    assert b.position_qty(SymbolConfig("SPY", market="us")) == 0
    b.buy(SymbolConfig("SPY", market="us"), 3, 100, PROT)
    order = s.calls[-1][2]["json"]
    assert order["order_class"] == "bracket" and order["stop_loss"]["stop_price"] == "90.00"
    b.close(SymbolConfig("SPY", market="us"), 100)
    assert ("DELETE", "https://paper-api.alpaca.markets/v2/orders/o1") in [(m, u) for m, u, _ in s.calls]


def test_openalgo_places_nse_order(monkeypatch):
    monkeypatch.setenv("OPENALGO_API_KEY", "key")
    s = FakeSession({("POST", "/api/v1/openposition"): {"status": "success", "quantity": "7"}})
    b = OpenAlgoBroker("india", session=s)
    sym = SymbolConfig("RELIANCE", market="india", exchange="NSE", product="CNC")
    b.buy(sym, 5, 2500, PROT)
    body = s.calls[-1][2]["json"]
    assert body == {"apikey": "key", "strategy": "AutoTrader", "symbol": "RELIANCE", "exchange": "NSE",
                    "product": "CNC", "action": "BUY", "pricetype": "MARKET", "quantity": "5"}
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


# -- LLM advisor --------------------------------------------------------------

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


# -- engine -------------------------------------------------------------------

def engine_for(tmp_path, frames, **signals):
    cfg = config_from_dict({
        "data_dir": str(tmp_path),
        "symbols": [{"symbol": "BTC-USD", "market": "crypto"}],
        "signals": {"technical_weight": 1.0, "forecast_weight": 0.0, "llm_weight": 0.0, **signals},
        "brokers": {"paper": {"starting_cash": 10000, "fee_rate": 0, "slippage": 0}},
    })
    return Engine(cfg, data_loader=lambda s: frames["df"])


def test_engine_buys_then_stops_out(tmp_path):
    frames = {"df": trending_df(up=True)}
    eng = engine_for(tmp_path, frames)
    sym = eng.cfg.symbols[0]
    broker = eng.brokers["paper"]
    eng.step()
    assert broker.position_qty(sym) > 0
    prot = eng.state["protections"]["BTC-USD"]
    # Next price is below the stop.
    crash = frames["df"].copy()
    crash.iloc[-1, crash.columns.get_loc("close")] = prot["stop"] * 0.98
    frames["df"] = crash
    eng.step()
    assert broker.position_qty(sym) == 0
    events = [line for line in (tmp_path / "journal.jsonl").read_text().splitlines() if '"exit"' in line]
    assert events and "stop_loss" in events[0]
    assert "BTC-USD" not in eng.state["protections"]


def test_engine_sell_signal_exits_and_downtrend_does_not_buy(tmp_path):
    frames = {"df": trending_df(up=True)}
    eng = engine_for(tmp_path, frames)
    eng.step()
    frames["df"] = trending_df(up=False) * 1.0
    # Keep the stop out of the way so the exit comes from the signal.
    eng.state["protections"]["BTC-USD"]["stop"] = 0.0
    eng.step()
    assert eng.brokers["paper"].position_qty(eng.cfg.symbols[0]) == 0
    assert '"signal_sell"' in (tmp_path / "journal.jsonl").read_text()
    eng.step()
    assert eng.brokers["paper"].position_qty(eng.cfg.symbols[0]) == 0


def test_engine_skips_closed_markets(tmp_path):
    cfg = config_from_dict({"data_dir": str(tmp_path), "symbols": [{"symbol": "SPY", "market": "us"}]})
    eng = Engine(cfg, data_loader=lambda s: pytest.fail("should not load data while closed"))
    assert eng.step(datetime(2026, 10, 17, 15, tzinfo=UTC)) == {}


# -- backtest -----------------------------------------------------------------

def test_backtest_runs_and_reports():
    df = make_df(n=500, drift=0.002, seed=3)
    res = run_backtest(df, sig_cfg=SignalConfig(forecast_neighbors=30), market="crypto")
    s = res.stats
    assert set(s) >= {"total_return", "max_drawdown", "sharpe", "trades", "win_rate", "buy_and_hold_return"}
    assert s["max_drawdown"] <= 0
    assert len(res.equity) == len(df) - 250
    for t in res.trades:
        assert t["exit_time"] >= t["entry_time"]
