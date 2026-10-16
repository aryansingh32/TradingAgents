"""The trading loop.

Two cadences:

* ``step`` (every ``loop_seconds``): when a symbol has a new closed candle,
  recompute its opinions and decision, then act on it (enter, exit, hold);
* ``risk_tick`` (every ``risk_loop_seconds``): read the live price of every
  open position and enforce its stop loss, trailing stop and take profit,
  so protection does not wait for the next candle.

Every decision and order is appended to ``<data_dir>/journal.jsonl``; the
latest decision per symbol is written to ``<data_dir>/signals.json`` (read by
the freqtrade strategy ``AutoTraderSignalStrategy``).
"""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from autotrader.brokers import Broker, build_broker
from autotrader.config import BotConfig, SymbolConfig
from autotrader.ensemble import BUY, SELL, Decision, combine
from autotrader.feeds import FeedError, MarketData, StaleData, build_feed, interval_delta
from autotrader.llm_advisor import LLMAdvisor
from autotrader.protections import Protections, ProtectionState
from autotrader.risk import (
    Protection,
    RiskManager,
    entry_window_open,
    market_open,
    mis_squareoff_due,
    vol_multiplier,
)
from autotrader.signals import (
    bars_per_year,
    forecast_opinion,
    realized_vol,
    technical_opinion,
    trend_opinion,
)

log = logging.getLogger(__name__)


def build_market_data(cfg: BotConfig) -> MarketData:
    if cfg.data_source in ("yfinance", "csv"):
        spec = {"type": cfg.data_source, **({"data_dir": cfg.data_dir} if cfg.data_source == "csv" else {})}
        feeds = {m: build_feed(spec) for m in ("crypto", "us", "india")}
    else:
        feeds = {m: build_feed(cfg.feeds[m]) for m in {s.market for s in cfg.symbols}}
    max_age = math.inf if cfg.data_source == "csv" else cfg.max_data_age_bars
    return MarketData(feeds, cfg.lookback_days, max_age)


class Engine:
    def __init__(self, cfg: BotConfig, brokers: dict[str, Broker] | None = None,
                 market_data: MarketData | None = None, llm: LLMAdvisor | None = None,
                 context=None):
        cfg.require_live_confirmation()
        self.cfg = cfg
        self.dir: Path = cfg.data_path
        live = cfg.mode == "live"
        self.brokers = brokers if brokers is not None else {
            name: build_broker(name, section, self.dir, live)
            for name, section in cfg.brokers.items()
            if name in {s.broker for s in cfg.symbols}
        }
        self.md = market_data or build_market_data(cfg)
        self.llm = llm
        if self.llm is None and cfg.signals.use_llm:
            self.llm = LLMAdvisor(self.dir, cfg.signals.llm_refresh_hours, cfg.signals.llm_analysts)
        self.context = context
        if self.context is None and cfg.signals.use_context and cfg.signals.context_weight > 0 \
                and any(s.market == "crypto" for s in cfg.symbols):
            from autotrader.context import CryptoContext

            self.context = CryptoContext()
        self.risk = RiskManager(cfg.risk)
        self.step_delta = interval_delta(cfg.interval)
        self.state_file = self.dir / "bot_state.json"
        self.state = self._load_json(self.state_file, {})
        for key, default in (("protections", {}), ("day", None), ("day_start_equity", {}),
                             ("guards", {}), ("last_bar", {})):
            self.state.setdefault(key, default)
        self.guards = {name: Protections(cfg.risk, ProtectionState.from_dict(self.state["guards"].get(name)))
                       for name in self.brokers}
        self.decisions: dict[str, Decision] = {}
        self.atr: dict[str, float] = {}
        self.vol: dict[str, float] = {}

    # -- persistence ---------------------------------------------------
    @staticmethod
    def _load_json(path: Path, default: dict) -> dict:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return default

    def _save_state(self) -> None:
        self.state["guards"] = {n: g.state.__dict__ for n, g in self.guards.items()}
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        tmp.replace(self.state_file)

    def journal(self, event: str, **fields) -> None:
        record = {"at": datetime.now(UTC).isoformat(), "event": event, **fields}
        with (self.dir / "journal.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        log.info("%s %s", event, {k: v for k, v in fields.items() if k != "decision"})

    def _write_signal(self, sym: SymbolConfig, price: float, decision: Decision, now: datetime) -> None:
        path = self.dir / "signals.json"
        signals = self._load_json(path, {})
        signals[sym.symbol] = {"at": now.isoformat(), "price": price, **decision.to_dict()}
        path.write_text(json.dumps(signals, indent=2), encoding="utf-8")

    # -- decision ------------------------------------------------------
    def decide(self, sym: SymbolConfig, df: pd.DataFrame, now: datetime) -> Decision:
        sc = self.cfg.signals
        opinions = [
            technical_opinion(df) if sc.technical_weight > 0 else None,
            forecast_opinion(df, sc.forecast_horizon, sc.forecast_neighbors, window=sc.forecast_window)
            if sc.forecast_weight > 0 else None,
        ]
        if self.llm is not None and sc.llm_weight > 0:
            opinions.append(self.llm.opinion(sym.symbol, sym.market, now))
        if self.context is not None and sym.market == "crypto" and sc.context_weight > 0:
            day_ago = df["close"][df.index <= df.index[-1] - timedelta(hours=23)]
            change = float(df["close"].iloc[-1] / day_ago.iloc[-1] - 1) if len(day_ago) else None
            try:
                opinions.append(self.context.opinion(sym.price_symbol, change))
            except Exception as exc:
                log.info("context unavailable for %s: %s", sym.symbol, exc)
        trend_up, note = None, ""
        if sc.trend_filter:
            htf = df if sc.trend_interval == self.cfg.interval else self.md.candles(
                sym.price_symbol, sym.market, sc.trend_interval,
                lookback=timedelta(days=max(self.cfg.lookback_days, int(sc.trend_ema * 1.6))))
            trend_up, note = trend_opinion(htf, sc.trend_ema)
        return combine(opinions, sc, trend_up, note)

    def _roll_day(self, now: datetime) -> None:
        today = now.date().isoformat()
        if self.state["day"] != today:
            self.state["day"] = today
            self.state["day_start_equity"] = {}
        for name, broker in self.brokers.items():
            if name not in self.state["day_start_equity"]:
                try:
                    self.state["day_start_equity"][name] = equity = broker.equity()
                    self.guards[name].record_equity(equity)
                except Exception as exc:  # retried next loop
                    self.journal("error", broker=name, error=repr(exc))

    def _price(self, sym: SymbolConfig, fallback: float | None):
        try:
            q = self.md.quote(sym.price_symbol, sym.market)
            return q.price, q
        except FeedError as exc:
            if fallback is None:
                raise
            log.info("no live quote for %s (%s); using last close", sym.symbol, exc)
            return fallback, None

    # -- exits -----------------------------------------------------------
    def _exit(self, sym: SymbolConfig, broker: Broker, qty: float, price: float, reason: str, now: datetime):
        prot = self.state["protections"].pop(sym.symbol, None)
        result = broker.close(sym, price)
        pnl = result.get("pnl") if isinstance(result, dict) else None
        if pnl is None and prot:
            pnl = (price - prot["entry"]) * qty
        r = None
        if prot:
            risk_unit = prot["entry"] - (prot.get("initial_stop") or prot["stop"])
            r = (price - prot["entry"]) / risk_unit if risk_unit > 0 else None
        self.guards[sym.broker].record_exit(sym.symbol, now, pnl or 0.0, reason)
        self.journal("exit", symbol=sym.symbol, reason=reason, qty=qty, price=price, pnl=pnl, r=r, result=result)

    def _manage_open(self, sym: SymbolConfig, broker: Broker, qty: float, price: float,
                     now: datetime, decision: Decision | None) -> None:
        prots = self.state["protections"]
        raw = prots.get(sym.symbol)
        if raw is None:
            atr = self.atr.get(sym.symbol)
            if not atr:
                return  # adopted on the next candle, once ATR is known
            prot = self.risk.levels(price, atr)
            prot.opened_at = now.isoformat()
            self.journal("adopted_position", symbol=sym.symbol, qty=qty, protection=prot.to_dict())
        else:
            prot = Protection(**raw)
        self.risk.update_trailing(prot, price)
        prots[sym.symbol] = prot.to_dict()
        hit = self.risk.exit_reason(prot, price, price)
        if hit:
            self._exit(sym, broker, qty, price, hit[0], now)
        elif decision is not None and decision.action == SELL:
            self._exit(sym, broker, qty, price, "signal_sell", now)
        elif sym.market == "india" and (sym.product or "").upper() == "MIS" \
                and mis_squareoff_due(now, self.cfg.session):
            self._exit(sym, broker, qty, price, "intraday_squareoff", now)

    # -- fast loop: protection on live prices -----------------------------
    def risk_tick(self, now: datetime | None = None) -> None:
        now = now or datetime.now(UTC)
        for sym in self.cfg.symbols:
            if sym.symbol not in self.state["protections"] or not market_open(sym.market, now):
                continue
            try:
                broker = self.brokers[sym.broker]
                price, _ = self._price(sym, None)
                broker.mark(sym, price)
                qty = broker.position_qty(sym)
                if qty <= 0:
                    self.state["protections"].pop(sym.symbol, None)
                    self.journal("closed_at_broker", symbol=sym.symbol)
                    continue
                self._manage_open(sym, broker, qty, price, now, None)
            except Exception as exc:
                log.warning("risk tick failed for %s: %s", sym.symbol, exc)
        self._save_state()

    # -- slow loop: decisions on closed candles ----------------------------
    def step(self, now: datetime | None = None) -> dict[str, Decision]:
        now = now or datetime.now(UTC)
        self._roll_day(now)
        out = {}
        for sym in self.cfg.symbols:
            try:
                d = self._step_symbol(sym, now)
                if d is not None:
                    out[sym.symbol] = d
            except StaleData as exc:
                self.journal("stale_data", symbol=sym.symbol, error=str(exc))
            except Exception as exc:  # one bad symbol or broker must not stop the rest
                log.exception("step failed for %s", sym.symbol)
                self.journal("error", symbol=sym.symbol, error=repr(exc))
        self._save_state()
        return out

    def _step_symbol(self, sym: SymbolConfig, now: datetime) -> Decision | None:
        if not market_open(sym.market, now):
            return None
        broker = self.brokers[sym.broker]
        df = self.md.candles(sym.price_symbol, sym.market, self.cfg.interval)
        last_bar = df.index[-1].isoformat()
        new_bar = self.state["last_bar"].get(sym.symbol) != last_bar or sym.symbol not in self.decisions
        if new_bar:
            decision = self.decide(sym, df, now)
            self.decisions[sym.symbol] = decision
            self.state["last_bar"][sym.symbol] = last_bar
            tech = next((o for o in decision.opinions if o.name == "technical"), None)
            if tech:
                self.atr[sym.symbol] = tech.details["atr"]
            self.vol[sym.symbol] = realized_vol(df, bars_per_year(self.cfg.interval, sym.market))
            self._write_signal(sym, float(df["close"].iloc[-1]), decision, now)
            self.journal("decision", symbol=sym.symbol, bar=last_bar, action=decision.action,
                         score=round(decision.score, 3), decision=decision.to_dict())
        decision = self.decisions[sym.symbol]

        price, quote = self._price(sym, float(df["close"].iloc[-1]))
        broker.mark(sym, price)
        qty = broker.position_qty(sym)
        if qty > 0:
            # A SELL is acted on once, on the candle that produced it.
            self._manage_open(sym, broker, qty, price, now, decision if new_bar else None)
            return decision
        if sym.symbol in self.state["protections"]:
            self.state["protections"].pop(sym.symbol)
            self.journal("closed_at_broker", symbol=sym.symbol)
        if decision.action == BUY and new_bar:
            self._try_entry(sym, broker, price, quote, now)
        return decision

    def _try_entry(self, sym: SymbolConfig, broker: Broker, price: float, quote, now: datetime) -> None:
        def blocked(reason):
            self.journal("entry_blocked", symbol=sym.symbol, reason=reason)

        if not entry_window_open(sym.market, now, self.cfg.session):
            return blocked("outside the entry window (session open/close buffer)")
        if quote is not None and quote.spread_pct is not None and quote.spread_pct > self.cfg.risk.max_spread_pct:
            return blocked(f"spread {quote.spread_pct:.3%} wider than {self.cfg.risk.max_spread_pct:.3%}")
        equity = broker.equity()
        guards = self.guards[sym.broker]
        guards.record_equity(equity)
        prots = self.state["protections"]
        open_here = sum(1 for s in self.cfg.symbols if s.broker == sym.broker and s.symbol in prots)
        ok, why = self.risk.can_open(open_here, self.state["day_start_equity"].get(sym.broker, equity), equity)
        if ok:
            ok, why = guards.allows_entry(sym.symbol, now, equity, self.step_delta)
        if not ok:
            return blocked(why)
        mult = guards.risk_multiplier(equity) * vol_multiplier(self.cfg.risk, self.vol.get(sym.symbol, float("nan")))
        plan = self.risk.plan_entry(price, self.atr.get(sym.symbol, 0.0), equity, broker.cash(), sym.market, mult)
        if plan is None:
            return blocked("position size rounds to zero")
        plan.protection.opened_at = now.isoformat()
        result = broker.buy(sym, plan.qty, price, plan.protection)
        prots[sym.symbol] = plan.protection.to_dict()
        self.journal("entry", symbol=sym.symbol, qty=plan.qty, price=price, size_mult=round(mult, 3),
                     protection=plan.protection.to_dict(), result=result)

    def run_forever(self) -> None:
        log.info("AutoTrader running in %s mode on %d symbols: decisions every %ss, stops every %ss",
                 self.cfg.mode, len(self.cfg.symbols), self.cfg.loop_seconds, self.cfg.risk_loop_seconds)
        next_step = 0.0
        while True:
            started = time.monotonic()
            try:
                if started >= next_step:
                    self.step()
                    next_step = started + self.cfg.loop_seconds
                else:
                    self.risk_tick()
            except Exception:
                log.exception("loop failed; retrying")
            time.sleep(max(1.0, self.cfg.risk_loop_seconds - (time.monotonic() - started)))
