"""The trading loop: data -> opinions -> decision -> risk -> orders.

Every decision and order is appended to ``<data_dir>/journal.jsonl`` and the
latest decision per symbol is written to ``<data_dir>/signals.json`` (which
the freqtrade strategy in ``autotrader/integrations`` can read).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from autotrader import indicators as ind
from autotrader.brokers import Broker, build_broker
from autotrader.config import BotConfig, SymbolConfig
from autotrader.data import load_history
from autotrader.ensemble import BUY, SELL, Decision, combine
from autotrader.llm_advisor import LLMAdvisor
from autotrader.risk import Protection, RiskManager, market_open
from autotrader.signals import forecast_opinion, technical_opinion

log = logging.getLogger(__name__)


class Engine:
    def __init__(self, cfg: BotConfig, brokers: dict[str, Broker] | None = None,
                 data_loader: Callable[[SymbolConfig], pd.DataFrame] | None = None,
                 llm: LLMAdvisor | None = None):
        cfg.require_live_confirmation()
        self.cfg = cfg
        self.dir: Path = cfg.data_path
        live = cfg.mode == "live"
        self.brokers = brokers or {
            name: build_broker(name, section, self.dir, live)
            for name, section in cfg.brokers.items()
            if name in {s.broker for s in cfg.symbols}
        }
        self.data_loader = data_loader or (lambda s: load_history(
            s.price_symbol, cfg.data_source, cfg.interval, cfg.lookback_days, cfg.data_dir))
        self.llm = llm
        if self.llm is None and cfg.signals.use_llm:
            self.llm = LLMAdvisor(self.dir, cfg.signals.llm_refresh_hours, cfg.signals.llm_analysts)
        self.risk = RiskManager(cfg.risk)
        self.state_file = self.dir / "bot_state.json"
        self.state = self._load_json(self.state_file, {"protections": {}, "day": None, "day_start_equity": {}})

    # -- persistence ---------------------------------------------------
    @staticmethod
    def _load_json(path: Path, default: dict) -> dict:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default

    def _save_state(self) -> None:
        self.state_file.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

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
            forecast_opinion(df, sc.forecast_horizon, sc.forecast_neighbors) if sc.forecast_weight > 0 else None,
        ]
        if self.llm is not None and sc.llm_weight > 0:
            opinions.append(self.llm.opinion(sym.symbol, sym.market, now))
        return combine(opinions, sc)

    def _roll_day(self, now: datetime) -> None:
        today = now.date().isoformat()
        if self.state["day"] != today:
            self.state["day"] = today
            self.state["day_start_equity"] = {}
        for name, broker in self.brokers.items():
            if name not in self.state["day_start_equity"]:
                try:
                    self.state["day_start_equity"][name] = broker.equity()
                except Exception as exc:  # retried next loop; entries use current equity meanwhile
                    self.journal("error", broker=name, error=repr(exc))

    # -- one pass over all symbols --------------------------------------
    def step(self, now: datetime | None = None) -> dict[str, Decision]:
        now = now or datetime.now(UTC)
        self._roll_day(now)
        decisions = {}
        for sym in self.cfg.symbols:
            try:
                decision = self._step_symbol(sym, now)
                if decision is not None:
                    decisions[sym.symbol] = decision
            except Exception as exc:  # one bad symbol or broker must not stop the rest
                log.exception("step failed for %s", sym.symbol)
                self.journal("error", symbol=sym.symbol, error=repr(exc))
        self._save_state()
        return decisions

    def _step_symbol(self, sym: SymbolConfig, now: datetime) -> Decision | None:
        if not market_open(sym.market, now):
            return None
        broker = self.brokers[sym.broker]
        df = self.data_loader(sym)
        if df.empty:
            self.journal("no_data", symbol=sym.symbol)
            return None
        price = float(df["close"].iloc[-1])
        atr_now = float(ind.atr(df).iloc[-1])
        broker.mark(sym, price)

        decision = self.decide(sym, df, now)
        self._write_signal(sym, price, decision, now)
        self.journal("decision", symbol=sym.symbol, price=price, action=decision.action,
                     score=round(decision.score, 3), decision=decision.to_dict())

        prots = self.state["protections"]
        qty = broker.position_qty(sym)
        prot_raw = prots.get(sym.symbol)

        if qty > 0:
            if prot_raw is None:
                # A position the bot did not open (or lost track of): protect it too.
                prot = self.risk.levels(price, atr_now)
                self.journal("adopted_position", symbol=sym.symbol, qty=qty, protection=prot.to_dict())
            else:
                prot = Protection(**prot_raw)
            prot = self.risk.update_trailing(prot, price)
            hit = self.risk.exit_reason(prot, price, price)
            if hit or decision.action == SELL:
                reason = hit[0] if hit else "signal_sell"
                result = broker.close(sym, price)
                prots.pop(sym.symbol, None)
                self.journal("exit", symbol=sym.symbol, reason=reason, qty=qty, price=price, result=result)
            else:
                prots[sym.symbol] = prot.to_dict()
            return decision

        if prot_raw is not None:
            # Flat at the broker but we had levels: its native stop/take fired.
            prots.pop(sym.symbol)
            self.journal("closed_at_broker", symbol=sym.symbol, protection=prot_raw)

        if decision.action == BUY:
            equity = broker.equity()
            open_here = sum(1 for s in self.cfg.symbols if s.broker == sym.broker and s.symbol in prots)
            ok, why = self.risk.can_open(open_here, self.state["day_start_equity"].get(sym.broker, equity), equity)
            if not ok:
                self.journal("entry_blocked", symbol=sym.symbol, reason=why)
                return decision
            plan = self.risk.plan_entry(price, atr_now, equity, broker.cash(), sym.market)
            if plan is None:
                self.journal("entry_blocked", symbol=sym.symbol, reason="position size rounds to zero")
                return decision
            result = broker.buy(sym, plan.qty, price, plan.protection)
            prots[sym.symbol] = plan.protection.to_dict()
            self.journal("entry", symbol=sym.symbol, qty=plan.qty, price=price,
                         protection=plan.protection.to_dict(), result=result)
        return decision

    def run_forever(self) -> None:
        log.info("AutoTrader running in %s mode on %d symbols, every %ss",
                 self.cfg.mode, len(self.cfg.symbols), self.cfg.loop_seconds)
        while True:
            started = time.monotonic()
            try:
                self.step()
            except Exception:
                log.exception("loop failed; retrying next interval")
            time.sleep(max(5.0, self.cfg.loop_seconds - (time.monotonic() - started)))
