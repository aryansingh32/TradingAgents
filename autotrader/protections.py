"""Trading discipline rules, applied before every new entry.

They encode what experienced traders do to survive losing streaks, the part
of trading that decides long-run results more than entry signals:

* cooldown: no re-entry into a symbol for a few bars after an exit
  (prevents revenge trading and whipsaw churn);
* stop-loss guard: after several stop-outs in a short window, pause all
  entries (the market regime has likely changed);
* drawdown guard: halve risk when equity is well below its peak, stop new
  entries when the drawdown deepens (protects capital to fight another day);
* losing-streak guard: halve risk after consecutive losses until a win
  (anti-martingale: bet less when wrong, not more).

The same class drives the live engine and the backtester.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from autotrader.config import RiskConfig


@dataclass
class ProtectionState:
    peak_equity: float = 0.0
    loss_streak: int = 0
    last_exit: dict[str, str] = field(default_factory=dict)  # symbol -> iso time
    stopouts: list[str] = field(default_factory=list)  # iso times
    paused_until: str | None = None

    @classmethod
    def from_dict(cls, raw: dict | None) -> ProtectionState:
        return cls(**raw) if raw else cls()


class Protections:
    def __init__(self, cfg: RiskConfig, state: ProtectionState | None = None):
        self.cfg = cfg
        self.state = state or ProtectionState()

    def record_equity(self, equity: float) -> None:
        self.state.peak_equity = max(self.state.peak_equity, equity)

    def record_exit(self, symbol: str, at: datetime, pnl: float, reason: str) -> None:
        s = self.state
        s.last_exit[symbol] = at.isoformat()
        s.loss_streak = s.loss_streak + 1 if pnl <= 0 else 0
        if reason == "stop_loss" and pnl <= 0:
            s.stopouts.append(at.isoformat())
            window = at - timedelta(hours=self.cfg.stoploss_guard_hours)
            s.stopouts = [t for t in s.stopouts if datetime.fromisoformat(t) >= window]
            if self.cfg.stoploss_guard_count and len(s.stopouts) >= self.cfg.stoploss_guard_count:
                s.paused_until = (at + timedelta(hours=self.cfg.stoploss_guard_pause_hours)).isoformat()
                s.stopouts = []

    def allows_entry(self, symbol: str, at: datetime, equity: float, bar: timedelta) -> tuple[bool, str]:
        s = self.state
        if s.paused_until and at < datetime.fromisoformat(s.paused_until):
            return False, f"stop-loss guard: {self.cfg.stoploss_guard_count} stop-outs, paused until {s.paused_until[:16]}"
        last = s.last_exit.get(symbol)
        if last and self.cfg.cooldown_bars and at < datetime.fromisoformat(last) + bar * self.cfg.cooldown_bars:
            return False, f"cooldown: exited {symbol} less than {self.cfg.cooldown_bars} bars ago"
        dd = self.drawdown(equity)
        if self.cfg.drawdown_pause_pct and dd >= self.cfg.drawdown_pause_pct:
            return False, f"drawdown guard: equity {dd:.1%} below peak, entries paused"
        return True, ""

    def drawdown(self, equity: float) -> float:
        peak = max(self.state.peak_equity, equity)
        return 1 - equity / peak if peak > 0 else 0.0

    def risk_multiplier(self, equity: float) -> float:
        mult = 1.0
        if self.cfg.drawdown_reduce_pct and self.drawdown(equity) >= self.cfg.drawdown_reduce_pct:
            mult *= 0.5
        if self.cfg.loss_streak_reduce and self.state.loss_streak >= self.cfg.loss_streak_reduce:
            mult *= 0.5
        return mult
