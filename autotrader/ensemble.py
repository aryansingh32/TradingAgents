"""Combine opinions into one trading decision."""

from __future__ import annotations

from dataclasses import dataclass, field

from autotrader.config import SignalConfig
from autotrader.signals import Opinion

BUY, SELL, HOLD = "BUY", "SELL", "HOLD"


@dataclass
class Decision:
    action: str
    score: float
    opinions: list[Opinion] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def reasons(self) -> list[str]:
        return [f"[{o.name} {o.score:+.2f}] {r}" for o in self.opinions for r in o.reasons] + self.notes

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "score": round(self.score, 4),
            "opinions": {o.name: {"score": round(o.score, 4), "reasons": o.reasons, **o.details}
                         for o in self.opinions},
            "notes": self.notes,
        }


def weights_of(cfg: SignalConfig) -> dict[str, float]:
    return {"technical": cfg.technical_weight, "forecast": cfg.forecast_weight,
            "llm": cfg.llm_weight, "context": cfg.context_weight, "news": cfg.news_weight,
            "patterns": cfg.patterns_weight, "volume": cfg.volume_weight,
            "trend_strength": cfg.trend_strength_weight, "mean_reversion": cfg.mean_reversion_weight}


def combine_scores(scores: dict[str, float | None], cfg: SignalConfig) -> float | None:
    weights = weights_of(cfg)
    present = {k: v for k, v in scores.items() if v is not None and v == v and weights.get(k, 0) > 0}
    total = sum(weights[k] for k in present)
    if not present or total == 0:
        return None
    return sum(weights[k] * v for k, v in present.items()) / total


def combine(opinions: list[Opinion | None], cfg: SignalConfig, trend_up: bool | None = None,
            trend_note: str = "") -> Decision:
    present = [o for o in opinions if o is not None and weights_of(cfg).get(o.name, 0) > 0]
    score = combine_scores({o.name: o.score for o in present}, cfg)
    if score is None:
        return Decision(HOLD, 0.0, [])
    notes = [trend_note] if trend_note else []
    llm = next((o for o in present if o.name == "llm"), None)
    if score >= cfg.buy_threshold:
        action = BUY
        if llm and llm.score <= -1.0:
            # A Sell from the debate (news risk the charts can't see) vetoes new buys.
            action, notes = HOLD, notes + ["buy vetoed: TradingAgents debate rated Sell"]
        elif cfg.trend_filter and trend_up is False:
            action = HOLD
    elif score <= cfg.sell_threshold:
        action = SELL
    else:
        action = HOLD
    mr = next((o for o in present if o.name == "mean_reversion"), None)
    if action != BUY and mr and mr.score >= cfg.mean_reversion_trigger and trend_up is True \
            and not (llm and llm.score <= -1.0):
        action = BUY
        notes = notes + ["mean-reversion setup: buying a dip inside a confirmed uptrend"]
    return Decision(action, score, present, notes)
