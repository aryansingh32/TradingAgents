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

    @property
    def reasons(self) -> list[str]:
        return [f"[{o.name} {o.score:+.2f}] {r}" for o in self.opinions for r in o.reasons]

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "score": round(self.score, 4),
            "opinions": {o.name: {"score": round(o.score, 4), "reasons": o.reasons, **o.details}
                         for o in self.opinions},
        }


def combine(opinions: list[Opinion | None], cfg: SignalConfig) -> Decision:
    weights = {"technical": cfg.technical_weight, "forecast": cfg.forecast_weight, "llm": cfg.llm_weight}
    present = [o for o in opinions if o is not None and weights.get(o.name, 0) > 0]
    total = sum(weights[o.name] for o in present)
    if not present or total == 0:
        return Decision(HOLD, 0.0, [])
    score = sum(weights[o.name] * o.score for o in present) / total
    # A Sell from the debate (news risk the charts can't see) vetoes new buys.
    llm = next((o for o in present if o.name == "llm"), None)
    if score >= cfg.buy_threshold and not (llm and llm.score <= -1.0):
        action = BUY
    elif score <= cfg.sell_threshold:
        action = SELL
    else:
        action = HOLD
    return Decision(action, score, present)
