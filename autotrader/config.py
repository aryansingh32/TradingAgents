"""Bot configuration, loaded from a JSON file.

Secrets (API keys) are never stored in the config file: each broker section
names the environment variables that hold them.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

LIVE_CONFIRM_ENV = "AUTOTRADER_LIVE_CONFIRM"
LIVE_CONFIRM_VALUE = "I_UNDERSTAND_THE_RISK"

MARKETS = ("crypto", "us", "india")


@dataclass
class RiskConfig:
    # Fraction of equity lost if a trade hits its stop (1% is a common ceiling).
    risk_per_trade: float = 0.01
    # Largest single position as a fraction of equity.
    max_position_pct: float = 0.20
    max_open_positions: int = 5
    # New entries stop for the day once the day's loss reaches this fraction.
    max_daily_loss_pct: float = 0.03
    # Stop loss distance in ATRs below entry, and take profit as a multiple
    # of that distance (reward:risk).
    atr_stop_mult: float = 2.0
    reward_risk: float = 2.0
    # Once price has moved this many ATRs in favour, the stop trails the
    # highest price at atr_stop_mult ATRs. 0 disables trailing. The bot is
    # long-only: a SELL decision exits a position, it never opens a short.
    trail_after_atr: float = 1.0


@dataclass
class SignalConfig:
    # Weights of each opinion in the final score; a disabled or unavailable
    # opinion is dropped and the rest are renormalised.
    technical_weight: float = 0.4
    forecast_weight: float = 0.3
    llm_weight: float = 0.3
    # Final score in [-1, 1] needed to buy / to sell.
    buy_threshold: float = 0.35
    sell_threshold: float = -0.25
    # Run the TradingAgents LLM debate (costs API credits, takes minutes).
    use_llm: bool = False
    llm_refresh_hours: float = 24.0
    llm_analysts: list[str] = field(
        default_factory=lambda: ["market", "social", "news", "fundamentals"]
    )
    forecast_horizon: int = 10
    forecast_neighbors: int = 50


@dataclass
class SymbolConfig:
    symbol: str
    market: str = "crypto"
    broker: str = "paper"
    # Symbol used for price data when it differs from the broker symbol,
    # e.g. "RELIANCE.NS" on Yahoo for "RELIANCE" on NSE, or "BTC-USD" for
    # freqtrade's "BTC/USDT".
    data_symbol: str | None = None
    exchange: str | None = None
    product: str | None = None

    def __post_init__(self):
        if self.market not in MARKETS:
            raise ValueError(f"{self.symbol}: market must be one of {MARKETS}, got {self.market!r}")

    @property
    def price_symbol(self) -> str:
        return self.data_symbol or self.symbol


@dataclass
class BotConfig:
    symbols: list[SymbolConfig]
    mode: str = "paper"
    interval: str = "1d"
    lookback_days: int = 1500
    loop_seconds: int = 900
    data_source: str = "yfinance"
    data_dir: str = "autotrader_data"
    risk: RiskConfig = field(default_factory=RiskConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    brokers: dict[str, dict] = field(default_factory=lambda: {"paper": {"starting_cash": 100000}})

    def __post_init__(self):
        if self.mode not in ("paper", "live"):
            raise ValueError(f"mode must be 'paper' or 'live', got {self.mode!r}")

    @property
    def data_path(self) -> Path:
        path = Path(self.data_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def require_live_confirmation(self) -> None:
        """Refuse to trade real money unless the operator opted in twice."""
        if self.mode == "live" and os.environ.get(LIVE_CONFIRM_ENV) != LIVE_CONFIRM_VALUE:
            raise RuntimeError(
                f"Live mode is set in the config but {LIVE_CONFIRM_ENV} is not "
                f"{LIVE_CONFIRM_VALUE!r}. Paper trade first; set it only when you "
                "accept that the bot can lose real money."
            )

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> BotConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return config_from_dict(raw)


def config_from_dict(raw: dict) -> BotConfig:
    raw = dict(raw)
    symbols = [SymbolConfig(**s) for s in raw.pop("symbols", [])]
    if not symbols:
        raise ValueError("config needs at least one entry in 'symbols'")
    risk = RiskConfig(**raw.pop("risk", {}))
    signals = SignalConfig(**raw.pop("signals", {}))
    cfg = BotConfig(symbols=symbols, risk=risk, signals=signals, **raw)
    unknown = {s.broker for s in symbols} - set(cfg.brokers)
    if unknown:
        raise ValueError(f"symbols use brokers missing from 'brokers': {sorted(unknown)}")
    return cfg
