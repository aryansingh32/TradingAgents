"""Bot configuration, loaded from a JSON file.

Secrets (API keys) are never stored in the config file: each broker or feed
section names the environment variables that hold them.
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
    # Volatility targeting: when the asset's annualised volatility is above
    # this, the position shrinks proportionally (never below min_size_mult).
    # 0 disables.
    vol_target_annual: float = 0.0
    min_size_mult: float = 0.25
    # Protections, the discipline rules most systematic traders run:
    # no re-entry for this many bars after an exit,
    cooldown_bars: int = 3
    # pause all entries after this many stop-outs within the lookback,
    stoploss_guard_count: int = 3
    stoploss_guard_hours: float = 24.0
    stoploss_guard_pause_hours: float = 12.0
    # halve risk when equity is this far below its peak, stop entries at the second,
    drawdown_reduce_pct: float = 0.10
    drawdown_pause_pct: float = 0.20
    # and halve risk after this many losses in a row, until the next win.
    loss_streak_reduce: int = 3
    # Skip entries when the bid/ask spread is wider than this fraction.
    max_spread_pct: float = 0.003


@dataclass
class SignalConfig:
    # Weights of each opinion in the final score; a disabled or unavailable
    # opinion is dropped and the rest are renormalised.
    technical_weight: float = 0.4
    forecast_weight: float = 0.3
    llm_weight: float = 0.3
    # Crypto derivatives and sentiment context (funding, open interest, Fear & Greed).
    context_weight: float = 0.15
    use_context: bool = True
    # World Monitor headline tilt (see NewsConfig); not replayed in backtests.
    news_weight: float = 0.1
    # Extra price-based opinions (see signals_extra.py): candlestick patterns,
    # volume/VWAP, ADX + Supertrend trend strength, and RSI(2) / Bollinger
    # mean reversion that only buys dips inside an uptrend.
    patterns_weight: float = 0.1
    volume_weight: float = 0.1
    trend_strength_weight: float = 0.15
    mean_reversion_weight: float = 0.15
    # A mean-reversion setup at or above this score buys on its own when the
    # higher timeframe is confirmed up, even if momentum still reads negative.
    # Above 1 disables the setup.
    mean_reversion_trigger: float = 0.7
    # Final score in [-1, 1] needed to buy / to sell.
    buy_threshold: float = 0.35
    sell_threshold: float = -0.25
    # Only buy when the higher timeframe is in an uptrend (close above its
    # trend EMA and that EMA rising): "trade with the trend".
    trend_filter: bool = True
    trend_interval: str = "1d"
    trend_ema: int = 200
    # Run the TradingAgents LLM debate (costs API credits, takes minutes).
    use_llm: bool = False
    llm_refresh_hours: float = 24.0
    llm_analysts: list[str] = field(
        default_factory=lambda: ["market", "social", "news", "fundamentals"]
    )
    forecast_horizon: int = 10
    forecast_neighbors: int = 50
    # Most recent bars the analog search looks through (bounds its cost).
    forecast_window: int = 8000


@dataclass
class NewsConfig:
    """World Monitor news and event-risk settings (key in WORLDMONITOR_API_KEY,
    or base_url for a self-hosted instance)."""

    enabled: bool = True
    base_url: str | None = None
    variant: str = "finance"
    refresh_seconds: int = 300
    lookback_hours: float = 24.0
    # Block new entries after a critical breaking story (with at least
    # shock_min_sources sources) or a high-severity market-stress signal.
    shock_guard: bool = True
    shock_hours: float = 3.0
    shock_min_sources: int = 3
    # Block new entries on the day of a high-impact release for the market's country.
    macro_guard: bool = True
    # Block new stock entries this many days before earnings.
    earnings_guard: bool = True
    earnings_days: int = 1
    blackout_dates: list[str] = field(default_factory=list)


@dataclass
class TradeManagementConfig:
    # Scale out: at partial_tp_r times the initial risk, sell partial_tp_fraction
    # of the position and move the stop to breakeven. 0 disables.
    partial_tp_r: float = 1.0
    partial_tp_fraction: float = 0.5
    breakeven_after_partial: bool = True


@dataclass
class AccumulateConfig:
    """Smart DCA: buy a fixed amount on a schedule, more when cheap, less when
    euphoric; never sells. For long-term stacking (e.g. Bitcoin)."""

    symbol: str
    market: str = "crypto"
    broker: str = "paper"
    data_symbol: str | None = None
    amount: float = 50.0  # quote currency per buy
    every_hours: float = 24.0 * 7
    # Multiply the amount when price is below its 200-day EMA, and when the
    # crypto Fear & Greed index shows extreme fear; shrink it in euphoria.
    dip_multiplier: float = 2.0
    fear_multiplier: float = 1.5
    greed_multiplier: float = 0.5
    max_multiplier: float = 3.0

    @property
    def price_symbol(self) -> str:
        return self.data_symbol or self.symbol


@dataclass
class UniverseConfig:
    """Dynamic symbol selection, as volume / momentum pair lists do in popular bots."""

    enabled: bool = False
    market: str = "crypto"
    broker: str = "paper"
    # crypto: top_n Binance USDT pairs by 24h quote volume (stablecoins and
    # leveraged tokens excluded); stocks: rank `watchlist` by momentum.
    top_n: int = 5
    min_quote_volume: float = 50_000_000
    watchlist: list[str] = field(default_factory=list)
    exchange: str | None = None
    product: str | None = None
    refresh_hours: float = 24.0


@dataclass
class NotifyConfig:
    """Telegram (token in TELEGRAM_BOT_TOKEN, chat in TELEGRAM_CHAT_ID) and/or
    a Discord webhook (DISCORD_WEBHOOK_URL)."""

    telegram: bool = False
    discord: bool = False
    events: list[str] = field(default_factory=lambda: [
        "entry", "exit", "partial_exit", "entry_blocked_news", "error", "daily_summary",
        "paused", "resumed", "accumulate", "stale_data"])
    # Accept /status /pause /resume /panic /daily from the configured chat.
    telegram_commands: bool = True


@dataclass
class SessionConfig:
    # Stock markets: no new entries in the first / last minutes of the
    # session, when spreads are wide and moves are noisy. Exits always run.
    skip_open_minutes: int = 15
    skip_close_minutes: int = 15
    # India intraday (MIS) positions are closed by the bot at this IST time,
    # before the broker's auto square-off (which charges a fee).
    india_mis_squareoff: str = "15:10"


@dataclass
class SymbolConfig:
    symbol: str
    market: str = "crypto"
    broker: str = "paper"
    # Symbol used for price data when it differs from the broker symbol,
    # e.g. "RELIANCE.NS" on Yahoo for "RELIANCE" on NSE.
    data_symbol: str | None = None
    exchange: str | None = None
    product: str | None = None

    def __post_init__(self):
        if self.market not in MARKETS:
            raise ValueError(f"{self.symbol}: market must be one of {MARKETS}, got {self.market!r}")

    @property
    def price_symbol(self) -> str:
        return self.data_symbol or self.symbol


def _default_feeds() -> dict:
    return {"crypto": {"type": "crypto"}, "us": {"type": "yfinance"}, "india": {"type": "yfinance"}}


def _default_costs() -> dict:
    # One-way cost per trade as a fraction of value, used by backtests:
    # crypto taker fee; US spread + slippage (commission-free brokers);
    # India delivery: STT 0.1% + stamp duty + exchange/SEBI fees + GST.
    return {"crypto": 0.001, "us": 0.0005, "india": 0.0012}


@dataclass
class BotConfig:
    symbols: list[SymbolConfig]
    mode: str = "paper"
    interval: str = "1h"
    lookback_days: int = 730
    # Signals are recomputed when a new candle closes; stops, take profits
    # and trailing stops are checked on the live price every risk_loop_seconds.
    loop_seconds: int = 60
    risk_loop_seconds: int = 10
    # "live" uses each market's real-time feed (see feeds); "yfinance" and
    # "csv" force one source for every symbol.
    data_source: str = "live"
    # A symbol is not traded when its newest closed candle is older than
    # this many candles (feed outage, exchange holiday).
    max_data_age_bars: float = 2.0
    data_dir: str = "autotrader_data"
    risk: RiskConfig = field(default_factory=RiskConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    trade: TradeManagementConfig = field(default_factory=TradeManagementConfig)
    accumulate: list[AccumulateConfig] = field(default_factory=list)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    notify: NotifyConfig = field(default_factory=NotifyConfig)
    # Written every loop; a watchdog restarts the bot when it goes stale.
    heartbeat_file: str = "heartbeat"
    feeds: dict[str, dict] = field(default_factory=_default_feeds)
    costs: dict[str, float] = field(default_factory=_default_costs)
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
    cfg = config_from_dict(raw)
    # Parameters chosen by `autotrader optimize --apply` sit next to the config.
    tuned = Path(path).with_suffix(".tuned.json")
    if tuned.exists():
        apply_tuned(cfg, json.loads(tuned.read_text(encoding="utf-8")))
    return cfg


def apply_tuned(cfg: BotConfig, tuned: dict) -> None:
    for section in ("risk", "signals", "trade"):
        target = getattr(cfg, section)
        for key, value in tuned.get(section, {}).items():
            if not hasattr(target, key):
                raise ValueError(f"tuned parameter {section}.{key} does not exist")
            setattr(target, key, value)


def config_from_dict(raw: dict) -> BotConfig:
    raw = dict(raw)
    symbols = [SymbolConfig(**s) for s in raw.pop("symbols", [])]
    if not symbols and not raw.get("accumulate") and not raw.get("universe", {}).get("enabled"):
        raise ValueError("config needs at least one entry in 'symbols' (or 'accumulate' / 'universe')")
    risk = RiskConfig(**raw.pop("risk", {}))
    signals = SignalConfig(**raw.pop("signals", {}))
    session = SessionConfig(**raw.pop("session", {}))
    news = NewsConfig(**raw.pop("news", {}))
    trade = TradeManagementConfig(**raw.pop("trade", {}))
    accumulate = [AccumulateConfig(**a) for a in raw.pop("accumulate", [])]
    universe = UniverseConfig(**raw.pop("universe", {}))
    notify = NotifyConfig(**raw.pop("notify", {}))
    feeds = {**_default_feeds(), **raw.pop("feeds", {})}
    costs = {**_default_costs(), **raw.pop("costs", {})}
    cfg = BotConfig(symbols=symbols, risk=risk, signals=signals, session=session, news=news,
                    trade=trade, accumulate=accumulate, universe=universe, notify=notify,
                    feeds=feeds, costs=costs, **raw)
    used = {s.broker for s in symbols} | {a.broker for a in accumulate}
    if universe.enabled:
        used.add(universe.broker)
    unknown = used - set(cfg.brokers)
    if unknown:
        raise ValueError(f"symbols use brokers missing from 'brokers': {sorted(unknown)}")
    traded = {(s.broker, s.symbol) for s in symbols}
    clash = [a.symbol for a in accumulate if (a.broker, a.symbol) in traded]
    if clash:
        raise ValueError(f"{clash}: an accumulated symbol cannot also be traded on the same broker "
                         "(the bot would put stops on the long-term stack); use a separate broker account")
    return cfg
