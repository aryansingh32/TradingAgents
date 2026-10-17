"""Dynamic universe: which symbols to trade today.

* crypto: the most liquid Binance USDT pairs by 24h quote volume, like
  freqtrade's VolumePairList (stablecoins and leveraged tokens excluded);
* stocks: a watchlist ranked by 12-1 month momentum (12-month return
  skipping the latest month, the classic cross-sectional momentum measure),
  keeping only names above their 200-day average.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import requests

from autotrader import indicators as ind
from autotrader.config import SymbolConfig, UniverseConfig

log = logging.getLogger(__name__)

STABLES = {"USDC", "FDUSD", "TUSD", "DAI", "BUSD", "USDP", "USDD", "EUR", "TRY", "BRL", "PYUSD",
           "USDE", "AEUR", "XUSD", "USD1", "RLUSD"}
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")


def crypto_top_volume(cfg: UniverseConfig, session: requests.Session | None = None,
                      base_url: str = "https://data-api.binance.vision") -> list[str]:
    session = session or requests.Session()
    rows = session.get(f"{base_url}/api/v3/ticker/24hr", timeout=15).json()
    picks = []
    for r in rows:
        sym = r.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        base = sym[:-4]
        if base in STABLES or base.endswith(LEVERAGED_SUFFIXES):
            continue
        qv = float(r.get("quoteVolume") or 0)
        if qv >= cfg.min_quote_volume:
            picks.append((qv, f"{base}/USDT"))
    picks.sort(reverse=True)
    return [s for _, s in picks[: cfg.top_n]]


def momentum_rank(cfg: UniverseConfig, md) -> list[str]:
    scored = []
    for sym in cfg.watchlist:
        try:
            df = md.candles(sym, cfg.market, "1d", lookback=timedelta(days=420))
        except Exception as exc:
            log.info("universe: skipping %s (%s)", sym, exc)
            continue
        close = df["close"]
        if len(close) < 260 or close.iloc[-1] <= ind.sma(close, 200).iloc[-1]:
            continue
        scored.append((float(close.iloc[-22] / close.iloc[-253] - 1), sym))
    scored.sort(reverse=True)
    return [s for _, s in scored[: cfg.top_n]]


def select(cfg: UniverseConfig, md, session=None) -> list[SymbolConfig]:
    names = crypto_top_volume(cfg, session) if cfg.market == "crypto" else momentum_rank(cfg, md)
    return [SymbolConfig(symbol=n, market=cfg.market, broker=cfg.broker, exchange=cfg.exchange,
                         product=cfg.product) for n in names]
