"""Real-time market data.

Each market has a primary feed and fallbacks:

* crypto: Binance public market data, then Coinbase, then Kraken (no keys);
* us: Alpaca market data (real-time IEX on the free plan; needs Alpaca keys)
  or Yahoo Finance;
* india: OpenAlgo (your broker's own real-time data) or Yahoo Finance
  (``.NS`` / ``.BO`` symbols, delayed).

Two rules keep signals accurate:

* only **closed** candles are used for signals, so a decision never changes
  as the current candle moves (no repainting) and backtests match live runs;
* a feed whose newest closed candle is too old raises ``StaleData`` and the
  symbol is skipped (outage, holiday, delisted symbol), instead of trading
  on old prices.

Live prices (``quote``) come from the ticker / order book and drive the
stop-loss loop between candles.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pandas as pd
import requests

from autotrader.data import OHLCV, normalize_ohlcv

log = logging.getLogger(__name__)

INTERVALS = {
    "1m": timedelta(minutes=1), "5m": timedelta(minutes=5), "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30), "1h": timedelta(hours=1), "4h": timedelta(hours=4),
    "1d": timedelta(days=1),
}


class FeedError(RuntimeError):
    pass


class StaleData(FeedError):
    pass


@dataclass
class Quote:
    price: float
    bid: float | None = None
    ask: float | None = None
    at: datetime | None = None
    source: str = ""

    @property
    def spread_pct(self) -> float | None:
        if self.bid and self.ask and self.bid > 0 and self.ask >= self.bid:
            return (self.ask - self.bid) / ((self.ask + self.bid) / 2)
        return None


def interval_delta(interval: str) -> timedelta:
    try:
        return INTERVALS[interval]
    except KeyError:
        raise FeedError(f"unsupported interval {interval!r}; use one of {list(INTERVALS)}") from None


def closed_only(df: pd.DataFrame, interval: str, now: datetime) -> pd.DataFrame:
    """Drop the still-forming candle: a bar indexed by its open time is closed
    once open + interval <= now."""
    if df.empty:
        return df
    return df[df.index + interval_delta(interval) <= pd.Timestamp(now)]


def check_fresh(df: pd.DataFrame, interval: str, now: datetime, max_age_bars: float,
                market: str, symbol: str) -> None:
    if df.empty:
        raise StaleData(f"{symbol}: no closed candles")
    if max_age_bars == float("inf"):
        return
    step = interval_delta(interval)
    allowance = step * max_age_bars
    if market != "crypto" and step >= timedelta(days=1):
        allowance += timedelta(days=3)  # weekend and a holiday for daily stock bars
    age = pd.Timestamp(now) - (df.index[-1] + step)
    if age > allowance:
        raise StaleData(f"{symbol}: newest closed {interval} candle is {age} old")


def _resample(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    rule = {"4h": "4h", "1d": "1D", "30m": "30min"}[interval]
    out = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return out.dropna(subset=["close"])


class HttpFeed:
    name = "http"
    timeout = 10
    retries = 2

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()

    def _get(self, url: str, **kwargs):
        last = None
        method = kwargs.pop("method", "GET")
        for attempt in range(self.retries + 1):
            try:
                resp = self.session.request(method, url, timeout=self.timeout, **kwargs)
                if resp.status_code == 429:
                    raise FeedError(f"{self.name}: rate limited")
                if resp.status_code >= 400:
                    raise FeedError(f"{self.name}: {url} -> {resp.status_code} {resp.text[:200]}")
                return resp.json()
            except (requests.RequestException, FeedError, ValueError) as exc:
                last = exc
                if attempt < self.retries:
                    time.sleep(0.5 * (attempt + 1))
        raise FeedError(f"{self.name}: {last}")


def _frame(rows: list[list], cols: list[str], unit: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=OHLCV)
    df = pd.DataFrame(rows, columns=cols)
    df.index = pd.to_datetime(df["time"].astype("int64"), unit=unit, utc=True)
    return normalize_ohlcv(df)


def crypto_parts(symbol: str) -> tuple[str, str]:
    """``BTC/USDT``, ``BTC-USD``, ``BTCUSDT`` -> (base, quote)."""
    s = symbol.upper().replace("-", "/")
    if "/" in s:
        base, quote = s.split("/", 1)
        return base, quote
    for quote in ("USDT", "USDC", "USD", "INR", "EUR"):
        if s.endswith(quote) and len(s) > len(quote):
            return s[: -len(quote)], quote
    return s, "USDT"


class BinanceFeed(HttpFeed):
    """Binance spot public market data (no key). ``data-api.binance.vision``
    serves the public endpoints; set ``base_url`` to ``https://api.binance.com``
    if you prefer the main host."""

    name = "binance"
    LIMIT = 1000

    def __init__(self, base_url: str = "https://data-api.binance.vision", session=None):
        super().__init__(session)
        self.base_url = base_url.rstrip("/")

    @staticmethod
    def pair(symbol: str) -> str:
        base, quote = crypto_parts(symbol)
        return base + ("USDT" if quote == "USD" else quote)

    def candles(self, symbol: str, interval: str, start: datetime, end: datetime | None = None) -> pd.DataFrame:
        interval_delta(interval)
        frames, since = [], int(start.timestamp() * 1000)
        end_ms = int((end or datetime.now(UTC)).timestamp() * 1000)
        cols = ["time", "open", "high", "low", "close", "volume", "close_time", "qv", "n", "tb", "tq", "x"]
        while since < end_ms:
            rows = self._get(f"{self.base_url}/api/v3/klines", params={
                "symbol": self.pair(symbol), "interval": interval, "startTime": since,
                "endTime": end_ms, "limit": self.LIMIT})
            if not rows:
                break
            frames.append(_frame(rows, cols, "ms"))
            since = int(rows[-1][0]) + 1
            if len(rows) < self.LIMIT:
                break
        return normalize_ohlcv(pd.concat(frames)) if frames else pd.DataFrame(columns=OHLCV)

    def quote(self, symbol: str) -> Quote:
        book = self._get(f"{self.base_url}/api/v3/ticker/bookTicker", params={"symbol": self.pair(symbol)})
        bid, ask = float(book["bidPrice"]), float(book["askPrice"])
        return Quote((bid + ask) / 2, bid, ask, datetime.now(UTC), self.name)


class CoinbaseFeed(HttpFeed):
    name = "coinbase"
    GRANULARITY = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "1d": 86400}
    LIMIT = 300

    def __init__(self, base_url: str = "https://api.exchange.coinbase.com", session=None):
        super().__init__(session)
        self.base_url = base_url.rstrip("/")

    @staticmethod
    def product(symbol: str) -> str:
        base, quote = crypto_parts(symbol)
        return f"{base}-{'USD' if quote in ('USDT', 'USDC') else quote}"

    def candles(self, symbol: str, interval: str, start: datetime, end: datetime | None = None) -> pd.DataFrame:
        if interval not in self.GRANULARITY:
            base = "1h" if interval in ("4h",) else "15m"
            return _resample(self.candles(symbol, base, start, end), interval)
        gran = self.GRANULARITY[interval]
        end = end or datetime.now(UTC)
        frames, cursor = [], start
        while cursor < end:
            chunk_end = min(end, cursor + timedelta(seconds=gran * self.LIMIT))
            rows = self._get(f"{self.base_url}/products/{self.product(symbol)}/candles", params={
                "granularity": gran, "start": cursor.isoformat(), "end": chunk_end.isoformat()})
            # [time, low, high, open, close, volume], newest first
            if rows:
                frames.append(_frame([[r[0], r[3], r[2], r[1], r[4], r[5]] for r in rows],
                                     ["time", "open", "high", "low", "close", "volume"], "s"))
            cursor = chunk_end
        return normalize_ohlcv(pd.concat(frames)) if frames else pd.DataFrame(columns=OHLCV)

    def quote(self, symbol: str) -> Quote:
        t = self._get(f"{self.base_url}/products/{self.product(symbol)}/ticker")
        return Quote(float(t["price"]), float(t["bid"]), float(t["ask"]), datetime.now(UTC), self.name)


class KrakenFeed(HttpFeed):
    name = "kraken"
    MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}

    def __init__(self, base_url: str = "https://api.kraken.com", session=None):
        super().__init__(session)
        self.base_url = base_url.rstrip("/")

    @staticmethod
    def pair(symbol: str) -> str:
        base, quote = crypto_parts(symbol)
        base = "XBT" if base == "BTC" else base
        return base + ("USD" if quote in ("USDT", "USDC") else quote)

    def _result(self, data: dict) -> list:
        if data.get("error"):
            raise FeedError(f"kraken: {data['error']}")
        return next(v for k, v in data["result"].items() if k != "last")

    def candles(self, symbol: str, interval: str, start: datetime, end: datetime | None = None) -> pd.DataFrame:
        # Kraken returns at most the latest 720 candles from `since`.
        data = self._get(f"{self.base_url}/0/public/OHLC", params={
            "pair": self.pair(symbol), "interval": self.MINUTES[interval], "since": int(start.timestamp())})
        rows = [[r[0], r[1], r[2], r[3], r[4], r[6]] for r in self._result(data)]
        return _frame(rows, ["time", "open", "high", "low", "close", "volume"], "s")

    def quote(self, symbol: str) -> Quote:
        t = self._result(self._get(f"{self.base_url}/0/public/Ticker", params={"pair": self.pair(symbol)}))
        if isinstance(t, list):
            raise FeedError("kraken: unexpected ticker shape")
        return Quote(float(t["c"][0]), float(t["b"][0]), float(t["a"][0]), datetime.now(UTC), self.name)


class AlpacaDataFeed(HttpFeed):
    """Alpaca market data. The free plan streams IEX in real time; ``feed="sip"``
    (all US exchanges) needs a paid plan."""

    name = "alpaca"
    TIMEFRAME = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "30m": "30Min", "1h": "1Hour",
                 "4h": "4Hour", "1d": "1Day"}

    def __init__(self, key_env: str = "APCA_API_KEY_ID", secret_env: str = "APCA_API_SECRET_KEY",
                 feed: str = "iex", base_url: str = "https://data.alpaca.markets", session=None, **_):
        super().__init__(session)
        key, secret = os.environ.get(key_env), os.environ.get(secret_env)
        if not key or not secret:
            raise FeedError(f"alpaca data: set {key_env} and {secret_env}")
        self.session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
        self.base_url, self.feed = base_url.rstrip("/"), feed

    def candles(self, symbol: str, interval: str, start: datetime, end: datetime | None = None) -> pd.DataFrame:
        params = {"timeframe": self.TIMEFRAME[interval], "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "limit": 10000, "adjustment": "all", "feed": self.feed}
        if end:
            params["end"] = end.strftime("%Y-%m-%dT%H:%M:%SZ")
        rows = []
        while True:
            data = self._get(f"{self.base_url}/v2/stocks/{symbol}/bars", params=params)
            rows += data.get("bars") or []
            if not data.get("next_page_token"):
                break
            params["page_token"] = data["next_page_token"]
        if not rows:
            return pd.DataFrame(columns=OHLCV)
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["t"], utc=True)
        return normalize_ohlcv(df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}))

    def quote(self, symbol: str) -> Quote:
        trade = self._get(f"{self.base_url}/v2/stocks/{symbol}/trades/latest", params={"feed": self.feed})["trade"]
        q = self._get(f"{self.base_url}/v2/stocks/{symbol}/quotes/latest", params={"feed": self.feed})["quote"]
        bid, ask = float(q.get("bp") or 0) or None, float(q.get("ap") or 0) or None
        return Quote(float(trade["p"]), bid, ask, pd.Timestamp(trade["t"]).to_pydatetime(), self.name)


class OpenAlgoFeed(HttpFeed):
    """Indian market data from your broker through OpenAlgo."""

    name = "openalgo"
    INTERVAL = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "1d": "D"}

    def __init__(self, host: str = "http://127.0.0.1:5000", key_env: str = "OPENALGO_API_KEY",
                 exchange: str = "NSE", session=None, **_):
        super().__init__(session)
        self.api_key = os.environ.get(key_env)
        if not self.api_key:
            raise FeedError(f"openalgo data: set {key_env}")
        self.host, self.exchange = host.rstrip("/"), exchange

    def _post(self, path: str, payload: dict) -> dict:
        data = self._get(f"{self.host}/api/v1/{path}", method="POST", json={"apikey": self.api_key, **payload})
        if data.get("status") != "success":
            raise FeedError(f"openalgo {path}: {data}")
        return data["data"]

    @staticmethod
    def _plain(symbol: str) -> str:
        return symbol.split(".")[0]

    def candles(self, symbol: str, interval: str, start: datetime, end: datetime | None = None) -> pd.DataFrame:
        if interval == "4h":
            return _resample(self.candles(symbol, "1h", start, end), "4h")
        rows = self._post("history", {
            "symbol": self._plain(symbol), "exchange": self.exchange, "interval": self.INTERVAL[interval],
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": (end or datetime.now(UTC)).strftime("%Y-%m-%d")})
        if not rows:
            return pd.DataFrame(columns=OHLCV)
        df = pd.DataFrame(rows)
        ts = df["timestamp"]
        df.index = (pd.to_datetime(ts.astype("int64"), unit="s", utc=True)
                    if pd.api.types.is_numeric_dtype(ts) else pd.to_datetime(ts, utc=True))
        return normalize_ohlcv(df)

    def quote(self, symbol: str) -> Quote:
        q = self._post("quotes", {"symbol": self._plain(symbol), "exchange": self.exchange})
        bid, ask = float(q.get("bid") or 0) or None, float(q.get("ask") or 0) or None
        return Quote(float(q["ltp"]), bid, ask, datetime.now(UTC), self.name)


class YahooFeed:
    """Yahoo Finance via yfinance: free, no key, but delayed for many
    exchanges (NSE about 15 minutes) and limited intraday history (1h: 730
    days). Fine for daily bars; prefer Alpaca / OpenAlgo for intraday."""

    name = "yfinance"
    YF = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "60m", "1d": "1d"}

    def candles(self, symbol: str, interval: str, start: datetime, end: datetime | None = None) -> pd.DataFrame:
        import yfinance as yf

        if interval == "4h":
            return _resample(self.candles(symbol, "1h", start, end), "4h")
        if interval == "1h":
            start = max(start, datetime.now(UTC) - timedelta(days=729))
        raw = yf.download(symbol, start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d") if end else None,
                          interval=self.YF[interval], progress=False, auto_adjust=True)
        df = normalize_ohlcv(raw)
        if not df.empty:
            df.index = (df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC"))
        return df

    def quote(self, symbol: str) -> Quote:
        import yfinance as yf

        info = yf.Ticker(symbol).fast_info
        return Quote(float(info["last_price"]), None, None, datetime.now(UTC), self.name)


class FallbackFeed:
    """Try each feed in order; remember the last one that worked."""

    def __init__(self, feeds: list):
        if not feeds:
            raise FeedError("no feeds configured")
        self.feeds = feeds
        self.name = "+".join(f.name for f in feeds)

    def _call(self, method: str, *args):
        errors = []
        for feed in self.feeds:
            try:
                return getattr(feed, method)(*args)
            except Exception as exc:
                errors.append(f"{feed.name}: {exc}")
                log.warning("%s %s failed, trying next feed: %s", feed.name, method, exc)
        raise FeedError("; ".join(errors))

    def candles(self, symbol, interval, start, end=None):
        return self._call("candles", symbol, interval, start, end)

    def quote(self, symbol):
        return self._call("quote", symbol)


def build_feed(spec: dict):
    spec = dict(spec)
    kind = spec.pop("type")
    if kind == "crypto":
        return FallbackFeed([BinanceFeed(**spec.get("binance", {})), CoinbaseFeed(), KrakenFeed()])
    if kind == "binance":
        return BinanceFeed(**spec)
    if kind == "coinbase":
        return CoinbaseFeed(**spec)
    if kind == "kraken":
        return KrakenFeed(**spec)
    if kind == "alpaca":
        return FallbackFeed([AlpacaDataFeed(**spec), YahooFeed()]) if spec.pop("fallback", True) else AlpacaDataFeed(**spec)
    if kind == "openalgo":
        return FallbackFeed([OpenAlgoFeed(**spec), YahooFeed()]) if spec.pop("fallback", False) else OpenAlgoFeed(**spec)
    if kind == "yfinance":
        return YahooFeed()
    if kind == "csv":
        return CsvFeed(**spec)
    raise FeedError(f"unknown feed type {kind!r}")


class MarketData:
    """Cached, incrementally refreshed candles plus live quotes per symbol."""

    def __init__(self, feeds: dict, lookback_days: int, max_age_bars: float, clock=None):
        self.feeds = feeds  # market -> feed
        self.lookback = timedelta(days=lookback_days)
        self.max_age_bars = max_age_bars
        self.clock = clock or (lambda: datetime.now(UTC))
        self._cache: dict[tuple[str, str], pd.DataFrame] = {}

    def candles(self, symbol: str, market: str, interval: str, lookback: timedelta | None = None) -> pd.DataFrame:
        now = self.clock()
        key = (symbol, interval)
        cached = self._cache.get(key)
        feed = self.feeds[market]
        if cached is not None and not cached.empty:
            # Refetch from the last cached bar: it may have been revised.
            fresh = feed.candles(symbol, interval, cached.index[-1].to_pydatetime())
            df = normalize_ohlcv(pd.concat([cached, fresh]))
        else:
            df = feed.candles(symbol, interval, now - (lookback or self.lookback))
        df = closed_only(df, interval, now)
        df = df[df.index >= pd.Timestamp(now - (lookback or self.lookback))]
        self._cache[key] = df
        check_fresh(df, interval, now, self.max_age_bars, market, symbol)
        return df

    def quote(self, symbol: str, market: str) -> Quote:
        return self.feeds[market].quote(symbol)


class CsvFeed:
    """Candles from ``<data_dir>/<symbol>_<interval>.csv`` (or ``<symbol>.csv``),
    ``/`` in the symbol written as ``_``. For research on exported data; the
    quote is the last close."""

    name = "csv"

    def __init__(self, data_dir: str):
        from pathlib import Path

        self.dir = Path(data_dir)

    def _path(self, symbol: str, interval: str):
        stem = symbol.replace("/", "_")
        for name in (f"{stem}_{interval}.csv", f"{stem}.csv"):
            if (self.dir / name).exists():
                return self.dir / name
        raise FeedError(f"csv: no {stem}_{interval}.csv or {stem}.csv in {self.dir}")

    def candles(self, symbol: str, interval: str, start: datetime, end: datetime | None = None) -> pd.DataFrame:
        from autotrader.data import load_csv

        try:
            df = load_csv(self._path(symbol, interval))
        except FeedError:
            # Build higher timeframes (4h, 1d) from a finer file when only that exists.
            finer = [b for b in ("1h", "15m", "5m", "1m") if interval_delta(b) < interval_delta(interval)]
            for base in finer:
                try:
                    df = _resample(load_csv(self._path(symbol, base)), interval)
                    break
                except FeedError:
                    continue
            else:
                raise
        df = df[df.index >= pd.Timestamp(start)]
        return df[df.index <= pd.Timestamp(end)] if end else df

    def quote(self, symbol: str) -> Quote:
        raise FeedError("csv feed has no live quotes")
