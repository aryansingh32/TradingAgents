# AutoTrader

An automated trading bot for **Bitcoin and crypto, US stocks and Indian
stocks**, built on TradingAgents. It runs on real-time exchange and broker
data, and decides only on closed candles. Every trade is sized from its stop
loss. Stop losses, trailing stops and take profits are checked on the live
price every few seconds. The discipline rules experienced traders use to
survive losing streaks are built in.

> **Read this first.** Nobody can predict the market, and this bot does not
> try to. Professional traders do not win by being right most of the time.
> Many trend followers are right less than half the time. They win because
> their average win is larger than their average loss, they size every
> position so a losing streak cannot ruin them, and they test before they
> trade. This bot is built around those rules. It does not guarantee profit
> and can lose money. Start on paper (simulated money), check the
> walk-forward results, and only go live with money you can afford to lose.
> Nothing here is financial advice.

## What successful traders do, and where it lives in the bot

| Principle | In the bot |
|---|---|
| **Trade with the trend.** Don't buy into a falling market. Time-series momentum is one of the best-documented effects in markets, crypto included. | Longs only while the higher timeframe (daily) closes above a rising 200 EMA (`signals.trend_filter`). In the freqtrade strategy too. |
| **Risk a small, fixed fraction per trade** (0.5–1%) | Size = equity × `risk_per_trade` ÷ stop distance (`risk.py`) |
| **Cut losers, let winners run.** Judge a system by expectancy in R (average win × win rate − average loss × loss rate), not by win rate. | ATR stop, 2–3R take profit, trailing stop after 1–2 ATR. Backtests report win rate, average win/loss in R, expectancy, SQN. |
| **Size down when volatility is high** | Volatility targeting: position × min(1, target vol ÷ realised vol) (`risk.vol_target_annual`) |
| **Stop after a bad day or a bad streak** | Daily loss limit; stop-loss guard (N stop-outs → pause); drawdown guard (−10% → half risk, −20% → no entries); losing-streak guard (3 losses → half risk until a win) (`protections.py`) |
| **Don't revenge-trade or churn** | Cooldown bars after every exit, one decision per closed candle |
| **Avoid the open and close auction noise** | No entries in the first/last 15 min of NYSE/NSE sessions; India MIS positions closed at 15:10 IST, before the broker's square-off fee |
| **Respect liquidity and costs** | Entries skipped when the live bid/ask spread is too wide; backtests charge realistic costs per market (crypto 0.10%, US 0.05%, India delivery 0.12% per side: STT, stamp duty, exchange fees, GST) |
| **Know positioning and sentiment** (crypto) | Funding rate, open interest change and Fear & Greed as a small contrarian tilt at extremes (`context.py`) |
| **Test out of sample, never fit the past** | Walk-forward optimizer: parameters picked on one stretch of history are scored on the next, unseen stretch; it refuses to apply parameters with no out-of-sample edge (`optimize.py`) |
| **Keep a journal** | Every decision with its reasons, every order, exit reason and R multiple in `journal.jsonl` |

## How it decides

```
 live candles (closed only)        live quote (bid/ask)          optional
        │                                 │                         │
  ┌─────▼─────────┐ ┌──────────────────┐  │  ┌─────────────────┐ ┌──▼────────────────────┐
  │ Technical     │ │ Analog forecast  │  │  │ Crypto context  │ │ TradingAgents debate  │
  │ EMA50/200,    │ │ 50 most similar  │  │  │ funding, OI,    │ │ news, sentiment,      │
  │ MACD, RSI,    │ │ past bars: P(up),│  │  │ Fear & Greed    │ │ fundamentals, bull vs │
  │ breakouts     │ │ median, range    │  │  │                 │ │ bear, risk team       │
  └─────┬─────────┘ └────────┬─────────┘  │  └───────┬─────────┘ └──────────┬────────────┘
        └──────── weighted score in [-1, +1] ◄───────┴──────────────────────┘
                              │  BUY ≥ 0.35 · SELL ≤ −0.25
                              │  daily trend down → no buy · debate "Sell" → no buy
                    ┌─────────▼──────────┐
                    │ Session window,    │  spread, daily loss, max positions,
                    │ protections, risk  │  cooldown, guards, vol-scaled size
                    └─────────┬──────────┘
         ┌──────────────┬─────┴───────────┬────────────────┐
       paper        freqtrade          Alpaca           OpenAlgo
                  (Binance, Bybit,   (US stocks)     (Zerodha, Angel One,
                   OKX, Kraken…)                     Upstox, Dhan, Fyers…)
```

The backtest and the live bot run the same signal code, and tests check
that they agree bar for bar and that no future data can leak into a
decision.

## Real-time data

| Market | Primary feed | Fallbacks | Notes |
|---|---|---|---|
| Crypto | Binance spot public API | Coinbase, Kraken | no keys; quotes from the order book, so the spread check works |
| US | Alpaca market data (`feeds.us.type = "alpaca"`) | Yahoo Finance | free Alpaca plan: real-time IEX; `"feed": "sip"` (all exchanges) needs a paid plan |
| India | OpenAlgo (your broker's real-time data) | Yahoo Finance | Yahoo NSE data is delayed about 15 min; use it for daily bars, OpenAlgo for intraday |

Guards:
- Only closed candles are used for signals.
- A symbol whose newest candle is older than `max_data_age_bars` is skipped and logged as `stale_data`. This covers feed outages and exchange holidays.
- History is cached and refreshed incrementally.

## Workflow

```bash
pip install .                                   # from the TradingAgents repo root
cp autotrader/examples/btc.json btc.json        # or us.json / india.json

autotrader monitor  -c btc.json --watch 60      # live table: returns, trend, forecast, context, action
autotrader backtest -c btc.json                 # full history with costs and protections
autotrader optimize -c btc.json                 # walk-forward; add --apply to save the parameters
autotrader once     -c btc.json                 # one decision pass with the reasons
autotrader run      -c btc.json                 # trade continuously (paper by default)
autotrader status   -c btc.json                 # positions, levels, guards, last decisions
```

- `optimize --apply` writes `btc.tuned.json` next to the config, and the bot
  loads it automatically. It only applies parameters whose out-of-sample
  result is positive.
- Keep one config per market. Their parameters should differ: Bitcoin trends
  hard and is volatile around the clock, large US stocks are smoother and
  intraday sessions matter, and Indian delivery trades carry high fixed costs,
  which favours daily bars and fewer, larger moves.
- Speed: decisions recompute only when a candle closes. Two years of hourly
  BTC take about 20 s to prepare once, then about 20 ms per backtest. A full
  walk-forward run over 144 parameter sets takes about 11 s.

### Turn on the AI debate

Set an LLM key as described in the main README (e.g. `ANTHROPIC_API_KEY`)
and set `"use_llm": true`. Each symbol is debated at most once per
`llm_refresh_hours` (default 24). If the debate rates a symbol Sell, the bot
won't open a new buy on it. If the LLM fails, the bot carries on with the
other opinions.

## Brokers and live trading

Live trading needs both `"mode": "live"` in the config and
`AUTOTRADER_LIVE_CONFIRM=I_UNDERSTAND_THE_RISK` in the environment. In paper
mode the bot refuses to build any broker that could place a real order.

| Market | Broker type | Setup | Stops held by |
|---|---|---|---|
| Crypto | `freqtrade` | freqtrade with `config_examples/config_autotrader.example.json` (REST API on, `force_entry_enable`, dry run first); `FREQTRADE_USERNAME`, `FREQTRADE_PASSWORD`. Or run freqtrade's `AutoTraderSignalStrategy` on the bot's `signals.json`. | freqtrade (stop loss, trailing, on-exchange stop) **and** the bot |
| US | `alpaca` | `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`; `"live": false` keeps the paper account | Alpaca bracket orders (GTC) **and** the bot |
| India | `openalgo` | self-hosted [OpenAlgo](https://github.com/marketcalls/openalgo) connected to your broker; `OPENALGO_API_KEY`; `"exchange": "NSE"`, `"product": "CNC"` or `"MIS"` | the bot (risk loop every `risk_loop_seconds`) |

- **India (SEBI rules since 1 April 2026):** broker APIs only accept orders
  from a static IP you have registered with the broker. Every session needs a
  daily OAuth + 2FA login, and orders carry an exchange algo ID. A personal
  algo under 10 orders per second needs no exchange registration. Run on a
  VPS with a fixed IP and log in to OpenAlgo each morning.
- **US:** accounts under $25,000 are limited by the pattern day trader rule;
  hourly bars can trigger it, so use daily bars or a larger account. Alpaca
  enforces it.
- **Crypto tax in India:** 30% on gains plus 1% TDS.
- Keep exchange keys **trade-only with withdrawals disabled**. The bot is long-only.

## Research behind this design

**"The repository with the highest success rate":** no open-source project
publishes a verified live track record, and GitHub stars measure popularity,
not profit. Popular free strategies are often overfitted to the past. So the
bot takes the *rules* that the most-used and best-maintained projects share,
and leaves out their hand-tuned entry conditions.

| Project | What was taken |
|---|---|
| [TradingAgents](https://github.com/TauricResearch/TradingAgents) | multi-agent news/sentiment/fundamentals debate as an opinion |
| [freqtrade](https://github.com/freqtrade/freqtrade) | execution on many exchanges; its protections (Cooldown, StoplossGuard, MaxDrawdown, LowProfitPairs), mirrored in `protections.py` and enabled in the strategy |
| [NostalgiaForInfinity](https://github.com/iterativv/NostalgiaForInfinity) (the most-used freqtrade strategy) | multi-timeframe confirmation, a market-regime filter, small sizing spread over trades |
| [OpenAlgo](https://github.com/marketcalls/openalgo) | one API for 30+ Indian brokers, data and orders |
| [Alpaca](https://alpaca.markets) | US broker and market data API, free paper trading |
| [hummingbot](https://github.com/hummingbot/hummingbot) | not used: market making and arbitrage, not directional trading |
| [World Monitor](https://github.com/koala73/worldmonitor), [Qlib](https://github.com/microsoft/qlib), [NautilusTrader](https://github.com/nautechsystems/nautilus_trader) | candidates for more news feeds, factor models, and tick-level execution |

Evidence behind each signal:
- Time-series momentum held up in crypto studies (moving-average momentum on
  eight coins, 2020–2025), with short-horizon trend persistence that has
  weakened over time. This is why the trend filter and trailing stops matter
  more than any single entry rule.
- Funding, open interest and Fear & Greed are documented gauges of leverage
  and sentiment, not proven standalone predictors, so they carry a small
  weight and only act at extremes.
- Time-series foundation models (Chronos, TimesFM) match simple baselines on
  stock prices but have not traded better in published tests. The bot uses a
  transparent analog forecast whose reasoning you can inspect.

## Extending

- New opinion: write a function returning `signals.Opinion`, add its weight in
  `SignalConfig` and `ensemble.weights_of`, and call it in `Engine.decide`.
- New broker: subclass `brokers.base.Broker` (five methods) and register it in
  `brokers.build_broker`.
- New data source: implement `candles(symbol, interval, start, end)` and
  `quote(symbol)`, and register it in `feeds.build_feed`.
- Tests: `pytest tests/test_autotrader.py`.
