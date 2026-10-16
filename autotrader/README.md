# AutoTrader

An automated trading bot for **Bitcoin and crypto, US stocks and Indian stocks**,
built on TradingAgents. It reads each market from three angles, combines them
into one decision, sizes the trade from its stop loss, and places the orders
through a broker. Positions get a stop loss, a take profit and a trailing stop.

> **Read this first.** No bot, including the ones big trading firms run, can
> predict the future. Their edge comes from data, speed, risk control and
> testing, not from certainty. This bot gives you that discipline: every
> decision is explained and logged, history is backtested, and risk limits are
> enforced. It does **not** guarantee a profit, and it can lose money. It
> starts in paper mode (simulated money). Run it on paper for weeks before
> you think about live trading, and then only with money you can afford to lose.
> Nothing here is financial advice.

## How it decides

```
            price history (years of daily or intraday bars)
                 │                         │
     ┌───────────▼──────────┐  ┌───────────▼────────────────────┐   ┌───────────────────────────┐
     │ Technical opinion    │  │ History-analog forecast        │   │ TradingAgents debate (opt) │
     │ EMA50/200 trend,     │  │ finds the 50 past days that    │   │ news, social sentiment,    │
     │ MACD momentum, RSI,  │  │ looked most like today and     │   │ fundamentals, Polymarket,  │
     │ 20-bar breakouts     │  │ measures what happened next:   │   │ bull vs bear researchers,  │
     │                      │  │ P(up), median, 10th-90th pct   │   │ risk team, final rating    │
     └───────────┬──────────┘  └───────────┬────────────────────┘   └─────────────┬─────────────┘
                 └──────── weighted score in [-1, +1] ◄─────────────────────────────┘
                                   │   BUY ≥ 0.35 · SELL ≤ -0.25 · a debate "Sell" vetoes buys
                         ┌─────────▼──────────┐
                         │ Risk manager       │ 1% of equity at risk per trade, stop = 2 ATR,
                         │                    │ take profit = 2× risk, trailing stop, max 20% per
                         │                    │ position, max 5 positions, -3% daily loss → no new trades,
                         │                    │ trades only during market hours (US / NSE), crypto 24/7
                         └─────────┬──────────┘
          ┌──────────────┬─────────┴───────┬──────────────────┐
       paper          freqtrade          Alpaca            OpenAlgo
     (simulated)   (Binance, Bybit,    (US stocks,      (Zerodha, Angel One,
                   Kraken, OKX, ...)    paper + live)    Upstox, Dhan, Fyers, ...)
```

The forecast is "the future is built on the past" made concrete: it never
sees data after the bar it is forecasting (tested), and it reports the spread
of outcomes, not one magic number.

## Quick start (paper trading)

```bash
pip install .                         # from the TradingAgents repo root
cp autotrader/examples/paper.json autotrader.json

autotrader monitor   # table: price, 1D/1W/1M/1Y change, drawdown, volatility, forecast, action
autotrader backtest  # walk-forward backtest of every symbol over its full history
autotrader once      # one decision pass, with the reasons behind each decision
autotrader run       # keep trading (paper money) every loop_seconds
autotrader status    # open positions, stop/take levels, last decisions
```

The paper config watches BTC-USD, ETH-USD, SPY, AAPL, RELIANCE.NS and TCS.NS.
Everything the bot does is appended to `autotrader_data/journal.jsonl`.

Price data comes from Yahoo Finance (`data_source: "yfinance"`). To use your
own data, set `data_source: "csv"` and put `<SYMBOL>.csv` files (date, open,
high, low, close, volume) in `data_dir`.

### Turn on the AI debate (news, sentiment, bull/bear)

Set an LLM key as described in the main README (e.g. `ANTHROPIC_API_KEY` or
`OPENAI_API_KEY`) and set `"use_llm": true` under `signals`. Each symbol is
debated at most once per `llm_refresh_hours` (default 24) because a run takes
minutes and costs API credits. If the LLM fails, the bot carries on with the
other two opinions.

## Connecting real brokers

Copy `autotrader/examples/live.json` and keep only the markets you trade.
Live trading needs **both** `"mode": "live"` in the config **and**
`AUTOTRADER_LIVE_CONFIRM=I_UNDERSTAND_THE_RISK` in the environment. In paper
mode the bot refuses to build any broker that could place a real order
(Alpaca is pointed at its paper endpoint).

| Market | Broker type | What you need | Stops held by |
|---|---|---|---|
| Bitcoin / crypto | `freqtrade` | A running freqtrade (repo `freqtrade`) with `config_examples/config_autotrader.example.json`: `api_server` on, `force_entry_enable: true`, start it in `dry_run` first. `FREQTRADE_USERNAME`, `FREQTRADE_PASSWORD`. Symbols as pairs: `"symbol": "BTC/USDT", "data_symbol": "BTC-USD"` | freqtrade (`stoploss`, trailing, `stoploss_on_exchange`) **and** the bot |
| US stocks | `alpaca` | An [Alpaca](https://alpaca.markets) account. `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`. `"live": false` uses the paper account even in live mode | Alpaca (bracket orders, GTC) **and** the bot |
| India (NSE/BSE) | `openalgo` | A self-hosted [OpenAlgo](https://github.com/marketcalls/openalgo) server connected to your broker. `OPENALGO_API_KEY`. `"exchange": "NSE", "product": "CNC"` (delivery) or `"MIS"` (intraday) | the bot (keep `loop_seconds` short) |

Notes:
- **Indian market rules (SEBI, in force since 1 April 2026):** broker APIs only
  accept orders from a static IP registered with the broker, each session
  needs a daily OAuth + 2FA login, and orders are tagged with an exchange algo
  ID. Under 10 orders per second a personal algo needs no exchange
  registration. Run the bot on a VPS with a fixed IP and log in to OpenAlgo
  each morning. Check your broker's current rules before going live.
- **Crypto in India** carries 30% tax on gains and 1% TDS; check the rules.
- Keep exchange API keys **trade-only with withdrawals disabled**.
- The bot is long-only: SELL closes a position, it never shorts.
- Exchange holidays are not modelled; the broker rejects orders on those days
  and the bot logs the error and carries on.

Instead of AutoTrader placing crypto orders itself, you can run freqtrade with
`AutoTraderSignalStrategy` (in the freqtrade repo, `user_data/strategies/`),
which reads AutoTrader's `signals.json` and enters and exits on its decisions.

## Research behind this design

Open-source building blocks reviewed for this bot, and where each fits:

| Project | Role | Used here |
|---|---|---|
| [TradingAgents](https://github.com/TauricResearch/TradingAgents) | Multi-agent LLM analysis: news, sentiment, fundamentals, debates | yes, as the debate opinion |
| [freqtrade](https://github.com/freqtrade/freqtrade) | Crypto execution on Binance, Bybit, OKX, Kraken and more; stop loss, backtests, Telegram | yes, as the crypto broker |
| [OpenAlgo](https://github.com/marketcalls/openalgo) | One REST API for 30+ Indian brokers (AGPL-3.0) | yes, as the India broker |
| [Alpaca](https://alpaca.markets) | US stock broker API with free paper trading | yes, as the US broker |
| [hummingbot](https://github.com/hummingbot/hummingbot) | Market making, arbitrage, grid bots on CEX/DEX | not needed for directional trading; good next step for market making |
| [World Monitor](https://github.com/koala73/worldmonitor) | Real-time global events dashboard (conflicts, flights, markets; AGPL-3.0) | candidate extra news feed for the debate |
| [Microsoft Qlib](https://github.com/microsoft/qlib) | ML alpha research platform, no live trading | candidate for richer factor models |
| [NautilusTrader](https://github.com/nautechsystems/nautilus_trader), [LEAN](https://github.com/QuantConnect/Lean) | Institutional-grade event-driven engines | worth migrating to if you need tick-level execution |
| Chronos / TimesFM | Time-series foundation models | studies find them competitive with simple baselines on prices, not better for trading, so the bot uses a transparent analog forecast instead |

## Extending

- New opinion: write a function returning `signals.Opinion`, add a weight in
  `SignalConfig` and `ensemble.combine`, and call it in `Engine.decide`.
- New broker: subclass `brokers.base.Broker` (five methods) and register it in
  `brokers.build_broker`.
- Tests: `pytest tests/test_autotrader.py`.
