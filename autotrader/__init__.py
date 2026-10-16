"""AutoTrader: an automated trading bot built on top of TradingAgents.

The bot combines three independent opinions about each symbol:

* a technical read of the price history (trend, momentum, volatility),
* a history-analog forecast that looks for past situations resembling today
  and measures what happened next, and
* optionally, the TradingAgents multi-agent LLM debate (news, sentiment,
  fundamentals, bull/bear researchers, risk team).

A risk manager sizes every trade from its stop distance and blocks new trades
after a daily loss limit, and a broker adapter executes the orders on a paper
account, Alpaca (US stocks), OpenAlgo (Indian brokers) or freqtrade (crypto).

Paper trading is the default. Live trading needs ``mode: "live"`` in the config
and ``AUTOTRADER_LIVE_CONFIRM=I_UNDERSTAND_THE_RISK`` in the environment.
"""

__version__ = "0.1.0"
