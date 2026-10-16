"""Command line: ``python -m autotrader <command> --config bot.json``.

Commands:
  run        trade continuously (paper by default)
  once       one pass over all symbols, then exit (for cron / schedulers)
  backtest   walk-forward backtest of each symbol over its full history
  monitor    market table: history, regime, forecast, current read
  status     open positions, protective levels and the last decisions
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from autotrader.config import load_config
from autotrader.data import load_history


def _load_df(cfg, sym):
    return load_history(sym.price_symbol, cfg.data_source, cfg.interval, cfg.lookback_days, cfg.data_dir)


def cmd_run(cfg, args):
    from autotrader.engine import Engine

    engine = Engine(cfg)
    if args.command == "once":
        for symbol, d in engine.step().items():
            print(f"{symbol:12s} {d.action:4s} score {d.score:+.2f}")
            for reason in d.reasons:
                print(f"    {reason}")
    else:
        engine.run_forever()


def cmd_backtest(cfg, args):
    from autotrader.backtest import run_backtest

    for sym in cfg.symbols:
        df = _load_df(cfg, sym)
        if df.empty:
            print(f"{sym.symbol}: no data")
            continue
        res = run_backtest(df, cfg.risk, cfg.signals, sym.market,
                           starting_cash=cfg.brokers.get("paper", {}).get("starting_cash", 100000.0))
        s = res.stats
        print(f"\n== {sym.symbol} ({df.index[0].date()} .. {df.index[-1].date()}, {len(df)} bars)")
        if not s:
            print("   not enough history")
            continue
        print(f"   return {s['total_return']:+.1%} (buy & hold {s['buy_and_hold_return']:+.1%})  "
              f"CAGR {s['cagr']:+.1%}  max drawdown {s['max_drawdown']:.1%}  Sharpe {s['sharpe']:.2f}")
        print(f"   trades {s['trades']}  win rate {s['win_rate']:.0%}  profit factor {s['profit_factor']:.2f}")
        if args.trades:
            for t in res.trades:
                print(f"   {t['entry_time']:%Y-%m-%d} -> {t['exit_time']:%Y-%m-%d} "
                      f"{t['return']:+.1%} {t['reason']}")


def cmd_monitor(cfg, args):
    from autotrader.monitor import render, snapshot

    rows = []
    for sym in cfg.symbols:
        try:
            rows.append(snapshot(sym.symbol, _load_df(cfg, sym), cfg))
        except Exception as exc:
            rows.append({"symbol": sym.symbol, "error": str(exc)})
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
    else:
        render(rows, cfg.signals.forecast_horizon)


def cmd_status(cfg, args):
    state_path = cfg.data_path / "bot_state.json"
    signals_path = cfg.data_path / "signals.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"protections": {}}
    signals = json.loads(signals_path.read_text()) if signals_path.exists() else {}
    print(f"mode: {cfg.mode}   data: {cfg.data_path.resolve()}")
    print("\nOpen positions (bot-managed levels):")
    for sym, p in state["protections"].items() or []:
        print(f"  {sym:12s} entry {p['entry']:.2f}  stop {p['stop']:.2f}  take {p['take']:.2f}  high {p['highest']:.2f}")
    if not state["protections"]:
        print("  none")
    print("\nLast decisions:")
    for sym, s in signals.items():
        print(f"  {sym:12s} {s['action']:4s} score {s['score']:+.2f} at {s['at'][:16]} price {s['price']:.2f}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="autotrader", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "once", "backtest", "monitor", "status"])
    parser.add_argument("--config", "-c", default="autotrader.json")
    parser.add_argument("--trades", action="store_true", help="backtest: list every trade")
    parser.add_argument("--json", action="store_true", help="monitor: print JSON")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose or args.command == "run" else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    handler = {"run": cmd_run, "once": cmd_run, "backtest": cmd_backtest,
               "monitor": cmd_monitor, "status": cmd_status}[args.command]
    handler(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
