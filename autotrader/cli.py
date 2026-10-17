"""Command line: ``autotrader <command> --config bot.json``.

Commands:
  run        trade continuously (paper by default)
  once       one decision pass over all symbols, then exit
  monitor    live market table: history, trend, forecast, context, action
  backtest   backtest every symbol over its history with costs and protections
  optimize   walk-forward optimization; --apply saves the chosen parameters
  status     open positions, protective levels and the last decisions
  doctor     check every feed, broker, World Monitor, LLM key and notifier
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import timedelta
from pathlib import Path

from autotrader.config import load_config


def _market_data(cfg):
    from autotrader.engine import build_market_data

    return build_market_data(cfg)


def _histories(cfg, md, sym):
    df = md.candles(sym.price_symbol, sym.market, cfg.interval)
    sc = cfg.signals
    # The trend timeframe is loaded even with the filter off: the monitor shows it.
    htf = df if sc.trend_interval == cfg.interval else md.candles(
        sym.price_symbol, sym.market, sc.trend_interval,
        lookback=timedelta(days=max(cfg.lookback_days, int(sc.trend_ema * 1.6))))
    return df, htf


def cmd_run(cfg, args):
    from autotrader.engine import Engine

    engine = Engine(cfg)
    if args.command == "once":
        for symbol, d in engine.step().items():
            print(f"{symbol:12s} {d.action:4s} score {d.score:+.2f}")
            for reason in d.reasons:
                print(f"    {reason}")
        print(f"\njournal: {engine.dir / 'journal.jsonl'}")
    else:
        engine.run_forever()


def cmd_monitor(cfg, args):
    from autotrader.monitor import render, snapshot

    md = _market_data(cfg)
    context = None
    if cfg.signals.use_context and any(s.market == "crypto" for s in cfg.symbols):
        from autotrader.context import CryptoContext

        context = CryptoContext()
    while True:
        rows = []
        for sym in cfg.symbols:
            try:
                df, htf = _histories(cfg, md, sym)
                try:
                    price = md.quote(sym.price_symbol, sym.market).price
                except Exception:
                    price = None
                rows.append(snapshot(sym, df, htf, cfg, price, context))
            except Exception as exc:
                rows.append({"symbol": sym.symbol, "error": str(exc)})
        if args.json:
            print(json.dumps(rows, indent=2, default=str))
        else:
            render(rows, cfg)
        if not args.watch:
            return
        time.sleep(args.watch)


def _prepared(cfg, md):
    from autotrader.backtest import prepare

    out = []
    for sym in cfg.symbols:
        t = time.perf_counter()
        df, htf = _histories(cfg, md, sym)
        p = prepare(df, cfg.signals, cfg.interval, sym.market, htf)
        print(f"  {sym.symbol}: {len(df)} {cfg.interval} candles "
              f"{df.index[0]:%Y-%m-%d} .. {df.index[-1]:%Y-%m-%d %H:%M}, prepared in {time.perf_counter() - t:.1f}s")
        out.append((sym, p, cfg.costs[sym.market]))
    return out


def _print_stats(name, s):
    if not s:
        print(f"   {name}: not enough history")
        return
    print(f"   return {s['total_return']:+.1%} (buy & hold {s['buy_and_hold_return']:+.1%})  "
          f"CAGR {s['cagr']:+.1%}  max drawdown {s['max_drawdown']:.1%}  Sharpe {s['sharpe']:.2f}")
    print(f"   trades {s['trades']}  win rate {s['win_rate']:.0%}  avg win {s['avg_win_r']:+.2f}R  "
          f"avg loss {s['avg_loss_r']:+.2f}R  expectancy {s['expectancy_r']:+.2f}R  SQN {s['sqn']:.2f}")
    print(f"   profit factor {s['profit_factor']:.2f}  longest losing streak {s['max_losing_streak']}  "
          f"time in market {s['exposure']:.0%}")


def cmd_backtest(cfg, args):
    from autotrader.backtest import simulate

    md = _market_data(cfg)
    print("loading history:")
    for sym, p, cost in _prepared(cfg, md):
        t = time.perf_counter()
        res = simulate(p, cfg.risk, cfg.signals, cost, session=cfg.session, trade_cfg=cfg.trade)
        print(f"\n== {sym.symbol} ({sym.market}, cost {cost:.2%} per side, simulated in "
              f"{(time.perf_counter() - t) * 1000:.0f} ms)")
        _print_stats(sym.symbol, res.stats)
        if args.trades:
            for tr in res.trades:
                print(f"   {tr['entry_time']:%Y-%m-%d %H:%M} -> {tr['exit_time']:%Y-%m-%d %H:%M} "
                      f"{tr['r']:+.2f}R {tr['return']:+.1%} {tr['reason']}")


def cmd_optimize(cfg, args):
    from autotrader.optimize import DEFAULT_GRID, tuned_file_payload, walk_forward

    grid = json.loads(Path(args.grid).read_text()) if args.grid else DEFAULT_GRID
    md = _market_data(cfg)
    print("loading history:")
    preps = _prepared(cfg, md)
    t = time.perf_counter()
    report = walk_forward([(p, c) for _, p, c in preps], cfg.risk, cfg.signals, grid, folds=args.folds,
                          base_trade=cfg.trade)
    print(f"\ntested {report['combos_tested']} parameter sets x {args.folds} folds "
          f"in {time.perf_counter() - t:.1f}s")
    for f in report["folds"]:
        print(f"  fold {f['fold']}: picked {f['params']}")
    print("\nout-of-sample (trades on data the parameters were not chosen on):")
    for (sym, _, _), s in zip(preps, report["out_of_sample"], strict=True):
        print(f"  {sym.symbol:12s} trades {s['trades']:4d}  win rate {s['win_rate']:.0%}  "
              f"expectancy {s['expectancy_r']:+.2f}R  SQN {s['sqn']:.2f}  total {s['total_r']:+.1f}R")
    rec = report["recommended"]
    print(f"\nrecommended for live (best over full history, SQN {report['recommended_full_history_sqn']:.2f}): {rec}")
    if rec is None:
        print("no parameter set met the minimum trades / drawdown limits; keep the defaults")
        return
    oos_edge = sum(s["total_r"] for s in report["out_of_sample"])
    if args.apply:
        if oos_edge <= 0:
            print("NOT applied: the out-of-sample result has no positive edge. "
                  "Re-applying would only fit noise.")
            return
        out = Path(args.config).with_suffix(".tuned.json")
        out.write_text(json.dumps(tuned_file_payload(rec), indent=2))
        print(f"saved to {out}; the bot loads it next to the config")


def cmd_doctor(cfg, args):
    import os

    from autotrader.engine import build_market_data

    ok = True

    def report(name, fn):
        nonlocal ok
        try:
            print(f"  OK    {name}: {fn()}")
        except Exception as exc:
            ok = False
            print(f"  FAIL  {name}: {exc}")

    print("market data:")
    md = build_market_data(cfg)
    seen = set()
    for sym in [*cfg.symbols, *cfg.accumulate]:
        if sym.market in seen:
            continue
        seen.add(sym.market)
        report(f"{sym.market} candles {sym.price_symbol}",
               lambda s=sym: f"{len(md.candles(s.price_symbol, s.market, cfg.interval))} closed {cfg.interval} candles")
        report(f"{sym.market} quote {sym.price_symbol}", lambda s=sym: f"{md.quote(s.price_symbol, s.market).price:,.4f}")
    print("brokers:")
    from autotrader.brokers import build_broker

    for name, section in cfg.brokers.items():
        report(name, lambda n=name, sec=section: f"equity {build_broker(n, sec, cfg.data_path, cfg.mode == 'live').equity():,.2f}")
    print("intelligence:")
    if cfg.news.enabled:
        from tradingagents.dataflows.vendors.worldmonitor import WorldMonitorClient

        client = WorldMonitorClient(base_url=cfg.news.base_url)
        report("World Monitor news", lambda: f"{len(client.news(cfg.news.variant))} stories")
    if cfg.signals.use_context and "crypto" in seen:
        from autotrader.context import CryptoContext

        report("crypto Fear & Greed", lambda: CryptoContext()._fear_greed())
    if cfg.signals.use_llm:
        keys = [k for k in os.environ if k.endswith("_API_KEY") and os.environ[k] and k != "WORLDMONITOR_API_KEY"]
        report("LLM key", lambda: ", ".join(keys) if keys else (_ for _ in ()).throw(RuntimeError("no *_API_KEY set")))
    print("notifications:")
    from autotrader.notify import Notifier

    n = Notifier(cfg.notify)
    print(f"  telegram {'on' if n.telegram_on else 'off'}, discord {'on' if n.discord_on else 'off'}")
    print("\nall checks passed" if ok else "\nsome checks failed; the bot skips what it cannot reach")
    return 0 if ok else 1


def cmd_status(cfg, args):
    state_path = cfg.data_path / "bot_state.json"
    signals_path = cfg.data_path / "signals.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"protections": {}}
    signals = json.loads(signals_path.read_text()) if signals_path.exists() else {}
    print(f"mode: {cfg.mode}   data: {cfg.data_path.resolve()}")
    print("\nOpen positions (bot-managed levels):")
    for sym, p in state.get("protections", {}).items():
        print(f"  {sym:12s} entry {p['entry']:.2f}  stop {p['stop']:.2f}  take {p['take']:.2f}  high {p['highest']:.2f}")
    if not state.get("protections"):
        print("  none")
    for name, g in state.get("guards", {}).items():
        print(f"\nprotections [{name}]: peak equity {g.get('peak_equity', 0):,.2f}, losing streak "
              f"{g.get('loss_streak', 0)}, paused until {g.get('paused_until') or '-'}")
    print("\nLast decisions:")
    for sym, s in signals.items():
        print(f"  {sym:12s} {s['action']:4s} score {s['score']:+.2f} at {s['at'][:16]} price {s['price']:.2f}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="autotrader", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "once", "monitor", "backtest", "optimize", "status", "doctor"])
    parser.add_argument("--config", "-c", default="autotrader.json")
    parser.add_argument("--trades", action="store_true", help="backtest: list every trade")
    parser.add_argument("--json", action="store_true", help="monitor: print JSON")
    parser.add_argument("--watch", type=int, default=0, help="monitor: refresh every N seconds")
    parser.add_argument("--folds", type=int, default=4, help="optimize: walk-forward folds")
    parser.add_argument("--grid", help="optimize: JSON file mapping 'risk.x'/'signals.y' to value lists")
    parser.add_argument("--apply", action="store_true", help="optimize: save the recommended parameters")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose or args.command == "run" else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    handler = {"run": cmd_run, "once": cmd_run, "backtest": cmd_backtest, "optimize": cmd_optimize,
               "monitor": cmd_monitor, "status": cmd_status, "doctor": cmd_doctor}[args.command]
    return handler(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
