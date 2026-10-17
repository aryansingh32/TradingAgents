"""Walk-forward parameter optimization.

Tuning parameters on all of history and reporting how well they did on that
same history is the most common way trading systems fool their authors. This
optimizer only reports results on data the parameters were not chosen on:

1. history is split into ``folds + 1`` consecutive blocks;
2. for each fold, every parameter set is simulated on all blocks before the
   test block (anchored training window) and the best by the objective is
   picked;
3. that set is then traded on the unseen test block;
4. the test blocks, stitched together, are the honest out-of-sample result.

The objective is Van Tharp's SQN (mean R / stdev R * sqrt(trades)), which
rewards consistent edge over many trades rather than one lucky run, with
parameter sets that trade too rarely or draw down too far disqualified. Every
symbol in the config is scored together (mean SQN), so parameters must work
across markets rather than fit one chart.

The parameters recommended for live use are the best set over the full
history, reported next to its out-of-sample record.
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np

from autotrader.backtest import Prepared, simulate
from autotrader.config import RiskConfig, SignalConfig, TradeManagementConfig

DEFAULT_GRID = {
    "signals.buy_threshold": [0.25, 0.35, 0.45],
    "signals.sell_threshold": [-0.15, -0.3],
    "signals.trend_filter": [True, False],
    "risk.atr_stop_mult": [1.5, 2.5, 3.5],
    "risk.reward_risk": [2.0, 3.0],
    "risk.trail_after_atr": [0.0, 1.5],
    "trade.partial_tp_r": [0.0, 1.0],
}


def expand(grid: dict[str, list]) -> list[dict[str, object]]:
    keys = list(grid)
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*(grid[k] for k in keys))]


def apply_params(risk: RiskConfig, sig: SignalConfig, params: dict,
                 trade: TradeManagementConfig | None = None):
    def part(prefix):
        return {k.split(".", 1)[1]: v for k, v in params.items() if k.startswith(prefix)}

    return (replace(risk, **part("risk.")), replace(sig, **part("signals.")),
            replace(trade or TradeManagementConfig(), **part("trade.")))


def objective(stats: dict, min_trades: int, max_dd: float) -> float:
    if not stats or stats["trades"] < min_trades or stats["max_drawdown"] < -max_dd:
        return -np.inf
    return stats["sqn"]


def _score(preps: list[tuple[Prepared, float]], cfgs, start_frac, end_frac, min_trades, max_dd):
    risk, sig, trade = cfgs
    scores = []
    for p, cost in preps:
        n = len(p.df)
        res = simulate(p, risk, sig, cost, start=int(n * start_frac), end=int(n * end_frac), trade_cfg=trade)
        scores.append(objective(res.stats, min_trades, max_dd))
    return float(np.mean(scores)) if scores else -np.inf


def walk_forward(preps: list[tuple[Prepared, float]], base_risk: RiskConfig, base_sig: SignalConfig,
                 grid: dict | None = None, folds: int = 4, min_trades: int = 10,
                 max_drawdown: float = 0.30, base_trade: TradeManagementConfig | None = None) -> dict:
    """``preps``: (prepared history, one-way cost) per symbol."""
    combos = expand(grid or DEFAULT_GRID)
    blocks = folds + 1
    fold_reports = []
    oos = {i: [] for i in range(len(preps))}
    for f in range(1, blocks):
        train_end, test_end = f / blocks, (f + 1) / blocks
        best, best_score = None, -np.inf
        for params in combos:
            cfgs = apply_params(base_risk, base_sig, params, base_trade)
            s = _score(preps, cfgs, 0.0, train_end, max(3, min_trades * f // folds), max_drawdown)
            if s > best_score:
                best, best_score = params, s
        if best is None:
            fold_reports.append({"fold": f, "params": None, "train_sqn": None, "test": None})
            continue
        risk, sig, trade = apply_params(base_risk, base_sig, best, base_trade)
        tests = []
        for i, (p, cost) in enumerate(preps):
            n = len(p.df)
            res = simulate(p, risk, sig, cost, start=int(n * train_end), end=int(n * test_end), trade_cfg=trade)
            oos[i] += res.trades
            tests.append(res.stats)
        fold_reports.append({"fold": f, "params": best, "train_sqn": best_score, "test": tests})

    final, final_score = None, -np.inf
    for params in combos:
        s = _score(preps, apply_params(base_risk, base_sig, params, base_trade), 0.0, 1.0, min_trades, max_drawdown)
        if s > final_score:
            final, final_score = params, s

    oos_summary = []
    for i in range(len(preps)):
        rs = np.array([t["r"] for t in oos[i]])
        oos_summary.append({
            "trades": len(rs),
            "win_rate": float((rs > 0).mean()) if len(rs) else 0.0,
            "expectancy_r": float(rs.mean()) if len(rs) else 0.0,
            "sqn": float(rs.mean() / rs.std(ddof=1) * np.sqrt(len(rs))) if len(rs) > 1 and rs.std(ddof=1) > 0 else 0.0,
            "total_r": float(rs.sum()) if len(rs) else 0.0,
        })
    return {"folds": fold_reports, "out_of_sample": oos_summary, "recommended": final,
            "recommended_full_history_sqn": final_score, "combos_tested": len(combos)}


def tuned_file_payload(params: dict) -> dict:
    out: dict[str, dict] = {"risk": {}, "signals": {}, "trade": {}}
    for k, v in params.items():
        section, key = k.split(".", 1)
        out[section][key] = v
    return out
