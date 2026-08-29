"""Command line interface.

    freqtrade-monte-carlo run backtest-result.json --iterations 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Sequence

from .engine import SimulationConfig, SimulationResult, run_simulation
from .freqtrade import FreqtradeResultError, load_trades


def _fmt(v: float) -> str:
    if abs(v) >= 1e9:
        return "inf" if v > 0 else "-inf"
    return f"{v:,.2f}"


def _render(result: SimulationResult, capital: float, ruin_pct: float) -> str:
    lines = [f"{result.iterations_completed:,} simulations", ""]
    rows = [
        ("Final balance", result.final_balance, ""),
        ("Total return %", result.total_return, ""),
        ("Max drawdown", result.max_drawdown, ""),
        ("Sharpe", result.sharpe_ratio, ""),
        ("Sortino", result.sortino_ratio, ""),
        ("Profit factor", result.profit_factor, ""),
        ("Win rate", result.win_rate, ""),
    ]
    keys = [k for k in result.final_balance.percentiles]
    header = "  ".join(f"{k:>12}" for k in keys)
    lines.append(f"{'':<16}{header}")
    for label, stats, _ in rows:
        cells = "  ".join(f"{_fmt(stats.percentiles[k]):>12}" for k in keys)
        lines.append(f"{label:<16}{cells}")
    lines += ["", f"Risk of ruin     {result.risk_of_ruin:.1%}  "
                  f"(chance of losing {ruin_pct:.0f}% of {_fmt(capital)} at any point)"]

    # trade_shuffle reorders the SAME fixed profits, and a sum does not care
    # about order — so every run ends on the same balance by arithmetic, not by
    # accident. Say so, or the identical percentiles read as a broken tool.
    fb = result.final_balance
    if fb.max - fb.min < 1e-9:
        lines += [
            "",
            "Final balance is identical across every run because reordering a fixed",
            "set of profits cannot change their sum. That is arithmetic, not a bug.",
            "Shuffling still tells you what you came for — look at max drawdown, which",
            "varies a lot: the same trades in a worse order dig a deeper hole.",
            "For a spread of RETURNS, resample with replacement: --method bootstrap.",
        ]
    lines += ["", "A backtest is one ordering of your trades. These are the others."]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="freqtrade-monte-carlo",
        description="Turn one Freqtrade backtest into a distribution of outcomes.",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="resample a backtest's trades")
    p.add_argument("result", help="Freqtrade backtest result (exported with --export trades)")
    p.add_argument("--strategy", help="strategy name, if the file holds more than one")
    p.add_argument("--iterations", type=int, default=1000, help="simulations to run (10-50000)")
    p.add_argument("--method", default="trade_shuffle",
                   choices=["trade_shuffle", "bootstrap", "parametric"])
    p.add_argument("--capital", type=float, default=1000.0, help="starting capital")
    p.add_argument("--seed", type=int, help="fix the draws for a reproducible run")
    p.add_argument("--ruin-threshold", type=float, default=50.0,
                   help="%% capital loss counted as ruin (default 50)")
    p.add_argument("--include-fees", action="store_true")
    p.add_argument("--risk-free-rate", type=float, default=0.0)

    args = parser.parse_args(argv)
    try:
        trades = load_trades(args.result, args.strategy)
        config = SimulationConfig(
            iterations=args.iterations, method=args.method, initial_capital=args.capital,
            seed=args.seed, risk_of_ruin_threshold=args.ruin_threshold,
            include_fees=args.include_fees, risk_free_rate=args.risk_free_rate,
        )
        result = run_simulation(trades, config)
    except (FreqtradeResultError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        payload = asdict(result)
        payload.pop("equity_curves", None)  # too large for a terminal; use the library
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(_render(result, args.capital, args.ruin_threshold))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
