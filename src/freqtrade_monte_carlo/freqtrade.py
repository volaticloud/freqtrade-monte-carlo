"""Read the trades out of Freqtrade's own backtest export.

Needs the per-trade detail, so the backtest must have been exported with
trades::

    freqtrade backtesting --strategy MyStrategy --export trades

then point at ``user_data/backtest_results/backtest-result-*.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .engine import TradeResult

__all__ = ["FreqtradeResultError", "trades_from_export", "load_trades"]


class FreqtradeResultError(ValueError):
    """The file is not a Freqtrade backtest result we can read."""


def _num(entry: Mapping[str, Any], *keys: str) -> float:
    for k in keys:
        v = entry.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        return float(v)
    return 0.0


def trades_from_export(
    result: Mapping[str, Any],
    strategy: Optional[str] = None,
) -> list[TradeResult]:
    """Extract trades from a parsed Freqtrade backtest result."""
    strategies = result.get("strategy")
    if not isinstance(strategies, Mapping) or not strategies:
        raise FreqtradeResultError(
            "no 'strategy' object — is this a Freqtrade backtest result?"
        )
    if strategy is not None:
        if strategy not in strategies:
            raise FreqtradeResultError(
                f"strategy {strategy!r} not in this result; found: {', '.join(sorted(strategies))}"
            )
        block = strategies[strategy]
    elif len(strategies) > 1:
        raise FreqtradeResultError(
            f"this result holds {len(strategies)} strategies ({', '.join(sorted(strategies))}); "
            "pass strategy=... to choose one"
        )
    else:
        block = next(iter(strategies.values()))

    raw_trades = block.get("trades") if isinstance(block, Mapping) else None
    if not isinstance(raw_trades, Sequence) or isinstance(raw_trades, (str, bytes)):
        raise FreqtradeResultError(
            "no 'trades' array — re-run the backtest with `--export trades`, "
            "which is what writes the per-trade detail this needs"
        )
    if not raw_trades:
        raise FreqtradeResultError("this backtest produced no trades")

    out: list[TradeResult] = []
    for i, entry in enumerate(raw_trades):
        if not isinstance(entry, Mapping):
            raise FreqtradeResultError(f"trade[{i}]: expected an object")
        profit_abs = _num(entry, "profit_abs")
        fee = _num(entry, "trade_fee")
        if fee == 0.0:
            # Some Freqtrade versions split the fee across open and close.
            fee = _num(entry, "fee_open_cost") + _num(entry, "fee_close_cost")
        out.append(
            TradeResult(
                profit_abs=profit_abs,
                profit_ratio=_num(entry, "profit_ratio"),
                fee_amount=fee,
            )
        )
    return out


def load_trades(path: str | Path, strategy: Optional[str] = None) -> list[TradeResult]:
    """Read a Freqtrade backtest result file and extract its trades."""
    p = Path(path)
    try:
        raw = p.read_text()
    except OSError as exc:
        raise FreqtradeResultError(f"{p}: {exc}") from exc
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FreqtradeResultError(f"{p}: not valid JSON ({exc})") from exc
    return trades_from_export(result, strategy)
