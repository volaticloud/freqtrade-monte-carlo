"""Monte Carlo resampling of a backtest's trades.

A backtest is **one** ordering of your trades — one path out of the many that
could plausibly have happened. Resampling that path thousands of times turns a
single number into a distribution, which is the honest way to read it: not
"this strategy returns 2.3%" but "this strategy returns between -4% and 9%,
with a 12% chance of losing half the account along the way".

Three methods:

``trade_shuffle`` (default)
    Same trades, different order. Answers: how much of the result depended on
    the sequence? A strategy whose drawdown explodes under reordering was
    surviving on luck.
``bootstrap``
    Resample with replacement. Some trades repeat, others vanish — this widens
    the distribution and asks what a *different but similar* period looks like.
``parametric``
    Draw synthetic returns from a normal distribution fitted to the real ones.
    The most aggressive assumption, and the least faithful to fat tails; useful
    as a sanity check, not as evidence.

Nothing here does I/O or touches the network.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

__all__ = [
    "TradeResult",
    "SimulationConfig",
    "DistributionStats",
    "SimulationResult",
    "run_simulation",
    "compute_equity_curve",
    "compute_distribution_stats",
    "CRYPTO_TRADING_DAYS_PER_YEAR",
]

#: Crypto markets trade 24/7, so returns annualise on 365 days rather than the
#: 252 trading days used for equities.
CRYPTO_TRADING_DAYS_PER_YEAR = 365.0

_METHODS = ("trade_shuffle", "bootstrap", "parametric")


@dataclass
class TradeResult:
    """One trade's contribution. Build these with
    :func:`freqtrade_monte_carlo.freqtrade.trades_from_export`."""

    profit_abs: float = 0.0
    profit_ratio: float = 0.0
    fee_amount: float = 0.0
    #: True for synthetic trades, whose PnL is derived from the ratio and the
    #: running balance rather than taken from ``profit_abs``.
    is_parametric: bool = False
    #: The fee already baked into ``profit_abs`` by the backtest. Only used when
    #: overriding the fee rate, to reverse the original deduction.
    original_fee_amount: float = 0.0


@dataclass
class SimulationConfig:
    """Simulation parameters."""

    iterations: int = 1000
    method: str = "trade_shuffle"
    initial_capital: float = 1000.0
    confidence_levels: Sequence[float] = (5, 25, 50, 75, 95)
    #: Fixes the draws so a run is reproducible. See the note in the README:
    #: a seed reproduces a run within THIS implementation only.
    seed: Optional[int] = None
    risk_free_rate: float = 0.0
    #: Percentage capital loss counted as ruin. 50 means "lost half the account".
    risk_of_ruin_threshold: float = 50.0
    include_fees: bool = False
    #: Override the per-trade fee. None keeps the backtest's own fees.
    fee_rate: Optional[float] = None


@dataclass
class DistributionStats:
    """A metric's distribution across all runs."""

    min: float = 0.0
    max: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    percentiles: dict[str, float] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """Aggregated outcome.

    Sharpe, Sortino and profit factor are capped at ±1e9 — those legitimately go
    infinite (zero downside deviation, zero gross loss) and JSON cannot carry an
    infinity. Read a value at or near 1e9 as "effectively infinite".
    """

    final_balance: DistributionStats = field(default_factory=DistributionStats)
    total_return: DistributionStats = field(default_factory=DistributionStats)
    max_drawdown: DistributionStats = field(default_factory=DistributionStats)
    sharpe_ratio: DistributionStats = field(default_factory=DistributionStats)
    sortino_ratio: DistributionStats = field(default_factory=DistributionStats)
    profit_factor: DistributionStats = field(default_factory=DistributionStats)
    win_rate: DistributionStats = field(default_factory=DistributionStats)
    #: Probability (0-1) that the balance fell below the ruin line at any point.
    risk_of_ruin: float = 0.0
    equity_curves: dict[str, list[float]] = field(default_factory=dict)
    iterations_completed: int = 0


@dataclass
class _RunResult:
    final_balance: float
    min_balance: float
    max_drawdown: float
    sharpe: float
    sortino: float
    profit_factor: float
    win_rate: float
    equity_curve: list[float] = field(default_factory=list)


# --- helpers ---------------------------------------------------------------


def _round_half_away_from_zero(v: float) -> int:
    """Round like Go's math.Round, NOT Python's banker's-rounding round().

    Used for percentile-curve indexing and curve downsampling, where a
    half-value landing on the wrong side silently selects a different run.
    """
    return int(math.floor(v + 0.5)) if v >= 0 else int(math.ceil(v - 0.5))


def _safe_float(v: float) -> float:
    """NaN becomes 0; infinities cap at ±1e9 so the result stays JSON-safe."""
    if math.isnan(v):
        return 0.0
    if math.isinf(v):
        return 1e9 if v > 0 else -1e9
    return v


def _sample_std(values: Sequence[float], mean: float) -> float:
    """Sample standard deviation (n-1). Population (n) would understate spread."""
    n = len(values)
    if n < 2:
        return 0.0
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def compute_drawdown(curve: Sequence[float]) -> float:
    """Largest peak-to-trough fall, as a fraction (0-1)."""
    if not curve:
        return 0.0
    peak = curve[0]
    max_dd = 0.0
    for val in curve:
        if val > peak:
            peak = val
        if peak > 0:
            dd = (peak - val) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd


def compute_sharpe(returns: Sequence[float], risk_free_rate: float = 0.0) -> float:
    """Annualised Sharpe from per-trade returns. Zero for fewer than 2 returns."""
    n = len(returns)
    if n < 2:
        return 0.0
    per_trade_rf = risk_free_rate / CRYPTO_TRADING_DAYS_PER_YEAR
    excess = [r - per_trade_rf for r in returns]
    mean = sum(excess) / n
    variance = sum((e - mean) ** 2 for e in excess) / (n - 1)
    std = math.sqrt(variance)
    if std == 0:
        return 0.0
    return (mean / std) * math.sqrt(CRYPTO_TRADING_DAYS_PER_YEAR)


def compute_sortino(returns: Sequence[float], risk_free_rate: float = 0.0) -> float:
    """Annualised Sortino — like Sharpe but punishing only downside deviation.

    Returns +inf when nothing went down and the mean is positive. The downside
    variance divides by the FULL sample count, not the count of losing trades:
    that is the semi-deviation convention, and dividing by the smaller count
    would flatter a strategy with few but severe losses.
    """
    n = len(returns)
    if n < 2:
        return 0.0
    per_trade_rf = risk_free_rate / CRYPTO_TRADING_DAYS_PER_YEAR
    excess = [r - per_trade_rf for r in returns]
    mean = sum(excess) / n

    downside = [e for e in excess if e < 0]
    if not downside:
        return math.inf if mean > 0 else 0.0
    downside_variance = sum(e * e for e in downside) / n
    downside_std = math.sqrt(downside_variance)
    if downside_std == 0:
        return 0.0
    return (mean / downside_std) * math.sqrt(CRYPTO_TRADING_DAYS_PER_YEAR)


def compute_profit_factor(pnls: Sequence[float]) -> float:
    """Gross profit / gross loss. +inf when there were no losses at all."""
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = sum(-p for p in pnls if p < 0)
    if gross_loss == 0:
        return math.inf if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def compute_equity_curve(
    trades: Sequence[TradeResult],
    initial_capital: float,
    include_fees: bool = False,
    fee_overridden: bool = False,
    risk_free_rate: float = 0.0,
) -> tuple[list[float], _RunResult]:
    """Walk one trade sequence and derive that run's metrics.

    Deterministic given the sequence — this is the part pinned by the shared
    golden vectors.
    """
    curve = [initial_capital]
    balance = initial_capital
    min_balance = initial_capital
    returns: list[float] = []
    pnls: list[float] = []
    wins = 0

    for t in trades:
        if t.is_parametric:
            # Synthetic trades compound: PnL comes from the ratio applied to the
            # RUNNING balance, not from a fixed absolute figure.
            pnl = balance * t.profit_ratio
            if include_fees:
                pnl -= t.fee_amount
        elif fee_overridden and include_fees:
            # profit_abs is already net of the ORIGINAL fee, so add that back
            # before taking off the replacement.
            pnl = t.profit_abs + t.original_fee_amount - t.fee_amount
        else:
            pnl = t.profit_abs

        pnls.append(pnl)
        prev_balance = balance
        balance += pnl
        if balance < 0:
            balance = 0.0  # you cannot lose more than the account holds
        curve.append(balance)
        if balance < min_balance:
            min_balance = balance
        returns.append(pnl / prev_balance if prev_balance > 0 else 0.0)
        if pnl > 0:
            wins += 1

    return curve, _RunResult(
        final_balance=balance,
        min_balance=min_balance,
        max_drawdown=compute_drawdown(curve),
        sharpe=compute_sharpe(returns, risk_free_rate),
        sortino=compute_sortino(returns, risk_free_rate),
        profit_factor=compute_profit_factor(pnls),
        win_rate=wins / len(trades) if trades else 0.0,
        equity_curve=curve,
    )


def _percentiles(sorted_values: Sequence[float], levels: Iterable[float]) -> dict[str, float]:
    """Linear-interpolated percentiles over an already-sorted sequence."""
    n = len(sorted_values)
    if n == 0:
        return {}
    out: dict[str, float] = {}
    for level in levels:
        rank = (level / 100.0) * (n - 1)
        lower = max(0, math.floor(rank))
        upper = min(n - 1, math.ceil(rank))
        if lower == upper:
            out[f"p{level:.0f}"] = sorted_values[lower]
        else:
            frac = rank - lower
            out[f"p{level:.0f}"] = sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac
    return out


def compute_distribution_stats(
    values: Sequence[float],
    levels: Iterable[float] = (5, 25, 50, 75, 95),
) -> DistributionStats:
    """Min, max, mean, sample std and interpolated percentiles."""
    n = len(values)
    if n == 0:
        return DistributionStats()
    mean = sum(values) / n
    ordered = sorted(values)
    return DistributionStats(
        min=ordered[0],
        max=ordered[-1],
        mean=mean,
        std=_sample_std(values, mean),
        percentiles=_percentiles(ordered, levels),
    )


def _downsample(curve: Sequence[float], max_points: int = 200) -> list[float]:
    """Thin a curve for plotting, always keeping the first and last points."""
    if len(curve) <= max_points:
        return list(curve)
    step = (len(curve) - 1) / (max_points - 1)
    out = [curve[_round_half_away_from_zero(i * step)] for i in range(max_points - 1)]
    out.append(curve[-1])
    return out


# --- resampling ------------------------------------------------------------


def _shuffle(trades: Sequence[TradeResult], rng: random.Random) -> list[TradeResult]:
    out = list(trades)
    rng.shuffle(out)
    return out


def _bootstrap(trades: Sequence[TradeResult], rng: random.Random) -> list[TradeResult]:
    n = len(trades)
    return [trades[rng.randrange(n)] for _ in range(n)]


def _parametric(trades: Sequence[TradeResult], rng: random.Random) -> list[TradeResult]:
    n = len(trades)
    ratios = [t.profit_ratio for t in trades]
    mean = sum(ratios) / n
    variance = sum((r - mean) ** 2 for r in ratios) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(variance)
    mean_fee = sum(t.fee_amount for t in trades) / n
    return [
        TradeResult(
            profit_ratio=rng.gauss(mean, std),
            profit_abs=0.0,
            is_parametric=True,
            fee_amount=mean_fee,
        )
        for _ in range(n)
    ]


def run_simulation(
    trades: Sequence[TradeResult],
    config: Optional[SimulationConfig] = None,
) -> SimulationResult:
    """Resample ``trades`` and aggregate the runs into distributions.

    Raises:
        ValueError: for an empty trade list or an invalid configuration.
    """
    config = config or SimulationConfig()
    if not trades:
        raise ValueError("at least one trade is required to simulate")
    if not 10 <= config.iterations <= 50000:
        raise ValueError(f"iterations must be between 10 and 50000, got {config.iterations}")
    if config.method not in _METHODS:
        raise ValueError(f"method must be one of {', '.join(_METHODS)}, got {config.method!r}")
    if config.initial_capital <= 0:
        raise ValueError(f"initial_capital must be positive, got {config.initial_capital}")

    sim_trades = [
        TradeResult(t.profit_abs, t.profit_ratio, t.fee_amount, t.is_parametric, t.original_fee_amount)
        for t in trades
    ]

    fee_overridden = config.fee_rate is not None
    if fee_overridden:
        # Approximate per-trade fee as rate x initial capital, and remember the
        # original so compute_equity_curve can reverse it.
        fee_per_trade = config.initial_capital * float(config.fee_rate)
        for t in sim_trades:
            t.original_fee_amount = t.fee_amount
            t.fee_amount = fee_per_trade

    rng = random.Random(config.seed)
    resample = {"bootstrap": _bootstrap, "parametric": _parametric}.get(config.method, _shuffle)

    runs: list[_RunResult] = []
    for _ in range(config.iterations):
        sequence = resample(sim_trades, rng)
        _, run = compute_equity_curve(
            sequence, config.initial_capital, config.include_fees, fee_overridden, config.risk_free_rate
        )
        runs.append(run)

    return _aggregate(runs, config)


def _aggregate(runs: Sequence[_RunResult], config: SimulationConfig) -> SimulationResult:
    n = len(runs)
    capital = config.initial_capital
    levels = list(config.confidence_levels)

    ruin_line = capital * (1 - config.risk_of_ruin_threshold / 100.0)
    ruin_count = sum(1 for r in runs if r.min_balance < ruin_line)

    stats = lambda vals: compute_distribution_stats(vals, levels)  # noqa: E731
    result = SimulationResult(
        final_balance=stats([r.final_balance for r in runs]),
        total_return=stats([((r.final_balance - capital) / capital) * 100.0 for r in runs]),
        max_drawdown=stats([r.max_drawdown for r in runs]),
        sharpe_ratio=stats([_safe_float(r.sharpe) for r in runs]),
        sortino_ratio=stats([_safe_float(r.sortino) for r in runs]),
        profit_factor=stats([_safe_float(r.profit_factor) for r in runs]),
        win_rate=stats([r.win_rate for r in runs]),
        risk_of_ruin=ruin_count / n,
        iterations_completed=n,
    )

    # One representative equity curve per percentile, ranked by final balance.
    order = sorted(range(n), key=lambda i: runs[i].final_balance)
    for level in levels:
        idx = min(max(_round_half_away_from_zero((level / 100.0) * (n - 1)), 0), n - 1)
        result.equity_curves[f"p{level:.0f}"] = _downsample(runs[order[idx]].equity_curve)
    return result
