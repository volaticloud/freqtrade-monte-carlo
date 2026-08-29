"""Turn one backtest into a distribution of outcomes.

    from freqtrade_monte_carlo import load_trades, run_simulation, SimulationConfig

    trades = load_trades("backtest-result.json")
    result = run_simulation(trades, SimulationConfig(iterations=5000, seed=42))

    print("median return", result.total_return.percentiles["p50"])
    print("risk of ruin ", result.risk_of_ruin)

Your backtest is one ordering of your trades. Resampling it thousands of times
shows how much of the result was the strategy and how much was the sequence.
"""

from .engine import (
    CRYPTO_TRADING_DAYS_PER_YEAR,
    DistributionStats,
    SimulationConfig,
    SimulationResult,
    TradeResult,
    compute_distribution_stats,
    compute_drawdown,
    compute_equity_curve,
    compute_profit_factor,
    compute_sharpe,
    compute_sortino,
    run_simulation,
)
from .freqtrade import FreqtradeResultError, load_trades, trades_from_export

__version__ = "0.1.0"

__all__ = [
    "CRYPTO_TRADING_DAYS_PER_YEAR",
    "DistributionStats",
    "FreqtradeResultError",
    "SimulationConfig",
    "SimulationResult",
    "TradeResult",
    "compute_distribution_stats",
    "compute_drawdown",
    "compute_equity_curve",
    "compute_profit_factor",
    "compute_sharpe",
    "compute_sortino",
    "load_trades",
    "run_simulation",
    "trades_from_export",
]
