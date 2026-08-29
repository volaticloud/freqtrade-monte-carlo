"""End-to-end behaviour of the resampling itself."""

from __future__ import annotations

import pytest

from freqtrade_monte_carlo import SimulationConfig, TradeResult, run_simulation

TRADES = [TradeResult(profit_abs=p, profit_ratio=p / 1000) for p in (50, -30, 80, -20, 60, -45, 25, 15)]


def _cfg(**kw):
    return SimulationConfig(iterations=200, seed=1, initial_capital=1000.0, **kw)


def test_a_seed_reproduces_a_run_within_this_implementation():
    a = run_simulation(TRADES, _cfg())
    b = run_simulation(TRADES, _cfg())
    assert a.max_drawdown.percentiles == b.max_drawdown.percentiles


def test_different_seeds_give_different_draws():
    a = run_simulation(TRADES, _cfg())
    b = run_simulation(TRADES, SimulationConfig(iterations=200, seed=2, initial_capital=1000.0))
    assert a.max_drawdown.percentiles != b.max_drawdown.percentiles


def test_shuffling_cannot_change_the_final_balance():
    """Reordering a fixed set of profits cannot change their sum.

    This is arithmetic, not a defect — and it is why the CLI explains itself
    rather than printing identical percentiles with no comment. Shuffling is
    still informative: the PATH varies, so drawdown varies.
    """
    result = run_simulation(TRADES, _cfg(method="trade_shuffle"))
    fb = result.final_balance
    assert fb.max - fb.min == pytest.approx(0.0)
    assert result.max_drawdown.max > result.max_drawdown.min


def test_bootstrap_does_spread_the_final_balance():
    result = run_simulation(TRADES, _cfg(method="bootstrap"))
    assert result.final_balance.max > result.final_balance.min


def test_risk_of_ruin_is_a_probability():
    result = run_simulation(TRADES, _cfg(method="bootstrap"))
    assert 0.0 <= result.risk_of_ruin <= 1.0


def test_a_strategy_that_cannot_lose_half_has_no_ruin():
    tiny = [TradeResult(profit_abs=1.0, profit_ratio=0.001) for _ in range(30)]
    assert run_simulation(tiny, _cfg()).risk_of_ruin == 0.0


def test_a_strategy_that_always_blows_up_has_certain_ruin():
    fatal = [TradeResult(profit_abs=-600.0, profit_ratio=-0.6) for _ in range(3)]
    assert run_simulation(fatal, _cfg()).risk_of_ruin == 1.0


def test_equity_curves_are_returned_per_percentile():
    result = run_simulation(TRADES, _cfg())
    assert set(result.equity_curves) == {"p5", "p25", "p50", "p75", "p95"}
    assert all(len(c) > 0 for c in result.equity_curves.values())


@pytest.mark.parametrize(
    "kw,msg",
    [
        ({"iterations": 5}, "iterations"),
        ({"iterations": 999999}, "iterations"),
        ({"method": "nonsense"}, "method"),
        ({"initial_capital": 0}, "initial_capital"),
    ],
)
def test_invalid_config_is_rejected(kw, msg):
    base = {"iterations": 200, "seed": 1, "initial_capital": 1000.0}
    base.update(kw)
    with pytest.raises(ValueError) as exc:
        run_simulation(TRADES, SimulationConfig(**base))
    assert msg in str(exc.value)


def test_no_trades_is_rejected():
    with pytest.raises(ValueError):
        run_simulation([], _cfg())


def test_the_input_trades_are_not_mutated():
    """A fee override rewrites fee_amount — on a copy, never the caller's list."""
    original = TRADES[0].fee_amount
    run_simulation(TRADES, _cfg(fee_rate=0.001, include_fees=True))
    assert TRADES[0].fee_amount == original
