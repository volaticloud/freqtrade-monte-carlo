"""Reading trades out of Freqtrade's backtest export."""

from __future__ import annotations

import pytest

from freqtrade_monte_carlo.freqtrade import FreqtradeResultError, trades_from_export

OK = {"strategy": {"S": {"trades": [
    {"profit_abs": 12.5, "profit_ratio": 0.0125, "trade_fee": 0.4},
    {"profit_abs": -8.0, "profit_ratio": -0.008, "fee_open_cost": 0.2, "fee_close_cost": 0.2},
]}}}


def test_reads_profit_and_fees():
    trades = trades_from_export(OK)
    assert [t.profit_abs for t in trades] == [12.5, -8.0]
    assert trades[0].fee_amount == pytest.approx(0.4)
    # split fee fields are summed when trade_fee is absent
    assert trades[1].fee_amount == pytest.approx(0.4)


def test_missing_trades_array_explains_the_export_flag():
    with pytest.raises(FreqtradeResultError) as exc:
        trades_from_export({"strategy": {"S": {"total_trades": 3}}})
    assert "--export trades" in str(exc.value)


def test_multi_strategy_requires_a_choice():
    with pytest.raises(FreqtradeResultError) as exc:
        trades_from_export({"strategy": {"A": {"trades": []}, "B": {"trades": []}}})
    assert "strategy=" in str(exc.value)


def test_named_strategy_is_selected():
    two = {"strategy": {"A": {"trades": [{"profit_abs": 1.0}]}, "B": {"trades": [{"profit_abs": 2.0}]}}}
    assert trades_from_export(two, "B")[0].profit_abs == 2.0


def test_not_a_backtest_result():
    with pytest.raises(FreqtradeResultError):
        trades_from_export({"nope": 1})


def test_empty_trades_is_rejected():
    with pytest.raises(FreqtradeResultError):
        trades_from_export({"strategy": {"S": {"trades": []}}})


def test_booleans_are_not_read_as_numbers():
    trades = trades_from_export({"strategy": {"S": {"trades": [{"profit_abs": True}]}}})
    assert trades[0].profit_abs == 0.0
