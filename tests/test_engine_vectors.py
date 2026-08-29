"""Cross-implementation golden vectors for the Monte Carlo maths.

**What these can and cannot pin.** Go's ``math/rand`` and Python's ``random``
produce different sequences from the same seed, so the resampled *draws* cannot
be made identical across the two implementations — a seed reproduces a run
within one implementation only. What must agree is everything downstream of the
draw: equity curve, drawdown, Sharpe, Sortino, profit factor, the distribution
statistics and the percentile interpolation. Those are deterministic given a
trade sequence, and those are what this file pins.

Two correct implementations then produce statistically equivalent distributions
from different draws, which is the honest guarantee for a Monte Carlo tool.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from freqtrade_monte_carlo.engine import (
    TradeResult,
    _safe_float,
    compute_distribution_stats,
    compute_equity_curve,
)

VECTORS = json.loads((Path(__file__).parent / "testdata" / "simulation_vectors.json").read_text())
CURVE_CASES = [c for c in VECTORS if c["kind"] == "equity_curve"]
DIST_CASES = [c for c in VECTORS if c["kind"] == "distribution"]


def _approx(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-9 * max(1.0, abs(b))


@pytest.mark.parametrize("case", CURVE_CASES, ids=[c["name"] for c in CURVE_CASES])
def test_equity_curve_matches_reference(case):
    trades = [
        TradeResult(
            profit_abs=t.get("profit_abs", 0.0),
            profit_ratio=t.get("profit_ratio", 0.0),
            fee_amount=t.get("fee_amount", 0.0),
            is_parametric=t.get("is_parametric", False),
            original_fee_amount=t.get("original_fee_amount", 0.0),
        )
        for t in case["trades"]
    ]
    curve, run = compute_equity_curve(
        trades,
        case.get("initialCapital", 0.0),
        case.get("includeFees", False),
        case.get("feeOverridden", False),
        case.get("riskFreeRate", 0.0),
    )
    want = case["expected"]

    assert len(curve) == len(want["curve"])
    for i, (got, exp) in enumerate(zip(curve, want["curve"])):
        assert _approx(got, exp), f"curve[{i}]"

    for name, got in (
        ("finalBalance", run.final_balance),
        ("minBalance", run.min_balance),
        ("maxDrawdown", run.max_drawdown),
        ("sharpe", _safe_float(run.sharpe)),
        ("sortino", _safe_float(run.sortino)),
        ("profitFactor", _safe_float(run.profit_factor)),
        ("winRate", run.win_rate),
    ):
        assert _approx(got, want[name]), f"{name}: {got} != {want[name]}"


@pytest.mark.parametrize("case", DIST_CASES, ids=[c["name"] for c in DIST_CASES])
def test_distribution_stats_match_reference(case):
    st = compute_distribution_stats(case["values"], case["levels"])
    want = case["expected"]
    assert _approx(st.min, want["min"])
    assert _approx(st.max, want["max"])
    assert _approx(st.mean, want["mean"])
    assert _approx(st.std, want["std"])
    for k, v in want["percentiles"].items():
        assert _approx(st.percentiles[k], v), f"percentile {k}"


def test_vector_file_keeps_its_dangerous_cases():
    names = {c["name"] for c in VECTORS}
    for required in (
        "blowup_floors_at_zero",
        "all_winners_infinite_ratios",
        "single_trade_no_ratios",
        "parametric_compounds_on_balance",
        "fee_override_reverses_original",
    ):
        assert required in names, f"vector file lost the {required!r} case"


def test_banker_rounding_would_pick_a_different_run():
    """Percentile-curve indexing and downsampling use Go's rounding, not Python's."""
    from freqtrade_monte_carlo.engine import _round_half_away_from_zero

    assert _round_half_away_from_zero(2.5) == 3
    assert round(2.5) == 2  # what we must NOT do
    assert _round_half_away_from_zero(-2.5) == -3


def test_sample_not_population_stddev():
    """Distribution std divides by n-1; population (n) understates the spread."""
    from freqtrade_monte_carlo.engine import _sample_std

    assert _sample_std([1.0, 2.0, 3.0], 2.0) == pytest.approx(1.0)


def test_infinities_are_capped_not_dropped():
    assert _safe_float(math.inf) == 1e9
    assert _safe_float(-math.inf) == -1e9
    assert _safe_float(math.nan) == 0.0
