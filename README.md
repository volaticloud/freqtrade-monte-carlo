# freqtrade-monte-carlo

Your backtest is **one** ordering of your trades — one path out of the many that
could plausibly have happened. Resampling it thousands of times turns a single
number into a distribution, which is the honest way to read it.

Not *"this strategy returns 82%"* but *"this strategy returns between 47% and
115%, and there's a 4% chance it halves the account on the way"*.

```console
$ freqtrade-monte-carlo run backtest-result.json --method bootstrap --iterations 5000
5,000 simulations

                          p5           p25           p50           p75           p95
Final balance       1,466.56      1,680.30      1,816.58      1,957.93      2,151.04
Total return %         46.66         68.03         81.66         95.79        115.10
Max drawdown            0.04          0.05          0.06          0.08          0.11
Sharpe                  4.71          5.30          5.62          5.94          6.44
Sortino                 7.95          9.15          9.86         10.58        11.71
Profit factor           1.63          1.80          1.90          2.01          2.19
Win rate                0.55          0.59          0.62          0.64          0.68

Risk of ruin      0.0%  (chance of losing 50% of 1,000.00 at any point)
```

Reads Freqtrade's own backtest export. No conversion step.

## Install

```bash
pip install freqtrade-monte-carlo
```

No dependencies. Python 3.9+.

The backtest must have been exported with per-trade detail:

```bash
freqtrade backtesting --strategy MyStrategy --export trades
```

## The three methods, and which question each answers

| Method | What it does | The question it answers |
|---|---|---|
| `trade_shuffle` *(default)* | Same trades, different order | How much of the result depended on the **sequence**? |
| `bootstrap` | Resample with replacement | What does a **different but similar** period look like? |
| `parametric` | Draw from a normal fitted to your returns | A stress test — the most aggressive assumption |

**Shuffling will not move your final balance, and that is arithmetic, not a bug.**
Reordering a fixed set of profits cannot change their sum. What it *does* move is
the path — and therefore drawdown, Sharpe and Sortino. The same trades dealt in a
crueller order dig a deeper hole, and that hole is what stops people out in real
life. For a spread of *returns*, use `--method bootstrap`.

`parametric` assumes returns are normally distributed. They are not — real
returns have fat tails — so treat it as a sanity check, never as evidence.

## Risk of ruin

The number most worth your attention. It is the share of simulations where the
balance dropped below the ruin line *at any point*, not just at the end:

```bash
freqtrade-monte-carlo run backtest-result.json --ruin-threshold 25   # losing a quarter
```

A strategy with a good median and a 15% risk of ruin is not a good strategy. It
is a strategy that works most of the time.

## As a library

```python
from freqtrade_monte_carlo import load_trades, run_simulation, SimulationConfig

trades = load_trades("backtest-result.json")
result = run_simulation(trades, SimulationConfig(
    method="bootstrap", iterations=5000, initial_capital=1000, seed=42,
))

print(result.total_return.percentiles["p5"])   # the bad-luck case
print(result.max_drawdown.percentiles["p95"])  # the deep hole
print(result.risk_of_ruin)
result.equity_curves["p50"]                    # a curve you can plot
```

## Reading the numbers

Sharpe and Sortino are annualised on **365 days** — crypto trades 24/7, so the
252-day equity convention would understate them.

Sortino's downside deviation divides by the full sample count, not the count of
losing trades. That is the semi-deviation convention; dividing by the smaller
count would flatter a strategy with few but severe losses.

Sharpe, Sortino and profit factor legitimately go infinite — zero downside
deviation, zero gross loss — and JSON cannot carry an infinity, so they are
capped at ±1e9. **Read a value at or near 1e9 as "effectively infinite", which
in practice means "too few losing trades to measure".**

Balance floors at zero. You cannot lose more than the account holds, and a
simulation that went negative would report a fictional drawdown.

## What this is not

Resampling explores the trades you *got*. It cannot tell you about the trades
your strategy never took because the period never offered them. It does not know
about look-ahead bias in your indicators, survivorship bias in your pair list,
fills you would not have received, or a regime that simply doesn't repeat. A
tight distribution over a fundamentally broken backtest is still a broken
backtest.

Two companions for the questions this one cannot answer:

- Did the result survive data the optimiser never saw? →
  [freqtrade-overfit-score](https://github.com/volaticloud/freqtrade-overfit-score)
- Were the parameters themselves worth trusting? →
  [freqtrade-hyperopt-guard](https://github.com/volaticloud/freqtrade-hyperopt-guard)

Trading involves risk of loss. Nothing here is financial advice.

## Accuracy, and one honest limitation

The same engine runs in Go inside [VolatiCloud](https://volaticloud.com). Both
implementations are pinned to the golden vectors in
`tests/testdata/simulation_vectors.json`.

**Those vectors pin the maths, not the draws.** Go's `math/rand` and Python's
`random` produce different sequences from the same seed, so a seed reproduces a
run *within* one implementation — it will not reproduce the platform's run.
What is guaranteed identical is everything downstream of the draw: the equity
curve, drawdown, Sharpe, Sortino, profit factor, the distribution statistics and
the percentile interpolation. Two correct implementations therefore produce
statistically equivalent distributions from different draws, which is the honest
guarantee a Monte Carlo tool can make.

## Releasing

Publishing uses PyPI Trusted Publishing, so no API token is stored here. Bump
`version` in `pyproject.toml`, tag it, cut a GitHub Release — the workflow
refuses a tag that disagrees with `pyproject.toml` and runs the vectors first.

## About

Built and maintained by [VolatiCloud](https://volaticloud.com), a managed
Freqtrade platform.

Self-hosting Freqtrade is great, and this tool is for you whether or not you ever
use anything of ours. It makes no network calls and collects nothing.

**Not affiliated with or endorsed by the Freqtrade project.** Freqtrade is an
independent open-source project; this is a third-party tool that reads its output.

## License

MIT — see [LICENSE](LICENSE).
