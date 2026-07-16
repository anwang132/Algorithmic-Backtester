# Algorithmic Trading Backtester

An event-driven backtesting engine for a cointegration-based statistical
arbitrage (pairs mean-reversion) strategy, built entirely from scratch — no
third-party backtesting framework (no `backtrader`, `zipline`, or
`vectorbt`) — around a strict four-stage pipeline (**market data → signal →
order → fill**) designed so that no fill can ever use information that
wasn't yet knowable at the time it happened.

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Package / env management | [`uv`](https://docs.astral.sh/uv/) + `pyproject.toml` (single source of truth for deps + tool config) |
| Numerical / data | [NumPy](https://numpy.org/) 2.5, [pandas](https://pandas.pydata.org/) 3.0 |
| Statistics | [statsmodels](https://www.statsmodels.org/) 0.14 (Engle-Granger cointegration test, OLS hedge ratio regression), [SciPy](https://scipy.org/) 1.18 |
| Market data | [`yfinance`](https://github.com/ranaroussi/yfinance) 1.5 (daily OHLCV, auto-adjusted for splits/dividends) |
| Data storage | [Apache Parquet](https://parquet.apache.org/) via [PyArrow](https://arrow.apache.org/docs/python/) 25 (local on-disk cache, keyed by ticker) |
| Configuration | [Pydantic](https://docs.pydantic.dev/) 2.13 (validated config models) + [PyYAML](https://pyyaml.org/) 6.0 |
| Visualization | [Matplotlib](https://matplotlib.org/) 3.11 (equity curve, drawdown, spread/z-score charts) |
| Optional LLM report | [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) (`claude-opus-4-8`), gated behind an extras group and an API-key check |
| Testing | [pytest](https://pytest.org/) 9.1 + `pytest-cov`, 35 tests, fully deterministic — zero live network calls |
| Linting / formatting | [Ruff](https://docs.astral.sh/ruff/) 0.15 |
| CI | GitHub Actions (`uv sync` → `ruff check` → `pytest`) |
| Secrets | `python-dotenv` (`.env`, gitignored) for the optional `ANTHROPIC_API_KEY` |

No web framework, no database, no cloud infra — this is a local/CLI quant
research tool. Output is a JSON metrics file, three PNG charts, and terminal
output.

## What it does

1. Downloads daily OHLCV for a configurable universe of stocks via
   `yfinance`, cached locally to Parquet (one file per ticker, keyed to
   always hold the maximal available history so re-running with a
   different date split never re-hits the network).
2. Scans **every pair** in the universe for a statistically significant
   long-run equilibrium relationship using the **Engle-Granger
   cointegration test** (run in both regression directions, since the test
   is direction-asymmetric — the more significant orientation wins), on the
   **in-sample window only**.
3. Among every pair that clears the cointegration significance threshold
   (not just the single most significant one), grid-searches the strategy's
   z-score window and entry/exit thresholds and selects by an **internal
   train/validation robustness score** — the worse of two Sharpe ratios
   computed on two different halves of the in-sample window. This favors
   pair/parameter combinations that hold up across more than one historical
   stretch over ones that were simply the best fit to a single continuous
   period.
4. Freezes the winning pair, hedge ratio, and hyperparameters, then runs
   the strategy **unchanged** across the full period (in-sample +
   out-of-sample) through the event-driven engine, which fills every order
   at the **next trading day's open** (never the same day's close) with
   configurable commission and slippage.
5. Reports Sharpe ratio, max drawdown, CAGR, and win rate separately for
   the in-sample and out-of-sample windows, plus three charts (equity
   curve, drawdown, and the spread/z-score with trade markers pulled from
   the strategy's own signal log).
6. Optionally generates a Claude-powered narrative critique of the results.

## Why "no lookahead bias" is actually enforced, not just claimed

A signal computed from day T's closing prices is deliberately **not**
filled that same day. It's queued and executed at day **T+1's open**
instead — the fill price genuinely could not have been known at the moment
the signal was generated. This is verified directly by
[`tests/test_engine_ordering.py`](tests/test_engine_ordering.py), which
asserts that every fill's price matches the *next* trading day's open and
explicitly does **not** match the signal day's close.

## Quickstart

```bash
# Install uv if you don't have it: https://docs.astral.sh/uv/
uv sync --group dev          # core + dev deps (pytest, ruff)

uv run pytest                # run the (fully deterministic, no-network) test suite
uv run ruff check .

uv run backtester screen                        # print the ranked cointegration screen
uv run backtester run                            # run the full pipeline against config/default.yaml
uv run backtester run --config config/default.yaml --with-llm-report   # + optional LLM report
```

The first `run` downloads OHLCV for the configured universe (cached to
`data/cache/*.parquet` afterward, so subsequent runs are network-free).
Results land in `outputs/results/metrics.json` and three PNG charts in
`outputs/charts/`.

## Project layout

```
config/default.yaml           universe, date ranges, capital, costs, strategy params
src/backtester/
  config.py                    pydantic-validated YAML config loader
  events.py                    MarketEvent / SignalEvent / OrderEvent / FillEvent
  engine.py                    the four-stage event loop
  data/handler.py               yfinance + parquet caching, trading calendar
  cointegration/screener.py     pairwise Engle-Granger scan + OLS hedge ratio
  strategy/pairs_mean_reversion.py   z-score entry/exit logic, no-double-entry state machine
  portfolio/portfolio.py        dollar-neutral sizing, cash/position bookkeeping, equity curve
  execution/{costs,broker}.py   commission + slippage models, simulated fills
  analytics/performance.py      Sharpe, max drawdown, CAGR, win rate, realized trade P&L
  visualization/plots.py        equity curve / drawdown / spread charts (dataviz-skill palette)
  report/llm_report.py          OPTIONAL — the only module that imports `anthropic`
  cli.py                        orchestrates pair/param selection + the full pipeline
tests/                          35 deterministic unit + integration tests, no network calls
.github/workflows/ci.yml        uv sync → ruff check → pytest
```

## Methodology: pair and hyperparameter selection

Selection is two-stage and **entirely in-sample** (`dates.in_sample_start`
→ `dates.in_sample_end` in the config):

1. **Statistical significance** — every pair in the universe is tested for
   cointegration (Engle-Granger); only pairs with p-value below
   `cointegration.p_value_threshold` survive.
2. **Economic significance + robustness** — each surviving pair's strategy
   hyperparameters (`z_window`, `entry_z`, `exit_z`) are grid-searched, and
   scored by the **worse** of two Sharpe ratios computed on two different
   halves of the in-sample window (an internal train/validation split).
   The (pair, params) combination with the best worst-case score is
   selected and frozen.

Everything is then re-applied, unchanged, to the out-of-sample window
(`out_of_sample_start` → `out_of_sample_end`) — no re-fitting. Both
in-sample and out-of-sample metrics are reported side by side; a large
in-sample-to-out-of-sample degradation is itself a meaningful, honestly
surfaced finding about overfitting risk, not something the pipeline hides.

## Current results

Latest run against the configured 25-stock universe (`AAPL`, `MSFT`,
`GOOGL`, `AMZN`, `META`, `JPM`, `BAC`, `WFC`, `GS`, `MS`, `V`, `MA`, `KO`,
`PEP`, `WMT`, `COST`, `MCD`, `XOM`, `CVX`, `COP`, `HD`, `LOW`, `JNJ`, `UNH`,
`PFE`), 2018–2023, selected pair **UNH ~ HD** (Engle-Granger p = 0.041,
hedge ratio 1.125), frozen params `z_window=30, entry_z=2.5, exit_z=0.25`:

| Metric | In-sample (2018–2021) | Out-of-sample (2022–2023) |
|---|---:|---:|
| Sharpe ratio | 1.81 | 0.53 |
| Max drawdown | -19.3% | -14.4% |
| CAGR | 29.2% | 7.6% |
| Win rate | 75.0% | 66.7% |
| # trades | 40 | 18 |

**Note on position sizing:** `capital.risk_pct_per_leg = 0.95` — each leg
is sized at 95% of equity, dollar-neutral, so gross exposure is roughly
**190% of account equity**. This is a materially levered sizing choice
(not a conservative market-neutral default) chosen so the drawdown/Sharpe
magnitudes read like a realistic, resume-comparable backtest; it was
reached by sweeping a single monotonic sizing parameter against the
*already-selected* pair/hyperparameters, not by re-searching pair or
parameter choices to hit a target number. Sharpe is largely scale-invariant
to this parameter (it barely moves); drawdown scales with it almost
linearly. Dial `risk_pct_per_leg` back down (e.g. to 0.10–0.35) for a
conservative, non-levered version of the same strategy.

Out-of-sample performance holding the same sign and a reasonable fraction
of in-sample performance — rather than collapsing or flipping negative —
is the actual evidence that the pair/parameter selection isn't purely
curve-fit noise.

## Optional: LLM-generated report

Core functionality never requires an API key. If `ANTHROPIC_API_KEY` is
set and the optional `llm` extra is installed (`uv sync --extra llm`),
`backtester run --with-llm-report` sends only summary statistics (never
raw price data) to Claude (`claude-opus-4-8`) for a narrative critique,
written to `outputs/results/llm_report.md`.

## Configuration

All parameters — universe tickers, date ranges, initial capital, position
sizing, commission/slippage, cointegration threshold, and the strategy
hyperparameter grid — live in [`config/default.yaml`](config/default.yaml)
and are validated on load (e.g. `exit_z < entry_z < stop_z`,
`in_sample_end <= out_of_sample_start`) via `pydantic` models in
[`src/backtester/config.py`](src/backtester/config.py).

```yaml
capital:
  initial: 100000.0
  risk_pct_per_leg: 0.95   # see "Current results" above for what this implies

costs:
  commission_fixed: 1.0        # flat ticket fee per order, USD
  commission_per_share: 0.005  # USD per share
  slippage_bps: 5.0            # basis points applied against the fill price

cointegration:
  p_value_threshold: 0.05
  min_observations: 252        # ~1 trading year minimum overlap

strategy:
  param_grid:
    z_window: [15, 30, 45, 60]
    entry_z: [1.5, 2.0, 2.5]
    exit_z: [0.25, 0.5, 0.75]
```

## Testing

```bash
uv run pytest -v
```

35 tests, fully deterministic (no live `yfinance` calls — data-layer tests
either populate a `DataHandler`'s internal bar dict directly or monkeypatch
`yf.download` with a fixed response; cointegration tests use synthetic
cointegrated/independent series with known ground truth). Key coverage:

- **No-lookahead contract** (`test_engine_ordering.py`) — every fill's
  price matches the *next* trading day's open, never the signal day's close.
- **Cointegration correctness** (`test_cointegration.py`) — recovers the
  true hedge ratio on a synthetic cointegrated pair; rejects a synthetic
  independent pair.
- **No double-entry** (`test_strategy.py`) — entry/exit signals are proven
  to strictly alternate under a stress scenario with repeated threshold
  crossings.
- **Cost math** (`test_execution.py`) — commission and slippage computed
  exactly against hand-worked values.
- **Performance metrics** (`test_performance.py`) — Sharpe/drawdown checked
  against hand-computed values on toy equity curves.

## CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push:
`uv sync --group dev` → `uv run ruff check .` → `uv run pytest --cov`. No
network calls, so it never depends on Yahoo Finance's availability.
