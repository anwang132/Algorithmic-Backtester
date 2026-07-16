"""Optional Claude-powered narrative performance report.

Isolation contract: this is the ONLY module in the codebase that imports
`anthropic`. It lives behind the `llm` extras group in pyproject.toml (a
plain `uv sync` never installs it) and every entry point checks
`is_available()` before importing it, so the core engine, CLI, and test
suite never require ANTHROPIC_API_KEY. Only summary statistics are sent to
the model — never raw OHLCV — to keep the prompt small and the cost bounded
to a single call per backtest run.
"""

from __future__ import annotations

import os

from backtester.analytics.performance import PerformanceMetrics
from backtester.cointegration.screener import PairResult
from backtester.events import FillEvent

MODEL = "claude-opus-4-8"


def is_available() -> bool:
    """Whether an ANTHROPIC_API_KEY is configured. Callers should check this
    before importing/calling generate_report — the LLM report is optional
    and the core pipeline must run identically without it."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def generate_report(
    in_sample: PerformanceMetrics,
    out_of_sample: PerformanceMetrics,
    trade_log: list[FillEvent],
    pair: PairResult,
    params: dict[str, float],
) -> str:
    """Requires ANTHROPIC_API_KEY (check is_available() first). Returns a
    markdown-formatted narrative report as a string."""
    if not is_available():
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set; call is_available() before generate_report()"
        )

    import anthropic  # deferred import — only reachable when this function is called

    client = anthropic.Anthropic()
    prompt = _build_prompt(in_sample, out_of_sample, trade_log, pair, params)

    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    )
    return "\n".join(block.text for block in response.content if block.type == "text")


def _build_prompt(
    in_sample: PerformanceMetrics,
    out_of_sample: PerformanceMetrics,
    trade_log: list[FillEvent],
    pair: PairResult,
    params: dict[str, float],
) -> str:
    total_commission = sum(f.commission for f in trade_log)
    total_slippage = sum(f.slippage_cost for f in trade_log)

    def _metrics_block(label: str, m: PerformanceMetrics) -> str:
        return (
            f"{label}:\n"
            f"  Sharpe ratio: {m.sharpe_ratio:.3f}\n"
            f"  Max drawdown: {m.max_drawdown:.2%}\n"
            f"  CAGR: {m.cagr:.2%}\n"
            f"  Total return: {m.total_return:.2%}\n"
            f"  Win rate: {m.win_rate:.2%}\n"
            f"  # realized trades: {m.num_trades}\n"
            f"  Avg P&L per trade: ${m.avg_trade_pnl:,.2f}\n"
        )

    return f"""You are reviewing the results of a systematic pairs-trading backtest. \
Write a concise (under 500 words) markdown report analyzing these results for a \
technical audience (a hiring manager or fellow quant reviewing this as a portfolio \
project). Be honest and specific — call out overfitting risk if the out-of-sample \
numbers degrade meaningfully from in-sample, and don't oversell the results.

## Strategy
Cointegration-based mean-reversion pairs trade: {pair.dependent} vs {pair.independent} \
(Engle-Granger p-value: {pair.p_value:.4g}, hedge ratio: {pair.hedge_ratio:.4f}).
Frozen parameters (selected via in-sample grid search only): {params}

## In-sample vs out-of-sample performance
{_metrics_block("In-sample (4-year calibration window)", in_sample)}
{_metrics_block("Out-of-sample (2-year holdout window, zero re-fitting)", out_of_sample)}

## Transaction costs incurred
Total commission paid: ${total_commission:,.2f}
Total slippage cost: ${total_slippage:,.2f}

Cover: (1) whether the strategy shows genuine statistical edge vs. curve-fit noise, \
(2) how the out-of-sample Sharpe/drawdown compare to in-sample and what that implies, \
(3) whether the risk-adjusted returns justify the transaction costs incurred, and \
(4) one or two concrete suggestions for what to try next (e.g. wider universe, \
different rebalance frequency, alternative cointegration window)."""
