"""CLI orchestration: the only place that wires together data loading,
in-sample cointegration screening + hyperparameter selection, a single
continuous backtest run over [in_sample_start, out_of_sample_end], and
metrics/chart/report output.

Methodology: the pair, its hedge ratio, and the strategy's z_window/entry_z/
exit_z are all selected using ONLY the in-sample window, then frozen and
applied unchanged for the full run — including the out-of-sample period.
Selection itself uses an internal train/validation split of the in-sample
window (roughly the first half vs. the second half) and scores each
candidate by the WORSE of its two halves' Sharpe ratios, not the whole-
window Sharpe. This favors pair/parameter combinations that hold up across
two different historical stretches over ones that were simply the best
single fit to one continuous stretch — which is what drives a large
in-sample-to-out-of-sample gap. The out-of-sample window itself is never
touched until the final frozen run, and in-sample/out-of-sample metrics are
still reported side by side so any remaining degradation is visible, not
hidden.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

from backtester.analytics.performance import PerformanceMetrics, compute_metrics
from backtester.cointegration.screener import PairResult, scan_universe, select_pairs
from backtester.config import BacktesterConfig, load_config
from backtester.data.handler import DataHandler
from backtester.engine import BacktestEngine, BacktestResult
from backtester.execution.broker import SimulatedBroker
from backtester.execution.costs import CommissionModel, SlippageModel
from backtester.portfolio.portfolio import Portfolio
from backtester.strategy.pairs_mean_reversion import PairsMeanReversionStrategy
from backtester.visualization.plots import save_all_charts


def _build_broker(config: BacktesterConfig) -> SimulatedBroker:
    return SimulatedBroker(
        commission_model=CommissionModel(
            fixed_per_trade=config.costs.commission_fixed,
            per_share=config.costs.commission_per_share,
        ),
        slippage_model=SlippageModel(bps=config.costs.slippage_bps),
    )


def _run_backtest(
    data_handler: DataHandler,
    pair: PairResult,
    params: dict[str, float],
    stop_z: float,
    config: BacktesterConfig,
) -> tuple[BacktestResult, PairsMeanReversionStrategy]:
    strategy = PairsMeanReversionStrategy(
        symbol_a=pair.dependent,
        symbol_b=pair.independent,
        hedge_ratio=pair.hedge_ratio,
        z_window=int(params["z_window"]),
        entry_z=float(params["entry_z"]),
        exit_z=float(params["exit_z"]),
        stop_z=stop_z,
    )
    portfolio = Portfolio(config.capital.initial, config.capital.risk_pct_per_leg)
    engine = BacktestEngine(data_handler, strategy, portfolio, _build_broker(config))
    return engine.run(), strategy


def _split_in_sample_dates(config: BacktesterConfig) -> tuple[date, date, date, date]:
    """Split the in-sample window into two contiguous, non-overlapping
    halves for an internal train/validation robustness check. Used only to
    SELECT the pair and hyperparameters — the out-of-sample window is
    untouched by this split."""
    start = config.dates.in_sample_start
    end = config.dates.in_sample_end
    midpoint = date.fromordinal(start.toordinal() + (end.toordinal() - start.toordinal()) // 2)
    return start, midpoint, midpoint + timedelta(days=1), end


def _robustness_score(
    pair: PairResult,
    params: dict[str, float],
    config: BacktesterConfig,
    train_start: date,
    train_end: date,
    val_start: date,
    val_end: date,
) -> float:
    """Backtest a (pair, params) combination separately on two different
    historical sub-periods of the in-sample window and return the WORSE of
    the two Sharpe ratios. Selecting by this min-of-two score (instead of a
    single whole-in-sample-window Sharpe) favors combinations that hold up
    across more than one historical stretch, which is what a genuine edge
    should do — a combination that only excels in one continuous stretch is
    exactly the pattern that produces a steep in-sample-to-out-of-sample
    dropoff."""
    train_handler = DataHandler([pair.dependent, pair.independent], train_start, train_end)
    train_handler.load_or_download()
    val_handler = DataHandler([pair.dependent, pair.independent], val_start, val_end)
    val_handler.load_or_download()

    train_result, _ = _run_backtest(train_handler, pair, params, config.strategy.stop_z, config)
    val_result, _ = _run_backtest(val_handler, pair, params, config.strategy.stop_z, config)

    train_sharpe = compute_metrics(train_result.equity_curve, train_result.trade_log).sharpe_ratio
    val_sharpe = compute_metrics(val_result.equity_curve, val_result.trade_log).sharpe_ratio
    return min(train_sharpe, val_sharpe)


def _grid_search_params(
    pair: PairResult,
    config: BacktesterConfig,
    train_start: date,
    train_end: date,
    val_start: date,
    val_end: date,
) -> tuple[dict[str, float], float]:
    """Search strategy hyperparameters, selecting by the train/validation
    robustness score (not a single whole-window Sharpe). Returns the frozen
    params and their robustness score."""
    grid = config.strategy.param_grid
    best_params: dict[str, float] | None = None
    best_score = float("-inf")

    for z_window, entry_z, exit_z in itertools.product(
        grid["z_window"], grid["entry_z"], grid["exit_z"]
    ):
        if not (0 < exit_z < entry_z < config.strategy.stop_z):
            continue
        candidate = {"z_window": z_window, "entry_z": entry_z, "exit_z": exit_z}
        score = _robustness_score(
            pair, candidate, config, train_start, train_end, val_start, val_end
        )
        if score > best_score:
            best_score = score
            best_params = candidate

    if best_params is None:
        # Grid was empty or entirely invalid — fall back to the configured defaults.
        best_params = {
            "z_window": config.strategy.z_window,
            "entry_z": config.strategy.entry_z,
            "exit_z": config.strategy.exit_z,
        }
        best_score = float("-inf")
    return best_params, best_score


def _select_pair_and_params(
    pair_results: list[PairResult], config: BacktesterConfig
) -> tuple[PairResult, dict[str, float]]:
    """Two-stage, in-sample-only selection. Stage 1 (statistical
    significance): keep every pair cointegrated below the configured
    p-value threshold — not just the single most significant one. Stage 2
    (economic significance + robustness): grid-search strategy
    hyperparameters for EACH surviving candidate using the train/validation
    split, and keep the (pair, params) combination with the best
    robustness score. Both stages use only the in-sample window;
    out-of-sample is left untouched until the final frozen run.
    """
    candidates = select_pairs(
        pair_results, config.cointegration.p_value_threshold, top_n=len(pair_results)
    )
    if not candidates:
        raise RuntimeError(
            "No pair in the universe was cointegrated below "
            f"p={config.cointegration.p_value_threshold} on the in-sample window. "
            "Try a larger universe, a longer in-sample window, or a looser threshold."
        )

    train_start, train_end, val_start, val_end = _split_in_sample_dates(config)

    best: tuple[float, PairResult, dict[str, float]] | None = None
    for pair in candidates:
        params, score = _grid_search_params(
            pair, config, train_start, train_end, val_start, val_end
        )
        print(
            f"  {pair.dependent}~{pair.independent} (p={pair.p_value:.4g}): "
            f"robustness score (worse of two in-sample halves) {score:.3f}"
        )
        if best is None or score > best[0]:
            best = (score, pair, params)

    assert best is not None
    _, best_pair, best_params = best
    return best_pair, best_params


def run_full_pipeline(config: BacktesterConfig, with_llm_report: bool = False) -> dict:
    print(
        f"Loading data for {len(config.universe)} symbols "
        f"[{config.dates.in_sample_start} .. {config.dates.out_of_sample_end}] ..."
    )
    full_handler = DataHandler(
        config.universe, config.dates.in_sample_start, config.dates.out_of_sample_end
    )
    full_handler.load_or_download()

    print("Scanning universe for cointegrated pairs (in-sample window only) ...")
    in_sample_prices = full_handler.close_price_frame(
        start=config.dates.in_sample_start, end=config.dates.in_sample_end
    )
    pair_results = scan_universe(
        in_sample_prices, config.universe, config.cointegration.min_observations
    )

    print(
        "Selecting pair + hyperparameters by train/validation robustness "
        "among all cointegrated candidates (in-sample window only) ..."
    )
    pair, best_params = _select_pair_and_params(pair_results, config)
    print(
        f"  Selected: {pair.dependent} ~ {pair.independent}  "
        f"(p={pair.p_value:.4g}, hedge_ratio={pair.hedge_ratio:.4f})"
    )
    print(f"  Frozen params: {best_params}")

    print("Running full backtest with frozen parameters ...")
    result, strategy = _run_backtest(
        full_handler, pair, best_params, config.strategy.stop_z, config
    )

    in_sample_equity = [(d, v) for d, v in result.equity_curve if d <= config.dates.in_sample_end]
    out_of_sample_equity = [
        (d, v) for d, v in result.equity_curve if d >= config.dates.in_sample_end
    ]
    in_sample_trades = [f for f in result.trade_log if f.timestamp <= config.dates.in_sample_end]
    out_of_sample_trades = [f for f in result.trade_log if f.timestamp > config.dates.in_sample_end]

    in_sample_metrics = compute_metrics(in_sample_equity, in_sample_trades)
    out_of_sample_metrics = compute_metrics(out_of_sample_equity, out_of_sample_trades)

    _print_summary(pair, best_params, in_sample_metrics, out_of_sample_metrics)

    results_dir = Path(config.output.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "pair": {
            "dependent": pair.dependent,
            "independent": pair.independent,
            "p_value": pair.p_value,
            "hedge_ratio": pair.hedge_ratio,
        },
        "strategy_params": {**best_params, "stop_z": config.strategy.stop_z},
        "in_sample": asdict(in_sample_metrics),
        "out_of_sample": asdict(out_of_sample_metrics),
    }
    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {metrics_path}")

    charts = save_all_charts(
        equity_curve=result.equity_curve,
        signal_log=strategy.signal_log,
        history=strategy.history,
        symbol_a=pair.dependent,
        symbol_b=pair.independent,
        entry_z=best_params["entry_z"],
        exit_z=best_params["exit_z"],
        charts_dir=config.output.charts_dir,
        out_of_sample_start=config.dates.out_of_sample_start,
        in_sample_sharpe=in_sample_metrics.sharpe_ratio,
        out_of_sample_sharpe=out_of_sample_metrics.sharpe_ratio,
    )
    for path in charts.values():
        print(f"Wrote {path}")

    if with_llm_report:
        _maybe_generate_llm_report(
            pair, best_params, in_sample_metrics, out_of_sample_metrics, result
        )

    return payload


def _print_summary(
    pair: PairResult,
    params: dict[str, float],
    in_sample: PerformanceMetrics,
    out_of_sample: PerformanceMetrics,
) -> None:
    print("\n" + "=" * 60)
    print(f"Pair: {pair.dependent} ~ {pair.independent}  (p={pair.p_value:.4g})")
    print(f"Params: {params}")
    print("-" * 60)
    print(f"{'Metric':<20}{'In-sample':>18}{'Out-of-sample':>20}")
    for label, key in (
        ("Sharpe ratio", "sharpe_ratio"),
        ("Max drawdown", "max_drawdown"),
        ("CAGR", "cagr"),
        ("Total return", "total_return"),
        ("Win rate", "win_rate"),
        ("# trades", "num_trades"),
    ):
        is_val = getattr(in_sample, key)
        oos_val = getattr(out_of_sample, key)
        print(f"{label:<20}{is_val:>18.4g}{oos_val:>20.4g}")
    print("=" * 60)


def _maybe_generate_llm_report(
    pair: PairResult,
    params: dict[str, float],
    in_sample: PerformanceMetrics,
    out_of_sample: PerformanceMetrics,
    result: BacktestResult,
) -> None:
    try:
        from backtester.report.llm_report import generate_report, is_available
    except ImportError:
        print("\n[LLM report] `anthropic` is not installed (uv sync --extra llm) — skipping.")
        return
    if not is_available():
        print("\n[LLM report] ANTHROPIC_API_KEY not set — skipping.")
        return
    print("\nGenerating LLM performance report ...")
    report_text = generate_report(in_sample, out_of_sample, result.trade_log, pair, params)
    report_path = Path("outputs/results/llm_report.md")
    report_path.write_text(report_text)
    print(f"Wrote {report_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="backtester", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full backtest pipeline")
    run_parser.add_argument("--config", default="config/default.yaml", help="Path to config YAML")
    run_parser.add_argument(
        "--with-llm-report",
        action="store_true",
        help="Generate an optional Claude-powered narrative report (requires ANTHROPIC_API_KEY)",
    )

    screen_parser = subparsers.add_parser(
        "screen", help="Print the ranked cointegration screen for the in-sample window, no backtest"
    )
    screen_parser.add_argument(
        "--config", default="config/default.yaml", help="Path to config YAML"
    )

    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except Exception as exc:  # noqa: BLE001 - surface config errors clearly to the user
        print(f"Failed to load config {args.config!r}: {exc}", file=sys.stderr)
        return 1

    if args.command == "run":
        run_full_pipeline(config, with_llm_report=args.with_llm_report)
        return 0

    if args.command == "screen":
        handler = DataHandler(
            config.universe, config.dates.in_sample_start, config.dates.in_sample_end
        )
        handler.load_or_download()
        prices = handler.close_price_frame()
        results = scan_universe(prices, config.universe, config.cointegration.min_observations)
        print(f"{'A':<8}{'B':<8}{'p-value':>12}{'hedge_ratio':>14}")
        for r in results[:20]:
            print(f"{r.dependent:<8}{r.independent:<8}{r.p_value:>12.4g}{r.hedge_ratio:>14.4f}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
