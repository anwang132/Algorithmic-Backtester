from datetime import date

import pytest

from backtester.analytics.performance import compute_metrics
from backtester.events import Direction, FillEvent


def test_max_drawdown_known_value():
    # 100 -> 110 -> 90 -> 120: running max hits 110 at day 2, trough 90 at
    # day 3 -> drawdown = (90-110)/110 = -18.1818...%
    equity_curve = [
        (date(2020, 1, 1), 100.0),
        (date(2020, 1, 2), 110.0),
        (date(2020, 1, 3), 90.0),
        (date(2020, 1, 4), 120.0),
    ]
    metrics = compute_metrics(equity_curve, trade_log=[])
    assert metrics.max_drawdown == pytest.approx(-18.1818 / 100, abs=1e-4)


def test_total_return_and_no_trades():
    equity_curve = [(date(2020, 1, 1), 100.0), (date(2020, 1, 2), 150.0)]
    metrics = compute_metrics(equity_curve, trade_log=[])
    assert metrics.total_return == pytest.approx(0.5)
    assert metrics.num_trades == 0
    assert metrics.win_rate == 0.0
    assert metrics.avg_trade_pnl == 0.0


def test_trade_stats_realized_pnl_and_win_rate():
    # Buy 10 @ 100 (open a position), sell 10 @ 110 (close it) -> a single
    # realized winning trade of 10*(110-100) - commission.
    fills = [
        FillEvent(
            date(2020, 1, 1),
            "A",
            10,
            Direction.BUY,
            fill_price=100.0,
            commission=1.0,
            slippage_cost=0.0,
        ),
        FillEvent(
            date(2020, 1, 2),
            "A",
            -10,
            Direction.SELL,
            fill_price=110.0,
            commission=1.0,
            slippage_cost=0.0,
        ),
    ]
    equity_curve = [(date(2020, 1, 1), 1000.0), (date(2020, 1, 2), 1098.0)]

    metrics = compute_metrics(equity_curve, fills)

    assert metrics.num_trades == 1
    assert metrics.win_rate == 1.0
    assert metrics.avg_trade_pnl == pytest.approx(10 * (110 - 100) - 1.0)


def test_trade_stats_losing_trade_is_not_a_win():
    fills = [
        FillEvent(
            date(2020, 1, 1),
            "A",
            10,
            Direction.BUY,
            fill_price=100.0,
            commission=0.0,
            slippage_cost=0.0,
        ),
        FillEvent(
            date(2020, 1, 2),
            "A",
            -10,
            Direction.SELL,
            fill_price=90.0,
            commission=0.0,
            slippage_cost=0.0,
        ),
    ]
    equity_curve = [(date(2020, 1, 1), 1000.0), (date(2020, 1, 2), 900.0)]

    metrics = compute_metrics(equity_curve, fills)

    assert metrics.num_trades == 1
    assert metrics.win_rate == 0.0
    assert metrics.avg_trade_pnl == pytest.approx(-100.0)


def test_sharpe_ratio_zero_when_no_variance():
    equity_curve = [(date(2020, 1, 1), 100.0), (date(2020, 1, 2), 100.0), (date(2020, 1, 3), 100.0)]
    metrics = compute_metrics(equity_curve, trade_log=[])
    assert metrics.sharpe_ratio == 0.0


def test_compute_metrics_requires_at_least_two_points():
    with pytest.raises(ValueError):
        compute_metrics([(date(2020, 1, 1), 100.0)], trade_log=[])
