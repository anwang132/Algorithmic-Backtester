from datetime import date

import pytest

from backtester.events import Direction, FillEvent, SignalEvent
from backtester.portfolio.portfolio import Portfolio


def test_entry_orders_are_dollar_neutral_two_legs():
    p = Portfolio(initial_capital=100_000, risk_pct_per_leg=0.10)
    p.mark_to_market("A", 50.0)
    p.mark_to_market("B", 25.0)
    signal = SignalEvent(
        date(2020, 1, 1), "A", Direction.LONG, "test", paired_symbol="B", hedge_ratio=2.0
    )

    orders = p.generate_orders(signal)

    assert len(orders) == 2
    order_a, order_b = orders
    assert order_a.symbol == "A" and order_a.direction == Direction.BUY and order_a.quantity > 0
    assert order_b.symbol == "B" and order_b.direction == Direction.SELL and order_b.quantity < 0

    dollars_a = order_a.quantity * 50.0
    dollars_b = abs(order_b.quantity) * 25.0
    # Dollar-neutral within integer-share-count rounding, NOT sized off the
    # raw hedge ratio applied directly to share counts.
    assert dollars_a == pytest.approx(dollars_b, rel=0.1)


def test_short_spread_signal_shorts_a_and_buys_b():
    p = Portfolio(100_000, 0.10)
    p.mark_to_market("A", 50.0)
    p.mark_to_market("B", 25.0)
    signal = SignalEvent(
        date(2020, 1, 1), "A", Direction.SHORT, "test", paired_symbol="B", hedge_ratio=2.0
    )
    orders = p.generate_orders(signal)
    order_a, order_b = orders
    assert order_a.direction == Direction.SELL and order_a.quantity < 0
    assert order_b.direction == Direction.BUY and order_b.quantity > 0


def test_exit_orders_flatten_current_positions():
    p = Portfolio(100_000, 0.1)
    p.positions = {"A": 200, "B": -100}
    signal = SignalEvent(date(2020, 1, 1), "A", Direction.EXIT, "test", paired_symbol="B")

    orders = p.generate_orders(signal)

    assert {(o.symbol, o.quantity, o.direction) for o in orders} == {
        ("A", -200, Direction.SELL),
        ("B", 100, Direction.BUY),
    }


def test_exit_with_no_open_position_produces_no_orders():
    p = Portfolio(100_000, 0.1)
    signal = SignalEvent(date(2020, 1, 1), "A", Direction.EXIT, "test", paired_symbol="B")
    assert p.generate_orders(signal) == []


def test_update_from_fill_updates_cash_and_positions():
    p = Portfolio(100_000, 0.1)
    fill = FillEvent(
        date(2020, 1, 1),
        "A",
        100,
        Direction.BUY,
        fill_price=50.0,
        commission=1.5,
        slippage_cost=0.5,
    )

    p.update_from_fill(fill)

    assert p.positions["A"] == 100
    assert p.cash == pytest.approx(100_000 - 100 * 50.0 - 1.5)
    assert p.trade_log == [fill]


def test_equity_curve_matches_manual_calculation():
    p = Portfolio(10_000, 0.1)
    p.mark_to_market("A", 100.0)
    p.record_equity_snapshot(date(2020, 1, 1))  # no positions yet -> equity == cash

    p.update_from_fill(
        FillEvent(
            date(2020, 1, 2),
            "A",
            10,
            Direction.BUY,
            fill_price=100.0,
            commission=1.0,
            slippage_cost=0.0,
        )
    )
    p.mark_to_market("A", 110.0)
    p.record_equity_snapshot(date(2020, 1, 2))

    assert p.equity_curve[0] == (date(2020, 1, 1), 10_000.0)
    expected_cash = 10_000 - 10 * 100.0 - 1.0
    expected_equity = expected_cash + 10 * 110.0
    assert p.equity_curve[1] == (date(2020, 1, 2), pytest.approx(expected_equity))
