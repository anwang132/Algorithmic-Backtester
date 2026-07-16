import dataclasses
from datetime import date

import pytest

from backtester.events import Direction, EventType, FillEvent, MarketEvent, OrderEvent, SignalEvent


def test_market_event_is_frozen_and_tagged():
    event = MarketEvent(date(2020, 1, 1), "AAA", 1.0, 2.0, 0.5, 1.5, 1000)
    assert event.type == EventType.MARKET
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.close = 999.0  # type: ignore[misc]


def test_signal_event_carries_paired_leg_info():
    event = SignalEvent(
        date(2020, 1, 1), "A", Direction.LONG, "strat", paired_symbol="B", hedge_ratio=1.2
    )
    assert event.type == EventType.SIGNAL
    assert event.paired_symbol == "B"
    assert event.hedge_ratio == 1.2
    assert event.strength == 1.0  # default


def test_order_and_fill_event_tags():
    order = OrderEvent(date(2020, 1, 1), "A", 10, Direction.BUY)
    assert order.type == EventType.ORDER
    assert order.quantity == 10

    fill = FillEvent(date(2020, 1, 2), "A", 10, Direction.BUY, 100.0, 1.0, 0.5)
    assert fill.type == EventType.FILL
    assert fill.timestamp == date(2020, 1, 2)
