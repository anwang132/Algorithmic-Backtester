"""Event types for the four-stage market data -> signal -> order -> fill pipeline.

All events are immutable (frozen dataclasses) so that once an event is placed
on the queue, nothing downstream can silently mutate the record a decision was
based on. Timestamps are plain `datetime.date` since the engine operates on
daily OHLCV bars, not intraday ticks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class EventType(Enum):
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"


class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT = "EXIT"
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True, slots=True)
class MarketEvent:
    """One day's OHLCV bar for a single symbol."""

    timestamp: date
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    type: EventType = field(default=EventType.MARKET, init=False)


@dataclass(frozen=True, slots=True)
class SignalEvent:
    """A trading signal emitted by a Strategy from data up to and including
    `timestamp`'s close. Carries both legs of a pairs trade explicitly (when
    applicable) so downstream consumers (Portfolio) never need to reach back
    into Strategy internals to know what to size.
    """

    timestamp: date
    symbol: str
    direction: Direction
    strategy_id: str
    strength: float = 1.0
    paired_symbol: str | None = None
    hedge_ratio: float | None = None
    type: EventType = field(default=EventType.SIGNAL, init=False)


@dataclass(frozen=True, slots=True)
class OrderEvent:
    """A sized order to be filled at the NEXT trading day's open."""

    timestamp: date
    symbol: str
    quantity: int  # signed: positive = buy, negative = sell/short
    direction: Direction  # BUY or SELL
    order_type: str = "MARKET"
    type: EventType = field(default=EventType.ORDER, init=False)


@dataclass(frozen=True, slots=True)
class FillEvent:
    """The realized result of executing an OrderEvent against a simulated
    broker, including transaction costs.
    """

    timestamp: date
    symbol: str
    quantity: int
    direction: Direction
    fill_price: float
    commission: float
    slippage_cost: float
    type: EventType = field(default=EventType.FILL, init=False)


Event = MarketEvent | SignalEvent | OrderEvent | FillEvent
