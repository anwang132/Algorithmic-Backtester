"""Simulated broker: turns an OrderEvent into a FillEvent, applying
commission and slippage. Fills always use the price the caller supplies —
the engine is responsible for ensuring that price is the NEXT trading day's
open relative to the signal that produced the order (see engine.py), never
the same-day close the signal was computed from.
"""

from __future__ import annotations

from datetime import date

from backtester.events import FillEvent, OrderEvent
from backtester.execution.costs import CommissionModel, SlippageModel


class SimulatedBroker:
    def __init__(
        self,
        commission_model: CommissionModel | None = None,
        slippage_model: SlippageModel | None = None,
    ) -> None:
        self.commission_model = commission_model or CommissionModel()
        self.slippage_model = slippage_model or SlippageModel()

    def execute_order(
        self, order: OrderEvent, fill_date: date, fill_reference_price: float
    ) -> FillEvent:
        """`fill_date`/`fill_reference_price` must be the NEXT trading day's
        date and open price relative to `order.timestamp` (the day the
        originating signal was generated) — enforced by the caller
        (BacktestEngine). The FillEvent carries `fill_date`, not
        `order.timestamp`, so downstream code (and tests) can directly
        verify the one-trading-day gap between signal and fill.
        """
        fill_price = self.slippage_model.apply(fill_reference_price, order.direction)
        commission = self.commission_model.compute(order.quantity)
        slippage_cost = abs(fill_price - fill_reference_price) * abs(order.quantity)
        return FillEvent(
            timestamp=fill_date,
            symbol=order.symbol,
            quantity=order.quantity,
            direction=order.direction,
            fill_price=fill_price,
            commission=commission,
            slippage_cost=slippage_cost,
        )
