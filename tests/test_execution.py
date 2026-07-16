from datetime import date

import pytest

from backtester.events import Direction, OrderEvent
from backtester.execution.broker import SimulatedBroker
from backtester.execution.costs import CommissionModel, SlippageModel


def test_commission_model_fixed_plus_per_share():
    model = CommissionModel(fixed_per_trade=1.0, per_share=0.005)
    assert model.compute(100) == pytest.approx(1.0 + 100 * 0.005)
    assert model.compute(-100) == pytest.approx(1.0 + 100 * 0.005)  # abs(quantity)


def test_slippage_model_buy_worse_sell_worse():
    model = SlippageModel(bps=10.0)  # 0.10%
    assert model.apply(100.0, Direction.BUY) == pytest.approx(100.10)
    assert model.apply(100.0, Direction.SELL) == pytest.approx(99.90)


def test_broker_execute_order_uses_supplied_fill_date_and_price():
    broker = SimulatedBroker(CommissionModel(1.0, 0.005), SlippageModel(bps=10.0))
    order = OrderEvent(date(2020, 1, 1), "AAA", 100, Direction.BUY)

    fill = broker.execute_order(order, fill_date=date(2020, 1, 2), fill_reference_price=50.0)

    # Fill is dated the day it was FILLED, not the day the order originated.
    assert fill.timestamp == date(2020, 1, 2)
    assert fill.fill_price == pytest.approx(50.05)  # 50.0 * 1.001
    assert fill.commission == pytest.approx(1.0 + 100 * 0.005)
    assert fill.slippage_cost == pytest.approx(0.05 * 100)


def test_broker_sell_slips_against_the_trader():
    broker = SimulatedBroker(CommissionModel(0.0, 0.0), SlippageModel(bps=20.0))
    order = OrderEvent(date(2020, 1, 1), "AAA", -50, Direction.SELL)
    fill = broker.execute_order(order, fill_date=date(2020, 1, 2), fill_reference_price=100.0)
    assert fill.fill_price == pytest.approx(99.80)  # sell fills below the reference price
    assert fill.fill_price < 100.0
