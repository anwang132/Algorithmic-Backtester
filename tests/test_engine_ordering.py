"""The explicit no-lookahead contract test for the whole engine: a signal
generated from day T's close must be filled at day T+1's open, never at T's
own close and never same-day.
"""

from datetime import date, timedelta

import pytest

from backtester.data.handler import Bar, DataHandler
from backtester.engine import BacktestEngine
from backtester.events import Direction, MarketEvent, SignalEvent
from backtester.execution.broker import SimulatedBroker
from backtester.execution.costs import CommissionModel, SlippageModel
from backtester.portfolio.portfolio import Portfolio
from backtester.strategy.base import Strategy


class _FireOnceStrategy(Strategy):
    """Test double: emits exactly one LONG signal on the Nth MarketEvent it
    sees for symbol_a, using only that day's data, then never signals again."""

    def __init__(self, symbol_a: str, symbol_b: str, fire_on_nth: int = 3):
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b
        self._count = 0
        self._fire_on_nth = fire_on_nth
        self.fired_on: date | None = None

    def on_market_event(self, event: MarketEvent) -> SignalEvent | None:
        if event.symbol != self.symbol_a:
            return None
        self._count += 1
        if self._count == self._fire_on_nth and self.fired_on is None:
            self.fired_on = event.timestamp
            return SignalEvent(
                event.timestamp,
                self.symbol_a,
                Direction.LONG,
                "test",
                paired_symbol=self.symbol_b,
                hedge_ratio=1.0,
            )
        return None


def _build_handler(tmp_path, n_days: int = 8) -> tuple[DataHandler, list[date]]:
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n_days)]
    bars_a = {
        d: Bar(open=100.0 + i, high=101.0 + i, low=99.0 + i, close=100.5 + i, volume=1000)
        for i, d in enumerate(dates)
    }
    bars_b = {
        d: Bar(open=50.0 + i, high=51.0 + i, low=49.0 + i, close=50.5 + i, volume=1000)
        for i, d in enumerate(dates)
    }
    handler = DataHandler(["A", "B"], dates[0], dates[-1], cache_dir=tmp_path)
    handler._bars = {"A": bars_a, "B": bars_b}
    return handler, dates


def test_fill_always_uses_next_trading_days_open_never_same_day_close(tmp_path):
    handler, _dates = _build_handler(tmp_path)
    strategy = _FireOnceStrategy("A", "B", fire_on_nth=3)
    portfolio = Portfolio(initial_capital=100_000, risk_pct_per_leg=0.10)
    # Zero commission/slippage so the fill price can be compared exactly
    # against the raw open price.
    broker = SimulatedBroker(CommissionModel(0.0, 0.0), SlippageModel(bps=0.0))
    engine = BacktestEngine(handler, strategy, portfolio, broker)

    result = engine.run()

    assert strategy.fired_on is not None
    signal_day = strategy.fired_on
    calendar = handler.trading_calendar
    next_day = calendar[calendar.index(signal_day) + 1]

    fills_for_a = [f for f in result.trade_log if f.symbol == "A"]
    assert len(fills_for_a) == 1
    fill = fills_for_a[0]

    # The fill is dated the NEXT trading day — never the signal day itself.
    assert fill.timestamp == next_day
    assert fill.timestamp != signal_day

    # With zero slippage, the fill price exactly equals next-day's OPEN and
    # must NOT equal the signal day's close — the direct proof there is no
    # same-bar lookahead.
    next_day_bar = handler.get_bar("A", next_day)
    signal_day_bar = handler.get_bar("A", signal_day)
    assert fill.fill_price == pytest.approx(next_day_bar.open)
    assert fill.fill_price != pytest.approx(signal_day_bar.close)


def test_pending_order_waits_for_symbol_to_actually_trade(tmp_path):
    """If a symbol has no bar on the trading day immediately after a
    signal, the order must stay queued rather than being silently dropped
    or filled against stale data."""
    handler, dates = _build_handler(tmp_path, n_days=8)
    signal_day = dates[2]
    gap_day = dates[3]
    fill_day = dates[4]
    # Remove symbol A's bar on the day right after the signal to force a gap.
    del handler._bars["A"][gap_day]

    strategy = _FireOnceStrategy("A", "B", fire_on_nth=3)
    portfolio = Portfolio(100_000, 0.10)
    broker = SimulatedBroker(CommissionModel(0.0, 0.0), SlippageModel(bps=0.0))
    engine = BacktestEngine(handler, strategy, portfolio, broker)

    result = engine.run()

    assert strategy.fired_on == signal_day
    fills_for_a = [f for f in result.trade_log if f.symbol == "A"]
    assert len(fills_for_a) == 1
    assert fills_for_a[0].timestamp == fill_day  # skipped the gap day entirely
