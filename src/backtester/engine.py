"""BacktestEngine: the four-stage market data -> signal -> order -> fill loop.

No-lookahead contract: a signal generated from day T's close is sized into
an OrderEvent tagged with day T, but that order is deliberately NOT filled
during day T's processing. It sits in `pending_orders` and is only executed
at the START of the next trading day the order's symbol actually trades,
priced at THAT day's open. This is what makes "no lookahead bias" a
verifiable property of the engine rather than a design intention: a fill can
never use information (a closing price) that wasn't yet knowable at the time
the fill price was determined.

Per-day sequence:
  1. Fill phase   — execute any orders left over from prior days, priced at
                     today's open, before anything else happens today.
  2. Market phase — stream today's MarketEvent for every symbol (fixed
                     order), mark each to market, and let the strategy react;
                     any resulting orders are queued for a FUTURE day, not
                     filled today.
  3. Snapshot     — record today's equity using today's close. Positions are
                     unchanged in this step (the new orders haven't filled).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from backtester.data.handler import DataHandler
from backtester.events import FillEvent, MarketEvent, OrderEvent
from backtester.execution.broker import SimulatedBroker
from backtester.portfolio.portfolio import Portfolio
from backtester.strategy.base import Strategy


@dataclass
class BacktestResult:
    equity_curve: list[tuple[date, float]]
    trade_log: list[FillEvent]


class BacktestEngine:
    def __init__(
        self,
        data_handler: DataHandler,
        strategy: Strategy,
        portfolio: Portfolio,
        broker: SimulatedBroker,
    ) -> None:
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.broker = broker

    def run(self) -> BacktestResult:
        pending_orders: list[OrderEvent] = []
        calendar = self.data_handler.trading_calendar
        if not calendar:
            raise RuntimeError(
                "trading calendar is empty — call data_handler.load_or_download() first"
            )

        for current_date in calendar:
            pending_orders = self._fill_pending_orders(pending_orders, current_date)
            new_orders = self._stream_market_events(current_date)
            pending_orders.extend(new_orders)
            self.portfolio.record_equity_snapshot(current_date)

        return BacktestResult(
            equity_curve=self.portfolio.equity_curve,
            trade_log=self.portfolio.trade_log,
        )

    def _fill_pending_orders(
        self, pending_orders: list[OrderEvent], current_date: date
    ) -> list[OrderEvent]:
        """Fill orders queued from a prior day, priced at TODAY's open.
        Orders whose symbol didn't trade today stay queued for the next day
        it does."""
        still_pending: list[OrderEvent] = []
        for order in pending_orders:
            bar = self.data_handler.get_bar(order.symbol, current_date)
            if bar is None:
                still_pending.append(order)
                continue
            fill = self.broker.execute_order(
                order, fill_date=current_date, fill_reference_price=bar.open
            )
            self.portfolio.update_from_fill(fill)
        return still_pending

    def _stream_market_events(self, current_date: date) -> list[OrderEvent]:
        """Push today's bar for every symbol through mark-to-market and the
        strategy, in a fixed universe order, collecting any resulting
        orders (queued for a future day, never filled today)."""
        new_orders: list[OrderEvent] = []
        for symbol in self.data_handler.universe:
            bar = self.data_handler.get_bar(symbol, current_date)
            if bar is None:
                continue
            self.portfolio.mark_to_market(symbol, bar.close)
            market_event = MarketEvent(
                timestamp=current_date,
                symbol=symbol,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
            )
            signal = self.strategy.on_market_event(market_event)
            if signal is not None:
                new_orders.extend(self.portfolio.generate_orders(signal))
        return new_orders
