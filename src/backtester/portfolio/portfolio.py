"""Portfolio: cash/position bookkeeping, order sizing, and the equity curve.

Sizing is fixed-fractional + dollar-neutral: each leg's target dollar
exposure is `equity * risk_pct_per_leg`, and the paired leg's share count is
derived from that SAME dollar target (not from applying the OLS hedge ratio
directly to share counts) so both legs of a pairs trade carry roughly equal
dollar exposure. Applying a price-ratio hedge ratio directly to share counts
is a common bug that leaves a "pairs" trade accidentally net-directional
whenever the two legs trade at very different price levels.
"""

from __future__ import annotations

from datetime import date

from backtester.events import Direction, FillEvent, OrderEvent, SignalEvent


class Portfolio:
    def __init__(self, initial_capital: float, risk_pct_per_leg: float = 0.10) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if not (0 < risk_pct_per_leg <= 1):
            raise ValueError("risk_pct_per_leg must be in (0, 1]")
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.risk_pct_per_leg = risk_pct_per_leg
        self.positions: dict[str, int] = {}
        self._latest_price: dict[str, float] = {}
        self.equity_curve: list[tuple[date, float]] = []
        self.trade_log: list[FillEvent] = []

    def mark_to_market(self, symbol: str, price: float) -> None:
        """Record the latest known price for a symbol. Does NOT change
        positions or cash — used purely to value the book for equity calc
        and for sizing the *next* order."""
        self._latest_price[symbol] = price

    def current_equity(self) -> float:
        holdings_value = sum(
            qty * self._latest_price.get(sym, 0.0) for sym, qty in self.positions.items()
        )
        return self.cash + holdings_value

    def record_equity_snapshot(self, dt: date) -> None:
        self.equity_curve.append((dt, self.current_equity()))

    def generate_orders(self, signal: SignalEvent) -> list[OrderEvent]:
        """Translate a SignalEvent into 0-2 sized OrderEvents. Entry signals
        (LONG/SHORT) produce a dollar-neutral two-leg order pair when the
        signal names a paired_symbol; EXIT signals flatten whatever is
        currently held in the named symbol(s)."""
        if signal.direction == Direction.EXIT:
            return self._generate_exit_orders(signal)
        return self._generate_entry_orders(signal)

    def _generate_exit_orders(self, signal: SignalEvent) -> list[OrderEvent]:
        orders: list[OrderEvent] = []
        for symbol in (signal.symbol, signal.paired_symbol):
            if symbol is None:
                continue
            held = self.positions.get(symbol, 0)
            if held == 0:
                continue
            direction = Direction.SELL if held > 0 else Direction.BUY
            orders.append(OrderEvent(signal.timestamp, symbol, -held, direction))
        return orders

    def _generate_entry_orders(self, signal: SignalEvent) -> list[OrderEvent]:
        price_a = self._latest_price.get(signal.symbol)
        if price_a is None or price_a <= 0:
            return []

        equity = self.current_equity()
        target_dollars = equity * self.risk_pct_per_leg
        qty_a = int(target_dollars / price_a)
        if qty_a == 0:
            return []

        # LONG spread = long A, short B. SHORT spread = short A, long B.
        a_direction = Direction.BUY if signal.direction == Direction.LONG else Direction.SELL
        signed_qty_a = qty_a if a_direction == Direction.BUY else -qty_a
        orders = [OrderEvent(signal.timestamp, signal.symbol, signed_qty_a, a_direction)]

        if signal.paired_symbol and signal.hedge_ratio:
            price_b = self._latest_price.get(signal.paired_symbol)
            if price_b and price_b > 0:
                # Dollar-neutral: size leg B off the SAME dollar notional as
                # leg A, not off the raw OLS price-ratio hedge ratio.
                qty_b = int((qty_a * price_a) / price_b)
                if qty_b != 0:
                    b_direction = Direction.SELL if a_direction == Direction.BUY else Direction.BUY
                    signed_qty_b = qty_b if b_direction == Direction.BUY else -qty_b
                    orders.append(
                        OrderEvent(
                            signal.timestamp, signal.paired_symbol, signed_qty_b, b_direction
                        )
                    )
        return orders

    def update_from_fill(self, fill: FillEvent) -> None:
        self.cash -= fill.quantity * fill.fill_price + fill.commission
        self.positions[fill.symbol] = self.positions.get(fill.symbol, 0) + fill.quantity
        self.trade_log.append(fill)
